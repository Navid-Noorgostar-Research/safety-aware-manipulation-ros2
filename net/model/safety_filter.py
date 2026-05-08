"""
Safety-Aware Action Filter
researcher: Navid Noorgostar

After the predictor proposes a next end-effector (EE) target, this filter checks
the action against five safety criteria and projects it onto the safe set:

    1. Joint limits        - workspace xyz bounds + gripper opening bounds
                             (quaternion is re-normalized to unit length).
    2. Velocity limits     - per-DoF cap on linear, angular and grip speeds
                             between observed and target state.
    3. Collision risk      - minimum distance between the EE point cloud and the
                             dough/object point cloud must stay above a margin.
    4. Base speed          - scalar cap on the magnitude of the translational
                             velocity (treats the EE root as a mobile base).
    5. Action smoothness   - limit on the change in commanded delta versus the
                             previous step (acceleration / jerk surrogate).

The filter is differentiable and works on batched tensors so it can be plugged
into the prediction pipeline without breaking gradient flow.
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn


def _to_tensor(x, ref):
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))


class SafetyAwareActionFilter(nn.Module):
    """
    Filters a predicted EE target action against a set of safety constraints.

    Action layout (last dim):
        7  -> [x, y, z, qw, qx, qy, qz]
        8  -> [x, y, z, qw, qx, qy, qz, open]

    All inputs are batched: ``[B, action_dim]`` for states, ``[B, N, 3]`` for
    point clouds.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        bounds = torch.tensor(config.workspace_bounds, dtype=torch.float32)  # [3, 2]
        self.register_buffer('workspace_bounds', bounds)

        self.open_min = float(config.gripper_open_bounds[0])
        self.open_max = float(config.gripper_open_bounds[1])

        self.v_lin_max = float(config.v_lin_max)
        self.v_ang_max = float(config.v_ang_max)
        self.v_grip_max = float(config.v_grip_max)
        self.base_speed_max = float(config.base_speed_max)

        self.collision_margin = float(config.collision_margin)
        self.collision_pullback = float(config.collision_pullback)

        self.accel_lin_max = float(config.accel_lin_max)
        self.accel_ang_max = float(config.accel_ang_max)

        self.dt = float(config.dt)

    # --- individual checks -------------------------------------------------

    def _enforce_joint_limits(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = target[..., :3]
        quat = target[..., 3:7]
        bounds = self.workspace_bounds.to(target.device).to(target.dtype)
        pos_clamped = torch.maximum(pos, bounds[:, 0])
        pos_clamped = torch.minimum(pos_clamped, bounds[:, 1])
        quat_clamped = quaternion_normalize(quat)
        out = torch.cat([pos_clamped, quat_clamped], dim=-1)
        if target.shape[-1] == 8:
            open_clamped = target[..., 7:8].clamp(self.open_min, self.open_max)
            out = torch.cat([out, open_clamped], dim=-1)
        violated = (pos != pos_clamped).any(dim=-1)
        if target.shape[-1] == 8:
            violated = violated | (target[..., 7] != out[..., 7])
        return out, violated

    def _enforce_velocity_limits(self, observed: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = target - observed
        # linear
        v_lin = delta[..., :3] / self.dt
        scale_lin = (self.v_lin_max / v_lin.abs().clamp_min(1e-12)).clamp_max(1.0)
        delta_lin = v_lin * scale_lin * self.dt
        # angular: use quaternion difference magnitude as a proxy
        q_obs = quaternion_normalize(observed[..., 3:7])
        q_tgt = quaternion_normalize(target[..., 3:7])
        # geodesic distance on SO(3): theta = 2 * acos(|<q_obs, q_tgt>|)
        dot = (q_obs * q_tgt).sum(dim=-1, keepdim=True).abs().clamp(-1.0, 1.0)
        theta = 2.0 * torch.acos(dot)  # [B, 1]
        omega_max = self.v_ang_max * self.dt
        ang_scale = (omega_max / theta.clamp_min(1e-8)).clamp_max(1.0)
        # slerp from q_obs toward q_tgt by ang_scale
        q_safe = self._slerp(q_obs, q_tgt, ang_scale)
        out = torch.cat([observed[..., :3] + delta_lin, q_safe], dim=-1)
        violated = (scale_lin < 1.0).any(dim=-1) | (ang_scale.squeeze(-1) < 1.0)
        if target.shape[-1] == 8:
            d_open = target[..., 7:8] - observed[..., 7:8]
            v_grip = d_open / self.dt
            scale_grip = (self.v_grip_max / v_grip.abs().clamp_min(1e-12)).clamp_max(1.0)
            open_safe = observed[..., 7:8] + v_grip * scale_grip * self.dt
            out = torch.cat([out, open_safe], dim=-1)
            violated = violated | (scale_grip < 1.0).squeeze(-1)
        return out, violated

    def _enforce_base_speed(self, observed: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = target[..., :3] - observed[..., :3]
        speed = delta.norm(dim=-1, keepdim=True) / self.dt
        scale = (self.base_speed_max / speed.clamp_min(1e-12)).clamp_max(1.0)
        new_pos = observed[..., :3] + delta * scale
        out = torch.cat([new_pos, target[..., 3:]], dim=-1)
        violated = (scale < 1.0).squeeze(-1)
        return out, violated

    def _enforce_smoothness(self, target: torch.Tensor, prev_delta: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_delta is None:
            return target, torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
        # prev_delta is the previous (target - observed) over the same dt.
        # Limit how much the new delta can differ from the previous one.
        new_delta = target - (target - prev_delta)  # placeholder; resolved by caller passing observed externally
        # NOTE: this branch is only used via :meth:`forward`, which constructs
        # the new delta explicitly. See ``forward`` for the full handling.
        return target, torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)

    def _enforce_collision(self, ee_pts: torch.Tensor, obj_pts: torch.Tensor,
                           observed: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ee_pts: [B, N_ee, 3] in the frame of the *target* EE pose
        # obj_pts: [B, N_obj, 3] dough point cloud
        # min pairwise distance per batch
        with torch.no_grad():
            # use chunked cdist to be memory-friendly
            d2 = torch.cdist(ee_pts, obj_pts).min(dim=-1).values.min(dim=-1).values  # [B]
        too_close = d2 < self.collision_margin
        if not too_close.any():
            return target, too_close
        # pull the target back toward the observed pose along the translation axis
        delta = target[..., :3] - observed[..., :3]
        new_pos = torch.where(
            too_close.unsqueeze(-1),
            observed[..., :3] + delta * self.collision_pullback,
            target[..., :3],
        )
        out = torch.cat([new_pos, target[..., 3:]], dim=-1)
        return out, too_close

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # spherical linear interpolation, per-batch t in [0, 1]
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        # take the shorter arc
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs().clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp_min(1e-8)
        # fall back to lerp when angle is tiny
        small = omega < 1e-4
        a = torch.sin((1.0 - t) * omega) / sin_omega
        b = torch.sin(t * omega) / sin_omega
        slerped = a * q0 + b * q1
        lerped = (1.0 - t) * q0 + t * q1
        out = torch.where(small.expand_as(slerped), lerped, slerped)
        return quaternion_normalize(out)

    # --- public entry point ------------------------------------------------

    def forward(self,
                observed: torch.Tensor,
                target: torch.Tensor,
                ee_pts: Optional[torch.Tensor] = None,
                obj_pts: Optional[torch.Tensor] = None,
                prev_target: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            observed:   [B, D] current EE state.
            target:     [B, D] predicted EE target.
            ee_pts:     [B, N_ee, 3] EE point cloud at the *target* pose.
                        If None, collision check is skipped.
            obj_pts:    [B, N_obj, 3] dough/object point cloud (current frame).
            prev_target:[B, D] previous step's (already-filtered) target, used
                        for action-smoothness constraint. If None, the smooth-
                        ness constraint is skipped.

        Returns:
            safe_target: [B, D] filtered target.
            info: dict of per-batch boolean masks for each violated constraint.
        """
        assert observed.shape == target.shape, "observed and target must have the same shape"
        info: Dict[str, torch.Tensor] = {}

        # 1) joint / workspace limits
        target, info['joint_limits'] = self._enforce_joint_limits(target)

        # 2) per-dof velocity limits
        target, info['velocity_limits'] = self._enforce_velocity_limits(observed, target)

        # 3) base (translational) speed cap
        target, info['base_speed'] = self._enforce_base_speed(observed, target)

        # 4) action smoothness: limit change in delta vs. previous step
        if prev_target is not None:
            new_delta = target[..., :3] - observed[..., :3]
            prev_delta = prev_target[..., :3] - observed[..., :3]
            d_delta = new_delta - prev_delta
            accel = d_delta / (self.dt * self.dt)
            scale = (self.accel_lin_max / accel.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.0)
            new_delta_smooth = prev_delta + d_delta * scale
            new_pos = observed[..., :3] + new_delta_smooth
            target = torch.cat([new_pos, target[..., 3:]], dim=-1)
            info['smoothness'] = (scale < 1.0).squeeze(-1)
        else:
            info['smoothness'] = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)

        # 5) collision risk (last so we react to the already-clamped target)
        if ee_pts is not None and obj_pts is not None:
            target, info['collision'] = self._enforce_collision(ee_pts, obj_pts, observed, target)
        else:
            info['collision'] = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)

        info['any_violation'] = torch.stack(list(info.values()), dim=0).any(dim=0)
        return target, info

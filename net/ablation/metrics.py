"""Evaluators that the ablation framework can plug into.

Currently ships one evaluator: :func:`safety_filter_evaluator`. It runs the
:class:`net.model.safety_filter.SafetyAwareActionFilter` on a seeded
synthetic dataset that spans all five violation modes, and reports the
fraction of samples flagged for each constraint plus the mean L2
correction magnitude on position.

Synthetic-data design (frozen for reproducibility):

- ``observed`` poses are sampled inside the workspace box (clamped to the
  box defined in ``net/config/safety.yaml``).
- ``target`` poses are sampled with deliberately wide tails so each
  constraint fires on a non-trivial fraction of samples even with default
  thresholds.
- ``ee_pts`` are fixed at the target translation and ``obj_pts`` at the
  origin, so the collision distance equals the target's translation norm.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Mapping


def _build_filter(cfg: Mapping[str, Any]):
    from net.model.safety_filter import SafetyAwareActionFilter  # noqa: WPS433

    ns = SimpleNamespace(**dict(cfg))
    return SafetyAwareActionFilter(ns)


def _sample_dataset(seed: int, n: int = 256):
    import torch  # local import

    g = torch.Generator().manual_seed(seed)
    # observed poses: tightly clustered around the workspace center
    obs_pos = (torch.rand((n, 3), generator=g) - 0.5) * 0.2  # +-0.1 in each axis
    obs_pos[:, 2] = obs_pos[:, 2].abs() * 0.5 + 0.02         # always above ground
    obs_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    observed = torch.cat([obs_pos, obs_quat], dim=-1)

    # target poses: wide tails to deliberately exceed constraints
    tgt_pos = obs_pos + (torch.rand((n, 3), generator=g) - 0.5) * 0.6  # large jumps
    tgt_quat = torch.nn.functional.normalize(
        torch.randn((n, 4), generator=g), dim=-1
    )
    target = torch.cat([tgt_pos, tgt_quat], dim=-1)

    # collision points: half the samples have EE very near the object so
    # the collision check actually fires; the other half are far away.
    ee_pts = target[:, :3].unsqueeze(1).clone()    # [n, 1, 3]
    obj_pts = torch.zeros((n, 1, 3))               # at origin
    near = torch.arange(n) % 2 == 0
    ee_pts[near] = (torch.rand((near.sum(), 1, 3), generator=g) - 0.5) * 0.005

    # previous target = observed for some samples, far for others, to give
    # the smoothness check something to fire on
    prev_target = observed.clone()
    flip = torch.rand((n,), generator=g) > 0.5
    prev_target[flip, :3] += (torch.rand((flip.sum(), 3), generator=g) - 0.5) * 0.5

    return observed, target, ee_pts, obj_pts, prev_target


def safety_filter_evaluator(config: Mapping[str, Any], seed: int) -> Dict[str, float]:
    """Run the safety filter on a seeded synthetic dataset.

    If ``config["enabled"]`` is False, the filter is bypassed and all
    violation rates and the correction magnitude are returned as 0.0
    (so the "disabled" ablation row is informative without crashing on a
    constructor that demands real numbers).

    Returns the following keys, in order::

        joint_violations
        velocity_violations
        base_speed_violations
        smoothness_violations
        collision_violations
        any_violations
        mean_correction
    """
    import torch  # local import

    if not config.get("enabled", True):
        return dict(
            joint_violations=0.0,
            velocity_violations=0.0,
            base_speed_violations=0.0,
            smoothness_violations=0.0,
            collision_violations=0.0,
            any_violations=0.0,
            mean_correction=0.0,
        )

    cfg = {k: v for k, v in config.items() if k != "enabled"}
    filt = _build_filter(cfg)
    observed, target, ee_pts, obj_pts, prev_target = _sample_dataset(seed)

    with torch.no_grad():
        safe, info = filt(
            observed, target,
            ee_pts=ee_pts, obj_pts=obj_pts, prev_target=prev_target,
        )
        correction = (safe[..., :3] - target[..., :3]).norm(dim=-1).mean().item()

    n = float(target.shape[0])
    return dict(
        joint_violations=info["joint_limits"].sum().item() / n,
        velocity_violations=info["velocity_limits"].sum().item() / n,
        base_speed_violations=info["base_speed"].sum().item() / n,
        smoothness_violations=info["smoothness"].sum().item() / n,
        collision_violations=info["collision"].sum().item() / n,
        any_violations=info["any_violation"].sum().item() / n,
        mean_correction=correction,
    )

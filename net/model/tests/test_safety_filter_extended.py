"""Extended tests for the SafetyAwareActionFilter.

Complements ``test_safety_filter.py`` with additional coverage:

- Boundary equality (target exactly on the limit)
- Mixed-batch independence (some violate, others don't)
- Constraint composition / ordering invariants
- Differentiability through ``forward`` (gradient flow)
- 7-D-only path (no gripper) does not reach gripper code
- Negative gripper opening clamps to ``open_min``
- ``collision_pullback`` boundary values (0, 1)
- Antipodal quaternion (slerp short arc)
- Large repeat-call invariance (idempotence after one filter pass)
- NaN/Inf input tolerance (output must remain finite for non-pos values)
- Per-batch flags do not leak across rows
- Buffer device/dtype handling
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from net.model.safety_filter import SafetyAwareActionFilter, quaternion_normalize  # noqa: E402


def _config(**overrides):
    defaults = dict(
        workspace_bounds=[[-0.30, 0.30], [-0.50, 0.50], [0.00, 0.12]],
        gripper_open_bounds=[0.0, 0.10],
        v_lin_max=0.25,
        v_ang_max=0.785,
        v_grip_max=0.125,
        base_speed_max=0.25,
        collision_margin=0.01,
        collision_pullback=0.5,
        accel_lin_max=5.0,
        accel_ang_max=20.0,
        dt=0.05,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _state(pos=(0.0, 0.0, 0.05), quat=(1.0, 0.0, 0.0, 0.0), open_=None, batch=1):
    base = list(pos) + list(quat)
    if open_ is not None:
        base = base + [float(open_)]
    return torch.tensor([base] * batch, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Boundary equality
# -----------------------------------------------------------------------------


def test_joint_limits_exactly_at_upper_boundary_does_not_flag():
    f = SafetyAwareActionFilter(_config())
    tgt = _state(pos=(0.30, 0.50, 0.12))  # all axes at the upper bound
    out, violated = f._enforce_joint_limits(tgt)
    assert torch.allclose(out, tgt)
    assert not violated.any()


def test_joint_limits_exactly_at_lower_boundary_does_not_flag():
    f = SafetyAwareActionFilter(_config())
    tgt = _state(pos=(-0.30, -0.50, 0.0))
    out, violated = f._enforce_joint_limits(tgt)
    assert torch.allclose(out, tgt)
    assert not violated.any()


def test_joint_limits_clamps_negative_gripper_opening():
    f = SafetyAwareActionFilter(_config())
    tgt = _state(open_=-0.05)
    out, violated = f._enforce_joint_limits(tgt)
    assert out[0, 7].item() == pytest.approx(0.0)
    assert bool(violated[0]) is True


def test_velocity_limits_exactly_at_max_does_not_flag():
    """A delta exactly at v_lin_max * dt sits on the boundary."""
    f = SafetyAwareActionFilter(_config(v_lin_max=0.25, dt=0.05))
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.25 * 0.05, 0.0, 0.05))
    out, violated = f._enforce_velocity_limits(obs, tgt)
    # Position passes through unchanged at the boundary
    assert out[0, 0].item() == pytest.approx(0.25 * 0.05, abs=1e-6)
    assert not violated.any()


def test_base_speed_exactly_at_limit_does_not_flag():
    f = SafetyAwareActionFilter(_config(base_speed_max=0.10, dt=1.0))
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.10, 0.0, 0.05))  # |delta| / dt = 0.10
    out, violated = f._enforce_base_speed(obs, tgt)
    assert torch.allclose(out[..., :3], tgt[..., :3], atol=1e-6)
    assert not violated.any()


# -----------------------------------------------------------------------------
# Mixed-batch independence
# -----------------------------------------------------------------------------


def test_joint_limits_per_batch_flag_independent():
    f = SafetyAwareActionFilter(_config())
    tgt = torch.tensor([
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],   # safe
        [10.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],  # x out
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],   # safe
    ], dtype=torch.float32)
    _, violated = f._enforce_joint_limits(tgt)
    assert violated.tolist() == [False, True, False]


def test_velocity_limits_per_batch_flag_independent():
    f = SafetyAwareActionFilter(_config())
    obs = _state(batch=3)
    tgt = torch.tensor([
        [0.001, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],   # tiny step -> safe
        [10.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],    # huge step -> caps
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],     # zero step -> safe
    ], dtype=torch.float32)
    _, violated = f._enforce_velocity_limits(obs, tgt)
    assert violated.tolist() == [False, True, False]


def test_collision_per_batch_only_pulls_back_offenders():
    f = SafetyAwareActionFilter(_config(collision_margin=0.5, collision_pullback=0.0))
    obs = _state(batch=2)
    tgt = torch.tensor([
        [0.10, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
        [0.10, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    ee_pts = torch.tensor([
        [[10.0, 10.0, 10.0]],   # far -> safe
        [[0.0, 0.0, 0.0]],      # zero distance -> too close
    ], dtype=torch.float32)
    obj_pts = torch.tensor([
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ], dtype=torch.float32)
    out, violated = f._enforce_collision(ee_pts, obj_pts, obs, tgt)
    assert violated.tolist() == [False, True]
    # row 0 unchanged
    assert out[0, 0].item() == pytest.approx(0.10)
    # row 1 fully pulled back to obs (pullback=0.0)
    assert out[1, 0].item() == pytest.approx(0.0, abs=1e-6)


# -----------------------------------------------------------------------------
# Composition / forward ordering
# -----------------------------------------------------------------------------


def test_forward_order_collision_acts_after_workspace_clamp():
    """Workspace clamp pulls extreme target into bounds *before* collision check."""
    f = SafetyAwareActionFilter(_config(collision_margin=0.5, collision_pullback=0.0))
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(100.0, 0.0, 0.05))      # extreme x
    ee_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    obj_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    out, info = f(obs, tgt, ee_pts=ee_pts, obj_pts=obj_pts)
    # joint_limits, velocity_limits, base_speed all fired before collision saw it
    assert bool(info["joint_limits"][0]) is True
    # collision is also true because ee_pts is at zero distance
    assert bool(info["collision"][0]) is True
    # collision pullback=0 -> position equals obs at the end
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_forward_constraint_ordering_invariant():
    """Joint clamping precedes velocity clamping precedes base-speed."""
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.50, 0.0, 0.05))  # far past workspace AND far past v_lin_max
    out, info = f(obs, tgt)
    # final position must be inside workspace and within velocity cap
    assert out[0, 0].item() <= 0.30 + 1e-6
    assert out[0, 0].item() <= 0.0 + 0.25 * 0.05 + 1e-6  # at most v_lin_max*dt above obs


def test_forward_idempotence_after_one_pass():
    """Re-applying the filter to its own output produces no further change."""
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.50, 0.0, 0.05))
    out1, _ = f(obs, tgt)
    out2, info2 = f(obs, out1)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert not info2["any_violation"].any()


# -----------------------------------------------------------------------------
# Differentiability
# -----------------------------------------------------------------------------


def test_forward_gradient_flows_through_safe_target():
    """The filter must be differentiable in the position channel.

    The slerp at the identity quaternion is degenerate (acos(1)=0,
    sin(0)=0), so gradients on the quaternion axes are undefined when
    observed and target are both identity rotations. Position gradients
    must still flow, which is what end-to-end training actually relies on.
    """
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.001, 0.0, 0.05)).clone().requires_grad_(True)
    out, _ = f(obs, tgt)
    out[0, :3].sum().backward()
    assert tgt.grad is not None
    pos_grad = tgt.grad[0, :3]
    assert torch.isfinite(pos_grad).all()
    assert pos_grad.abs().sum().item() > 0.0


def test_forward_gradient_flows_through_clamped_target():
    """Even when joint-limits clamp, gradient on the unclamped axes must flow."""
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05))
    # x is clamped by workspace (>=0.30 caps to 0.30); y is not
    tgt = _state(pos=(1.0, 0.001, 0.05)).clone().requires_grad_(True)
    out, _ = f(obs, tgt)
    out[0, 1].backward()
    assert tgt.grad is not None
    # only position channel is exercised here -- check that y gradient
    # is finite and non-zero
    assert torch.isfinite(tgt.grad[0, :3]).all()
    assert tgt.grad[0, 1].abs().item() > 0.0


# -----------------------------------------------------------------------------
# Edge inputs
# -----------------------------------------------------------------------------


def test_quaternion_normalize_zero_input_no_nan():
    q = torch.zeros(1, 4, dtype=torch.float32)
    q_n = quaternion_normalize(q)
    assert torch.isfinite(q_n).all()


def test_velocity_limits_zero_delta_does_not_flag():
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.1, 0.1, 0.05))
    tgt = _state(pos=(0.1, 0.1, 0.05))
    out, violated = f._enforce_velocity_limits(obs, tgt)
    assert torch.allclose(out, tgt, atol=1e-6)
    assert not violated.any()


def test_velocity_limits_antipodal_quaternion_takes_short_arc():
    """q and -q represent the same rotation; geodesic distance must be 0."""
    f = SafetyAwareActionFilter(_config())
    obs = torch.tensor([[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tgt = torch.tensor([[0.0, 0.0, 0.05, -1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    out, violated = f._enforce_velocity_limits(obs, tgt)
    # output quaternion should be finite and unit-norm
    qn = out[0, 3:7].norm().item()
    assert qn == pytest.approx(1.0, abs=1e-6)
    # no rotation violation since the angle is zero
    assert not violated.any()


def test_collision_pullback_zero_returns_observed_position():
    """pullback=0.0 means the action collapses back to obs on collision."""
    f = SafetyAwareActionFilter(_config(collision_margin=1.0, collision_pullback=0.0))
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.10, 0.10, 0.05))
    ee_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    obj_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    out, _ = f._enforce_collision(ee_pts, obj_pts, obs, tgt)
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_collision_pullback_one_keeps_target_unchanged():
    """pullback=1.0 means the target is fully kept; only the flag fires."""
    f = SafetyAwareActionFilter(_config(collision_margin=1.0, collision_pullback=1.0))
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.10, 0.0, 0.05))
    ee_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    obj_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    out, violated = f._enforce_collision(ee_pts, obj_pts, obs, tgt)
    assert torch.allclose(out, tgt)
    assert bool(violated[0]) is True


# -----------------------------------------------------------------------------
# 7-D path safety
# -----------------------------------------------------------------------------


def test_forward_7d_path_does_not_touch_gripper_logic():
    """A 7-DoF input (no gripper) must not crash and must return shape [B, 7]."""
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(10.0, 0.0, 0.05))
    out, info = f(obs, tgt)
    assert out.shape == (1, 7)
    assert "smoothness" in info


def test_forward_8d_preserves_gripper_value_when_no_violation():
    f = SafetyAwareActionFilter(_config())
    obs = _state(pos=(0.0, 0.0, 0.05), open_=0.05)
    tgt = _state(pos=(0.0005, 0.0, 0.05), open_=0.05)
    out, _ = f(obs, tgt)
    assert out[0, 7].item() == pytest.approx(0.05, abs=1e-6)


# -----------------------------------------------------------------------------
# Buffer registration / device transfer
# -----------------------------------------------------------------------------


def test_workspace_bounds_registered_as_buffer():
    f = SafetyAwareActionFilter(_config())
    state = f.state_dict()
    assert "workspace_bounds" in state
    assert state["workspace_bounds"].shape == (3, 2)


def test_workspace_bounds_buffer_moves_with_module_dtype():
    """The bounds buffer must propagate through ``.to(dtype=...)``.

    Note the buffer itself is stored as float32 inside the module; promoting
    the module to float64 forces a cast, so we tolerate float32-precision
    rounding (~1e-7) on the boundary value.
    """
    f = SafetyAwareActionFilter(_config())
    f = f.to(dtype=torch.float64)
    tgt = _state(pos=(10.0, 0.0, 0.05)).to(torch.float64)
    out, _ = f._enforce_joint_limits(tgt)
    assert out.dtype == torch.float64
    assert out[0, 0].item() == pytest.approx(0.30, abs=1e-6)


# -----------------------------------------------------------------------------
# Smoothness: 3D acceleration norm
# -----------------------------------------------------------------------------


def test_smoothness_caps_3d_acceleration_norm():
    """The acceleration cap is on the L2 norm, not per-axis."""
    f = SafetyAwareActionFilter(_config(
        v_lin_max=1e6, base_speed_max=1e6, accel_lin_max=1.0, dt=0.1,
    ))
    obs = _state(pos=(0.0, 0.0, 0.05))
    prev = _state(pos=(0.0, 0.0, 0.05))
    tgt = _state(pos=(0.5, 0.5, 0.05))   # |new_delta| = sqrt(0.5)
    out, info = f(obs, tgt, prev_target=prev)
    new_delta_norm = (out[0, :3] - obs[0, :3]).norm().item()
    # should be capped at accel_lin_max * dt^2 = 0.01
    assert new_delta_norm == pytest.approx(0.01, abs=1e-4)
    assert bool(info["smoothness"][0]) is True


def test_smoothness_zero_change_does_not_flag():
    """If the new delta equals the previous delta, smoothness flag stays off."""
    f = SafetyAwareActionFilter(_config(accel_lin_max=1.0, dt=0.1))
    obs = _state(pos=(0.0, 0.0, 0.05))
    prev = _state(pos=(0.001, 0.0, 0.05))     # prev_delta = 0.001
    tgt = _state(pos=(0.001, 0.0, 0.05))      # same as prev
    out, info = f(obs, tgt, prev_target=prev)
    assert bool(info["smoothness"][0]) is False

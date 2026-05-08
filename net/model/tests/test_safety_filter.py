"""Tests for the SafetyAwareActionFilter.

Covers all five safety checks (joint limits, velocity limits, base speed,
action smoothness, collision risk) in isolation, plus the integrated
``forward()`` entry point.

The filter is a torch ``nn.Module``, so this suite requires torch. It runs
on CPU only -- no GPU needed.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from net.model.safety_filter import SafetyAwareActionFilter, quaternion_normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config(**overrides):
    """Build a SimpleNamespace mirroring net/config/safety.yaml.

    Defaults match the YAML; ``overrides`` lets a test loosen or tighten
    individual limits to isolate behavior.
    """
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


def _identity_state(pos=(0.0, 0.0, 0.05), open_=None):
    """Build a [1, 7] or [1, 8] state with identity orientation."""
    base = list(pos) + [1.0, 0.0, 0.0, 0.0]
    if open_ is not None:
        base = base + [float(open_)]
    return torch.tensor([base], dtype=torch.float32)


@pytest.fixture
def filt():
    return SafetyAwareActionFilter(_config())


# ---------------------------------------------------------------------------
# 1) Joint / workspace limits
# ---------------------------------------------------------------------------


def test_joint_limits_inside_workspace_pass_through(filt):
    """A target inside the workspace box must not be modified."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.1, 0.1, 0.05))
    out, violated = filt._enforce_joint_limits(tgt)
    assert torch.allclose(out, tgt)
    assert not violated.any()


def test_joint_limits_clamps_out_of_bounds_position(filt):
    """A target outside the workspace must be clamped to the bounds."""
    tgt = _identity_state(pos=(10.0, -10.0, 1.0))  # all axes way out
    out, violated = filt._enforce_joint_limits(tgt)
    assert out[0, 0] == pytest.approx(0.30)   # x clamped to upper
    assert out[0, 1] == pytest.approx(-0.50)  # y clamped to lower
    assert out[0, 2] == pytest.approx(0.12)   # z clamped to upper
    assert bool(violated[0]) is True


def test_joint_limits_renormalizes_quaternion(filt):
    """Quaternions must come out unit-norm even if the input wasn't."""
    tgt = torch.tensor([[0.0, 0.0, 0.05, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    out, _ = filt._enforce_joint_limits(tgt)
    norm = out[0, 3:7].norm().item()
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_joint_limits_clamps_gripper_opening(filt):
    """When 8-dim, the gripper opening must respect [open_min, open_max]."""
    tgt = _identity_state(open_=0.5)  # > open_max=0.10
    out, violated = filt._enforce_joint_limits(tgt)
    assert out[0, 7] == pytest.approx(0.10)
    assert bool(violated[0]) is True


# ---------------------------------------------------------------------------
# 2) Velocity limits
# ---------------------------------------------------------------------------


def test_velocity_limits_small_step_passes_through(filt):
    """A step within v_lin_max * dt must be left unchanged."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    # max linear step = 0.25 * 0.05 = 0.0125 units; use half of that
    tgt = _identity_state(pos=(0.005, 0.0, 0.05))
    out, violated = filt._enforce_velocity_limits(obs, tgt)
    assert torch.allclose(out[..., :3], tgt[..., :3], atol=1e-6)
    assert not violated.any()


def test_velocity_limits_caps_linear_delta(filt):
    """A step larger than v_lin_max * dt must be scaled to that limit."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    # asking for 1.0 in x; should be clipped to v_lin_max * dt = 0.0125
    tgt = _identity_state(pos=(1.0, 0.0, 0.05))
    out, violated = filt._enforce_velocity_limits(obs, tgt)
    assert out[0, 0].item() == pytest.approx(0.0125, abs=1e-5)
    assert bool(violated[0]) is True


def test_velocity_limits_slerps_large_angular_delta():
    """A 180 deg quaternion jump must be slerp-limited to v_ang_max * dt."""
    f = SafetyAwareActionFilter(_config(v_ang_max=0.785, dt=0.05))  # max = 0.03925 rad
    obs = torch.tensor([[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tgt = torch.tensor([[0.0, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)  # 180deg about x
    out, violated = f._enforce_velocity_limits(obs, tgt)
    # geodesic distance between out and obs should be <= max + epsilon
    q_obs = quaternion_normalize(obs[..., 3:7])
    q_out = quaternion_normalize(out[..., 3:7])
    dot = (q_obs * q_out).sum(dim=-1).abs().clamp(-1.0, 1.0)
    theta = (2.0 * torch.acos(dot)).item()
    assert theta == pytest.approx(0.03925, abs=1e-3)
    assert bool(violated[0]) is True


def test_velocity_limits_caps_gripper_opening_speed():
    """Abrupt gripper open/close gets scaled by v_grip_max."""
    f = SafetyAwareActionFilter(_config(v_grip_max=0.125, dt=0.05))  # max delta = 0.00625
    obs = _identity_state(open_=0.0)
    tgt = _identity_state(open_=0.10)
    out, violated = f._enforce_velocity_limits(obs, tgt)
    assert out[0, 7].item() == pytest.approx(0.00625, abs=1e-5)
    assert bool(violated[0]) is True


# ---------------------------------------------------------------------------
# 3) Base speed
# ---------------------------------------------------------------------------


def test_base_speed_passes_within_limit(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    # |delta|/dt = sqrt(0.005^2)/0.05 = 0.1 < 0.25
    tgt = _identity_state(pos=(0.005, 0.0, 0.05))
    out, violated = filt._enforce_base_speed(obs, tgt)
    assert torch.allclose(out[..., :3], tgt[..., :3], atol=1e-6)
    assert not violated.any()


def test_base_speed_caps_translational_magnitude():
    """Combined |xyz| velocity above base_speed_max must be scaled."""
    f = SafetyAwareActionFilter(_config(base_speed_max=0.10, dt=1.0))  # max delta = 0.10
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.30, 0.40, 0.05))  # |delta|=0.5
    out, violated = f._enforce_base_speed(obs, tgt)
    new_delta = (out[0, :3] - obs[0, :3]).norm().item()
    assert new_delta == pytest.approx(0.10, abs=1e-5)
    assert bool(violated[0]) is True


# ---------------------------------------------------------------------------
# 4) Smoothness (acceleration cap)
#    Only exercised via forward() because the standalone helper is a stub.
# ---------------------------------------------------------------------------


def test_smoothness_skipped_when_no_prev_target(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.01, 0.0, 0.05))
    _, info = filt(obs, tgt, prev_target=None)
    assert "smoothness" in info
    assert not info["smoothness"].any()


def test_smoothness_caps_acceleration():
    """When |d_delta|/dt^2 exceeds accel_lin_max, position gets damped."""
    # loose translational/velocity limits so they don't dominate this test
    f = SafetyAwareActionFilter(_config(
        v_lin_max=1e6, base_speed_max=1e6, accel_lin_max=1.0, dt=0.1,
    ))
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    prev = _identity_state(pos=(0.0, 0.0, 0.05))     # prev_delta = 0
    tgt = _identity_state(pos=(0.5, 0.0, 0.05))      # new_delta = 0.5
    # accel = 0.5 / 0.01 = 50 units/s^2 >> 1.0 -> heavy scaling
    out, info = f(obs, tgt, prev_target=prev)
    # max permitted new_delta after smoothing = accel_lin_max * dt^2 = 0.01
    assert out[0, 0].item() == pytest.approx(0.01, abs=1e-4)
    assert bool(info["smoothness"][0]) is True


# ---------------------------------------------------------------------------
# 5) Collision risk
# ---------------------------------------------------------------------------


def test_collision_no_pullback_when_far(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.01, 0.0, 0.05))
    ee_pts = torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float32)   # 1 point, far
    obj_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    out, violated = filt._enforce_collision(ee_pts, obj_pts, obs, tgt)
    assert torch.allclose(out, tgt)
    assert not violated.any()


def test_collision_pullback_when_too_close():
    """Trigger collision and verify pos = obs + delta * collision_pullback."""
    f = SafetyAwareActionFilter(_config(collision_margin=0.5, collision_pullback=0.5))
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.10, 0.0, 0.05))
    ee_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    obj_pts = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)  # zero distance
    out, violated = f._enforce_collision(ee_pts, obj_pts, obs, tgt)
    # delta x = 0.10; pulled back by 0.5 -> new_x = 0.0 + 0.05
    assert out[0, 0].item() == pytest.approx(0.05, abs=1e-6)
    assert bool(violated[0]) is True


# ---------------------------------------------------------------------------
# Integrated forward()
# ---------------------------------------------------------------------------


def test_forward_returns_all_info_keys(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.005, 0.0, 0.05))
    out, info = filt(obs, tgt)
    expected_keys = {
        "joint_limits",
        "velocity_limits",
        "base_speed",
        "smoothness",
        "collision",
        "any_violation",
    }
    assert expected_keys == set(info.keys())


def test_forward_any_violation_is_logical_or(filt):
    """any_violation must be True iff any individual flag is True."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    # out-of-bounds in z forces joint_limits to fire
    tgt = _identity_state(pos=(0.0, 0.0, 5.0))
    _, info = filt(obs, tgt)
    assert bool(info["joint_limits"][0]) is True
    assert bool(info["any_violation"][0]) is True


def test_forward_no_violation_when_inputs_safe(filt):
    """Tiny delta within all limits -> no flags fire."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.001, 0.0, 0.05))
    _, info = filt(obs, tgt)
    assert not info["any_violation"].any()


def test_forward_preserves_input_shape_7d(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.001, 0.0, 0.05))
    out, _ = filt(obs, tgt)
    assert out.shape == tgt.shape == (1, 7)


def test_forward_preserves_input_shape_8d(filt):
    obs = _identity_state(pos=(0.0, 0.0, 0.05), open_=0.05)
    tgt = _identity_state(pos=(0.001, 0.0, 0.05), open_=0.05)
    out, _ = filt(obs, tgt)
    assert out.shape == tgt.shape == (1, 8)


def test_forward_handles_batch(filt):
    """Per-batch flags must be independent."""
    obs = torch.tensor([
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    tgt = torch.tensor([
        [0.001, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],   # safe
        [0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0],      # joint-limit violation
    ], dtype=torch.float32)
    out, info = filt(obs, tgt)
    assert out.shape == (2, 7)
    assert bool(info["any_violation"][0]) is False
    assert bool(info["any_violation"][1]) is True


def test_forward_collision_gated_on_pointcloud_args(filt):
    """If ee_pts/obj_pts aren't supplied, the collision check is a no-op."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(0.001, 0.0, 0.05))
    _, info = filt(obs, tgt, ee_pts=None, obj_pts=None)
    assert not info["collision"].any()


def test_forward_output_is_inside_workspace_after_extreme_target(filt):
    """End-to-end: an extreme target ends up inside the workspace box."""
    obs = _identity_state(pos=(0.0, 0.0, 0.05))
    tgt = _identity_state(pos=(100.0, -100.0, 100.0))
    out, _ = filt(obs, tgt)
    bounds = torch.tensor([[-0.30, 0.30], [-0.50, 0.50], [0.00, 0.12]])
    for axis in range(3):
        assert out[0, axis].item() >= bounds[axis, 0].item() - 1e-6
        assert out[0, axis].item() <= bounds[axis, 1].item() + 1e-6

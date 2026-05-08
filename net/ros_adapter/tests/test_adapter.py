"""Tests for the pure-Python ROS2 adapter.

These tests are dependency-free (no torch / numpy / rclpy) so they can run in
any environment.
"""

import math

import pytest

from net.ros_adapter import EEActionToROS2, PoseStamped


def _identity_action():
    return [0.5, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]


def test_returns_list_for_single_sample():
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped(_identity_action())
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], PoseStamped)


def test_sim_to_meter_scaling_default():
    """Default factor 0.2 matches the project sim->meter scaling."""
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped([5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert out[0].pose.position.x == pytest.approx(1.0)


def test_sim_to_meter_scaling_override():
    adapter = EEActionToROS2(sim_to_meter=1.0)
    out = adapter.to_pose_stamped([5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert out[0].pose.position.x == pytest.approx(5.0)


def test_quaternion_order_swap():
    """Model is [w, x, y, z] scalar-first; ROS2 is scalar-last."""
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped([0.0, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0])
    q = out[0].pose.orientation
    assert q.w == pytest.approx(0.7071, abs=1e-4)
    assert q.x == pytest.approx(0.7071, abs=1e-4)
    assert q.y == pytest.approx(0.0)
    assert q.z == pytest.approx(0.0)


def test_quaternion_renormalized():
    """Non-unit quaternions are projected back onto the unit sphere."""
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    q = out[0].pose.orientation
    norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_zero_quaternion_falls_back_to_identity():
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q = out[0].pose.orientation
    assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0)


def test_batched_input():
    adapter = EEActionToROS2()
    batch = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
    ]
    out = adapter.to_pose_stamped(batch)
    assert len(out) == 2
    assert out[1].pose.position.x == pytest.approx(0.2)
    assert out[1].pose.position.y == pytest.approx(0.4)
    assert out[1].pose.position.z == pytest.approx(0.6)


def test_8d_input_ignores_gripper_on_pose_channel():
    """Eighth dim (gripper opening) must not leak into the pose."""
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04])
    assert out[0].pose.position.x == pytest.approx(0.2)
    # round-trip via to_dict to confirm there is no gripper field on the pose
    d = out[0].to_dict()
    assert set(d["pose"].keys()) == {"position", "orientation"}


def test_invalid_dim_raises():
    adapter = EEActionToROS2()
    with pytest.raises(ValueError, match="last dim 7 or 8"):
        adapter.to_pose_stamped([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def test_frame_id_propagated():
    adapter = EEActionToROS2(frame_id="panda_link0")
    out = adapter.to_pose_stamped(_identity_action())
    assert out[0].header.frame_id == "panda_link0"


def test_stamp_split_into_sec_and_nanosec():
    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped(_identity_action(), stamp=1730000000.5)
    assert out[0].header.stamp.sec == 1730000000
    assert out[0].header.stamp.nanosec == 500_000_000


def test_torch_like_input_via_duck_typing():
    """The adapter accepts anything with .detach()/.cpu()/.tolist() chain."""

    class FakeTensor:
        def __init__(self, data):
            self._data = data

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self._data

    adapter = EEActionToROS2()
    out = adapter.to_pose_stamped(FakeTensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert len(out) == 1
    assert out[0].pose.position.x == pytest.approx(0.2)

"""Tests for the ROS2 / MoveIt2 export layer.

Covers:

- ``MoveIt2Exporter`` per-robot defaults (frame, group, joint names, EE link)
- MoveIt pose-goal construction, planner-knob validation, batched input
- Twist construction from explicit linear/angular vectors and from a
  finite-difference of two consecutive pose actions
- JointTrajectory construction with uniform dt and explicit timestamps,
  optional velocities/accelerations, custom joint names, defensive
  validation (width mismatch, negative time, missing joint names, etc.)
- Message dataclass round-trips via ``to_dict``
- End-to-end: a single AC-DiT-style EE prediction is converted into all
  three message types with the expected frame and scaling.

The tests have no torch / numpy / rclpy dependency.
"""

import math

import pytest

from net.hardware import load_robot_config
from net.ros_adapter import (
    ARM_DEFAULTS,
    Duration,
    JointTrajectory,
    JointTrajectoryPoint,
    MoveIt2Exporter,
    MoveItPoseGoal,
    PoseStamped,
    Twist,
    TwistStamped,
    Vector3,
    arm_defaults,
)


def _identity_action():
    return [0.5, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]


# -- arm defaults -----------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARM_DEFAULTS.keys()))
def test_arm_defaults_keys_present(arm):
    d = arm_defaults(arm)
    assert set(d.keys()) >= {"joint_names", "move_group", "end_effector_link"}


def test_arm_defaults_unknown_raises():
    with pytest.raises(KeyError, match="Unknown arm"):
        arm_defaults("frobnicator")


def test_arm_defaults_panda_has_seven_joints():
    assert len(arm_defaults("franka_panda")["joint_names"]) == 7


def test_arm_defaults_iiwa_has_seven_joints():
    assert len(arm_defaults("kuka_iiwa14")["joint_names"]) == 7


def test_arm_defaults_ur5_has_six_joints():
    assert len(arm_defaults("ur5")["joint_names"]) == 6


# -- exporter construction --------------------------------------------------

@pytest.mark.parametrize("kind", ["sim", "franka_mobile", "kuka_mobile", "husky_arm"])
def test_exporter_built_from_each_preset(kind):
    exp = MoveIt2Exporter.from_robot_config(kind)
    cfg = load_robot_config(kind)
    assert exp.frame_id == cfg.base_frame_id
    assert exp.sim_to_meter == cfg.sim_to_meter
    expected = arm_defaults(cfg.arm)
    assert tuple(exp.joint_names) == tuple(expected["joint_names"])
    assert exp.move_group == expected["move_group"]
    assert exp.end_effector_link == expected["end_effector_link"]


def test_exporter_constructor_overrides_defaults():
    exp = MoveIt2Exporter.from_robot_config(
        "franka_mobile",
        joint_names=["a", "b"],
        move_group="custom_group",
        end_effector_link="custom_ee",
    )
    assert exp.joint_names == ["a", "b"]
    assert exp.move_group == "custom_group"
    assert exp.end_effector_link == "custom_ee"


# -- pose stamped (delegated) ----------------------------------------------

def test_to_pose_stamped_propagates_frame_and_scaling():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    out = exp.to_pose_stamped([5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert out[0].header.frame_id == "panda_link0"
    assert out[0].pose.position.x == pytest.approx(1.0)


# -- MoveIt pose goal ------------------------------------------------------

def test_pose_goal_uses_per_arm_defaults():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exp.to_pose_goal(_identity_action())[0]
    assert isinstance(goal, MoveItPoseGoal)
    assert goal.group_name == "panda_arm"
    assert goal.end_effector_link == "panda_link8"
    assert goal.target_pose.header.frame_id == "panda_link0"


def test_pose_goal_per_call_overrides_take_precedence():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exp.to_pose_goal(
        _identity_action(),
        group_name="alt_group",
        end_effector_link="alt_ee",
    )[0]
    assert goal.group_name == "alt_group"
    assert goal.end_effector_link == "alt_ee"


def test_pose_goal_planner_knobs_propagate():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exp.to_pose_goal(
        _identity_action(),
        allowed_planning_time=2.5,
        max_velocity_scaling_factor=0.5,
        max_acceleration_scaling_factor=0.4,
        num_planning_attempts=3,
    )[0]
    assert goal.allowed_planning_time == 2.5
    assert goal.max_velocity_scaling_factor == 0.5
    assert goal.max_acceleration_scaling_factor == 0.4
    assert goal.num_planning_attempts == 3


def test_pose_goal_batched_input_yields_one_goal_per_row():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    batch = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
    ]
    goals = exp.to_pose_goal(batch)
    assert len(goals) == 2
    assert goals[1].target_pose.pose.position.x == pytest.approx(0.2)


def test_pose_goal_without_group_name_raises():
    exp = MoveIt2Exporter.from_robot_config("sim")  # no MoveIt group
    with pytest.raises(ValueError, match="group_name"):
        exp.to_pose_goal(_identity_action())


def test_pose_goal_invalid_velocity_scale_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="max_velocity_scaling_factor"):
        exp.to_pose_goal(_identity_action(), max_velocity_scaling_factor=0.0)
    with pytest.raises(ValueError, match="max_velocity_scaling_factor"):
        exp.to_pose_goal(_identity_action(), max_velocity_scaling_factor=1.5)


def test_pose_goal_invalid_acceleration_scale_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="max_acceleration_scaling_factor"):
        exp.to_pose_goal(_identity_action(), max_acceleration_scaling_factor=2.0)


def test_pose_goal_invalid_planning_time_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="allowed_planning_time"):
        exp.to_pose_goal(_identity_action(), allowed_planning_time=-1.0)


def test_pose_goal_invalid_planning_attempts_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="num_planning_attempts"):
        exp.to_pose_goal(_identity_action(), num_planning_attempts=0)


# -- Twist -----------------------------------------------------------------

def test_twist_zero_input_produces_zero_twist():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    t = exp.to_twist((0.0, 0.0, 0.0))
    assert isinstance(t, Twist)
    assert t.linear.x == 0.0 and t.linear.y == 0.0 and t.linear.z == 0.0
    assert t.angular.x == 0.0 and t.angular.y == 0.0 and t.angular.z == 0.0


def test_twist_does_not_scale_by_default():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    t = exp.to_twist((1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
    assert t.linear.x == 1.0
    assert t.linear.y == 2.0
    assert t.linear.z == 3.0
    assert t.angular.x == 0.1
    assert t.angular.z == 0.3


def test_twist_scale_to_meters_applies_sim_to_meter():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    t = exp.to_twist((5.0, 0.0, 0.0), scale_to_meters=True)
    assert t.linear.x == pytest.approx(5.0 * exp.sim_to_meter)


def test_twist_invalid_length_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="length 3"):
        exp.to_twist((1.0, 2.0))
    with pytest.raises(ValueError, match="length 3"):
        exp.to_twist((1.0, 2.0, 3.0), (0.1, 0.2))


def test_twist_stamped_carries_frame_and_timestamp():
    exp = MoveIt2Exporter.from_robot_config("husky_arm")
    ts = exp.to_twist_stamped((1.0, 0.0, 0.0), stamp=1730000000.5)
    assert isinstance(ts, TwistStamped)
    assert ts.header.frame_id == "ur_arm_base_link"
    assert ts.header.stamp.sec == 1730000000
    assert ts.header.stamp.nanosec == 500_000_000


def test_twist_from_action_pair_pure_translation():
    """A pure-translation step yields linear vel = (next-prev)*scale/dt, zero angular."""
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")  # sim_to_meter=0.2
    prev = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    nxt = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    t = exp.twist_from_action_pair(prev, nxt, dt=0.1)
    # linear: (1.0 - 0.0) * 0.2 / 0.1 = 2.0
    assert t.linear.x == pytest.approx(2.0)
    assert t.linear.y == pytest.approx(0.0)
    assert t.angular.x == pytest.approx(0.0)
    assert t.angular.y == pytest.approx(0.0)
    assert t.angular.z == pytest.approx(0.0)


def test_twist_from_action_pair_pure_rotation():
    """A pure 90deg rotation about z over 1 s yields wz = pi/2 rad/s."""
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    # quaternion for 90deg about z: (cos(pi/4), 0, 0, sin(pi/4))
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    prev = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    nxt = [0.0, 0.0, 0.0, c, 0.0, 0.0, s]
    t = exp.twist_from_action_pair(prev, nxt, dt=1.0)
    assert t.linear.x == pytest.approx(0.0, abs=1e-6)
    assert t.angular.x == pytest.approx(0.0, abs=1e-6)
    assert t.angular.y == pytest.approx(0.0, abs=1e-6)
    assert t.angular.z == pytest.approx(math.pi / 2, abs=1e-6)


def test_twist_from_action_pair_zero_rotation_handles_identity():
    """Identical orientations yield zero angular velocity (no NaN)."""
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    prev = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    nxt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    t = exp.twist_from_action_pair(prev, nxt, dt=0.1)
    assert math.isfinite(t.angular.x)
    assert math.isfinite(t.angular.y)
    assert math.isfinite(t.angular.z)
    assert (t.angular.x, t.angular.y, t.angular.z) == (0.0, 0.0, 0.0)


def test_twist_from_action_pair_negative_dt_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="dt"):
        exp.twist_from_action_pair(
            [0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], dt=0.0,
        )


def test_twist_from_action_pair_short_input_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="\\[x, y, z, qw"):
        exp.twist_from_action_pair([0, 0, 0], [1, 2, 3], dt=0.1)


# -- JointTrajectory -------------------------------------------------------

def test_joint_trajectory_uniform_dt():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    positions = [
        [0.0] * 7,
        [0.1] * 7,
        [0.2] * 7,
    ]
    traj = exp.to_joint_trajectory(positions, dt=0.5)
    assert isinstance(traj, JointTrajectory)
    assert traj.joint_names == list(arm_defaults("franka_panda")["joint_names"])
    assert len(traj.points) == 3
    assert traj.points[0].time_from_start == Duration(sec=0, nanosec=500_000_000)
    assert traj.points[1].time_from_start == Duration(sec=1, nanosec=0)
    assert traj.points[2].time_from_start == Duration(sec=1, nanosec=500_000_000)
    assert traj.points[1].positions == [0.1] * 7


def test_joint_trajectory_explicit_time_from_start():
    exp = MoveIt2Exporter.from_robot_config("kuka_mobile")
    positions = [[0.0] * 7, [0.5] * 7]
    traj = exp.to_joint_trajectory(positions, time_from_start=[0.0, 1.5])
    assert traj.points[0].time_from_start.sec == 0
    assert traj.points[0].time_from_start.nanosec == 0
    assert traj.points[1].time_from_start.sec == 1
    assert traj.points[1].time_from_start.nanosec == 500_000_000


def test_joint_trajectory_dt_and_explicit_mutually_exclusive():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="either dt or time_from_start"):
        exp.to_joint_trajectory(
            [[0.0] * 7, [0.1] * 7], dt=0.1, time_from_start=[0.1, 0.2],
        )


def test_joint_trajectory_negative_time_from_start_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="non-negative"):
        exp.to_joint_trajectory(
            [[0.0] * 7, [0.1] * 7], time_from_start=[-0.1, 0.2],
        )


def test_joint_trajectory_time_from_start_length_mismatch_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="time_from_start has"):
        exp.to_joint_trajectory(
            [[0.0] * 7, [0.1] * 7], time_from_start=[0.1],
        )


def test_joint_trajectory_negative_dt_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="dt must be positive"):
        exp.to_joint_trajectory([[0.0] * 7], dt=-0.1)


def test_joint_trajectory_width_mismatch_raises():
    """If joint_names has 7 entries but positions has 6, fail loudly."""
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="width 6 but 7 joint"):
        exp.to_joint_trajectory([[0.0] * 6], dt=0.1)


def test_joint_trajectory_inconsistent_row_widths_raise():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="must have width"):
        exp.to_joint_trajectory([[0.0] * 7, [0.0] * 6], dt=0.1)


def test_joint_trajectory_empty_input_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="at least one waypoint"):
        exp.to_joint_trajectory([], dt=0.1)


def test_joint_trajectory_no_joint_names_raises():
    """sim has no joint names by default; demanding a trajectory must fail clearly."""
    exp = MoveIt2Exporter.from_robot_config("sim")
    with pytest.raises(ValueError, match="joint_names"):
        exp.to_joint_trajectory([[0.0]], dt=0.1)


def test_joint_trajectory_custom_joint_names():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    custom = ["j1", "j2", "j3"]
    traj = exp.to_joint_trajectory([[0.0, 0.1, 0.2]], dt=0.1, joint_names=custom)
    assert traj.joint_names == custom


def test_joint_trajectory_velocities_and_accelerations_propagate():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    traj = exp.to_joint_trajectory(
        [[0.0] * 7, [0.1] * 7],
        dt=0.1,
        velocities=[[0.0] * 7, [1.0] * 7],
        accelerations=[[0.0] * 7, [10.0] * 7],
    )
    assert traj.points[1].velocities == [1.0] * 7
    assert traj.points[1].accelerations == [10.0] * 7


def test_joint_trajectory_velocities_width_mismatch_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="velocities row width"):
        exp.to_joint_trajectory(
            [[0.0] * 7], dt=0.1, velocities=[[0.0] * 6],
        )


def test_joint_trajectory_velocities_count_mismatch_raises():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    with pytest.raises(ValueError, match="velocities must have"):
        exp.to_joint_trajectory(
            [[0.0] * 7, [0.1] * 7], dt=0.1, velocities=[[0.0] * 7],
        )


def test_joint_trajectory_header_carries_frame():
    exp = MoveIt2Exporter.from_robot_config("husky_arm")
    traj = exp.to_joint_trajectory([[0.0] * 6], dt=0.1)
    assert traj.header.frame_id == "ur_arm_base_link"


def test_joint_trajectory_torch_like_input_via_duck_typing():
    class FakeTensor:
        def __init__(self, data):
            self._data = data
        def detach(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return self._data

    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    traj = exp.to_joint_trajectory(
        FakeTensor([[0.0] * 7, [0.5] * 7]), dt=0.2,
    )
    assert len(traj.points) == 2
    assert traj.points[1].positions == [0.5] * 7


# -- Duration --------------------------------------------------------------

def test_duration_from_seconds_round_trip():
    d = Duration.from_seconds(2.75)
    assert d.sec == 2
    assert d.nanosec == 750_000_000


def test_duration_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        Duration.from_seconds(-0.1)


def test_duration_handles_nanosecond_overflow():
    """0.999999999s should not overflow nanosec into 1e9."""
    d = Duration.from_seconds(0.9999999995)  # rounds up to 1s exactly
    assert d.sec == 1
    assert d.nanosec == 0


# -- Message round-trip ----------------------------------------------------

def test_pose_goal_to_dict_round_trip():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exp.to_pose_goal(_identity_action())[0]
    d = goal.to_dict()
    assert d["group_name"] == "panda_arm"
    assert d["target_pose"]["header"]["frame_id"] == "panda_link0"
    assert "position" in d["target_pose"]["pose"]
    assert "orientation" in d["target_pose"]["pose"]


def test_joint_trajectory_to_dict_round_trip():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    traj = exp.to_joint_trajectory([[0.0] * 7, [0.1] * 7], dt=0.5)
    d = traj.to_dict()
    assert "header" in d
    assert "joint_names" in d
    assert d["points"][0]["time_from_start"] == {"sec": 0, "nanosec": 500_000_000}


def test_twist_to_dict_round_trip():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    t = exp.to_twist((0.5, 0.0, 0.0), (0.0, 0.0, 0.1))
    d = t.to_dict()
    assert d == {
        "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.1},
    }


# -- end-to-end ------------------------------------------------------------

@pytest.mark.parametrize("kind", ["franka_mobile", "kuka_mobile", "husky_arm"])
def test_end_to_end_predict_to_three_messages(kind):
    """A single AC-DiT-style prediction round-trips into all three messages."""
    cfg = load_robot_config(kind)
    exp = MoveIt2Exporter.from_robot_config(cfg)
    action = [0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]

    # 1) PoseStamped (the base channel)
    poses = exp.to_pose_stamped(action)
    assert len(poses) == 1
    assert isinstance(poses[0], PoseStamped)
    assert poses[0].header.frame_id == cfg.base_frame_id

    # 2) MoveIt pose goal
    goal = exp.to_pose_goal(action)[0]
    assert goal.target_pose.header.frame_id == cfg.base_frame_id
    assert goal.group_name == arm_defaults(cfg.arm)["move_group"]

    # 3) Twist for the mobile base (cmd_vel)
    twist = exp.to_twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert isinstance(twist, Twist)
    assert twist.linear.x == 0.1


def test_vector3_default_is_zero():
    v = Vector3()
    assert (v.x, v.y, v.z) == (0.0, 0.0, 0.0)


def test_pose_goal_supports_eight_dim_input_and_drops_gripper():
    exp = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exp.to_pose_goal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04])[0]
    # the gripper open value must not leak into the pose dict
    pose = goal.target_pose.pose
    assert set(goal.target_pose.to_dict()["pose"].keys()) == {"position", "orientation"}
    assert pose.position.x == 0.0


def test_joint_trajectory_point_default_lists_are_empty():
    """Optional fields default to empty lists, not None."""
    p = JointTrajectoryPoint()
    assert p.positions == []
    assert p.velocities == []
    assert p.accelerations == []
    assert p.effort == []
    assert isinstance(p.time_from_start, Duration)

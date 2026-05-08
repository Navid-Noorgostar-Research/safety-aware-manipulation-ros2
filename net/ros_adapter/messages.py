"""Dataclass mirrors of ROS2 messages used by the EE-target adapter.

The fields, names, and types match the geometry_msgs / std_msgs wire format so
that ``to_dict()`` produces a payload that can be fed straight into rclpy's
``message_to_ordereddict`` consumers (or into rosbridge / DDS via JSON).
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Time:
    """std_msgs/Time -- POSIX seconds split into integer + nanosecond parts."""

    sec: int = 0
    nanosec: int = 0

    @classmethod
    def from_seconds(cls, t: float) -> "Time":
        sec = int(t)
        nanosec = int(round((t - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return cls(sec=sec, nanosec=nanosec)


@dataclass
class Header:
    """std_msgs/Header."""

    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Point:
    """geometry_msgs/Point -- XYZ in meters."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """geometry_msgs/Quaternion -- scalar-last (x, y, z, w) convention."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    """geometry_msgs/Pose."""

    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped:
    """geometry_msgs/PoseStamped -- pose + frame + timestamp."""

    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Vector3:
    """geometry_msgs/Vector3."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    """geometry_msgs/Twist -- linear + angular velocity."""

    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwistStamped:
    """geometry_msgs/TwistStamped -- Twist with header for cmd_vel timing."""

    header: Header = field(default_factory=Header)
    twist: Twist = field(default_factory=Twist)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Duration:
    """builtin_interfaces/Duration -- POSIX seconds split into sec + nsec."""

    sec: int = 0
    nanosec: int = 0

    @classmethod
    def from_seconds(cls, t: float) -> "Duration":
        if t < 0:
            raise ValueError(f"Duration must be non-negative (got {t}).")
        sec = int(t)
        nanosec = int(round((t - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return cls(sec=sec, nanosec=nanosec)


@dataclass
class JointTrajectoryPoint:
    """trajectory_msgs/JointTrajectoryPoint."""

    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    effort: List[float] = field(default_factory=list)
    time_from_start: Duration = field(default_factory=Duration)


@dataclass
class JointTrajectory:
    """trajectory_msgs/JointTrajectory."""

    header: Header = field(default_factory=Header)
    joint_names: List[str] = field(default_factory=list)
    points: List[JointTrajectoryPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MoveItPoseGoal:
    """A MoveIt2-compatible Cartesian motion plan goal.

    Mirrors the relevant subset of ``moveit_msgs/MotionPlanRequest`` /
    ``moveit_msgs/MoveGroup``: the planning group, target pose (with frame
    via the embedded :class:`PoseStamped`), end-effector link, and a couple
    of planner knobs that real MoveIt servers expose.
    """

    group_name: str = ""
    end_effector_link: str = ""
    target_pose: PoseStamped = field(default_factory=PoseStamped)
    allowed_planning_time: float = 5.0
    max_velocity_scaling_factor: float = 1.0
    max_acceleration_scaling_factor: float = 1.0
    num_planning_attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""ROS2 adapter for converting predicted EE target actions into ROS2 messages.

Public API:
    EEActionToROS2  -- the converter (pure Python, no rclpy dependency).
    PoseStamped, Pose, Point, Quaternion, Header, Time -- message dataclasses
                       mirroring the wire format of geometry_msgs/std_msgs.

The optional rclpy-based publisher lives in :mod:`net.ros_adapter.node` and is
imported on demand so the rest of the package stays importable without ROS2.
"""

from .messages import (
    Duration,
    Header,
    JointTrajectory,
    JointTrajectoryPoint,
    MoveItPoseGoal,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Time,
    Twist,
    TwistStamped,
    Vector3,
)
from .adapter import EEActionToROS2
from .exporters import ARM_DEFAULTS, MoveIt2Exporter, arm_defaults

__all__ = [
    "ARM_DEFAULTS",
    "Duration",
    "EEActionToROS2",
    "Header",
    "JointTrajectory",
    "JointTrajectoryPoint",
    "MoveIt2Exporter",
    "MoveItPoseGoal",
    "Point",
    "Pose",
    "PoseStamped",
    "Quaternion",
    "Time",
    "Twist",
    "TwistStamped",
    "Vector3",
    "arm_defaults",
]

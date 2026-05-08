"""Dataclass mirrors of ROS2 messages used by the EE-target adapter.

The fields, names, and types match the geometry_msgs / std_msgs wire format so
that ``to_dict()`` produces a payload that can be fed straight into rclpy's
``message_to_ordereddict`` consumers (or into rosbridge / DDS via JSON).
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


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

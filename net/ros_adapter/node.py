"""rclpy publisher for the EE target adapter.

This module imports rclpy at module-import time and is only meant to be loaded
in an environment with ROS2 installed. The adapter logic itself
(:mod:`net.ros_adapter.adapter`) has no rclpy dependency, so it can be tested
and used in pure-Python pipelines.

Example:
    Run a tiny demo that publishes a single hard-coded pose so you can verify
    the topic is reachable from another ROS2 node:

        python -m net.ros_adapter.node --topic /ee_target --frame base_link

    In your own inference loop:

        from net.ros_adapter import EEActionToROS2
        from net.ros_adapter.node import EEPoseBridge

        bridge = EEPoseBridge(topic="/ee_target", frame_id="panda_link0")
        for batch in loader:
            target = pipeline.predict(batch)["ee_target_nxt"]   # [B, 7] or [B, 8]
            bridge.publish(target)
        bridge.shutdown()
"""

from __future__ import annotations

import argparse
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import (
        PoseStamped as RosPoseStamped,
        Pose as RosPose,
        Point as RosPoint,
        Quaternion as RosQuaternion,
    )
    from std_msgs.msg import Header as RosHeader
    from builtin_interfaces.msg import Time as RosTime
except ImportError as e:  # pragma: no cover -- only relevant on machines without ROS2
    raise ImportError(
        "net.ros_adapter.node requires ROS2 (rclpy + geometry_msgs). "
        "Install ROS2 Humble or newer, or use net.ros_adapter.adapter directly "
        "for the pure-Python conversion without publishing."
    ) from e

from .adapter import EEActionToROS2
from .messages import PoseStamped as PyPoseStamped


def _py_to_ros(msg: PyPoseStamped) -> RosPoseStamped:
    """Translate our dataclass mirror into the actual ROS2 message type."""
    out = RosPoseStamped()
    out.header = RosHeader()
    out.header.frame_id = msg.header.frame_id
    out.header.stamp = RosTime(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec)
    out.pose = RosPose()
    out.pose.position = RosPoint(x=msg.pose.position.x, y=msg.pose.position.y, z=msg.pose.position.z)
    out.pose.orientation = RosQuaternion(
        x=msg.pose.orientation.x,
        y=msg.pose.orientation.y,
        z=msg.pose.orientation.z,
        w=msg.pose.orientation.w,
    )
    return out


class EEPoseBridge(Node):
    """ROS2 node that publishes EE target actions on a configurable topic.

    The bridge holds a single publisher and exposes :meth:`publish`, which
    accepts the same tensor / array shapes as :class:`EEActionToROS2`. Use it
    from your inference loop to forward predictions to a real robot.
    """

    def __init__(
        self,
        topic: str = "/ee_target",
        frame_id: str = "base_link",
        sim_to_meter: float = 0.2,
        queue_depth: int = 10,
        node_name: str = "ee_pose_bridge",
    ) -> None:
        if not rclpy.ok():
            rclpy.init()
        super().__init__(node_name)
        self.adapter = EEActionToROS2(frame_id=frame_id, sim_to_meter=sim_to_meter)
        self.topic = topic
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=queue_depth,
        )
        self.pub = self.create_publisher(RosPoseStamped, topic, qos)
        self.get_logger().info(f"EEPoseBridge publishing PoseStamped on '{topic}' (frame={frame_id})")

    @classmethod
    def from_robot_config(
        cls,
        robot,
        *,
        topic: Optional[str] = None,
        queue_depth: int = 10,
        node_name: str = "ee_pose_bridge",
    ) -> "EEPoseBridge":
        """Build a bridge from a :class:`net.hardware.RobotConfig` (or its name).

        ``robot`` accepts the same inputs as
        :func:`net.hardware.load_robot_config`. ``topic`` defaults to the
        config's ``ee_topic`` but can be overridden when remapping.
        """
        from net.hardware import load_robot_config  # local import to avoid cycle
        cfg = load_robot_config(robot)
        return cls(
            topic=topic if topic is not None else cfg.ee_topic,
            frame_id=cfg.base_frame_id,
            sim_to_meter=cfg.sim_to_meter,
            queue_depth=queue_depth,
            node_name=node_name,
        )

    def publish(self, action, stamp: Optional[float] = None) -> int:
        """Convert and publish; returns the number of messages sent."""
        msgs = self.adapter.to_pose_stamped(action, stamp=stamp)
        for m in msgs:
            self.pub.publish(_py_to_ros(m))
        return len(msgs)

    def shutdown(self) -> None:
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _demo_main() -> None:
    """Publish a single identity-pose at the workspace origin -- smoke test."""
    parser = argparse.ArgumentParser(description="EE pose bridge smoke test.")
    parser.add_argument("--topic", default="/ee_target")
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--rate", type=float, default=10.0, help="Hz; <=0 to publish once.")
    args = parser.parse_args()

    bridge = EEPoseBridge(topic=args.topic, frame_id=args.frame)
    # identity pose: [x, y, z, qw, qx, qy, qz]
    sample = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    try:
        if args.rate <= 0:
            bridge.publish(sample)
            return
        timer_period = 1.0 / args.rate
        bridge.create_timer(timer_period, lambda: bridge.publish(sample))
        rclpy.spin(bridge)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    _demo_main()

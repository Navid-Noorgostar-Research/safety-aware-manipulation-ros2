"""ROS2 / MoveIt2 export layer for AC-DiT outputs.

Turns the predicted (and safety-filtered) action into the four message
types a downstream stack actually consumes:

- ``geometry_msgs/PoseStamped`` -- already produced by
  :class:`net.ros_adapter.adapter.EEActionToROS2`; re-exported here so the
  whole export path is one object.
- ``geometry_msgs/Twist`` (and ``TwistStamped``) -- the ``cmd_vel`` channel
  for the mobile base. Exposed both as a direct constructor and as a
  finite-difference helper that derives a Twist from two consecutive EE
  pose targets.
- ``trajectory_msgs/JointTrajectory`` -- joint-space waypoints. AC-DiT
  emits Cartesian targets, so the caller is expected to have run IK; the
  exporter handles the per-arm joint-name defaults and the
  ``time_from_start`` accounting.
- A MoveIt2 ``MoveItPoseGoal`` (a focused subset of
  ``moveit_msgs/MotionPlanRequest``) -- group name, planning frame, target
  pose, planner knobs.

The exporter knows about per-arm joint name conventions and MoveIt group
names so it can populate sensible defaults from a :class:`RobotConfig`.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

from .adapter import EEActionToROS2, _to_float_list
from .messages import (
    Duration,
    Header,
    JointTrajectory,
    JointTrajectoryPoint,
    MoveItPoseGoal,
    PoseStamped,
    Time,
    Twist,
    TwistStamped,
    Vector3,
)


# Per-arm conventions matching the URDFs of the supported robots. Used to
# populate joint names, MoveIt group names, and end-effector links when the
# caller does not provide them explicitly.
ARM_DEFAULTS = {
    "franka_panda": {
        "joint_names": (
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ),
        "move_group": "panda_arm",
        "end_effector_link": "panda_link8",
    },
    "kuka_iiwa14": {
        "joint_names": (
            "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
            "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7",
        ),
        "move_group": "iiwa_arm",
        "end_effector_link": "iiwa_link_ee",
    },
    "ur5": {
        "joint_names": (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        "move_group": "manipulator",
        "end_effector_link": "tool0",
    },
    "none": {
        "joint_names": (),
        "move_group": "",
        "end_effector_link": "",
    },
}


def arm_defaults(arm: str) -> dict:
    """Return the joint-name / move-group / EE-link defaults for ``arm``."""
    if arm not in ARM_DEFAULTS:
        raise KeyError(
            f"Unknown arm {arm!r}. Available: {sorted(ARM_DEFAULTS)}."
        )
    return ARM_DEFAULTS[arm]


def _quat_conjugate(qw: float, qx: float, qy: float, qz: float):
    return qw, -qx, -qy, -qz


def _quat_mul(a, b):
    """Hamilton product for two scalar-first quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_axis_angle(qw: float, qx: float, qy: float, qz: float):
    """Return ``(axis, angle)`` -- angle in radians, axis a 3-tuple."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return (1.0, 0.0, 0.0), 0.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    # short-arc convention
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    angle = 2.0 * math.acos(max(-1.0, min(1.0, qw)))
    s = math.sqrt(max(0.0, 1.0 - qw * qw))
    if s < 1e-8:
        return (1.0, 0.0, 0.0), 0.0
    return (qx / s, qy / s, qz / s), angle


class MoveIt2Exporter:
    """Turn AC-DiT EE / joint / base outputs into ROS2 messages.

    A single exporter is bound to one robot (frame, scaling, group_name,
    joint_names). Build it via :meth:`from_robot_config` for the supported
    presets, or pass the four values directly.
    """

    def __init__(
        self,
        frame_id: str = "base_link",
        sim_to_meter: float = 0.2,
        joint_names: Optional[Sequence[str]] = None,
        move_group: str = "",
        end_effector_link: str = "",
    ) -> None:
        self.frame_id = frame_id
        self.sim_to_meter = float(sim_to_meter)
        self.joint_names: List[str] = list(joint_names or [])
        self.move_group = move_group
        self.end_effector_link = end_effector_link
        self._adapter = EEActionToROS2(
            frame_id=frame_id, sim_to_meter=self.sim_to_meter
        )

    # -- factories --------------------------------------------------------

    @classmethod
    def from_robot_config(
        cls,
        robot,
        *,
        joint_names: Optional[Sequence[str]] = None,
        move_group: Optional[str] = None,
        end_effector_link: Optional[str] = None,
    ) -> "MoveIt2Exporter":
        """Build an exporter from a :class:`net.hardware.RobotConfig`.

        ``robot`` accepts a ``RobotConfig``, a preset name, a YAML path, or
        a dict (anything :func:`net.hardware.load_robot_config` accepts).
        Per-arm defaults are derived from ``cfg.arm`` but each can be
        overridden with the matching keyword.
        """
        from net.hardware import load_robot_config  # local import to avoid cycle
        cfg = load_robot_config(robot)
        defs = arm_defaults(cfg.arm)
        return cls(
            frame_id=cfg.base_frame_id,
            sim_to_meter=cfg.sim_to_meter,
            joint_names=joint_names if joint_names is not None else list(defs["joint_names"]),
            move_group=move_group if move_group is not None else defs["move_group"],
            end_effector_link=end_effector_link if end_effector_link is not None else defs["end_effector_link"],
        )

    # -- pose stamped (delegated) ----------------------------------------

    def to_pose_stamped(self, action, stamp: Optional[float] = None) -> List[PoseStamped]:
        """Convert an EE action ``[B, 7]`` / ``[B, 8]`` into ``PoseStamped`` messages."""
        return self._adapter.to_pose_stamped(action, stamp=stamp)

    # -- MoveIt2 pose goal ------------------------------------------------

    def to_pose_goal(
        self,
        action,
        *,
        group_name: Optional[str] = None,
        end_effector_link: Optional[str] = None,
        allowed_planning_time: float = 5.0,
        max_velocity_scaling_factor: float = 1.0,
        max_acceleration_scaling_factor: float = 1.0,
        num_planning_attempts: int = 1,
        stamp: Optional[float] = None,
    ) -> List[MoveItPoseGoal]:
        """Wrap each EE action row into a MoveIt2 motion-plan pose goal.

        ``group_name`` / ``end_effector_link`` default to the values set on
        the exporter (which themselves default to the per-arm defaults).
        Planner knobs (``allowed_planning_time`` etc.) follow MoveIt2
        conventions.
        """
        gn = group_name if group_name is not None else self.move_group
        eel = end_effector_link if end_effector_link is not None else self.end_effector_link
        if not gn:
            raise ValueError(
                "MoveIt pose goal needs a group_name; pass one explicitly or "
                "use a robot config whose arm defines a MoveIt group."
            )
        if max_velocity_scaling_factor <= 0 or max_velocity_scaling_factor > 1:
            raise ValueError("max_velocity_scaling_factor must be in (0, 1].")
        if max_acceleration_scaling_factor <= 0 or max_acceleration_scaling_factor > 1:
            raise ValueError("max_acceleration_scaling_factor must be in (0, 1].")
        if allowed_planning_time <= 0:
            raise ValueError("allowed_planning_time must be > 0.")
        if num_planning_attempts < 1:
            raise ValueError("num_planning_attempts must be >= 1.")

        poses = self.to_pose_stamped(action, stamp=stamp)
        return [
            MoveItPoseGoal(
                group_name=gn,
                end_effector_link=eel,
                target_pose=p,
                allowed_planning_time=float(allowed_planning_time),
                max_velocity_scaling_factor=float(max_velocity_scaling_factor),
                max_acceleration_scaling_factor=float(max_acceleration_scaling_factor),
                num_planning_attempts=int(num_planning_attempts),
            )
            for p in poses
        ]

    # -- Twist (mobile base) ---------------------------------------------

    def to_twist(
        self,
        linear: Sequence[float],
        angular: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        scale_to_meters: bool = False,
    ) -> Twist:
        """Wrap explicit linear/angular velocities into a Twist.

        Args:
            linear:  3-tuple ``(vx, vy, vz)``, units depend on
                ``scale_to_meters``.
            angular: 3-tuple ``(wx, wy, wz)`` in rad/s.
            scale_to_meters: If True, treat ``linear`` as sim units / s and
                multiply by ``sim_to_meter`` to produce m/s. The default is
                False since most callers already work in SI.
        """
        if len(linear) != 3 or len(angular) != 3:
            raise ValueError("linear and angular must each have length 3.")
        s = self.sim_to_meter if scale_to_meters else 1.0
        return Twist(
            linear=Vector3(
                x=float(linear[0]) * s,
                y=float(linear[1]) * s,
                z=float(linear[2]) * s,
            ),
            angular=Vector3(
                x=float(angular[0]),
                y=float(angular[1]),
                z=float(angular[2]),
            ),
        )

    def to_twist_stamped(
        self,
        linear: Sequence[float],
        angular: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        scale_to_meters: bool = False,
        stamp: Optional[float] = None,
    ) -> TwistStamped:
        twist = self.to_twist(linear, angular, scale_to_meters=scale_to_meters)
        stamp_msg = Time.from_seconds(time.time() if stamp is None else stamp)
        return TwistStamped(
            header=Header(stamp=stamp_msg, frame_id=self.frame_id),
            twist=twist,
        )

    def twist_from_action_pair(
        self,
        prev_action: Sequence[float],
        next_action: Sequence[float],
        dt: float,
    ) -> Twist:
        """Derive a body-frame Twist from two consecutive EE pose actions.

        Linear velocity = ``(next_xyz - prev_xyz) * sim_to_meter / dt``.
        Angular velocity is the relative-rotation axis-angle / dt; the
        result is expressed in the previous-frame coordinate system, the
        same convention ``geometry_msgs/Twist`` follows.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive (got {dt}).")
        prev = list(prev_action)
        nxt = list(next_action)
        if len(prev) < 7 or len(nxt) < 7:
            raise ValueError(
                "actions must have layout [x, y, z, qw, qx, qy, qz, ...]; "
                f"got prev len {len(prev)}, next len {len(nxt)}."
            )
        # linear
        vx = (nxt[0] - prev[0]) * self.sim_to_meter / dt
        vy = (nxt[1] - prev[1]) * self.sim_to_meter / dt
        vz = (nxt[2] - prev[2]) * self.sim_to_meter / dt
        # angular: q_rel = q_prev^-1 * q_next, then axis-angle / dt
        q_prev_conj = _quat_conjugate(prev[3], prev[4], prev[5], prev[6])
        q_next = (nxt[3], nxt[4], nxt[5], nxt[6])
        q_rel = _quat_mul(q_prev_conj, q_next)
        axis, angle = _quat_to_axis_angle(*q_rel)
        wx = axis[0] * angle / dt
        wy = axis[1] * angle / dt
        wz = axis[2] * angle / dt
        return Twist(
            linear=Vector3(x=vx, y=vy, z=vz),
            angular=Vector3(x=wx, y=wy, z=wz),
        )

    # -- JointTrajectory --------------------------------------------------

    def to_joint_trajectory(
        self,
        joint_positions,
        *,
        dt: Optional[float] = None,
        time_from_start: Optional[Sequence[float]] = None,
        joint_names: Optional[Sequence[str]] = None,
        velocities=None,
        accelerations=None,
        stamp: Optional[float] = None,
    ) -> JointTrajectory:
        """Wrap a sequence of joint-space waypoints into a JointTrajectory.

        Args:
            joint_positions: ``[T, J]`` array / nested sequence. ``T``
                waypoints, each with ``J`` joint positions.
            dt: Uniform timestep between successive waypoints (seconds).
                Mutually exclusive with ``time_from_start``.
            time_from_start: Explicit per-waypoint timestamps relative to
                the trajectory start. Length must match ``T``.
            joint_names: Override the exporter's joint names. Length must
                match ``J`` (the trajectory width).
            velocities / accelerations: Optional same-shape arrays.
            stamp: Header timestamp; defaults to call time.
        """
        if dt is not None and time_from_start is not None:
            raise ValueError("pass either dt or time_from_start, not both.")
        rows = _to_float_list(joint_positions)
        if not rows:
            raise ValueError("joint_positions must contain at least one waypoint.")

        names = list(joint_names) if joint_names is not None else list(self.joint_names)
        if not names:
            raise ValueError(
                "JointTrajectory needs joint_names; pass an explicit list or "
                "configure them on the exporter / robot config."
            )
        width = len(rows[0])
        if width != len(names):
            raise ValueError(
                f"joint_positions has width {width} but {len(names)} joint "
                f"names were provided."
            )
        for r in rows:
            if len(r) != width:
                raise ValueError(
                    f"All waypoints must have width {width}; got {len(r)}."
                )

        if time_from_start is None:
            step = float(dt) if dt is not None else 0.1
            if step <= 0:
                raise ValueError(f"dt must be positive (got {step}).")
            tfs = [step * (i + 1) for i in range(len(rows))]
        else:
            tfs = [float(t) for t in time_from_start]
            if len(tfs) != len(rows):
                raise ValueError(
                    f"time_from_start has {len(tfs)} entries but there are "
                    f"{len(rows)} waypoints."
                )
            if any(t < 0 for t in tfs):
                raise ValueError("time_from_start values must be non-negative.")

        vel_rows = _to_float_list(velocities) if velocities is not None else None
        acc_rows = _to_float_list(accelerations) if accelerations is not None else None
        for label, arr in (("velocities", vel_rows), ("accelerations", acc_rows)):
            if arr is not None:
                if len(arr) != len(rows):
                    raise ValueError(f"{label} must have the same number of rows as joint_positions.")
                for r in arr:
                    if len(r) != width:
                        raise ValueError(f"{label} row width {len(r)} != joint count {width}.")

        points: List[JointTrajectoryPoint] = []
        for i, row in enumerate(rows):
            points.append(
                JointTrajectoryPoint(
                    positions=list(row),
                    velocities=list(vel_rows[i]) if vel_rows is not None else [],
                    accelerations=list(acc_rows[i]) if acc_rows is not None else [],
                    effort=[],
                    time_from_start=Duration.from_seconds(tfs[i]),
                )
            )

        stamp_msg = Time.from_seconds(time.time() if stamp is None else stamp)
        return JointTrajectory(
            header=Header(stamp=stamp_msg, frame_id=self.frame_id),
            joint_names=names,
            points=points,
        )

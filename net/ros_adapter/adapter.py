"""Convert predicted EE target actions into ROS2 PoseStamped messages.

The dynamics predictor (and SafetyAwareActionFilter) emits an EE target with
shape ``[..., 7]`` -- ``[x, y, z, qw, qx, qy, qz]`` -- or ``[..., 8]`` when the
gripper opening is included. Two transforms are required to make this consumable
on real hardware:

1. **Units.** Simulation positions are in scaled units (``0.1 sim == 2 cm``),
   while ROS2 messages are always in SI meters. The default conversion factor
   ``sim_to_meter = 0.2`` reproduces this scaling and can be overridden.
2. **Quaternion convention.** The model uses scalar-first ``[w, x, y, z]``;
   ROS2's ``geometry_msgs/Quaternion`` is scalar-last ``[x, y, z, w]``.

The adapter has no PyTorch / NumPy import-time dependency: tensor inputs are
detected duck-typed via ``__array__``/``tolist`` so the adapter stays usable
in environments without those libraries installed.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional, Sequence, Union

from .messages import Header, Point, Pose, PoseStamped, Quaternion, Time

# Numeric input may be a torch.Tensor, np.ndarray, or nested Python sequence.
ActionLike = Union[Sequence[float], "Iterable[float]"]


def _to_float_list(x) -> List[List[float]]:
    """Coerce torch/np/list input into a 2D Python ``[B, D]`` list of floats.

    A 1D input of length 7 or 8 is treated as a single (un-batched) sample and
    promoted to ``[1, D]``.
    """
    # torch.Tensor / np.ndarray both expose .detach()/.cpu()/.numpy()/.tolist()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    if not isinstance(x, list):
        x = list(x)
    if len(x) == 0:
        return []
    if not isinstance(x[0], (list, tuple)):
        x = [x]
    return [[float(v) for v in row] for row in x]


class EEActionToROS2:
    """Converter from an EE target action tensor to ROS2 ``PoseStamped``.

    Args:
        frame_id:        TF frame the published pose is expressed in. Defaults
                         to ``"base_link"`` -- the robot manipulator base, which
                         is the conventional target frame for MoveIt move_group
                         goals and Franka Cartesian impedance commands.
        sim_to_meter:    Multiplicative factor applied to XYZ when converting
                         from sim units to meters. Default ``0.2`` matches the
                         scaling documented in the project README
                         (``0.1 sim = 2 cm``).
        quat_eps:        Re-normalization epsilon. The safety filter already
                         normalizes, but we re-normalize defensively because a
                         non-unit quaternion is rejected by most ROS2 consumers.
    """

    def __init__(
        self,
        frame_id: str = "base_link",
        sim_to_meter: float = 0.2,
        quat_eps: float = 1e-8,
    ) -> None:
        self.frame_id = frame_id
        self.sim_to_meter = float(sim_to_meter)
        self.quat_eps = float(quat_eps)

    @classmethod
    def from_robot_config(cls, robot, *, quat_eps: float = 1e-8) -> "EEActionToROS2":
        """Construct an adapter from a :class:`net.hardware.RobotConfig` (or its name).

        ``robot`` may be a ``RobotConfig`` instance, a preset name like
        ``"franka_mobile"``, or anything else accepted by
        :func:`net.hardware.load_robot_config`.
        """
        from net.hardware import load_robot_config  # local import to avoid cycle
        cfg = load_robot_config(robot)
        return cls(
            frame_id=cfg.base_frame_id,
            sim_to_meter=cfg.sim_to_meter,
            quat_eps=quat_eps,
        )

    def to_pose_stamped(
        self,
        action: ActionLike,
        stamp: Optional[float] = None,
    ) -> List[PoseStamped]:
        """Convert one or many EE target actions to ``PoseStamped`` messages.

        Args:
            action: Tensor / array / sequence of shape ``[B, 7]`` or ``[B, 8]``.
                A 1D input of length 7 or 8 is accepted and treated as a single
                sample. The 8th element (gripper opening) is ignored here --
                it should be published on a separate gripper command topic.
            stamp:  POSIX seconds for the message header. Defaults to
                    ``time.time()`` at call time.

        Returns:
            A list of ``PoseStamped`` -- one per row of the input. Always a
            list, even for a single sample, so downstream loops are uniform.
        """
        rows = _to_float_list(action)
        if not rows:
            return []

        dim = len(rows[0])
        if dim not in (7, 8):
            raise ValueError(
                f"EE action must have last dim 7 or 8 (got {dim}). "
                "Expected layout [x, y, z, qw, qx, qy, qz] or "
                "[x, y, z, qw, qx, qy, qz, open]."
            )

        stamp_msg = Time.from_seconds(time.time() if stamp is None else stamp)

        out: List[PoseStamped] = []
        for row in rows:
            if len(row) != dim:
                raise ValueError(
                    f"Inconsistent action dimensions in batch: "
                    f"expected {dim}, got {len(row)}."
                )
            x, y, z, qw, qx, qy, qz = row[:7]
            # 1) sim units -> meters
            x *= self.sim_to_meter
            y *= self.sim_to_meter
            z *= self.sim_to_meter
            # 2) re-normalize quaternion defensively
            qn = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if qn < self.quat_eps:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                qw, qx, qy, qz = qw / qn, qx / qn, qy / qn, qz / qn
            # 3) scalar-first [w, x, y, z] -> ROS2 scalar-last [x, y, z, w]
            out.append(
                PoseStamped(
                    header=Header(stamp=stamp_msg, frame_id=self.frame_id),
                    pose=Pose(
                        position=Point(x=x, y=y, z=z),
                        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                    ),
                )
            )
        return out

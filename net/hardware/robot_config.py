"""Dataclasses describing a robot configuration.

The structure is intentionally small and pure-Python: no torch, numpy, or
hydra dependency at import time so it can be used both inside the training
pipeline and from the standalone ROS2 adapter (which runs in a ROS2-only
environment that may not have the full project conda env).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, replace
from typing import Any, Dict, Optional, Tuple


# Canonical robot kinds. Anything else raises during validation.
ROBOT_KINDS: Tuple[str, ...] = ("sim", "franka_mobile", "kuka_mobile", "husky_arm")


@dataclass(frozen=True)
class SafetyOverrides:
    """Per-robot safety thresholds.

    Each field is optional: only the keys present here override the matching
    entries in ``net/config/safety.yaml``. ``None`` means "use the safety
    yaml default". This keeps the global safety config as the single source
    of truth for tuning while still letting a Husky declare a 4 m workspace.
    """

    workspace_bounds: Optional[Tuple[Tuple[float, float], ...]] = None
    gripper_open_bounds: Optional[Tuple[float, float]] = None
    v_lin_max: Optional[float] = None
    v_ang_max: Optional[float] = None
    v_grip_max: Optional[float] = None
    base_speed_max: Optional[float] = None
    accel_lin_max: Optional[float] = None
    accel_ang_max: Optional[float] = None
    collision_margin: Optional[float] = None
    collision_pullback: Optional[float] = None
    dt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def apply_to(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of ``base`` with non-None overrides applied."""
        merged = dict(base)
        for k, v in self.to_dict().items():
            merged[k] = v
        return merged


@dataclass(frozen=True)
class RobotConfig:
    """Complete description of one robot target.

    The ROS2 adapter uses ``base_frame_id``, ``ee_topic`` and
    ``sim_to_meter``. The safety filter uses ``safety``. A planner / driver
    consumes the mobile-base fields when ``has_mobile_base`` is true.
    """

    name: str                      # "franka_mobile_lab1", free-form
    kind: str                      # one of ROBOT_KINDS
    arm: str                       # "none", "franka_panda", "kuka_iiwa14", "ur5"
    arm_dof: int                   # 0 for sim/no-arm
    has_mobile_base: bool
    base_frame_id: str             # frame the EE pose is expressed in
    ee_topic: str                  # PoseStamped command topic
    sim_to_meter: float = 0.2      # 1 sim unit -> meters
    gripper_topic: Optional[str] = None
    mobile_base_frame_id: Optional[str] = None  # e.g. "odom" or "map"
    base_cmd_topic: Optional[str] = None         # e.g. "/cmd_vel"
    description: str = ""
    safety: SafetyOverrides = field(default_factory=SafetyOverrides)

    # ---- validation ------------------------------------------------------

    def __post_init__(self) -> None:
        if self.kind not in ROBOT_KINDS:
            raise ValueError(
                f"Unknown robot kind {self.kind!r}; must be one of {ROBOT_KINDS}."
            )
        if self.arm_dof < 0:
            raise ValueError(f"arm_dof must be >= 0 (got {self.arm_dof}).")
        if self.sim_to_meter <= 0:
            raise ValueError(
                f"sim_to_meter must be positive (got {self.sim_to_meter})."
            )
        if self.has_mobile_base and self.mobile_base_frame_id is None:
            raise ValueError(
                f"{self.name}: has_mobile_base=True requires mobile_base_frame_id."
            )
        if not self.has_mobile_base and self.base_cmd_topic is not None:
            raise ValueError(
                f"{self.name}: base_cmd_topic is set but has_mobile_base=False."
            )
        if self.kind == "sim" and self.arm != "none":
            # sim is intentionally hardware-agnostic; the arm comes from the
            # generation config, not the robot config.
            raise ValueError("kind='sim' must have arm='none'.")

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Strip None overrides for a clean YAML round-trip.
        d["safety"] = self.safety.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        # Tolerate either a SafetyOverrides instance or a plain dict.
        safety_in = data.get("safety", {}) or {}
        if isinstance(safety_in, SafetyOverrides):
            safety = safety_in
        else:
            safety = _safety_from_dict(safety_in)
        known = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            raise ValueError(f"Unknown RobotConfig fields: {sorted(unknown)}")
        kwargs = {k: v for k, v in data.items() if k in known and k != "safety"}
        # Tuple coercion for workspace_bounds in safety would happen there.
        return cls(safety=safety, **kwargs)

    def replace(self, **changes: Any) -> "RobotConfig":
        return replace(self, **changes)


def _safety_from_dict(data: Dict[str, Any]) -> SafetyOverrides:
    known = {f.name for f in fields(SafetyOverrides)}
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(f"Unknown safety override fields: {sorted(unknown)}")
    norm: Dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            continue
        if k == "workspace_bounds":
            v = tuple(tuple(float(x) for x in row) for row in v)
        elif k == "gripper_open_bounds":
            v = tuple(float(x) for x in v)
        else:
            v = float(v)
        norm[k] = v
    return SafetyOverrides(**norm)

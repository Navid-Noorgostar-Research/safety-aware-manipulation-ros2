"""Hardware abstraction layer for the supported robot kinds.

Provides a single ``RobotConfig`` that captures everything a downstream
consumer (the ROS2 adapter, the safety filter, an inference loop) needs to
know about the target hardware:

- frame names and ROS2 topics
- sim-to-meter unit scaling
- mobile-base presence and command channel
- workspace, velocity and acceleration limits

Four canonical kinds ship as built-in presets:

- ``sim``           -- the simulator/inference default (no real hardware)
- ``franka_mobile`` -- Franka Panda (7-DoF) on a mobile base
- ``kuka_mobile``   -- KUKA iiwa14 (7-DoF) on a mobile base
- ``husky_arm``     -- Clearpath Husky base + 6-DoF arm

Lookup goes via :func:`load_robot_config`, which accepts either the preset
name (``"franka_mobile"``) or a path to a YAML file overriding any subset of
fields.
"""

from .robot_config import ROBOT_KINDS, RobotConfig, SafetyOverrides
from .registry import (
    PRESETS,
    list_presets,
    load_robot_config,
    register_preset,
)

__all__ = [
    "RobotConfig",
    "SafetyOverrides",
    "PRESETS",
    "ROBOT_KINDS",
    "list_presets",
    "load_robot_config",
    "register_preset",
]

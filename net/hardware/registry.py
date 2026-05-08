"""Registry of robot configs and the YAML loader.

The registry holds the four canonical presets. ``load_robot_config`` is the
single entry point and accepts either a preset name, a path to a YAML file
that may override any subset of fields, or a dict that has already been
parsed elsewhere (e.g. from a hydra config).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Union

from .robot_config import RobotConfig, SafetyOverrides

# A path-like or string with a ".yaml" suffix is treated as a YAML file;
# anything else is looked up in PRESETS.
PathLike = Union[str, os.PathLike]


# --- canonical presets ----------------------------------------------------

# Each preset captures the *minimum* sensible defaults for the kind. The
# safety overrides below only set values that genuinely differ from the
# global safety.yaml defaults (which assume a fixed-base tabletop arm).

_SIM = RobotConfig(
    name="sim",
    kind="sim",
    arm="none",
    arm_dof=0,
    has_mobile_base=False,
    base_frame_id="world",
    ee_topic="/sim/ee_target",
    sim_to_meter=0.2,
    description="Default simulator target. No real hardware; ROS2 adapter "
    "still works for end-to-end smoke tests.",
)

_FRANKA_MOBILE = RobotConfig(
    name="franka_mobile",
    kind="franka_mobile",
    arm="franka_panda",
    arm_dof=7,
    has_mobile_base=True,
    base_frame_id="panda_link0",
    mobile_base_frame_id="odom",
    ee_topic="/panda/equilibrium_pose",
    gripper_topic="/panda/franka_gripper/move",
    base_cmd_topic="/mobile_base/cmd_vel",
    sim_to_meter=0.2,
    description="Franka Panda 7-DoF arm mounted on a differential-drive "
    "mobile base. EE poses are commanded in the panda_link0 frame; the "
    "mobile base receives Twist commands on /mobile_base/cmd_vel.",
    safety=SafetyOverrides(
        # mobile reach extends past the tabletop workspace
        workspace_bounds=((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.5)),
        v_lin_max=0.5,
        base_speed_max=0.8,
    ),
)

_KUKA_MOBILE = RobotConfig(
    name="kuka_mobile",
    kind="kuka_mobile",
    arm="kuka_iiwa14",
    arm_dof=7,
    has_mobile_base=True,
    base_frame_id="iiwa_link_0",
    mobile_base_frame_id="odom",
    ee_topic="/iiwa/cartesian_pose",
    gripper_topic="/iiwa/gripper/command",
    base_cmd_topic="/mobile_base/cmd_vel",
    sim_to_meter=0.2,
    description="KUKA iiwa14 7-DoF arm on a mobile base. Same Cartesian "
    "interface as Franka via iiwa_ros2; differs in base frame name and "
    "Cartesian topic. FRI-bridged controllers are also compatible.",
    safety=SafetyOverrides(
        workspace_bounds=((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.5)),
        v_lin_max=0.5,
        v_ang_max=1.0,
        base_speed_max=0.8,
    ),
)

_HUSKY_ARM = RobotConfig(
    name="husky_arm",
    kind="husky_arm",
    arm="ur5",
    arm_dof=6,
    has_mobile_base=True,
    base_frame_id="ur_arm_base_link",
    mobile_base_frame_id="odom",
    ee_topic="/ur_arm/cartesian_pose",
    gripper_topic="/robotiq_2f_gripper/command",
    base_cmd_topic="/husky_velocity_controller/cmd_vel",
    sim_to_meter=0.2,
    description="Clearpath Husky A200 with a 6-DoF UR5 arm and Robotiq "
    "gripper. Husky cmd_vel is on the velocity controller topic; the arm "
    "frame is mounted to the Husky's top plate via ur_arm_base_link.",
    safety=SafetyOverrides(
        # Husky's outdoor envelope is large; the arm stays within UR5 reach
        workspace_bounds=((-3.0, 3.0), (-3.0, 3.0), (0.0, 1.2)),
        v_lin_max=0.4,
        base_speed_max=1.0,
        accel_lin_max=2.0,  # heavy mobile base, smoother accel
    ),
)


PRESETS: Dict[str, RobotConfig] = {
    cfg.kind: cfg for cfg in (_SIM, _FRANKA_MOBILE, _KUKA_MOBILE, _HUSKY_ARM)
}


def list_presets() -> List[str]:
    """Names of all built-in presets."""
    return sorted(PRESETS.keys())


def register_preset(cfg: RobotConfig, *, overwrite: bool = False) -> None:
    """Register a custom config under its kind. Raises if already registered."""
    if cfg.kind in PRESETS and not overwrite:
        raise ValueError(
            f"Preset {cfg.kind!r} already registered; pass overwrite=True."
        )
    PRESETS[cfg.kind] = cfg


# --- loader ---------------------------------------------------------------

def load_robot_config(spec: Union[str, PathLike, Mapping[str, Any], RobotConfig]) -> RobotConfig:
    """Resolve a spec to a ``RobotConfig``.

    Accepts:
        * a preset name, e.g. ``"franka_mobile"``
        * a path to a ``.yaml`` file
        * a dict (already parsed yaml or hydra DictConfig converted via
          ``OmegaConf.to_container``)
        * an existing ``RobotConfig`` (returned unchanged)
    """
    if isinstance(spec, RobotConfig):
        return spec
    if isinstance(spec, Mapping):
        return _from_mapping(dict(spec))
    if isinstance(spec, (str, os.PathLike)):
        s = os.fspath(spec)
        # YAML path?
        if s.endswith((".yaml", ".yml")) or os.sep in s or os.path.isfile(s):
            return _from_yaml_file(s)
        # otherwise treat as preset key
        if s not in PRESETS:
            raise KeyError(
                f"Unknown robot preset {s!r}. Available: {list_presets()}."
            )
        return PRESETS[s]
    raise TypeError(f"Cannot load robot config from {type(spec).__name__}.")


def _from_mapping(data: Dict[str, Any]) -> RobotConfig:
    # Allow nested ``robot:`` for hydra users that group fields.
    if set(data.keys()) == {"robot"}:
        data = dict(data["robot"])
    base_kind = data.get("kind") or data.get("base") or data.get("preset")
    # If a preset key is given, start from that preset and apply overrides.
    if base_kind in PRESETS and "name" not in data:
        cfg = PRESETS[base_kind]
        merged = cfg.to_dict()
        # safety overrides merge field-by-field
        if "safety" in data and data["safety"]:
            merged_safety = dict(merged.get("safety") or {})
            merged_safety.update(data["safety"])
            merged["safety"] = merged_safety
        for k, v in data.items():
            if k in ("safety", "preset"):
                continue
            merged[k] = v
        return RobotConfig.from_dict(merged)
    return RobotConfig.from_dict(data)


def _from_yaml_file(path: str) -> RobotConfig:
    import yaml  # imported lazily so the package works without pyyaml installed

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Robot config YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top-level YAML must be a mapping.")
    return _from_mapping(dict(data))

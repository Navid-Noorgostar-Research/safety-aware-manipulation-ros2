"""Tests for the robot hardware abstraction layer.

These tests exercise:

- the ``RobotConfig`` dataclass (validation, defaults, serialization)
- the four built-in presets (``sim``, ``franka_mobile``, ``kuka_mobile``,
  ``husky_arm``)
- the YAML loader on every shipped config file
- the safety-overrides merge semantics
- the ``EEActionToROS2.from_robot_config`` integration

They have no torch / numpy / rclpy dependency so they run in any
environment that has pyyaml installed.
"""

import os
from pathlib import Path

import pytest

from net.hardware import (
    PRESETS,
    ROBOT_KINDS,
    RobotConfig,
    SafetyOverrides,
    list_presets,
    load_robot_config,
    register_preset,
)
from net.ros_adapter import EEActionToROS2


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONFIG_DIR = REPO_ROOT / "net" / "config" / "robot"


# -- preset registry -------------------------------------------------------

def test_all_canonical_kinds_present():
    """The four robot kinds requested in the task are all registered."""
    expected = {"sim", "franka_mobile", "kuka_mobile", "husky_arm"}
    assert set(ROBOT_KINDS) == expected
    assert set(PRESETS.keys()) == expected
    assert set(list_presets()) == expected


@pytest.mark.parametrize("kind", ["sim", "franka_mobile", "kuka_mobile", "husky_arm"])
def test_preset_loads_by_name(kind):
    cfg = load_robot_config(kind)
    assert isinstance(cfg, RobotConfig)
    assert cfg.kind == kind


def test_preset_loads_unchanged_when_passed_as_config():
    """Passing a RobotConfig through ``load_robot_config`` is a no-op."""
    cfg = PRESETS["franka_mobile"]
    assert load_robot_config(cfg) is cfg


def test_unknown_preset_name_raises():
    with pytest.raises(KeyError, match="Unknown robot preset"):
        load_robot_config("nonexistent_robot")


# -- per-kind invariants ---------------------------------------------------

def test_sim_preset_invariants():
    cfg = PRESETS["sim"]
    assert cfg.arm == "none"
    assert cfg.arm_dof == 0
    assert cfg.has_mobile_base is False
    assert cfg.mobile_base_frame_id is None
    assert cfg.base_cmd_topic is None


def test_franka_mobile_preset_invariants():
    cfg = PRESETS["franka_mobile"]
    assert cfg.arm == "franka_panda"
    assert cfg.arm_dof == 7
    assert cfg.has_mobile_base is True
    assert cfg.base_frame_id == "panda_link0"
    assert cfg.mobile_base_frame_id is not None
    assert cfg.base_cmd_topic is not None
    assert "panda" in cfg.ee_topic


def test_kuka_mobile_preset_invariants():
    cfg = PRESETS["kuka_mobile"]
    assert cfg.arm == "kuka_iiwa14"
    assert cfg.arm_dof == 7
    assert cfg.has_mobile_base is True
    assert "iiwa" in cfg.base_frame_id
    assert "iiwa" in cfg.ee_topic


def test_husky_arm_preset_invariants():
    cfg = PRESETS["husky_arm"]
    assert cfg.arm == "ur5"
    assert cfg.arm_dof == 6
    assert cfg.has_mobile_base is True
    assert "husky" in (cfg.base_cmd_topic or "")
    # outdoor envelope is bigger than tabletop
    bounds = cfg.safety.workspace_bounds
    assert bounds is not None
    x_lo, x_hi = bounds[0]
    assert (x_hi - x_lo) >= 4.0


@pytest.mark.parametrize("kind", ROBOT_KINDS)
def test_every_preset_has_positive_sim_to_meter(kind):
    assert PRESETS[kind].sim_to_meter > 0


@pytest.mark.parametrize("kind", ROBOT_KINDS)
def test_mobile_kinds_declare_a_base_frame(kind):
    cfg = PRESETS[kind]
    if cfg.has_mobile_base:
        assert cfg.mobile_base_frame_id, kind
        assert cfg.base_cmd_topic, kind


# -- YAML loading ----------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_name",
    ["sim.yaml", "franka_mobile.yaml", "kuka_mobile.yaml", "husky_arm.yaml"],
)
def test_yaml_files_load(yaml_name):
    path = ROBOT_CONFIG_DIR / yaml_name
    assert path.is_file(), f"missing {path}"
    cfg = load_robot_config(str(path))
    # YAML's kind matches the filename stem
    assert cfg.kind == yaml_name.replace(".yaml", "")


def test_yaml_loader_accepts_pathlib_path():
    path = ROBOT_CONFIG_DIR / "sim.yaml"
    cfg = load_robot_config(path)
    assert cfg.kind == "sim"


def test_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_robot_config(str(ROBOT_CONFIG_DIR / "does_not_exist.yaml"))


def test_yaml_franka_overrides_match_preset():
    """Yaml-loaded config has the same safety overrides as the in-code preset."""
    yml = load_robot_config(str(ROBOT_CONFIG_DIR / "franka_mobile.yaml"))
    code = PRESETS["franka_mobile"]
    assert yml.safety.workspace_bounds == code.safety.workspace_bounds
    assert yml.safety.v_lin_max == code.safety.v_lin_max
    assert yml.safety.base_speed_max == code.safety.base_speed_max


# -- dict ingestion --------------------------------------------------------

def test_dict_with_only_kind_falls_back_to_preset():
    cfg = load_robot_config({"kind": "kuka_mobile"})
    assert cfg.base_frame_id == "iiwa_link_0"


def test_dict_kind_with_overrides_merges_on_top_of_preset():
    cfg = load_robot_config(
        {"kind": "franka_mobile", "ee_topic": "/custom/topic", "sim_to_meter": 1.0}
    )
    # Override applied
    assert cfg.ee_topic == "/custom/topic"
    assert cfg.sim_to_meter == 1.0
    # preset bits preserved
    assert cfg.arm == "franka_panda"
    assert cfg.has_mobile_base is True


def test_dict_safety_override_merges_field_by_field():
    cfg = load_robot_config(
        {"kind": "franka_mobile", "safety": {"v_lin_max": 0.1}}
    )
    # explicit override
    assert cfg.safety.v_lin_max == 0.1
    # preset overrides for *other* safety fields are preserved
    assert cfg.safety.workspace_bounds == ((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.5))
    assert cfg.safety.base_speed_max == 0.8


def test_nested_robot_key_is_unwrapped():
    cfg = load_robot_config({"robot": {"kind": "sim"}})
    assert cfg.kind == "sim"


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="Unknown RobotConfig fields"):
        RobotConfig.from_dict({"name": "x", "kind": "sim", "arm": "none",
                               "arm_dof": 0, "has_mobile_base": False,
                               "base_frame_id": "world", "ee_topic": "/x",
                               "totally_made_up": True})


def test_unknown_safety_field_raises():
    with pytest.raises(ValueError, match="Unknown safety override fields"):
        load_robot_config({"kind": "sim", "safety": {"made_up": 1.0}})


# -- validation ------------------------------------------------------------

def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown robot kind"):
        RobotConfig(
            name="x", kind="frobnicator", arm="none", arm_dof=0,
            has_mobile_base=False, base_frame_id="world", ee_topic="/x",
        )


def test_negative_arm_dof_raises():
    with pytest.raises(ValueError, match="arm_dof"):
        RobotConfig(
            name="x", kind="sim", arm="none", arm_dof=-1,
            has_mobile_base=False, base_frame_id="world", ee_topic="/x",
        )


def test_zero_sim_to_meter_raises():
    with pytest.raises(ValueError, match="sim_to_meter"):
        RobotConfig(
            name="x", kind="sim", arm="none", arm_dof=0,
            has_mobile_base=False, base_frame_id="world", ee_topic="/x",
            sim_to_meter=0.0,
        )


def test_mobile_base_without_frame_raises():
    with pytest.raises(ValueError, match="mobile_base_frame_id"):
        RobotConfig(
            name="x", kind="franka_mobile", arm="franka_panda", arm_dof=7,
            has_mobile_base=True, base_frame_id="panda_link0",
            ee_topic="/x",
        )


def test_base_cmd_topic_without_mobile_base_raises():
    with pytest.raises(ValueError, match="has_mobile_base=False"):
        RobotConfig(
            name="x", kind="sim", arm="none", arm_dof=0,
            has_mobile_base=False, base_frame_id="world", ee_topic="/x",
            base_cmd_topic="/cmd_vel",
        )


def test_sim_must_have_no_arm():
    with pytest.raises(ValueError, match="kind='sim'"):
        RobotConfig(
            name="x", kind="sim", arm="franka_panda", arm_dof=7,
            has_mobile_base=False, base_frame_id="world", ee_topic="/x",
        )


# -- safety overrides ------------------------------------------------------

def test_safety_overrides_apply_to_skips_none_fields():
    base = {"v_lin_max": 0.25, "v_ang_max": 0.785, "base_speed_max": 0.25}
    overrides = SafetyOverrides(v_lin_max=0.5)
    merged = overrides.apply_to(base)
    assert merged == {"v_lin_max": 0.5, "v_ang_max": 0.785, "base_speed_max": 0.25}


def test_safety_overrides_to_dict_strips_nones():
    overrides = SafetyOverrides(v_lin_max=0.5, v_ang_max=None)
    d = overrides.to_dict()
    assert d == {"v_lin_max": 0.5}


def test_safety_workspace_bounds_coerced_to_nested_tuples():
    cfg = load_robot_config(
        {"kind": "sim", "safety": {"workspace_bounds": [[-1, 1], [-2, 2], [0, 3]]}}
    )
    bounds = cfg.safety.workspace_bounds
    assert isinstance(bounds, tuple)
    assert all(isinstance(row, tuple) for row in bounds)
    assert bounds == ((-1.0, 1.0), (-2.0, 2.0), (0.0, 3.0))


# -- serialization round-trip ---------------------------------------------

@pytest.mark.parametrize("kind", ROBOT_KINDS)
def test_to_dict_round_trip(kind):
    cfg = PRESETS[kind]
    d = cfg.to_dict()
    rebuilt = RobotConfig.from_dict(d)
    assert rebuilt == cfg


# -- registration ----------------------------------------------------------

def test_register_preset_rejects_duplicate(monkeypatch):
    snapshot = dict(PRESETS)
    monkeypatch.setattr("net.hardware.registry.PRESETS", snapshot.copy())
    # importing again so the module-level reference matches the patched copy
    from net.hardware import registry as reg
    cfg = RobotConfig(
        name="sim_alt", kind="sim", arm="none", arm_dof=0,
        has_mobile_base=False, base_frame_id="world",
        ee_topic="/sim/ee_target",
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register_preset(cfg)
    reg.register_preset(cfg, overwrite=True)
    assert reg.PRESETS["sim"].name == "sim_alt"


# -- ROS2 adapter integration ---------------------------------------------

@pytest.mark.parametrize("kind", ROBOT_KINDS)
def test_ros_adapter_built_from_robot_config(kind):
    cfg = PRESETS[kind]
    adapter = EEActionToROS2.from_robot_config(cfg)
    assert adapter.frame_id == cfg.base_frame_id
    assert adapter.sim_to_meter == cfg.sim_to_meter


def test_ros_adapter_built_from_preset_name():
    adapter = EEActionToROS2.from_robot_config("husky_arm")
    assert adapter.frame_id == "ur_arm_base_link"


def test_ros_adapter_emits_correct_frame_for_franka():
    """End-to-end: convert a sample action and confirm header.frame_id."""
    adapter = EEActionToROS2.from_robot_config("franka_mobile")
    out = adapter.to_pose_stamped([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert out[0].header.frame_id == "panda_link0"


def test_ros_adapter_scaling_consistent_for_each_kind():
    """The adapter's sim_to_meter must match the config it was built from."""
    for kind in ROBOT_KINDS:
        cfg = PRESETS[kind]
        adapter = EEActionToROS2.from_robot_config(kind)
        out = adapter.to_pose_stamped([5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert out[0].pose.position.x == pytest.approx(5.0 * cfg.sim_to_meter)


# -- topics differ for each kind (sanity check the abstraction works) ------

def test_each_kind_has_distinct_ee_topic():
    topics = {PRESETS[k].ee_topic for k in ROBOT_KINDS}
    assert len(topics) == len(ROBOT_KINDS)


def test_each_mobile_kind_has_a_base_command_topic():
    for k in ROBOT_KINDS:
        cfg = PRESETS[k]
        if cfg.has_mobile_base:
            assert cfg.base_cmd_topic, f"{k} missing base_cmd_topic"

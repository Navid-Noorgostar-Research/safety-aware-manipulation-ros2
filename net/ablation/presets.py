"""Canonical ablation studies that ship with the framework.

These are the studies the paper / appendix would cite. They use the same
defaults as :file:`net/config/safety.yaml` and the robot presets in
:mod:`net.hardware.registry`, so re-running them reproduces the numbers
quoted in the README.
"""

from __future__ import annotations

from typing import Mapping

from .ablation import AblationConfig, AblationStudy


# Defaults mirror net/config/safety.yaml. Frozen here so the ablation study
# is reproducible without depending on a hydra config resolver.
SAFETY_FILTER_BASE = dict(
    enabled=True,
    workspace_bounds=[[-0.30, 0.30], [-0.50, 0.50], [0.00, 0.12]],
    gripper_open_bounds=[0.0, 0.10],
    v_lin_max=0.25,
    v_ang_max=0.785,
    v_grip_max=0.125,
    base_speed_max=0.25,
    collision_margin=0.01,
    collision_pullback=0.5,
    accel_lin_max=5.0,
    accel_ang_max=20.0,
    dt=0.05,
)

# A "loose" workspace effectively disables the joint-limits check.
_HUGE_WORKSPACE = [[-1e6, 1e6], [-1e6, 1e6], [-1e6, 1e6]]
_HUGE_GRIPPER = [-1e6, 1e6]


def safety_filter_study(seed: int = 0) -> AblationStudy:
    """Standard 5-constraint ablation of the safety filter.

    Each row removes (or weakens to a no-op) one of the five checks; the
    final ``disabled`` row turns the filter off entirely. This is the
    table you would publish to argue that every constraint contributes.
    """
    return AblationStudy(
        name="safety_filter",
        base_config=SAFETY_FILTER_BASE,
        ablations=[
            AblationConfig(
                name="full",
                description="all five constraints active (baseline)",
            ),
            AblationConfig(
                name="no_joint_limits",
                overrides={
                    "workspace_bounds": _HUGE_WORKSPACE,
                    "gripper_open_bounds": _HUGE_GRIPPER,
                },
                description="disable workspace + gripper clamps",
            ),
            AblationConfig(
                name="no_velocity_limits",
                overrides={"v_lin_max": 1e6, "v_ang_max": 1e6, "v_grip_max": 1e6},
                description="disable per-DoF velocity caps",
            ),
            AblationConfig(
                name="no_base_speed",
                overrides={"base_speed_max": 1e6},
                description="disable scalar |xyz| velocity cap",
            ),
            AblationConfig(
                name="no_smoothness",
                overrides={"accel_lin_max": 1e6},
                description="disable acceleration smoothness cap",
            ),
            AblationConfig(
                name="no_collision",
                overrides={"collision_margin": 0.0, "collision_pullback": 1.0},
                description="disable collision pullback",
            ),
            AblationConfig(
                name="disabled",
                overrides={"enabled": False},
                description="bypass filter entirely (lower bound)",
            ),
        ],
        seed=seed,
    )


def robot_workspace_study(seed: int = 0) -> AblationStudy:
    """Workspace-tightness sweep on the default tabletop robot.

    Uses the safety filter's joint-limits check to demonstrate that
    tightening / loosening the workspace box trades off correction
    magnitude vs. flagged-sample rate, with all other checks held fixed.
    """
    return AblationStudy(
        name="robot_workspace",
        base_config=SAFETY_FILTER_BASE,
        ablations=[
            AblationConfig(
                name="tight",
                overrides={"workspace_bounds": [[-0.10, 0.10], [-0.20, 0.20], [0.00, 0.06]]},
                description="half the default workspace box",
            ),
            AblationConfig(
                name="default",
                description="net/config/safety.yaml workspace box",
            ),
            AblationConfig(
                name="loose",
                overrides={"workspace_bounds": [[-0.60, 0.60], [-1.00, 1.00], [0.00, 0.24]]},
                description="2x the default workspace box",
            ),
        ],
        seed=seed,
    )

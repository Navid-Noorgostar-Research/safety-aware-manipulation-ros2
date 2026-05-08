"""Cross-feature integration tests.

Each isolated feature has its own dense test file. This file exercises
*compositions* -- the place where bugs hide because each side passes
on its own but the contract between them quietly drifts.

Coverage matrix (rows = features, cols = combinations):

    1. Safety-Aware Action Filter
    2. ROS2 + MoveIt2 Export Layer
    3. Franka/Husky Hardware Abstraction
    4. Uncertainty-Aware AC-DiT
    5. Closed-Loop Replanning
    6. Ablation Framework

Pairings exercised below: 1+2, 2+3, 1+3, 4+5, 1+4, 1+5, 6+1, 1+2+3,
1+2+3+4+5, plus a final 1+2+3+4+5+6 end-to-end run.
"""

from types import SimpleNamespace
import math
import random

import pytest

torch = pytest.importorskip("torch")

from net.ablation import (  # noqa: E402
    safety_filter_evaluator,
    safety_filter_study,
)
from net.control import (  # noqa: E402
    HalvingPredictor,
    NoisyWorld,
    ReplannerConfig,
    compare_loops,
    run_closed_loop,
)
from net.hardware import (  # noqa: E402
    PRESETS,
    ROBOT_KINDS,
    load_robot_config,
)
from net.model.safety_filter import SafetyAwareActionFilter  # noqa: E402
from net.ros_adapter import (  # noqa: E402
    EEActionToROS2,
    JointTrajectory,
    MoveIt2Exporter,
    MoveItPoseGoal,
    PoseStamped,
    Twist,
    arm_defaults,
)
from net.uncertainty import (  # noqa: E402
    ensemble_predict,
    make_uncertainty_aware_predict_fn,
    mean_std_score,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safety_config(**overrides):
    """Default safety config matching net/config/safety.yaml."""
    defaults = dict(
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
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _identity_action_tensor(pos=(0.0, 0.0, 0.05)):
    return torch.tensor([list(pos) + [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)


# ===========================================================================
# 1 + 2: Safety filter -> ROS2/MoveIt2 export
# ===========================================================================


def test_safety_filter_output_round_trips_through_pose_stamped():
    """Filtered action is convertible to PoseStamped without dim-mangling."""
    filt = SafetyAwareActionFilter(_safety_config())
    obs = _identity_action_tensor((0.0, 0.0, 0.05))
    tgt = _identity_action_tensor((10.0, 10.0, 10.0))   # extreme target
    safe, _ = filt(obs, tgt)
    safe_list = safe.detach().cpu().tolist()

    adapter = EEActionToROS2(frame_id="panda_link0", sim_to_meter=0.2)
    poses = adapter.to_pose_stamped(safe_list)
    assert len(poses) == 1
    p = poses[0]
    assert isinstance(p, PoseStamped)
    # filtered position must be inside workspace box (verified in m)
    bounds = ((-0.30, 0.30), (-0.50, 0.50), (0.00, 0.12))
    for i, axis in enumerate(("x", "y", "z")):
        v_meters = getattr(p.pose.position, axis)
        v_units = v_meters / 0.2
        assert bounds[i][0] - 1e-6 <= v_units <= bounds[i][1] + 1e-6


def test_safety_filter_output_to_moveit_pose_goal():
    """The full chain: predict + filter + export to a MoveIt pose goal."""
    filt = SafetyAwareActionFilter(_safety_config())
    obs = _identity_action_tensor()
    tgt = _identity_action_tensor((100.0, 0.0, 0.05))
    safe, info = filt(obs, tgt)

    exporter = MoveIt2Exporter.from_robot_config("franka_mobile")
    goal = exporter.to_pose_goal(safe.detach().cpu().tolist())[0]
    assert isinstance(goal, MoveItPoseGoal)
    assert goal.group_name == "panda_arm"
    assert goal.target_pose.header.frame_id == "panda_link0"
    # safety filter intervened on at least one constraint
    assert bool(info["any_violation"][0]) is True


def test_safety_filter_output_to_joint_trajectory_post_ik():
    """A joint-space placeholder (post-IK) flows through the exporter cleanly."""
    # filter doesn't touch joint-space; it's the EE pose that gets clamped.
    # Here we just confirm the JointTrajectory builder works with the same
    # robot the safety filter was configured for.
    exporter = MoveIt2Exporter.from_robot_config("kuka_mobile")
    positions = [[0.0] * 7, [0.1] * 7, [0.2] * 7]
    traj = exporter.to_joint_trajectory(positions, dt=0.05)
    assert isinstance(traj, JointTrajectory)
    assert traj.joint_names == list(arm_defaults("kuka_iiwa14")["joint_names"])
    assert len(traj.points) == 3


# ===========================================================================
# 2 + 3: Robot config drives the exporter for every kind
# ===========================================================================


@pytest.mark.parametrize("kind", list(ROBOT_KINDS))
def test_exporter_built_from_each_robot_config_emits_correct_frame(kind):
    cfg = load_robot_config(kind)
    exporter = MoveIt2Exporter.from_robot_config(cfg)
    poses = exporter.to_pose_stamped([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert poses[0].header.frame_id == cfg.base_frame_id


@pytest.mark.parametrize("kind", ["franka_mobile", "kuka_mobile", "husky_arm"])
def test_pose_goal_uses_correct_arm_group_per_kind(kind):
    cfg = load_robot_config(kind)
    exporter = MoveIt2Exporter.from_robot_config(cfg)
    goal = exporter.to_pose_goal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[0]
    expected_group = arm_defaults(cfg.arm)["move_group"]
    assert goal.group_name == expected_group
    assert goal.target_pose.header.frame_id == cfg.base_frame_id


def test_sim_robot_does_not_support_pose_goal_or_joint_trajectory():
    """sim has arm='none' -> no MoveIt group, no joint names."""
    exporter = MoveIt2Exporter.from_robot_config("sim")
    with pytest.raises(ValueError, match="group_name"):
        exporter.to_pose_goal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="joint_names"):
        exporter.to_joint_trajectory([[0.0]], dt=0.1)


# ===========================================================================
# 1 + 3: Robot config's safety overrides drive the safety filter
# ===========================================================================


def test_robot_config_safety_overrides_actually_change_filter_behaviour():
    """The franka_mobile preset declares a much wider workspace than the
    tabletop default; a target inside that wider envelope must be passed
    by the franka_mobile filter and clamped by the default-safety filter."""
    franka_safety = PRESETS["franka_mobile"].safety
    bounds = list(map(list, franka_safety.workspace_bounds))

    wide_filter = SafetyAwareActionFilter(_safety_config(workspace_bounds=bounds))
    narrow_filter = SafetyAwareActionFilter(_safety_config())

    obs = _identity_action_tensor((0.0, 0.0, 0.05))
    tgt = _identity_action_tensor((1.5, 0.0, 1.0))   # outside narrow, inside wide

    _, narrow_info = narrow_filter(obs, tgt)
    _, wide_info = wide_filter(obs, tgt)

    assert bool(narrow_info["joint_limits"][0]) is True
    assert bool(wide_info["joint_limits"][0]) is False


# ===========================================================================
# 4 + 5: Uncertainty-aware closed-loop replanning
# ===========================================================================


def _halving_with_seed_noise(noise_std):
    base = HalvingPredictor()
    def sample_fn(obs, horizon, goal, sample_id):
        rng = random.Random(sample_id * 1000 + 7)
        actions = base(obs, horizon, goal)
        return [tuple(a + rng.gauss(0, noise_std) for a in row) for row in actions]
    return sample_fn


def test_uncertainty_aware_closed_loop_converges():
    """An ensemble-driven predict_fn drives run_closed_loop to the goal."""
    sample_fn = _halving_with_seed_noise(noise_std=0.1)
    predict_fn = make_uncertainty_aware_predict_fn(sample_fn, n_samples=8)
    res = run_closed_loop(
        predict_fn, NoisyWorld(noise_std=0.0, seed=0), [0.0],
        config=ReplannerConfig(horizon=5, execute_steps=1, max_steps=20, goal_tolerance=0.05),
        goal=[10.0],
    )
    assert res.success is True
    assert abs(res.final_observation[0] - 10.0) < 0.05


def test_uncertainty_gate_increases_replan_frequency_under_disagreement():
    """High-noise sampler triggers the gate -> per-step horizon is short_horizon."""
    sample_fn = _halving_with_seed_noise(noise_std=1.0)
    predict_fn = make_uncertainty_aware_predict_fn(
        sample_fn, n_samples=8, score_threshold=0.001, short_horizon=1,
    )
    res = run_closed_loop(
        predict_fn, NoisyWorld(noise_std=0.0, seed=0), [0.0],
        config=ReplannerConfig(horizon=5, execute_steps=1, max_steps=15, goal_tolerance=0.05),
        goal=[10.0],
    )
    assert all(len(r.predicted) == 1 for r in res.records)


def test_compare_loops_with_uncertainty_aware_predict_fn():
    """Closed-loop with ensemble still beats open-loop under systematic drift."""
    sample_fn = _halving_with_seed_noise(noise_std=0.0)   # deterministic-enough
    predict_fn = make_uncertainty_aware_predict_fn(sample_fn, n_samples=4)
    cmp = compare_loops(
        predict_fn,
        execute_fn_factory=lambda: NoisyWorld(noise_std=0.0, drift=0.3, seed=0),
        initial_observation=[0.0],
        config=ReplannerConfig(horizon=8, execute_steps=1, max_steps=30, goal_tolerance=0.0),
        goal=[10.0],
    )
    assert cmp.closed_final_error < cmp.open_final_error


# ===========================================================================
# 1 + 4: Safety filter inside the ensemble's sample_fn
# ===========================================================================


def test_safety_filter_applied_inside_each_ensemble_sample():
    """Each ensemble sample is post-filter; mean must still be inside the workspace."""
    filt = SafetyAwareActionFilter(_safety_config())

    def sample_fn(obs, horizon, goal, sample_id):
        rng = random.Random(sample_id)
        actions = []
        cur = list(obs)
        for _ in range(horizon):
            # raw proposal: random target outside the workspace
            raw = [c + rng.gauss(0, 1.0) + 5.0 for c in cur]
            quat = [1.0, 0.0, 0.0, 0.0]
            target = torch.tensor(
                [raw + quat], dtype=torch.float32,
            )
            observed = torch.tensor(
                [list(cur) + quat], dtype=torch.float32,
            )
            safe, _ = filt(observed, target)
            safe_pos = safe[0, :3].detach().cpu().tolist()
            actions.append(tuple(safe_pos))
            cur = safe_pos
        return actions

    res = ensemble_predict(sample_fn, [0.0, 0.0, 0.05], horizon=4, n_samples=8)
    bounds = ((-0.30, 0.30), (-0.50, 0.50), (0.00, 0.12))
    for action in res.mean_actions:
        for i, v in enumerate(action):
            assert bounds[i][0] - 1e-3 <= v <= bounds[i][1] + 1e-3, \
                f"action axis {i} = {v} outside workspace {bounds[i]}"


# ===========================================================================
# 1 + 5: Safety filter inside the closed-loop predict_fn
# ===========================================================================


def test_closed_loop_with_safety_filtered_predict_fn():
    """A closed-loop run whose predict_fn applies the safety filter to every
    proposed action stays inside the workspace at all times."""
    filt = SafetyAwareActionFilter(_safety_config())

    def predict_fn(obs, horizon, goal):
        # naive: each step tries to step 50% toward goal in xyz, then clamp
        actions = []
        cur = list(obs)
        for _ in range(horizon):
            raw = [c + 0.5 * (g - c) for c, g in zip(cur, goal)]
            quat = [1.0, 0.0, 0.0, 0.0]
            observed = torch.tensor([list(cur) + quat], dtype=torch.float32)
            target = torch.tensor([raw + quat], dtype=torch.float32)
            safe, _ = filt(observed, target)
            safe_pos = safe[0, :3].detach().cpu().tolist()
            actions.append(tuple(safe_pos))
            cur = safe_pos
        return actions

    def execute_fn(obs, action):
        # action is the new position directly (predict_fn already produced absolute targets)
        return tuple(action), {}

    res = run_closed_loop(
        predict_fn, execute_fn, (0.0, 0.0, 0.05),
        config=ReplannerConfig(horizon=4, execute_steps=1, max_steps=20, goal_tolerance=0.0),
        goal=(10.0, 10.0, 10.0),   # extreme goal; filter must clamp
    )
    bounds = ((-0.30, 0.30), (-0.50, 0.50), (0.00, 0.12))
    for r in res.records:
        for i, v in enumerate(r.new_observation):
            assert bounds[i][0] - 1e-3 <= v <= bounds[i][1] + 1e-3, \
                f"step {r.t} axis {i} = {v} outside workspace {bounds[i]}"


# ===========================================================================
# 6 + 1: Ablation framework actually drives safety filter behaviour
# ===========================================================================


def test_ablation_diagonal_property_end_to_end():
    """Each `no_X` row zeros only its own constraint; `disabled` zeros all."""
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    diagonal_pairs = [
        ("no_joint_limits",    "joint_violations"),
        ("no_velocity_limits", "velocity_violations"),
        ("no_base_speed",      "base_speed_violations"),
        ("no_smoothness",      "smoothness_violations"),
        ("no_collision",       "collision_violations"),
    ]
    full = results["full"].metrics
    for ablation_name, metric in diagonal_pairs:
        row = results[ablation_name].metrics
        assert row[metric] == 0.0, f"{ablation_name} did not zero {metric}"
        # other constraints still fire on the same data
        other_metrics = [m for _, m in diagonal_pairs if m != metric]
        assert any(row[m] > 0.0 for m in other_metrics), \
            f"{ablation_name} unexpectedly zeroed every constraint"
        # baseline fires every constraint
        assert full[metric] > 0.0
    # disabled row zeros everything
    disabled = results["disabled"].metrics
    for v in disabled.values():
        assert v == 0.0


# ===========================================================================
# 1 + 2 + 3 + 4 + 5 + 6: All six features wired together end-to-end
# ===========================================================================


def test_full_pipeline_smoke():
    """Smoke test: hardware config -> ensemble -> safety filter -> exporter
    -> closed-loop -> ablation row.

    The point is not to validate every numerical detail (each feature's
    own suite does that) -- it's to ensure the contracts hold across
    every interface so the full pipeline composes without errors.
    """
    # 3) Hardware abstraction picks a robot
    cfg = load_robot_config("franka_mobile")

    # 2) Exporter built from that config
    exporter = MoveIt2Exporter.from_robot_config(cfg)

    # 1) Safety filter built with default safety config (or robot overrides)
    safety_overrides = cfg.safety
    s_kwargs = _safety_config().__dict__
    if safety_overrides.workspace_bounds is not None:
        s_kwargs["workspace_bounds"] = list(map(list, safety_overrides.workspace_bounds))
    if safety_overrides.v_lin_max is not None:
        s_kwargs["v_lin_max"] = safety_overrides.v_lin_max
    filt = SafetyAwareActionFilter(SimpleNamespace(**s_kwargs))

    # 4) Ensemble sample_fn that filters every sample through the safety filter
    def sample_fn(obs, horizon, goal, sample_id):
        rng = random.Random(sample_id)
        actions = []
        cur = list(obs)
        for _ in range(horizon):
            quat = [1.0, 0.0, 0.0, 0.0]
            observed = torch.tensor([list(cur) + quat], dtype=torch.float32)
            raw = [c + 0.5 * (g - c) + rng.gauss(0, 0.01) for c, g in zip(cur, goal)]
            target = torch.tensor([raw + quat], dtype=torch.float32)
            safe, _ = filt(observed, target)
            cur = safe[0, :3].detach().cpu().tolist()
            actions.append(tuple(cur))
        return actions

    # 5) Closed-loop replanner with uncertainty-aware predict_fn
    predict_fn = make_uncertainty_aware_predict_fn(
        sample_fn, n_samples=4, score_fn=mean_std_score,
    )

    def execute_fn(obs, action):
        return tuple(action), {}

    res = run_closed_loop(
        predict_fn, execute_fn, (0.0, 0.0, 0.05),
        # max_steps must accommodate velocity-limited halving: each step is
        # capped at v_lin_max*dt, so reaching a goal 0.2 units away takes
        # many tiny steps even though the controller is geometric.
        config=ReplannerConfig(horizon=3, execute_steps=1, max_steps=60, goal_tolerance=0.05),
        goal=(0.2, 0.2, 0.1),   # inside workspace
    )
    assert res.success is True, (
        f"closed-loop did not converge within max_steps; "
        f"final={res.final_observation}, total_steps={res.total_steps}"
    )

    # 2) Final action exported as a MoveIt2 pose goal in the right frame
    final_action = list(res.final_observation) + [1.0, 0.0, 0.0, 0.0]
    goal_msg = exporter.to_pose_goal(final_action)[0]
    assert isinstance(goal_msg, MoveItPoseGoal)
    assert goal_msg.target_pose.header.frame_id == cfg.base_frame_id
    assert goal_msg.group_name == arm_defaults(cfg.arm)["move_group"]

    # 6) Ablation framework still produces a sensible study with the same
    #    safety filter underneath.
    ab = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert len(ab) == 7
    assert ab["full"].metrics["any_violations"] > 0.0
    assert ab["disabled"].metrics["any_violations"] == 0.0


# ===========================================================================
# Smoke test: every feature module imports cleanly
# ===========================================================================


def test_every_feature_package_imports_cleanly():
    """Importing the six feature packages must not have side effects."""
    import importlib
    for pkg in (
        "net.model.safety_filter",
        "net.ros_adapter",
        "net.ros_adapter.exporters",
        "net.hardware",
        "net.uncertainty",
        "net.control",
        "net.ablation",
    ):
        importlib.import_module(pkg)


def test_every_robot_kind_round_trips_through_full_export_chain():
    """For each robot kind, build exporter from config and emit all 4 messages."""
    for kind in ROBOT_KINDS:
        cfg = load_robot_config(kind)
        exporter = MoveIt2Exporter.from_robot_config(cfg)
        action = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # PoseStamped works for every kind
        poses = exporter.to_pose_stamped(action)
        assert poses[0].header.frame_id == cfg.base_frame_id

        # Twist works for every kind
        twist = exporter.to_twist((0.1, 0.0, 0.0))
        assert isinstance(twist, Twist)

        # MoveIt + JointTrajectory require an arm
        if cfg.arm != "none":
            goal = exporter.to_pose_goal(action)[0]
            assert goal.group_name == arm_defaults(cfg.arm)["move_group"]
            j = len(arm_defaults(cfg.arm)["joint_names"])
            traj = exporter.to_joint_trajectory([[0.0] * j, [0.1] * j], dt=0.1)
            assert len(traj.joint_names) == j

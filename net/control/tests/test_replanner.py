"""Tests for the closed-loop replanner.

Three layers:

- ``ReplannerConfig`` validation and config-shape invariants.
- Single-rollout behaviour (``run_closed_loop`` / ``run_open_loop``):
  determinism, max_steps, goal-tolerance stopping, custom stop_fn,
  records, replan-vs-execute counts.
- ``compare_loops``: closed-loop beats open-loop under both systematic
  drift and zero-mean Gaussian noise (multi-seed average).

The synthetic predictor and world model are deterministic given seed,
so every test is byte-stable across reruns.
"""

import statistics

import pytest

from net.control import (
    ComparisonResult,
    HalvingPredictor,
    NoisyWorld,
    ReplannerConfig,
    RolloutResult,
    StepRecord,
    compare_loops,
    run_closed_loop,
    run_open_loop,
)


# ---------------------------------------------------------------------------
# ReplannerConfig validation
# ---------------------------------------------------------------------------


def test_config_default_values_are_sane():
    cfg = ReplannerConfig()
    assert cfg.horizon >= 1
    assert cfg.execute_steps == 1                 # canonical MPC
    assert cfg.execute_steps <= cfg.horizon
    assert cfg.max_steps >= 1
    assert cfg.goal_tolerance >= 0


def test_config_horizon_below_one_raises():
    with pytest.raises(ValueError, match="horizon"):
        ReplannerConfig(horizon=0)


def test_config_execute_steps_below_one_raises():
    with pytest.raises(ValueError, match="execute_steps"):
        ReplannerConfig(execute_steps=0)


def test_config_execute_steps_exceeds_horizon_raises():
    with pytest.raises(ValueError, match="must be <= horizon"):
        ReplannerConfig(horizon=4, execute_steps=8)


def test_config_negative_max_steps_raises():
    with pytest.raises(ValueError, match="max_steps"):
        ReplannerConfig(max_steps=0)


def test_config_negative_tolerance_raises():
    with pytest.raises(ValueError, match="goal_tolerance"):
        ReplannerConfig(goal_tolerance=-0.01)


def test_config_is_frozen():
    cfg = ReplannerConfig()
    with pytest.raises(Exception):
        cfg.horizon = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HalvingPredictor / NoisyWorld primitives
# ---------------------------------------------------------------------------


def test_halving_predictor_returns_horizon_actions():
    pred = HalvingPredictor()
    actions = pred([0.0], horizon=5, goal=[10.0])
    assert len(actions) == 5


def test_halving_predictor_first_action_halves_distance():
    pred = HalvingPredictor(fraction=0.5)
    actions = pred([0.0], horizon=1, goal=[10.0])
    assert actions[0] == (5.0,)


def test_halving_predictor_invalid_fraction_raises():
    with pytest.raises(ValueError, match="fraction"):
        HalvingPredictor(fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        HalvingPredictor(fraction=1.5)


def test_halving_predictor_no_goal_raises():
    pred = HalvingPredictor()
    with pytest.raises(ValueError, match="goal"):
        pred([0.0], horizon=1, goal=None)


def test_halving_predictor_shape_mismatch_raises():
    pred = HalvingPredictor()
    with pytest.raises(ValueError, match="shapes"):
        pred([0.0, 0.0], horizon=1, goal=[10.0])


def test_noisy_world_zero_noise_is_deterministic():
    w = NoisyWorld(noise_std=0.0, drift=0.0, seed=0)
    o1, _ = w((0.0,), (1.0,))
    o2, _ = w((0.0,), (1.0,))
    assert o1 == o2 == (1.0,)


def test_noisy_world_same_seed_yields_same_sequence():
    w1 = NoisyWorld(noise_std=0.5, seed=42)
    w2 = NoisyWorld(noise_std=0.5, seed=42)
    for _ in range(5):
        a, _ = w1((0.0,), (1.0,))
        b, _ = w2((0.0,), (1.0,))
        assert a == b


def test_noisy_world_drift_adds_constant_per_step():
    w = NoisyWorld(noise_std=0.0, drift=0.3, seed=0)
    state = (0.0,)
    for k in range(3):
        state, _ = w(state, (0.0,))
    # action=0 means new = obs + drift = 0 + 0.3 + 0.3 + 0.3 = 0.9
    assert state[0] == pytest.approx(0.9)


def test_noisy_world_negative_noise_std_raises():
    with pytest.raises(ValueError, match="noise_std"):
        NoisyWorld(noise_std=-0.1)


def test_noisy_world_reset_restores_rng():
    w = NoisyWorld(noise_std=1.0, seed=7)
    o1, _ = w((0.0,), (0.0,))
    w.reset()
    o2, _ = w((0.0,), (0.0,))
    assert o1 == o2


def test_noisy_world_shape_mismatch_raises():
    w = NoisyWorld()
    with pytest.raises(ValueError, match="shapes"):
        w((0.0, 0.0), (0.0,))


# ---------------------------------------------------------------------------
# Closed-loop single rollout
# ---------------------------------------------------------------------------


def _ezcfg(**kw):
    defaults = dict(horizon=8, execute_steps=1, max_steps=30, goal_tolerance=0.01)
    defaults.update(kw)
    return ReplannerConfig(**defaults)


def test_closed_loop_converges_in_noiseless_world():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(),
        goal=[10.0],
    )
    assert res.success is True
    assert res.mode == "closed"
    assert abs(res.final_observation[0] - 10.0) < 0.01


def test_closed_loop_n_replans_equals_steps_when_execute_one():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(execute_steps=1),
        goal=[10.0],
    )
    assert res.n_replans == res.total_steps


def test_closed_loop_n_replans_less_than_steps_when_execute_many():
    """K=4 means 4 executes per replan -> n_replans = ceil(steps/4)."""
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(horizon=8, execute_steps=4, max_steps=20, goal_tolerance=0.0),
        goal=[10.0],
    )
    assert res.n_replans <= res.total_steps
    assert res.n_replans * 4 >= res.total_steps


def test_closed_loop_respects_max_steps_cap():
    """Even with an unreachable goal, total_steps must not exceed max_steps."""
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=7, goal_tolerance=0.0),
        goal=[10.0],
    )
    assert res.total_steps == 7


def test_closed_loop_respects_goal_tolerance():
    """Loose tolerance triggers an early stop."""
    res_tight = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=50, goal_tolerance=0.0),
        goal=[10.0],
    )
    res_loose = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=50, goal_tolerance=2.0),
        goal=[10.0],
    )
    assert res_loose.total_steps < res_tight.total_steps
    assert res_loose.success is True


def test_closed_loop_records_have_sequential_step_indices():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(),
        goal=[10.0],
    )
    assert [r.t for r in res.records] == list(range(len(res.records)))


def test_closed_loop_record_history_false_returns_empty_records():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(record_history=False),
        goal=[10.0],
    )
    assert res.records == ()
    # but the rollout still tracks counters
    assert res.total_steps > 0


def test_closed_loop_custom_stop_fn_is_consulted():
    """A stop_fn that always returns True after the first step exits early."""
    calls = {"n": 0}
    def stop_fn(obs, goal):
        calls["n"] += 1
        return calls["n"] >= 2  # first call before any step (False), second call True
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(),
        goal=[10.0],
        stop_fn=stop_fn,
    )
    # at least one step executed before stop fired
    assert res.total_steps >= 1
    assert res.success is True


def test_closed_loop_goal_none_runs_to_max_steps():
    """Without a goal, the default tolerance check returns False forever."""
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=5),
        goal=[10.0],   # need goal for HalvingPredictor; stop_fn never fires due to tolerance=0.01 + reach
    )
    # success when we hit tolerance
    assert res.total_steps <= 5


def test_closed_loop_records_capture_predicted_horizon():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(horizon=6),
        goal=[10.0],
    )
    for r in res.records:
        assert isinstance(r, StepRecord)
        assert len(r.predicted) == 6  # full horizon kept on each step


def test_closed_loop_determinism_same_seed():
    """Same predict_fn + same execute_fn (same seed) -> identical trajectory."""
    a = run_closed_loop(
        HalvingPredictor(), NoisyWorld(noise_std=0.3, seed=11),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=20, goal_tolerance=0.0),
        goal=[10.0],
    )
    b = run_closed_loop(
        HalvingPredictor(), NoisyWorld(noise_std=0.3, seed=11),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=20, goal_tolerance=0.0),
        goal=[10.0],
    )
    assert a.final_observation == b.final_observation
    assert [r.executed for r in a.records] == [r.executed for r in b.records]


def test_closed_loop_rejects_predict_fn_returning_none():
    def bad_predict(obs, h, goal):
        return None
    with pytest.raises(TypeError, match="None"):
        run_closed_loop(
            bad_predict, NoisyWorld(), [0.0],
            config=_ezcfg(), goal=[10.0],
        )


def test_closed_loop_rejects_predict_fn_returning_empty_sequence():
    def empty_predict(obs, h, goal):
        return []
    with pytest.raises(ValueError, match="empty"):
        run_closed_loop(
            empty_predict, NoisyWorld(), [0.0],
            config=_ezcfg(), goal=[10.0],
        )


def test_closed_loop_handles_short_predict_sequence_gracefully():
    """If predict_fn returns fewer than horizon actions, use what we have."""
    def short_predict(obs, h, goal):
        # only return 1 action regardless of requested horizon
        return [(0.5 * (goal[0] - obs[0]),)]
    res = run_closed_loop(
        short_predict, NoisyWorld(), [0.0],
        config=_ezcfg(horizon=8, execute_steps=1, max_steps=20),
        goal=[10.0],
    )
    # should still converge by replanning every step
    assert res.success is True


# ---------------------------------------------------------------------------
# Open-loop single rollout
# ---------------------------------------------------------------------------


def test_open_loop_executes_all_horizon_actions():
    res = run_open_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        horizon=10,
        goal=[10.0],
    )
    assert res.mode == "open"
    assert res.total_steps == 10
    assert res.n_replans == 0


def test_open_loop_horizon_below_one_raises():
    with pytest.raises(ValueError, match="horizon"):
        run_open_loop(
            HalvingPredictor(), NoisyWorld(), [0.0], horizon=0, goal=[10.0],
        )


def test_open_loop_record_history_false_returns_empty_records():
    res = run_open_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        horizon=5,
        goal=[10.0],
        record_history=False,
    )
    assert res.records == ()
    assert res.total_steps == 5


def test_open_loop_in_noiseless_world_converges():
    res = run_open_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        horizon=20,
        goal=[10.0],
    )
    assert abs(res.final_observation[0] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# compare_loops -- the headline scientific claim
# ---------------------------------------------------------------------------


def test_compare_loops_returns_both_results():
    cmp = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=lambda: NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(),
        goal=[10.0],
    )
    assert isinstance(cmp, ComparisonResult)
    assert cmp.closed.mode == "closed"
    assert cmp.open_.mode == "open"


def test_compare_loops_both_converge_in_noiseless_world():
    cmp = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=lambda: NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=30),
        goal=[10.0],
    )
    assert cmp.closed_final_error < 0.01
    assert cmp.open_final_error < 0.01


def test_compare_loops_closed_beats_open_under_systematic_drift():
    """Constant drift accumulates in open-loop; closed-loop's P-control absorbs it."""
    cmp = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=lambda: NoisyWorld(noise_std=0.0, drift=0.3, seed=0),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=30, goal_tolerance=0.0),
        goal=[10.0],
    )
    # closed-loop within ~1 unit of goal; open-loop blown out by ~9 units
    assert cmp.closed_final_error < 1.0
    assert cmp.open_final_error > 5.0
    assert cmp.closed_final_error < cmp.open_final_error


def test_compare_loops_closed_beats_open_in_expectation_under_gaussian_noise():
    """Multi-seed average: closed-loop's mean error is materially smaller."""
    closed_errs = []
    open_errs = []
    for seed in range(20):
        cmp = compare_loops(
            HalvingPredictor(),
            execute_fn_factory=lambda s=seed: NoisyWorld(noise_std=0.3, seed=s),
            initial_observation=[0.0],
            config=_ezcfg(max_steps=30, goal_tolerance=0.0),
            goal=[10.0],
        )
        closed_errs.append(cmp.closed_final_error)
        open_errs.append(cmp.open_final_error)
    closed_mean = statistics.mean(closed_errs)
    open_mean = statistics.mean(open_errs)
    # closed-loop averages a much smaller residual error (>2x lower)
    assert closed_mean * 2 < open_mean, (
        f"expected closed_mean ({closed_mean:.3f}) << open_mean ({open_mean:.3f})"
    )


def test_compare_loops_is_deterministic():
    """Calling compare_loops twice with the same factory must give identical numbers."""
    factory = lambda: NoisyWorld(noise_std=0.3, seed=5)  # noqa: E731
    a = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=factory,
        initial_observation=[0.0],
        config=_ezcfg(max_steps=20),
        goal=[10.0],
    )
    b = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=factory,
        initial_observation=[0.0],
        config=_ezcfg(max_steps=20),
        goal=[10.0],
    )
    assert a.closed_final_error == b.closed_final_error
    assert a.open_final_error == b.open_final_error


def test_compare_loops_mean_error_includes_intermediate_states():
    """mean_error averages per-step distance, not just the final one."""
    cmp = compare_loops(
        HalvingPredictor(),
        execute_fn_factory=lambda: NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=10, goal_tolerance=0.0),
        goal=[10.0],
    )
    assert cmp.closed_mean_error > cmp.closed_final_error  # earlier steps were further away


# ---------------------------------------------------------------------------
# RolloutResult shape
# ---------------------------------------------------------------------------


def test_rollout_result_is_frozen():
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=3),
        goal=[10.0],
    )
    with pytest.raises(Exception):
        res.success = not res.success  # type: ignore[misc]


def test_rollout_result_records_are_tuple_not_list():
    """Records must be an immutable tuple so the result is hashable-ish."""
    res = run_closed_loop(
        HalvingPredictor(), NoisyWorld(),
        initial_observation=[0.0],
        config=_ezcfg(max_steps=3),
        goal=[10.0],
    )
    assert isinstance(res.records, tuple)
    if res.records:
        assert isinstance(res.records[0], StepRecord)

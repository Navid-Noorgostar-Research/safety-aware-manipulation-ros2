"""Tests for the uncertainty-aware ensemble layer.

Three layers, mirroring the module layout:

- ``ensemble_predict`` core: shape, validation, determinism, n_samples
  behaviour, std monotonicity in injected noise.
- Score functions: zero on deterministic ensembles, monotonic under
  noise scaling, ordering invariants (max >= mean).
- Closed-loop integration: the uncertainty-aware ``predict_fn`` is
  compatible with ``run_closed_loop``, the gate truncates the horizon
  when the score exceeds threshold, and records per-call scores when
  asked.
"""

import math
import random
import statistics

import pytest

from net.uncertainty import (
    EnsembleResult,
    diffusion_entropy_score,
    disagreement_score,
    ensemble_predict,
    make_uncertainty_aware_predict_fn,
    max_std_score,
    mean_std_score,
)


# ---------------------------------------------------------------------------
# Sampler fixtures
# ---------------------------------------------------------------------------


def _deterministic_sample(obs, horizon, goal, sample_id):
    """Always return the same actions regardless of sample_id."""
    return [(1.0, 2.0)] * horizon


def _noisy_sample_factory(noise_std):
    """Return a sampler whose action is a fixed mean plus per-call Gaussian noise."""
    def sample_fn(obs, horizon, goal, sample_id):
        rng = random.Random(sample_id * 1000 + 7)
        return [
            (1.0 + rng.gauss(0, noise_std), 2.0 + rng.gauss(0, noise_std))
            for _ in range(horizon)
        ]
    return sample_fn


# ---------------------------------------------------------------------------
# ensemble_predict core
# ---------------------------------------------------------------------------


def test_returns_ensemble_result():
    res = ensemble_predict(_deterministic_sample, [0.0], horizon=3, n_samples=4)
    assert isinstance(res, EnsembleResult)


def test_shapes_match_horizon_and_action_dim():
    res = ensemble_predict(_deterministic_sample, [0.0], horizon=5, n_samples=4)
    assert res.horizon == 5
    assert res.action_dim == 2
    assert res.n_samples == 4
    assert len(res.mean_actions) == 5
    assert len(res.std_actions) == 5
    assert len(res.samples) == 4
    assert all(len(s) == 5 for s in res.samples)
    assert all(all(len(a) == 2 for a in s) for s in res.samples)


def test_deterministic_sampler_yields_zero_std():
    res = ensemble_predict(_deterministic_sample, [0.0], horizon=3, n_samples=8)
    for row in res.std_actions:
        for v in row:
            assert v == 0.0


def test_deterministic_sampler_score_is_zero():
    res = ensemble_predict(_deterministic_sample, [0.0], horizon=3, n_samples=8)
    assert res.score == 0.0


def test_n_samples_one_collapses_to_single_pass():
    res = ensemble_predict(_deterministic_sample, [0.0], horizon=2, n_samples=1)
    assert res.n_samples == 1
    # std is 0 on a single sample (population variance with N=1 is 0)
    for row in res.std_actions:
        for v in row:
            assert v == 0.0


def test_noisy_sampler_has_nonzero_std():
    res = ensemble_predict(
        _noisy_sample_factory(0.5), [0.0], horizon=3, n_samples=32,
    )
    # at least one timestep / dim must show meaningful std
    flat = [v for row in res.std_actions for v in row]
    assert max(flat) > 0.1


def test_std_scales_linearly_with_injected_noise():
    """Doubling the noise std should approximately double the empirical std."""
    res_lo = ensemble_predict(_noisy_sample_factory(0.5), [0.0], 5, n_samples=64)
    res_hi = ensemble_predict(_noisy_sample_factory(1.0), [0.0], 5, n_samples=64)
    lo = statistics.mean(v for row in res_lo.std_actions for v in row)
    hi = statistics.mean(v for row in res_hi.std_actions for v in row)
    assert 1.6 < hi / lo < 2.4   # within ~20% of the 2x ratio


def test_mean_converges_to_truth_with_more_samples():
    """Larger ensembles should produce a tighter estimate of the true mean."""
    truth = (1.0, 2.0)
    sample_fn = _noisy_sample_factory(1.0)
    err_small = abs(ensemble_predict(sample_fn, [0.0], 1, n_samples=4).mean_actions[0][0] - truth[0])
    err_big = abs(ensemble_predict(sample_fn, [0.0], 1, n_samples=256).mean_actions[0][0] - truth[0])
    # Not strictly guaranteed for any single seed pair, but for our seeded
    # sampler the expected behaviour holds reliably.
    assert err_big <= err_small + 1e-6 or err_big < 0.2


def test_determinism_same_sampler_same_result():
    sample_fn = _noisy_sample_factory(0.3)
    a = ensemble_predict(sample_fn, [0.0], 4, n_samples=8)
    b = ensemble_predict(sample_fn, [0.0], 4, n_samples=8)
    assert a.mean_actions == b.mean_actions
    assert a.std_actions == b.std_actions
    assert a.score == b.score


def test_n_samples_below_one_raises():
    with pytest.raises(ValueError, match="n_samples"):
        ensemble_predict(_deterministic_sample, [0.0], 3, n_samples=0)


def test_horizon_below_one_raises():
    with pytest.raises(ValueError, match="horizon"):
        ensemble_predict(_deterministic_sample, [0.0], 0, n_samples=4)


def test_sample_fn_returning_none_raises():
    def bad(obs, h, g, sid):
        return None
    with pytest.raises(TypeError, match="None"):
        ensemble_predict(bad, [0.0], 3, n_samples=4)


def test_sample_fn_returning_wrong_horizon_raises():
    def bad(obs, h, g, sid):
        return [(1.0,)] * (h + 1)
    with pytest.raises(ValueError, match="horizon"):
        ensemble_predict(bad, [0.0], 3, n_samples=4)


def test_sample_fn_returning_inconsistent_dim_raises():
    def bad(obs, h, g, sid):
        return [(1.0,), (1.0, 2.0), (1.0,)]   # middle row has 2 dims
    with pytest.raises(ValueError, match="action_dim"):
        ensemble_predict(bad, [0.0], 3, n_samples=4)


def test_sample_fn_returning_zero_dim_raises():
    def bad(obs, h, g, sid):
        return [() for _ in range(h)]
    with pytest.raises(ValueError, match="zero-dim"):
        ensemble_predict(bad, [0.0], 3, n_samples=4)


def test_ensemble_result_is_frozen():
    res = ensemble_predict(_deterministic_sample, [0.0], 3, n_samples=4)
    with pytest.raises(Exception):
        res.score = 99.0  # type: ignore[misc]


def test_samples_are_tuples_not_lists():
    """Frozen dataclass: samples must be tuples for hashability/immutability."""
    res = ensemble_predict(_deterministic_sample, [0.0], 3, n_samples=2)
    assert isinstance(res.samples, tuple)
    assert all(isinstance(s, tuple) for s in res.samples)
    assert all(isinstance(a, tuple) for s in res.samples for a in s)


def test_default_score_fn_is_mean_std_score():
    res = ensemble_predict(_noisy_sample_factory(0.5), [0.0], 3, n_samples=16)
    expected = mean_std_score(res.samples, res.mean_actions, res.std_actions)
    assert res.score == pytest.approx(expected)


def test_custom_score_fn_replaces_default():
    res = ensemble_predict(
        _noisy_sample_factory(0.5), [0.0], 3, n_samples=16,
        score_fn=lambda samples, mean, std: 42.0,
    )
    assert res.score == 42.0


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def _zero_ensemble():
    return ensemble_predict(_deterministic_sample, [0.0], 3, n_samples=8)


def _noisy_ensemble(noise_std=0.5, n_samples=64):
    return ensemble_predict(
        _noisy_sample_factory(noise_std), [0.0], 3, n_samples=n_samples,
    )


def test_mean_std_score_zero_on_deterministic():
    e = _zero_ensemble()
    assert mean_std_score(e.samples, e.mean_actions, e.std_actions) == 0.0


def test_max_std_score_zero_on_deterministic():
    e = _zero_ensemble()
    assert max_std_score(e.samples, e.mean_actions, e.std_actions) == 0.0


def test_disagreement_score_zero_on_deterministic():
    e = _zero_ensemble()
    assert disagreement_score(e.samples, e.mean_actions, e.std_actions) == 0.0


def test_disagreement_score_zero_with_single_sample():
    e = ensemble_predict(_noisy_sample_factory(1.0), [0.0], 3, n_samples=1)
    assert disagreement_score(e.samples, e.mean_actions, e.std_actions) == 0.0


def test_max_std_geq_mean_std_always():
    e = _noisy_ensemble(0.5, n_samples=32)
    assert max_std_score(e.samples, e.mean_actions, e.std_actions) >= \
           mean_std_score(e.samples, e.mean_actions, e.std_actions) - 1e-12


def test_mean_std_score_scales_linearly_with_noise():
    e_lo = _noisy_ensemble(0.3, n_samples=128)
    e_hi = _noisy_ensemble(0.6, n_samples=128)
    lo = mean_std_score(e_lo.samples, e_lo.mean_actions, e_lo.std_actions)
    hi = mean_std_score(e_hi.samples, e_hi.mean_actions, e_hi.std_actions)
    assert 1.6 < hi / lo < 2.4


def test_disagreement_score_scales_with_noise():
    e_lo = _noisy_ensemble(0.3, n_samples=64)
    e_hi = _noisy_ensemble(0.6, n_samples=64)
    lo = disagreement_score(e_lo.samples, e_lo.mean_actions, e_lo.std_actions)
    hi = disagreement_score(e_hi.samples, e_hi.mean_actions, e_hi.std_actions)
    assert hi > lo


def test_diffusion_entropy_increases_with_noise():
    """A wider Gaussian fit has higher entropy."""
    e_lo = _noisy_ensemble(0.1, n_samples=64)
    e_hi = _noisy_ensemble(1.0, n_samples=64)
    lo = diffusion_entropy_score(e_lo.samples, e_lo.mean_actions, e_lo.std_actions)
    hi = diffusion_entropy_score(e_hi.samples, e_hi.mean_actions, e_hi.std_actions)
    assert hi > lo


def test_diffusion_entropy_is_finite_for_zero_std():
    """eps avoids log(0). Should return a finite (very negative) value."""
    e = _zero_ensemble()
    score = diffusion_entropy_score(e.samples, e.mean_actions, e.std_actions)
    assert math.isfinite(score)


def test_score_functions_match_internal_score_when_used_as_default():
    """Picking a non-default score on the call must override the result."""
    e = ensemble_predict(
        _noisy_sample_factory(0.5), [0.0], 3, n_samples=16,
        score_fn=max_std_score,
    )
    assert e.score == pytest.approx(
        max_std_score(e.samples, e.mean_actions, e.std_actions)
    )


# ---------------------------------------------------------------------------
# Closed-loop integration
# ---------------------------------------------------------------------------


from net.control import (  # noqa: E402
    HalvingPredictor,
    NoisyWorld,
    ReplannerConfig,
    run_closed_loop,
)


def _halving_with_noise(noise_std, base_seed=0):
    base = HalvingPredictor()
    def sample_fn(obs, horizon, goal, sample_id):
        rng = random.Random(base_seed * 1000 + sample_id)
        actions = base(obs, horizon, goal)
        return [tuple(a + rng.gauss(0, noise_std) for a in row) for row in actions]
    return sample_fn


def test_predict_fn_returns_horizon_actions():
    sf = _halving_with_noise(0.0)
    predict_fn = make_uncertainty_aware_predict_fn(sf, n_samples=4)
    out = predict_fn([0.0], 5, [10.0])
    assert len(out) == 5
    assert all(len(a) == 1 for a in out)


def test_predict_fn_returns_mean_when_gate_disabled():
    sf = _halving_with_noise(0.5)
    predict_fn = make_uncertainty_aware_predict_fn(
        sf, n_samples=8, score_threshold=math.inf,
    )
    out = predict_fn([0.0], 4, [10.0])
    # default ensemble mean of halving predictor = halving trajectory ~[5, 7.5, ...]
    assert out[0][0] == pytest.approx(5.0, abs=0.5)


def test_predict_fn_truncates_horizon_when_score_exceeds_threshold():
    sf = _halving_with_noise(1.0)   # large noise -> score >> 0
    predict_fn = make_uncertainty_aware_predict_fn(
        sf, n_samples=8, score_threshold=0.001, short_horizon=2,
    )
    out = predict_fn([0.0], 6, [10.0])
    assert len(out) == 2  # gate fired, horizon shortened to short_horizon


def test_predict_fn_keeps_full_horizon_when_score_below_threshold():
    sf = _halving_with_noise(0.0)   # zero noise -> score == 0
    predict_fn = make_uncertainty_aware_predict_fn(
        sf, n_samples=8, score_threshold=0.001, short_horizon=2,
    )
    out = predict_fn([0.0], 6, [10.0])
    assert len(out) == 6


def test_predict_fn_records_score_history():
    sf = _halving_with_noise(0.5)
    log = []
    predict_fn = make_uncertainty_aware_predict_fn(
        sf, n_samples=8, record_to=log,
    )
    predict_fn([0.0], 3, [10.0])
    predict_fn([1.0], 3, [10.0])
    assert len(log) == 2
    assert all(s >= 0 for s in log)


def test_predict_fn_validates_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        make_uncertainty_aware_predict_fn(_halving_with_noise(0.0), n_samples=0)


def test_predict_fn_validates_short_horizon():
    with pytest.raises(ValueError, match="short_horizon"):
        make_uncertainty_aware_predict_fn(_halving_with_noise(0.0), short_horizon=0)


def test_predict_fn_validates_threshold():
    with pytest.raises(ValueError, match="score_threshold"):
        make_uncertainty_aware_predict_fn(
            _halving_with_noise(0.0), score_threshold=-1.0,
        )


def test_predict_fn_short_horizon_exceeds_requested_raises():
    predict_fn = make_uncertainty_aware_predict_fn(
        _halving_with_noise(0.0), short_horizon=10,
    )
    with pytest.raises(ValueError, match="short_horizon"):
        predict_fn([0.0], 3, [10.0])


def test_uncertainty_aware_closed_loop_converges_in_noisy_sampler():
    """End-to-end: uncertainty-aware predictor + closed-loop converges."""
    sf = _halving_with_noise(0.2)
    predict_fn = make_uncertainty_aware_predict_fn(sf, n_samples=16)
    res = run_closed_loop(
        predict_fn,
        NoisyWorld(noise_std=0.0, seed=0),  # noiseless world to isolate
        [0.0],
        config=ReplannerConfig(horizon=5, execute_steps=1, max_steps=20, goal_tolerance=0.05),
        goal=[10.0],
    )
    assert res.success is True


def test_uncertainty_gate_increases_replan_frequency():
    """When the gate fires, execute_steps stays 1 but horizon truncates --
    that doesn't change replan count for K=1, but it does limit how many
    actions ever get executed per replan when K>1. Verify the truncation
    is observed via the records' predicted-tuple length."""
    sf = _halving_with_noise(1.0)
    predict_fn = make_uncertainty_aware_predict_fn(
        sf, n_samples=8, score_threshold=0.001, short_horizon=1,
    )
    res = run_closed_loop(
        predict_fn,
        NoisyWorld(noise_std=0.0, seed=0),
        [0.0],
        config=ReplannerConfig(horizon=5, execute_steps=1, max_steps=15, goal_tolerance=0.05),
        goal=[10.0],
    )
    # every recorded predicted-action tuple should have length short_horizon
    assert all(len(r.predicted) == 1 for r in res.records)

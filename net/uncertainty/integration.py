"""Adapter that turns an ensemble sampler into a closed-loop ``predict_fn``.

The :class:`net.control.run_closed_loop` orchestrator calls
``predict_fn(obs, horizon, goal) -> Sequence[action]``. This module wraps
an uncertainty-aware sampler so the orchestrator gets the ensemble mean
trajectory, with an optional gate that *shortens the horizon* whenever
the uncertainty score crosses a threshold -- forcing the receding-
horizon controller to re-observe sooner under disagreement.

The shortening is a soft form of "the model isn't sure, so don't commit
to a long plan from a stale observation". The threshold is a knob, not
a magic number: tune it on a validation set, or set it to ``inf`` to
disable the gate while still using the ensemble mean.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .ensemble import SampleFn, ScoreFn, ensemble_predict


PredictFn = Callable[[Any, int, Any], List[Tuple[float, ...]]]


def make_uncertainty_aware_predict_fn(
    sample_fn: SampleFn,
    *,
    n_samples: int = 8,
    score_fn: Optional[ScoreFn] = None,
    score_threshold: float = math.inf,
    short_horizon: int = 1,
    record_to: Optional[List[float]] = None,
) -> PredictFn:
    """Build a ``predict_fn`` that consults ``sample_fn`` ``n_samples`` times.

    Args:
        sample_fn: stochastic sampler, signature
                   ``(obs, horizon, goal, sample_id) -> [T, D]``.
        n_samples: ensemble size. ``1`` means no uncertainty (single
                   forward pass; std=0 so score=0; gate never triggers).
        score_fn: scalar score over (samples, mean, std). Defaults to
                  :func:`net.uncertainty.scores.mean_std_score`.
        score_threshold: when the score *strictly exceeds* this value,
                         the returned action sequence is truncated to
                         ``short_horizon`` so the closed-loop replanner
                         re-observes sooner. ``inf`` (the default)
                         disables the gate.
        short_horizon: minimum horizon returned when the gate fires.
                       Must be >= 1 and <= the requested horizon.
        record_to: optional list to which the per-call score is
                   appended. Useful for plotting or for tying scores
                   into an analysis loop without piping them through
                   the predict_fn return signature.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1 (got {n_samples}).")
    if short_horizon < 1:
        raise ValueError(f"short_horizon must be >= 1 (got {short_horizon}).")
    if score_threshold < 0 and not math.isinf(score_threshold):
        raise ValueError(
            f"score_threshold must be non-negative or +inf (got {score_threshold})."
        )

    def predict_fn(observation: Any, horizon: int, goal: Any) -> List[Tuple[float, ...]]:
        if short_horizon > horizon:
            raise ValueError(
                f"short_horizon ({short_horizon}) must be <= requested "
                f"horizon ({horizon})."
            )
        result = ensemble_predict(
            sample_fn, observation, horizon,
            goal=goal, n_samples=n_samples, score_fn=score_fn,
        )
        if record_to is not None:
            record_to.append(result.score)

        actions = list(result.mean_actions)
        if result.score > score_threshold:
            actions = actions[:short_horizon]
        return actions

    return predict_fn

"""Ensemble sampling over action trajectories.

The standalone, dependency-light core. ``ensemble_predict`` calls a user-
provided ``sample_fn`` ``n_samples`` times and aggregates the resulting
trajectories into an :class:`EnsembleResult` with mean, per-step std, the
raw samples, and a scalar score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple


SampleFn = Callable[[Any, int, Any, int], Sequence[Sequence[float]]]
ScoreFn = Callable[
    [Sequence[Sequence[Sequence[float]]],   # samples [N, T, D]
     Sequence[Sequence[float]],              # mean    [T, D]
     Sequence[Sequence[float]]],             # std     [T, D]
    float,
]


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregated output of :func:`ensemble_predict`.

    Attributes:
        mean_actions: ``[T, D]`` averaged trajectory.
        std_actions:  ``[T, D]`` per-element standard deviation across
                      ensemble samples.
        samples:      ``[N, T, D]`` raw samples.
        score:        Scalar uncertainty score chosen by the caller.
    """

    mean_actions: Tuple[Tuple[float, ...], ...]
    std_actions: Tuple[Tuple[float, ...], ...]
    samples: Tuple[Tuple[Tuple[float, ...], ...], ...]
    score: float

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def horizon(self) -> int:
        return len(self.mean_actions)

    @property
    def action_dim(self) -> int:
        return len(self.mean_actions[0]) if self.mean_actions else 0


def _to_nested_floats(traj: Sequence[Sequence[float]]) -> List[List[float]]:
    if traj is None:
        raise TypeError("sample_fn returned None.")
    rows: List[List[float]] = []
    for a in traj:
        if a is None:
            raise TypeError("sample_fn produced a None action.")
        rows.append([float(v) for v in a])
    return rows


def _column_stats(samples: List[List[List[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Return (mean, std) shaped ``[T, D]`` over the ``[N, T, D]`` input."""
    n = len(samples)
    t_dim = len(samples[0])
    d_dim = len(samples[0][0])
    mean = [
        [sum(samples[k][t][d] for k in range(n)) / n for d in range(d_dim)]
        for t in range(t_dim)
    ]
    # population std (divide by N) -- simpler, monotonic in spread, and
    # matches the diffusion convention of computing variance across
    # samples directly. Sample std (N-1) is also fine for N>=2; we
    # picked population for the N==1 corner case (yields 0.0).
    var = [
        [
            sum((samples[k][t][d] - mean[t][d]) ** 2 for k in range(n)) / n
            for d in range(d_dim)
        ]
        for t in range(t_dim)
    ]
    std = [[v ** 0.5 for v in row] for row in var]
    return mean, std


def ensemble_predict(
    sample_fn: SampleFn,
    observation: Any,
    horizon: int,
    *,
    goal: Any = None,
    n_samples: int = 8,
    score_fn: Optional[ScoreFn] = None,
) -> EnsembleResult:
    """Sample a trajectory ``n_samples`` times and aggregate.

    Args:
        sample_fn: ``(observation, horizon, goal, sample_id) -> [T, D]``.
                   Caller is responsible for making this stochastic
                   (seed by ``sample_id``, dropout, ensemble member
                   index, diffusion noise schedule, ...).
        observation: passed through to ``sample_fn`` unchanged.
        horizon: number of actions per trajectory.
        goal: passed through; may be None.
        n_samples: ensemble size. Must be >= 1; ``1`` collapses to a
                   single deterministic forward pass with std=0.
        score_fn: scalar score over (samples, mean, std). Defaults to
                  :func:`net.uncertainty.scores.mean_std_score`, the
                  L2-norm-of-std averaged across time -- a sensible
                  default both for diffusion-style and dropout
                  ensembles.

    Returns:
        :class:`EnsembleResult` with mean, std, samples and score.

    Raises:
        ValueError: ``n_samples < 1``, ``horizon < 1``, or
                    ``sample_fn`` returns inconsistent shapes.
        TypeError:  ``sample_fn`` returns ``None`` or a non-numeric
                    value somewhere in the trajectory.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1 (got {n_samples}).")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 (got {horizon}).")

    raw: List[List[List[float]]] = []
    for k in range(n_samples):
        traj = _to_nested_floats(sample_fn(observation, horizon, goal, k))
        if len(traj) != horizon:
            raise ValueError(
                f"sample_fn returned {len(traj)} actions but horizon={horizon} "
                f"was requested (sample_id={k})."
            )
        if k == 0:
            d = len(traj[0])
            if d == 0:
                raise ValueError("sample_fn returned zero-dim actions.")
        for row in traj:
            if len(row) != d:
                raise ValueError(
                    f"inconsistent action_dim in sample {k}: expected {d}, "
                    f"got {len(row)}."
                )
        raw.append(traj)

    mean, std = _column_stats(raw)

    # Default score: mean of per-step L2 norms of std vectors. Importing
    # locally to avoid a circular import (scores.py imports nothing here).
    if score_fn is None:
        from .scores import mean_std_score
        score_fn = mean_std_score
    score_value = float(score_fn(raw, mean, std))

    return EnsembleResult(
        mean_actions=tuple(tuple(a) for a in mean),
        std_actions=tuple(tuple(s) for s in std),
        samples=tuple(tuple(tuple(a) for a in s) for s in raw),
        score=score_value,
    )

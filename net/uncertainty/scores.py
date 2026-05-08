"""Scalar uncertainty scores over an ensemble of action trajectories.

Each score takes the same ``(samples, mean, std)`` triple produced by
:func:`net.uncertainty.ensemble_predict` and reduces it to a single float.
The ensembling is otherwise model-agnostic, so picking the score is the
one place where modelling assumptions live.

Recommended use:

- :func:`mean_std_score` -- average disagreement, the safe default.
- :func:`max_std_score`  -- worst-case disagreement; useful as a gate
  that says "any single highly-uncertain action triggers replanning".
- :func:`disagreement_score` -- mean pairwise sample distance; closest
  to the diffusion-literature notion of "spread".
- :func:`diffusion_entropy_score` -- approximates the differential
  entropy of a per-step Gaussian fit to the samples; mirrors how
  diffusion models report uncertainty via the noise schedule.
"""

from __future__ import annotations

import math
from typing import Sequence


def _l2(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in v))


def mean_std_score(samples, mean, std) -> float:
    """Mean over time of the per-step L2 norm of the std vector.

    A flat trajectory with no per-axis variation gives 0; doubling all
    sample noise doubles the score.
    """
    if not std:
        return 0.0
    return sum(_l2(row) for row in std) / len(std)


def max_std_score(samples, mean, std) -> float:
    """Maximum over time of the per-step L2 norm of the std vector.

    Useful as a gate: a single highly-uncertain timestep dominates.
    """
    if not std:
        return 0.0
    return max(_l2(row) for row in std)


def disagreement_score(samples, mean, std) -> float:
    """Mean pairwise L2 distance between samples, averaged over time.

    Equivalent in spirit to a population variance, but expressed as a
    direct disagreement metric that doesn't bake in the Gaussian
    assumption ``std`` does.
    """
    n = len(samples)
    if n < 2:
        return 0.0
    t_dim = len(samples[0])
    pair_total = 0.0
    pair_count = 0
    for t in range(t_dim):
        for i in range(n):
            for j in range(i + 1, n):
                diff = [samples[i][t][d] - samples[j][t][d]
                        for d in range(len(samples[i][t]))]
                pair_total += _l2(diff)
                pair_count += 1
    return pair_total / pair_count if pair_count else 0.0


def diffusion_entropy_score(samples, mean, std, eps: float = 1e-6) -> float:
    """Approximate differential entropy of a per-step Gaussian fit.

    For a Gaussian with diagonal covariance the differential entropy is
    proportional to ``sum(log(std_i))``. We average that quantity over
    time to give a per-step entropy estimate. This mirrors how diffusion
    models report uncertainty via the noise schedule; for a non-diffusion
    ensemble it still has the property that wider sample spread
    produces a larger score.
    """
    if not std:
        return 0.0
    total = 0.0
    count = 0
    for row in std:
        for v in row:
            total += math.log(float(v) + eps)
            count += 1
    return total / count if count else 0.0

"""Uncertainty-aware action prediction.

Adds ensemble sampling and diffusion-style uncertainty scores over action
trajectories. The framework is model-agnostic: the only thing it needs is
a ``sample_fn(observation, horizon, goal, sample_id)`` that produces one
candidate trajectory per call. For an ensemble of K models you call it
with ``sample_id`` 0..K-1; for a stochastic single model (dropout, noise-
conditioned diffusion) you seed each call with ``sample_id``.

Public entry points:

- :func:`ensemble_predict` -- run ``sample_fn`` ``n_samples`` times and
  collect mean trajectory, per-step std, and a scalar uncertainty score.
- :class:`EnsembleResult`  -- frozen result of one ensemble call.
- score functions in :mod:`net.uncertainty.scores`:
  ``mean_std_score``, ``max_std_score``, ``disagreement_score``,
  ``diffusion_entropy_score``.
- :func:`make_uncertainty_aware_predict_fn` -- adapter that turns a
  ``sample_fn`` into a ``predict_fn`` compatible with
  :func:`net.control.run_closed_loop`. Uses the ensemble mean as the
  predicted action sequence and (optionally) shortens the horizon when
  the uncertainty score crosses a threshold so the replanner sees more
  frequent re-observations under disagreement.
"""

from .ensemble import EnsembleResult, ensemble_predict
from .integration import make_uncertainty_aware_predict_fn
from .scores import (
    diffusion_entropy_score,
    disagreement_score,
    max_std_score,
    mean_std_score,
)

__all__ = [
    "EnsembleResult",
    "diffusion_entropy_score",
    "disagreement_score",
    "ensemble_predict",
    "make_uncertainty_aware_predict_fn",
    "max_std_score",
    "mean_std_score",
]

"""Closed-loop replanning utilities.

Wraps a one-shot predictor (the dynamics model) into a receding-horizon
controller that re-observes between steps. The standard MPC pattern:

    while not done:
        observation = sense()
        actions = predict(observation, horizon=H)
        execute(actions[:K])      # K=1 is canonical MPC
        # repeat

vs the open-loop baseline (predict once, execute the entire horizon).

Public entry points:

- :class:`ReplannerConfig` -- horizon / execute_steps / stopping knobs.
- :func:`run_closed_loop`  -- receding-horizon rollout.
- :func:`run_open_loop`    -- predict once, execute all (baseline).
- :func:`compare_loops`    -- run both with the same start and report
                              divergence metrics.
- :class:`RolloutResult`   -- frozen record of one rollout (records,
                              final state, success, replan count).

The framework is torch-agnostic: the predictor and world-step are
plain callables, so the same orchestrator drives both the synthetic
test fixtures and a real model + sim/hardware loop.
"""

from .replanner import (
    ComparisonResult,
    ReplannerConfig,
    RolloutResult,
    StepRecord,
    compare_loops,
    run_closed_loop,
    run_open_loop,
)
from .synthetic import HalvingPredictor, NoisyWorld

__all__ = [
    "ComparisonResult",
    "HalvingPredictor",
    "NoisyWorld",
    "ReplannerConfig",
    "RolloutResult",
    "StepRecord",
    "compare_loops",
    "run_closed_loop",
    "run_open_loop",
]

"""Synthetic predictor and world model.

Used by the test suite and the README example to demonstrate the
closed- vs open-loop story without depending on the trained dynamics
model. Both classes are deterministic given their seed, so a rollout
driven by them is byte-stable across reruns.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


class HalvingPredictor:
    """Predicts a horizon of "halve the remaining distance" actions.

    Each predicted action moves the predictor's *internal* state by
    ``fraction * (goal - state)``. In a noiseless world, executing all
    predicted actions converges to the goal geometrically. In a noisy
    world, the open-loop predictor's internal state diverges from the
    true state on every step, so its later predictions become wrong --
    that is the canonical pathology that closed-loop replanning fixes.

    Args:
        fraction: Step size as a fraction of remaining distance. The
                  default 0.5 gives the textbook halving trajectory.
    """

    def __init__(self, fraction: float = 0.5) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1] (got {fraction}).")
        self.fraction = float(fraction)

    def __call__(
        self,
        observation: Sequence[float],
        horizon: int,
        goal: Sequence[float],
    ) -> List[Tuple[float, ...]]:
        if goal is None:
            raise ValueError("HalvingPredictor requires a goal.")
        if len(observation) != len(goal):
            raise ValueError(
                f"obs and goal shapes differ: {len(observation)} vs {len(goal)}."
            )
        cur: List[float] = [float(o) for o in observation]
        actions: List[Tuple[float, ...]] = []
        for _ in range(horizon):
            action = tuple(self.fraction * (g - c) for c, g in zip(cur, goal))
            cur = [c + a for c, a in zip(cur, action)]
            actions.append(action)
        return actions


class NoisyWorld:
    """Deterministic world: ``new_obs = obs + action + drift + Gaussian noise``.

    The noise is seeded so two ``NoisyWorld`` instances built with the
    same ``seed``, ``noise_std`` and ``drift`` produce identical
    sequences -- this is what makes ``compare_loops`` a fair head-to-head:
    closed and open loops see the same noise schedule.

    Args:
        noise_std: Per-axis standard deviation of additive Gaussian noise.
                   ``0.0`` (the default) means no Gaussian noise.
        drift:     Per-step systematic bias added to every axis.
                   ``0.0`` (the default) means no drift.
                   A constant drift is the cleanest demonstration of the
                   closed-loop win: open-loop's predictions can't react to
                   the bias, while a P-controller MPC absorbs it.
        seed:      Seed for the internal :class:`random.Random` instance.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        drift: float = 0.0,
        seed: int = 0,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0 (got {noise_std}).")
        self.noise_std = float(noise_std)
        self.drift = float(drift)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def __call__(
        self,
        observation: Sequence[float],
        action: Sequence[float],
    ) -> Tuple[Tuple[float, ...], dict]:
        if len(observation) != len(action):
            raise ValueError(
                f"obs and action shapes differ: {len(observation)} vs {len(action)}."
            )
        if self.noise_std > 0:
            noise = [self._rng.gauss(0.0, self.noise_std) for _ in observation]
        else:
            noise = [0.0] * len(observation)
        new_obs = tuple(
            float(o) + float(a) + n + self.drift
            for o, a, n in zip(observation, action, noise)
        )
        return new_obs, {"noise": tuple(noise), "drift": self.drift}

    def reset(self) -> None:
        """Reset the internal RNG to the original seed."""
        self._rng = random.Random(self.seed)

"""Closed-loop receding-horizon replanner + open-loop baseline.

Three callables compose every rollout:

- ``predict_fn(observation, horizon, goal) -> Sequence[action]``
- ``execute_fn(observation, action) -> (new_observation, info)``
- ``stop_fn(observation, goal) -> bool`` (optional)

The default ``stop_fn`` evaluates ``||observation - goal||_2 <
goal_tolerance``, treating both as flat sequences of floats. Pass an
explicit ``stop_fn`` for non-numeric observations (e.g. point clouds,
dicts).

Determinism: given deterministic ``predict_fn`` / ``execute_fn`` (or
``execute_fn`` driven by a seeded RNG), the rollout is byte-stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import (
    Any, Callable, List, Optional, Sequence, Tuple,
)


PredictFn = Callable[[Any, int, Any], Sequence[Any]]
ExecuteFn = Callable[[Any, Any], Tuple[Any, dict]]
StopFn = Callable[[Any, Any], bool]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplannerConfig:
    """Knobs for :func:`run_closed_loop`.

    Args:
        horizon: How many actions ``predict_fn`` is asked to produce per
            replan. Must be >= 1.
        execute_steps: How many of those to execute before replanning.
            ``1`` is canonical MPC (replan after every step). Equal to
            ``horizon`` collapses to open-loop.
        max_steps: Hard cap on total executed actions, regardless of
            replans. Stops a non-converging rollout from running forever.
        goal_tolerance: Default ``stop_fn`` returns True once
            ``||obs - goal||_2 < goal_tolerance``.
        record_history: When False, the rollout returns an empty
            ``records`` tuple -- useful for fast batch sweeps where the
            per-step trace is not needed.
    """

    horizon: int = 8
    execute_steps: int = 1
    max_steps: int = 100
    goal_tolerance: float = 0.01
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 (got {self.horizon}).")
        if self.execute_steps < 1:
            raise ValueError(
                f"execute_steps must be >= 1 (got {self.execute_steps})."
            )
        if self.execute_steps > self.horizon:
            raise ValueError(
                f"execute_steps ({self.execute_steps}) must be <= "
                f"horizon ({self.horizon})."
            )
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 (got {self.max_steps}).")
        if self.goal_tolerance < 0:
            raise ValueError(
                f"goal_tolerance must be >= 0 (got {self.goal_tolerance})."
            )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One executed step in a rollout.

    ``predicted`` is the *full* sequence of actions returned by
    ``predict_fn`` at the most recent replan; ``executed`` is the slice
    actually applied this step.
    """

    t: int
    replan_index: int
    observation: Any
    predicted: Tuple[Any, ...]
    executed: Any
    new_observation: Any
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutResult:
    """Frozen result of a single rollout (closed- or open-loop)."""

    mode: str                          # "closed" or "open"
    initial_observation: Any
    goal: Any
    final_observation: Any
    success: bool
    n_replans: int
    total_steps: int
    records: Tuple[StepRecord, ...] = ()


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side outcome of :func:`compare_loops`."""

    closed: RolloutResult
    open_: RolloutResult
    closed_final_error: float
    open_final_error: float
    closed_mean_error: float
    open_mean_error: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Cannot take L2 norm: shape mismatch {len(a)} vs {len(b)}."
        )
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _default_stop_fn(obs: Any, goal: Any, tol: float) -> bool:
    if goal is None:
        return False
    try:
        return _l2(obs, goal) < tol
    except TypeError as e:
        raise TypeError(
            "default stop_fn requires observation and goal to be sequences "
            "of numbers. Pass an explicit stop_fn for richer types."
        ) from e


def _validate_predicted(actions: Any, horizon: int, where: str) -> List[Any]:
    if actions is None:
        raise TypeError(f"{where}: predict_fn returned None.")
    try:
        seq = list(actions)
    except TypeError as e:
        raise TypeError(
            f"{where}: predict_fn must return a sequence of actions, "
            f"got {type(actions).__name__}."
        ) from e
    if len(seq) < 1:
        raise ValueError(f"{where}: predict_fn returned an empty sequence.")
    if len(seq) < horizon:
        # Permissive: allow short sequences (caller may be horizon-truncated),
        # but require at least one element.
        return seq
    return seq[:horizon]


# ---------------------------------------------------------------------------
# Closed-loop (receding horizon)
# ---------------------------------------------------------------------------


def run_closed_loop(
    predict_fn: PredictFn,
    execute_fn: ExecuteFn,
    initial_observation: Any,
    *,
    config: ReplannerConfig,
    goal: Any = None,
    stop_fn: Optional[StopFn] = None,
) -> RolloutResult:
    """Run a receding-horizon rollout.

    At every replan, ``predict_fn`` is asked for ``horizon`` actions; the
    first ``execute_steps`` are applied through ``execute_fn``; then we
    re-observe and replan. The loop ends when ``stop_fn`` returns True or
    ``max_steps`` is hit, whichever comes first.
    """
    obs = initial_observation
    records: List[StepRecord] = []
    t = 0
    replan_idx = 0
    success = False
    last_obs = obs

    while t < config.max_steps:
        if (stop_fn or _stop_with_tol(config.goal_tolerance))(obs, goal):
            success = True
            break

        actions = _validate_predicted(
            predict_fn(obs, config.horizon, goal),
            horizon=config.horizon,
            where=f"closed_loop step {t}",
        )

        # execute the first execute_steps actions before replanning
        executed_this_replan = 0
        for k in range(min(config.execute_steps, len(actions))):
            if t >= config.max_steps:
                break
            action = actions[k]
            new_obs, info = execute_fn(obs, action)
            if config.record_history:
                records.append(
                    StepRecord(
                        t=t, replan_index=replan_idx,
                        observation=obs,
                        predicted=tuple(actions),
                        executed=action,
                        new_observation=new_obs,
                        info=dict(info or {}),
                    )
                )
            obs = new_obs
            t += 1
            executed_this_replan += 1
            # short-circuit if stop_fn fires mid-execute (only for K>1 case)
            if config.execute_steps > 1 and (stop_fn or _stop_with_tol(config.goal_tolerance))(obs, goal):
                success = True
                break

        replan_idx += 1
        last_obs = obs
        if success:
            break
        # if predict returned fewer than execute_steps, advance anyway to
        # avoid an infinite loop -- one replan with no executed action
        # would otherwise spin forever.
        if executed_this_replan == 0:
            break

    # final stop check (in case loop ended on max_steps but we just landed
    # on the goal)
    if not success and (stop_fn or _stop_with_tol(config.goal_tolerance))(obs, goal):
        success = True

    return RolloutResult(
        mode="closed",
        initial_observation=initial_observation,
        goal=goal,
        final_observation=obs,
        success=success,
        n_replans=replan_idx,
        total_steps=t,
        records=tuple(records),
    )


def _stop_with_tol(tol: float) -> StopFn:
    return lambda obs, goal: _default_stop_fn(obs, goal, tol)


# ---------------------------------------------------------------------------
# Open-loop baseline
# ---------------------------------------------------------------------------


def run_open_loop(
    predict_fn: PredictFn,
    execute_fn: ExecuteFn,
    initial_observation: Any,
    *,
    horizon: int,
    goal: Any = None,
    record_history: bool = True,
) -> RolloutResult:
    """Predict ``horizon`` actions once, execute them all without replanning.

    This is the baseline against which closed-loop should look better in
    the presence of process noise. ``success`` follows the default
    L2-tolerance test if ``goal`` is set; otherwise it is False.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 (got {horizon}).")

    actions = _validate_predicted(
        predict_fn(initial_observation, horizon, goal),
        horizon=horizon,
        where="open_loop",
    )

    obs = initial_observation
    records: List[StepRecord] = []
    for t, action in enumerate(actions):
        new_obs, info = execute_fn(obs, action)
        if record_history:
            records.append(
                StepRecord(
                    t=t, replan_index=0,
                    observation=obs,
                    predicted=tuple(actions),
                    executed=action,
                    new_observation=new_obs,
                    info=dict(info or {}),
                )
            )
        obs = new_obs

    success = False
    if goal is not None:
        try:
            success = _l2(obs, goal) < 0.01  # default tol; caller can override via stop check
        except TypeError:
            success = False

    return RolloutResult(
        mode="open",
        initial_observation=initial_observation,
        goal=goal,
        final_observation=obs,
        success=success,
        n_replans=0,
        total_steps=len(actions),
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _trajectory_errors(result: RolloutResult, goal: Any) -> Tuple[float, float]:
    """Return (final_error, mean_error). 0 / 0 if goal is None or non-numeric."""
    if goal is None:
        return 0.0, 0.0
    try:
        final = _l2(result.final_observation, goal)
        if not result.records:
            return final, final
        per_step = [
            _l2(r.new_observation, goal) for r in result.records
        ]
        return final, sum(per_step) / len(per_step)
    except TypeError:
        return float("nan"), float("nan")


def compare_loops(
    predict_fn: PredictFn,
    execute_fn_factory: Callable[[], ExecuteFn],
    initial_observation: Any,
    *,
    config: ReplannerConfig,
    goal: Any = None,
    stop_fn: Optional[StopFn] = None,
) -> ComparisonResult:
    """Run closed- and open-loop rollouts on independently seeded worlds.

    ``execute_fn_factory`` is called once per loop to produce a fresh
    ``execute_fn``; this lets a stochastic world be reset between
    rollouts so the two sides see *the same noise schedule* (assuming the
    factory is deterministic).
    """
    closed = run_closed_loop(
        predict_fn, execute_fn_factory(), initial_observation,
        config=config, goal=goal, stop_fn=stop_fn,
    )
    open_ = run_open_loop(
        predict_fn, execute_fn_factory(), initial_observation,
        horizon=config.max_steps, goal=goal,
    )
    c_final, c_mean = _trajectory_errors(closed, goal)
    o_final, o_mean = _trajectory_errors(open_, goal)
    return ComparisonResult(
        closed=closed,
        open_=open_,
        closed_final_error=c_final,
        open_final_error=o_final,
        closed_mean_error=c_mean,
        open_mean_error=o_mean,
    )

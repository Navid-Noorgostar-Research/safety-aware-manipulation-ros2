"""Evaluators that the ablation framework can plug into.

Two evaluators ship:

- :func:`safety_filter_evaluator` -- runs the actual
  :class:`net.model.safety_filter.SafetyAwareActionFilter` on a seeded
  synthetic dataset that spans all five violation modes, and reports the
  fraction of samples flagged for each constraint plus the mean L2
  correction magnitude on position.
- :func:`model_ablation_evaluator` -- a *scaffolded* evaluator for
  model-level ablations (3D input, mobility-to-body conditioning, safety
  filter on/off). Until you register a runner via
  :func:`set_model_runner`, it returns ``NaN`` for every metric so the
  paper-style table renders with the correct layout while the lab box
  fills in real numbers later.

Synthetic-data design for the safety-filter evaluator (frozen for
reproducibility):

- ``observed`` poses are sampled inside the workspace box (clamped to the
  box defined in ``net/config/safety.yaml``).
- ``target`` poses are sampled with deliberately wide tails so each
  constraint fires on a non-trivial fraction of samples even with default
  thresholds.
- ``ee_pts`` is split half/half: half the samples sit at the target
  translation (far from the dough-at-origin), the other half are placed
  within ~5 mm of the origin so the collision check actually fires.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional


def _build_filter(cfg: Mapping[str, Any]):
    from net.model.safety_filter import SafetyAwareActionFilter  # noqa: WPS433

    ns = SimpleNamespace(**dict(cfg))
    return SafetyAwareActionFilter(ns)


def _sample_dataset(seed: int, n: int = 256):
    import torch  # local import

    g = torch.Generator().manual_seed(seed)
    # observed poses: tightly clustered around the workspace center
    obs_pos = (torch.rand((n, 3), generator=g) - 0.5) * 0.2  # +-0.1 in each axis
    obs_pos[:, 2] = obs_pos[:, 2].abs() * 0.5 + 0.02         # always above ground
    obs_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone()
    observed = torch.cat([obs_pos, obs_quat], dim=-1)

    # target poses: wide tails to deliberately exceed constraints
    tgt_pos = obs_pos + (torch.rand((n, 3), generator=g) - 0.5) * 0.6  # large jumps
    tgt_quat = torch.nn.functional.normalize(
        torch.randn((n, 4), generator=g), dim=-1
    )
    target = torch.cat([tgt_pos, tgt_quat], dim=-1)

    # collision points: half the samples have EE very near the object so
    # the collision check actually fires; the other half are far away.
    ee_pts = target[:, :3].unsqueeze(1).clone()    # [n, 1, 3]
    obj_pts = torch.zeros((n, 1, 3))               # at origin
    near = torch.arange(n) % 2 == 0
    ee_pts[near] = (torch.rand((near.sum(), 1, 3), generator=g) - 0.5) * 0.005

    # previous target = observed for some samples, far for others, to give
    # the smoothness check something to fire on
    prev_target = observed.clone()
    flip = torch.rand((n,), generator=g) > 0.5
    prev_target[flip, :3] += (torch.rand((flip.sum(), 3), generator=g) - 0.5) * 0.5

    return observed, target, ee_pts, obj_pts, prev_target


def safety_filter_evaluator(config: Mapping[str, Any], seed: int) -> Dict[str, float]:
    """Run the safety filter on a seeded synthetic dataset.

    If ``config["enabled"]`` is False, the filter is bypassed and all
    violation rates and the correction magnitude are returned as 0.0
    (so the "disabled" ablation row is informative without crashing on a
    constructor that demands real numbers).

    Returns the following keys, in order::

        joint_violations
        velocity_violations
        base_speed_violations
        smoothness_violations
        collision_violations
        any_violations
        mean_correction
    """
    import torch  # local import

    if not config.get("enabled", True):
        return dict(
            joint_violations=0.0,
            velocity_violations=0.0,
            base_speed_violations=0.0,
            smoothness_violations=0.0,
            collision_violations=0.0,
            any_violations=0.0,
            mean_correction=0.0,
        )

    cfg = {k: v for k, v in config.items() if k != "enabled"}
    filt = _build_filter(cfg)
    observed, target, ee_pts, obj_pts, prev_target = _sample_dataset(seed)

    with torch.no_grad():
        safe, info = filt(
            observed, target,
            ee_pts=ee_pts, obj_pts=obj_pts, prev_target=prev_target,
        )
        correction = (safe[..., :3] - target[..., :3]).norm(dim=-1).mean().item()

    n = float(target.shape[0])
    return dict(
        joint_violations=info["joint_limits"].sum().item() / n,
        velocity_violations=info["velocity_limits"].sum().item() / n,
        base_speed_violations=info["base_speed"].sum().item() / n,
        smoothness_violations=info["smoothness"].sum().item() / n,
        collision_violations=info["collision"].sum().item() / n,
        any_violations=info["any_violation"].sum().item() / n,
        mean_correction=correction,
    )


# ---------------------------------------------------------------------------
# Model-level ablation: scaffolded evaluator.
#
# The metrics below are the canonical headline numbers for this project's
# dynamics predictor. They are reported in the order a paper appendix would
# present them: shape fidelity, semantic accuracy, then the safety-related
# numbers that justify the filter being on the action path.
# ---------------------------------------------------------------------------

MODEL_METRIC_NAMES = (
    "chamfer_l1",            # mean L1 chamfer between predicted and GT shape
    "iou",                   # voxel IoU on the predicted occupancy grid
    "topology_accuracy",     # 1 if predicted genus == GT genus, averaged
    "safety_violation_rate", # how often the safety filter intervenes downstream
    "mean_correction",       # mean L2 correction magnitude on EE position
)


# Pluggable runner: register a callable taking ``(effective_config, seed)``
# and returning a dict keyed by ``MODEL_METRIC_NAMES``. Until set, the
# evaluator returns NaN for every metric so the table layout reproduces
# without weights or data on hand.
_MODEL_RUNNER: Optional[Callable[[Mapping[str, Any], int], Dict[str, float]]] = None


def set_model_runner(
    fn: Optional[Callable[[Mapping[str, Any], int], Dict[str, float]]],
) -> None:
    """Register the runner the model-ablation evaluator delegates to.

    The runner receives the effective ablation config (already merged from
    the base + overrides) and a per-row seed. It must return a dict whose
    keys are exactly :data:`MODEL_METRIC_NAMES`. Pass ``None`` to reset to
    the NaN stub.
    """
    global _MODEL_RUNNER
    if fn is not None and not callable(fn):
        raise TypeError(f"model runner must be callable or None, got {type(fn).__name__}.")
    _MODEL_RUNNER = fn


def get_model_runner():
    """Return the currently registered runner (or ``None``)."""
    return _MODEL_RUNNER


def _stub_metrics() -> Dict[str, float]:
    nan = float("nan")
    return {k: nan for k in MODEL_METRIC_NAMES}


def model_ablation_evaluator(
    config: Mapping[str, Any], seed: int,
) -> Dict[str, float]:
    """Scaffolded evaluator for model-level ablations.

    By default this returns ``NaN`` for every metric in
    :data:`MODEL_METRIC_NAMES`, so the ablation table layout is
    reproducible without trained weights or a dataset on the current
    machine. To produce real numbers, register a runner via
    :func:`set_model_runner`; it is invoked with the same arguments and
    must return a dict keyed by ``MODEL_METRIC_NAMES``.
    """
    runner = _MODEL_RUNNER
    if runner is None:
        return _stub_metrics()
    out = runner(config, seed)
    if not isinstance(out, dict):
        raise TypeError(
            f"model runner must return a dict, got {type(out).__name__}."
        )
    missing = set(MODEL_METRIC_NAMES) - set(out.keys())
    if missing:
        raise ValueError(
            f"model runner output is missing required metrics: {sorted(missing)}."
        )
    extra = set(out.keys()) - set(MODEL_METRIC_NAMES)
    if extra:
        raise ValueError(
            f"model runner returned unexpected metrics: {sorted(extra)}. "
            f"Expected exactly {list(MODEL_METRIC_NAMES)}."
        )
    return {k: float(out[k]) for k in MODEL_METRIC_NAMES}

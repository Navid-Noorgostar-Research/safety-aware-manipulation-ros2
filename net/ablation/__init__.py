"""Ablation study framework.

A small, dependency-light framework for declaring and running ablation
studies. The intent is to keep ablations *declarative*: each ablation is
named, has a documented set of config overrides, and produces a reproducible
row of metrics. Studies render as Markdown tables that drop straight into a
paper appendix.

Public entry points:

- :class:`AblationConfig` -- one named config delta on top of a base.
- :class:`AblationStudy`  -- a base config, a list of ablations, a seed, and
                              a runner that applies an evaluator to each.
- :class:`AblationResults` -- a frozen, ordered table of metric rows with
                              ``to_markdown`` / ``to_csv`` / ``to_dict``.

Canonical evaluators (in :mod:`net.ablation.metrics`):

- :func:`safety_filter_evaluator` -- runs the safety filter on a seeded
                                    synthetic dataset and reports per-
                                    constraint violation rates and mean
                                    correction magnitude.

Canonical studies (in :mod:`net.ablation.presets`):

- :func:`safety_filter_study` -- ablates each of the five safety checks.
- :func:`robot_workspace_study` -- ablates workspace tightness for one robot.
"""

from .ablation import AblationConfig, AblationResults, AblationStudy, MetricRow
from .metrics import (
    MODEL_METRIC_NAMES,
    get_model_runner,
    model_ablation_evaluator,
    safety_filter_evaluator,
    set_model_runner,
)
from .presets import (
    model_ablation_study,
    robot_workspace_study,
    safety_filter_study,
)
from .report import to_csv, to_markdown

__all__ = [
    "AblationConfig",
    "AblationResults",
    "AblationStudy",
    "MODEL_METRIC_NAMES",
    "MetricRow",
    "get_model_runner",
    "model_ablation_evaluator",
    "model_ablation_study",
    "robot_workspace_study",
    "safety_filter_evaluator",
    "safety_filter_study",
    "set_model_runner",
    "to_csv",
    "to_markdown",
]

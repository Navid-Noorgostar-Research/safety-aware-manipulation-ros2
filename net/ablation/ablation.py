"""Core ablation primitives -- declarative configs, study runner, results table.

The framework is torch-agnostic: ``AblationConfig`` and ``AblationStudy``
work with plain dicts. Concrete evaluators (e.g. the safety filter
evaluator in :mod:`net.ablation.metrics`) bring in their own dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Type aliases for clarity.
ConfigDict = Dict[str, Any]
Evaluator = Callable[[ConfigDict, int], Dict[str, float]]


@dataclass(frozen=True)
class AblationConfig:
    """One named ablation: a list of overrides applied to a base config.

    Args:
        name: Stable identifier used as the row label in result tables.
              Must be unique within a study and contain no whitespace.
        overrides: Flat dict of key -> value pairs that replace the
                   matching keys in the base config. Keys not in the base
                   are allowed (the evaluator may consume them) but emit a
                   warning at study-validation time when ``strict=True``.
        description: Free-form one-liner explaining what the ablation
                     removes / weakens. Surfaces in Markdown reports.
    """

    name: str
    overrides: ConfigDict = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AblationConfig.name must be non-empty.")
        if any(c.isspace() for c in self.name):
            raise ValueError(
                f"AblationConfig.name {self.name!r} must not contain whitespace."
            )

    def apply_to(self, base: Mapping[str, Any]) -> ConfigDict:
        """Return a shallow copy of ``base`` with overrides applied."""
        out = dict(base)
        out.update(self.overrides)
        return out


@dataclass(frozen=True)
class MetricRow:
    """One row of a results table.

    ``metrics`` is an ordered dict of metric name -> float, plus optional
    string metadata (sample count, evaluator name) under ``meta``.
    """

    name: str
    metrics: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def with_meta(self, **kv: Any) -> "MetricRow":
        new_meta = dict(self.meta)
        new_meta.update(kv)
        return replace(self, meta=new_meta)


class AblationResults:
    """Frozen ordered table of :class:`MetricRow`.

    Use :meth:`to_markdown` for paper-ready output, :meth:`to_csv` for
    downstream tooling, or :meth:`to_dict` for programmatic consumption.
    """

    def __init__(
        self,
        study_name: str,
        rows: Sequence[MetricRow],
        seed: Optional[int] = None,
    ) -> None:
        if not rows:
            raise ValueError("AblationResults requires at least one row.")
        self.study_name = study_name
        self.rows: Tuple[MetricRow, ...] = tuple(rows)
        self.seed = seed

    # ------------------------------------------------------------------

    @property
    def metric_names(self) -> Tuple[str, ...]:
        """Column order, taken from the first row to keep output stable."""
        return tuple(self.rows[0].metrics.keys())

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, name: str) -> MetricRow:
        for r in self.rows:
            if r.name == name:
                return r
        raise KeyError(f"No ablation row named {name!r}.")

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable nested dict."""
        return {
            "study": self.study_name,
            "seed": self.seed,
            "metric_names": list(self.metric_names),
            "rows": [
                {
                    "name": r.name,
                    "metrics": dict(r.metrics),
                    "meta": dict(r.meta),
                    "description": r.description,
                }
                for r in self.rows
            ],
        }

    def to_markdown(self) -> str:
        from .report import to_markdown as _to_md
        return _to_md(self)

    def to_csv(self) -> str:
        from .report import to_csv as _to_csv
        return _to_csv(self)


class AblationStudy:
    """A base config + a list of ablations + a seed.

    Calling :meth:`run` materializes each ablation's effective config and
    feeds it to the user-provided evaluator. Results are returned in the
    same order as the ablations were declared, so paper tables stay
    deterministic across reruns.
    """

    def __init__(
        self,
        name: str,
        base_config: Mapping[str, Any],
        ablations: Iterable[AblationConfig],
        seed: int = 0,
        strict: bool = True,
    ) -> None:
        self.name = name
        self.base_config: ConfigDict = dict(base_config)
        self.ablations: Tuple[AblationConfig, ...] = tuple(ablations)
        self.seed = int(seed)
        self.strict = bool(strict)
        self._validate()

    def _validate(self) -> None:
        if not self.ablations:
            raise ValueError(f"Study {self.name!r}: ablations list must be non-empty.")
        seen = set()
        for a in self.ablations:
            if a.name in seen:
                raise ValueError(
                    f"Study {self.name!r}: duplicate ablation name {a.name!r}."
                )
            seen.add(a.name)
        if self.strict:
            base_keys = set(self.base_config.keys())
            for a in self.ablations:
                unknown = set(a.overrides.keys()) - base_keys
                if unknown:
                    raise ValueError(
                        f"Study {self.name!r}: ablation {a.name!r} overrides "
                        f"keys not in base_config: {sorted(unknown)}. "
                        "Pass strict=False to allow extension keys."
                    )

    # ------------------------------------------------------------------

    def materialize(self) -> Dict[str, ConfigDict]:
        """Return ``{ablation_name: effective_config}``."""
        return {a.name: a.apply_to(self.base_config) for a in self.ablations}

    def run(self, evaluator: Evaluator) -> AblationResults:
        """Run ``evaluator(effective_config, seed)`` for each ablation.

        The seed passed to the evaluator is a per-ablation value derived
        from ``self.seed`` and the ablation index, so that each row is
        reproducible *and* independent (i.e. changing the order of
        ablations doesn't change a row's metrics).
        """
        rows: List[MetricRow] = []
        for i, a in enumerate(self.ablations):
            cfg = a.apply_to(self.base_config)
            sub_seed = self.seed + i  # stable, per-ablation
            metrics = evaluator(cfg, sub_seed)
            if not isinstance(metrics, dict):
                raise TypeError(
                    f"Evaluator must return a dict, got {type(metrics).__name__} "
                    f"for ablation {a.name!r}."
                )
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"Evaluator metric {k!r} for ablation {a.name!r} "
                        f"must be numeric, got {type(v).__name__}."
                    )
            row = MetricRow(
                name=a.name,
                metrics={k: float(v) for k, v in metrics.items()},
                meta={"seed": sub_seed},
                description=a.description,
            )
            rows.append(row)
        # Sanity: every row must have the same metric keys for a stable table.
        first_keys = tuple(rows[0].metrics.keys())
        for r in rows[1:]:
            if tuple(r.metrics.keys()) != first_keys:
                raise ValueError(
                    f"Study {self.name!r}: ablation {r.name!r} returned a "
                    f"different metric set ({tuple(r.metrics.keys())}) than "
                    f"the first row ({first_keys})."
                )
        return AblationResults(study_name=self.name, rows=rows, seed=self.seed)

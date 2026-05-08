"""CLI entry point for running canonical ablation studies.

Usage::

    python -m net.ablation.run --study safety_filter
    python -m net.ablation.run --study robot_workspace --format csv --seed 42
    python -m net.ablation.run --study safety_filter --out results.md

The two studies that ship are :func:`safety_filter_study` (5-constraint
ablation) and :func:`robot_workspace_study` (workspace tightness sweep).
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict

from .ablation import AblationStudy
from .metrics import safety_filter_evaluator
from .presets import robot_workspace_study, safety_filter_study


_STUDIES: Dict[str, Callable[[int], AblationStudy]] = {
    "safety_filter": safety_filter_study,
    "robot_workspace": robot_workspace_study,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run an ablation study.")
    parser.add_argument(
        "--study", choices=sorted(_STUDIES), default="safety_filter",
        help="Which canonical study to run.",
    )
    parser.add_argument(
        "--format", choices=("md", "csv"), default="md",
        help="Output format.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed; per-row seeds are derived from this.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    study = _STUDIES[args.study](seed=args.seed)
    results = study.run(safety_filter_evaluator)
    text = results.to_markdown() if args.format == "md" else results.to_csv()

    if args.out is None:
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

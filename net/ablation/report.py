"""Render :class:`AblationResults` as Markdown / CSV.

The Markdown renderer is what makes the framework "look scientific":
right-aligned numeric columns, fixed precision, and a description column
so the reader knows what each row removes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ablation import AblationResults


def _format_number(v: float, precision: int = 4) -> str:
    if v != v:  # NaN
        return "NaN"
    return f"{v:.{precision}f}"


def to_markdown(results: "AblationResults", precision: int = 4) -> str:
    """Render as a Markdown table.

    Layout::

        ## Ablation: <study>
        seed: <n>

        | Ablation | <metric1> | <metric2> | ... | Description |
        | -------- | --------- | --------- | --- | ----------- |
        | full     | 0.0123    | 1.2345    |     | (full)      |
        | no_X     | 0.5678    | 2.3456    |     | disable X   |
    """
    metric_cols = list(results.metric_names)
    headers = ["Ablation"] + metric_cols + ["Description"]

    body_rows = []
    for r in results.rows:
        cells = [r.name]
        for m in metric_cols:
            cells.append(_format_number(r.metrics[m], precision=precision))
        cells.append(r.description or "")
        body_rows.append(cells)

    # column widths
    widths = [len(h) for h in headers]
    for row in body_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def _fmt(row):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"

    lines = [
        f"## Ablation: {results.study_name}",
    ]
    if results.seed is not None:
        lines.append(f"seed: {results.seed}")
    lines.append("")
    lines.append(_fmt(headers))
    lines.append(sep)
    for row in body_rows:
        lines.append(_fmt(row))
    return "\n".join(lines) + "\n"


def to_csv(results: "AblationResults", precision: int = 6) -> str:
    """Render as RFC4180-style CSV (no quoting needed for our content)."""
    metric_cols = list(results.metric_names)
    headers = ["ablation"] + metric_cols + ["description"]
    lines = [",".join(headers)]
    for r in results.rows:
        cells = [r.name]
        for m in metric_cols:
            cells.append(_format_number(r.metrics[m], precision=precision))
        desc = r.description.replace(",", " ").replace("\n", " ")
        cells.append(desc)
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"

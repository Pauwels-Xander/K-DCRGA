"""Aggregate per-config / per-seed runs into mean ± std and LaTeX-ready tables."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def aggregate_runs(root: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Walk `root/<config>/seed_*/test_metrics.json` and aggregate per metric.

    Returns: `{config: {metric: {"mean": ..., "std": ..., "n": ...}}}`.
    """
    root = Path(root)
    out: dict[str, dict[str, dict[str, float]]] = {}
    if not root.exists():
        return out
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds: list[dict[str, float]] = []
        for seed_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            f = seed_dir / "test_metrics.json"
            if f.exists():
                seeds.append(json.loads(f.read_text()))
        if not seeds:
            continue
        per_metric: dict[str, dict[str, float]] = {}
        metric_names = set().union(*(s.keys() for s in seeds))
        for m in metric_names:
            vals = [s[m] for s in seeds
                    if m in s and isinstance(s[m], (int, float))
                    and not math.isnan(s[m])]
            if not vals:
                continue
            per_metric[m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        out[cfg_dir.name] = per_metric
    return out


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def format_latex_table(
    aggregated: dict[str, dict[str, dict[str, float]]],
    metrics: list[str],
) -> str:
    """Emit a LaTeX tabular with rows per config, columns per metric, cells as
    `mean $\\pm$ std`. Caller wraps it in a `table` environment for the thesis."""
    cols = "l" + "c" * len(metrics)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule"]
    lines.append("Config & " + " & ".join(_latex_escape(m) for m in metrics) + r" \\")
    lines.append(r"\midrule")
    for cfg, per_metric in aggregated.items():
        cells = [_latex_escape(cfg)]
        for m in metrics:
            if m in per_metric:
                cells.append(
                    f"{per_metric[m]['mean']:.3f} $\\pm$ {per_metric[m]['std']:.3f}"
                )
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

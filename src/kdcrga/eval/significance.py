"""Paired Wilcoxon signed-rank test for per-side-effect metric comparison.

Standard test in the polypharmacy literature: each side effect contributes one
paired observation (model_a metric, model_b metric); the signed-rank test asks
whether the median paired difference is non-zero. Two-sided p-value.
"""
from __future__ import annotations

from scipy.stats import wilcoxon


def wilcoxon_paired(
    metric_per_se_a: dict[int, float],
    metric_per_se_b: dict[int, float],
) -> tuple[float, float]:
    """Return (statistic, two-sided p-value). Side effects missing from either
    dict are dropped. No common keys, or identical paired values, -> p is NaN."""
    keys = sorted(set(metric_per_se_a) & set(metric_per_se_b))
    a = [metric_per_se_a[k] for k in keys]
    b = [metric_per_se_b[k] for k in keys]
    if not keys or all(ai == bi for ai, bi in zip(a, b)):
        return 0.0, float("nan")
    res = wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)

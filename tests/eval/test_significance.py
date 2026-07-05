import math

from kdcrga.eval.significance import wilcoxon_paired


def test_wilcoxon_returns_pvalue_for_paired_metrics():
    # Method A strictly dominates method B on every paired side effect.
    a = {k: 0.7 + 0.01 * k for k in range(200)}
    b = {k: 0.6 + 0.01 * k for k in range(200)}
    stat, p = wilcoxon_paired(a, b)
    assert p < 1e-10              # strong significance
    assert stat >= 0


def test_wilcoxon_p_is_one_on_identical_inputs():
    a = {k: 0.5 for k in range(20)}
    b = {k: 0.5 for k in range(20)}
    # all paired diffs zero -> not reject; scipy returns p == 1 or NaN.
    stat, p = wilcoxon_paired(a, b)
    assert math.isnan(p) or p > 0.5


def test_wilcoxon_skips_missing_keys():
    a = {1: 0.9, 2: 0.8, 3: 0.7}
    b = {1: 0.6, 2: 0.7}            # missing key 3 -> drop the pair
    stat, p = wilcoxon_paired(a, b)
    assert p is not None             # ran on the 2 paired entries


def test_wilcoxon_no_common_keys_returns_nan():
    stat, p = wilcoxon_paired({1: 0.9}, {2: 0.8})
    assert math.isnan(p)

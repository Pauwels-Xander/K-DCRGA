import json

import pytest

from kdcrga.eval.aggregate import aggregate_runs, format_latex_table


@pytest.fixture
def mock_runs(tmp_path):
    for cfg, vals in [("kdcrga_v4", [(0.80, 0.70), (0.82, 0.72), (0.81, 0.71)]),
                      ("dc_rgcn_v1", [(0.75, 0.62), (0.74, 0.64), (0.76, 0.63)])]:
        for i, (au, ap) in enumerate(vals):
            d = tmp_path / cfg / f"seed_{i}"
            d.mkdir(parents=True)
            (d / "test_metrics.json").write_text(json.dumps({
                "auroc": au, "auprc": ap, "ap50": au - 0.1,
                "auroc_q0": au + 0.05, "auroc_q3": au - 0.05,
            }))
    return tmp_path


def test_aggregate_runs_produces_mean_and_std(mock_runs):
    agg = aggregate_runs(mock_runs)
    assert set(agg) == {"kdcrga_v4", "dc_rgcn_v1"}
    kd = agg["kdcrga_v4"]
    assert abs(kd["auroc"]["mean"] - 0.81) < 1e-9
    assert kd["auroc"]["std"] > 0
    assert kd["auroc"]["n"] == 3


def test_format_latex_table_emits_tabular(mock_runs):
    agg = aggregate_runs(mock_runs)
    s = format_latex_table(agg, metrics=["auroc", "auprc"])
    assert s.startswith("\\begin{tabular}")
    assert "\\end{tabular}" in s
    # config names are underscore-escaped for valid LaTeX text mode
    assert "kdcrga\\_v4" in s
    assert "0.81" in s        # auroc mean
    assert "$\\pm$" in s       # mean ± std notation


def test_format_latex_table_escapes_underscores_in_headers(mock_runs):
    agg = aggregate_runs(mock_runs)
    s = format_latex_table(agg, metrics=["auroc_q0", "auroc_q3"])
    # Raw underscores in a LaTeX text-mode header would break compilation.
    assert "auroc_q0" not in s
    assert "auroc\\_q0" in s


def test_aggregate_runs_ignores_missing_and_empty(tmp_path):
    # A config dir with no seed subdirs is skipped; a metric present in only
    # some seeds still aggregates over the seeds that have it.
    (tmp_path / "empty_cfg").mkdir()
    d = tmp_path / "cfg" / "seed_0"
    d.mkdir(parents=True)
    (d / "test_metrics.json").write_text(json.dumps({"auroc": 0.9}))
    agg = aggregate_runs(tmp_path)
    assert "empty_cfg" not in agg
    assert agg["cfg"]["auroc"]["mean"] == 0.9
    assert agg["cfg"]["auroc"]["std"] == 0.0   # single seed -> std 0

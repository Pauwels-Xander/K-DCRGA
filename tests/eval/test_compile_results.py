"""Smoke test for the results compiler against a synthetic runs/ tree."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import compile_results as cr  # noqa: E402


@pytest.fixture
def mock_runs(tmp_path):
    configs = {
        "kdcrga_v4": [(0.80, 0.70), (0.82, 0.72), (0.81, 0.71)],
        "dc_rgcn_v1": [(0.75, 0.62), (0.74, 0.64), (0.76, 0.63)],
        "no_shared": [(0.78, 0.68), (0.79, 0.69), (0.77, 0.67)],
    }
    for cfg, vals in configs.items():
        for i, (au, ap) in enumerate(vals):
            d = tmp_path / cfg / f"seed_{i}"
            d.mkdir(parents=True)
            (d / "test_metrics.json").write_text(json.dumps({
                "auroc": au, "auprc": ap, "ap50": au - 0.1,
                "auroc_q0": au + 0.05, "auroc_q1": au, "auroc_q2": au - 0.02,
                "auroc_q3": au - 0.05,
                "auprc_q0": ap + 0.05, "auprc_q1": ap, "auprc_q2": ap - 0.02,
                "auprc_q3": ap - 0.05,
            }))
            # per-se breakdown for the Wilcoxon comparison (seed 0 only needed)
            (d / "per_se_metrics.json").write_text(json.dumps({
                str(k): {"auroc": au - 0.001 * k, "auprc": ap, "ap50": au - 0.1}
                for k in range(30)
            }))
    return tmp_path


def test_compile_results_writes_expected_tables(mock_runs, tmp_path, monkeypatch):
    # ablation detection reads configs/ablations/*.yaml from the real repo, which
    # includes no_shared.yaml; that's fine for this smoke.
    out = tmp_path / "out"
    written = cr.compile_results(mock_runs, out)
    names = {p.name for p in written}
    assert "results_rq1.tex" in names
    assert "results_rq2.tex" in names
    assert "results_rq3.tex" in names
    assert "results_rq1_wilcoxon.tex" in names

    rq1 = (out / "results_rq1.tex").read_text()
    assert rq1.startswith("\\begin{tabular}")
    assert "kdcrga\\_v4" in rq1

    wil = (out / "results_rq1_wilcoxon.tex").read_text()
    # K-DCRGA strictly dominates dc_rgcn_v1 per-SE -> small p-value present
    assert "dc\\_rgcn\\_v1" in wil


def test_rq1_excludes_ablation_configs(mock_runs, tmp_path):
    # no_shared is an ablation (configs/ablations/no_shared.yaml); it must appear
    # in the RQ3 ablation table but NOT in the RQ1 main-benchmark table, nor be a
    # Wilcoxon comparison target.
    out = tmp_path / "out"
    cr.compile_results(mock_runs, out)

    rq1 = (out / "results_rq1.tex").read_text()
    assert "dc\\_rgcn\\_v1" in rq1          # real baseline stays
    assert "no\\_shared" not in rq1          # ablation must be excluded

    rq3 = (out / "results_rq3.tex").read_text()
    assert "no\\_shared" in rq3              # ablation belongs here
    assert "dc\\_rgcn\\_v1" not in rq3       # baseline does not

    wil = (out / "results_rq1_wilcoxon.tex").read_text()
    assert "no\\_shared" not in wil          # don't compare vs ablations in RQ1


def test_compile_results_empty_tree_writes_nothing(tmp_path):
    written = cr.compile_results(tmp_path / "empty", tmp_path / "out")
    assert written == []

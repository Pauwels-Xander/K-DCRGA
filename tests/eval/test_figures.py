"""Tests for the thesis figures pipeline (Agg backend, synthetic data)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in CI / headless
import pytest  # noqa: E402

from kdcrga.eval import figures as fg  # noqa: E402


def _agg(configs):
    """{name: {metric: (mean, std)}} -> aggregate_runs-shaped dict."""
    return {c: {m: {"mean": mn, "std": sd, "n": 3} for m, (mn, sd) in metrics.items()}
            for c, metrics in configs.items()}


def test_palette_and_style_are_usable():
    assert len(fg.OKABE_ITO) == 8
    assert all(c.startswith("#") for c in fg.OKABE_ITO)
    assert fg.REFERENCE == "kdcrga_v4"
    fg._apply_style()   # idempotent, must not raise
    fg._apply_style()


def test_rq1_benchmark_bar_count_and_file(tmp_path):
    agg = _agg({"kdcrga_v5": {"auroc": (0.83, 0.01), "auprc": (0.81, 0.02)},
                "decagon_v1": {"auroc": (0.80, 0.01), "auprc": (0.78, 0.02)}})
    fig = fg._fig_rq1_benchmark(agg, metrics=("auroc", "auprc"))
    assert len(fig.axes[0].patches) == 2 * 2   # configs × metrics
    out = fg.plot_rq1_benchmark(agg, tmp_path / "rq1.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_rq2_quartiles_detects_and_plots(tmp_path):
    agg = _agg({"kdcrga_v5": {f"auprc_q{q}": (0.8 - 0.05 * q, 0.01) for q in range(4)},
                "decagon_v1": {f"auprc_q{q}": (0.7 - 0.05 * q, 0.01) for q in range(4)}})
    assert fg.detect_n_quartiles(agg, "auprc") == 4
    fig = fg._fig_rq2_quartiles(agg, metric="auprc")
    assert len(fig.axes[0].patches) == 4 * 2   # quartiles × configs
    out = fg.plot_rq2_quartiles(agg, tmp_path / "rq2.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_rq3_deltas_relative_to_reference(tmp_path):
    agg = _agg({"kdcrga_v4": {"auprc": (0.80, 0.01)},
                "no_shared": {"auprc": (0.75, 0.02)},
                "no_hierarchical": {"auprc": (0.78, 0.02)}})
    fig = fg._fig_rq3_ablation_deltas(agg, reference="kdcrga_v4", metric="auprc")
    assert len(fig.axes[0].patches) == 2   # one bar per ablation, reference excluded
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[0] == "no_shared"   # most-harmful (delta -0.05) sorted first
    out = fg.plot_rq3_ablation_deltas(agg, tmp_path / "rq3.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_dataset_frequency_line_and_bands(tmp_path):
    counts = list(range(200, 0, -1))   # 200 side effects, strictly descending
    fig = fg._fig_dataset_frequency(counts, n_bins=4)
    ax = fig.axes[0]
    line = ax.get_lines()[0]
    assert len(line.get_ydata()) == 200
    # y is sorted descending, first point is the max count
    assert line.get_ydata()[0] == 200
    # one shaded axvspan band per quartile (the only patches on the axes)
    assert len(ax.patches) == 4
    out = fg.plot_dataset_frequency(counts, tmp_path / "freq.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_rq3_missing_reference_raises():
    agg = _agg({"no_shared": {"auprc": (0.75, 0.02)}})
    with pytest.raises(ValueError):
        fg._fig_rq3_ablation_deltas(agg, reference="kdcrga_v4", metric="auprc")


def test_learning_curves_ragged_and_legend(tmp_path):
    histories = {
        "kdcrga_v5": [{"val_auprc": [0.7, 0.75, 0.8], "train_loss": [9, 8, 7]},
                      {"val_auprc": [0.71, 0.76], "train_loss": [9.1, 8.1]}],  # ragged
        "decagon_v1": [{"val_auprc": [0.6, 0.65], "train_loss": [9, 8]}],
    }
    fig = fg._fig_learning_curves(histories, metrics=("val_auprc", "train_loss"))
    assert len(fig.axes) == 2   # two metric panels
    _, labels = fig.axes[0].get_legend_handles_labels()
    assert set(labels) == {"kdcrga_v5", "decagon_v1"}   # one entry per config
    out = fg.plot_learning_curves(histories, tmp_path / "lc.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_learning_curves_drops_absent_metric():
    histories = {"a": [{"val_auprc": [0.7, 0.8]}]}   # no train_loss anywhere
    fig = fg._fig_learning_curves(histories, metrics=("val_auprc", "train_loss"))
    assert len(fig.axes) == 1   # train_loss panel dropped


def test_rq2_no_quartiles_raises():
    agg = _agg({"kdcrga_v5": {"auprc": (0.81, 0.02)}})  # no *_q* keys
    with pytest.raises(ValueError):
        fg._fig_rq2_quartiles(agg, metric="auprc")


def test_rq3_no_ablations_raises():
    agg = _agg({"kdcrga_v4": {"auprc": (0.80, 0.01)}})  # reference only
    with pytest.raises(ValueError):
        fg._fig_rq3_ablation_deltas(agg, reference="kdcrga_v4", metric="auprc")


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import make_figures as mf  # noqa: E402


def test_make_figures_skips_missing(tmp_path):
    # Partial tree: one benchmark config, one seed, history + flat test metrics,
    # NO quartile keys and NO ablations -> only learning curves + RQ1 emit.
    d = tmp_path / "runs" / "kdcrga_v5" / "seed_0"
    d.mkdir(parents=True)
    (d / "history.json").write_text(json.dumps(
        {"val_auprc": [0.7, 0.8], "train_loss": [9.0, 8.0]}))
    (d / "test_metrics.json").write_text(json.dumps({"auroc": 0.83, "auprc": 0.81}))
    out = tmp_path / "figs"
    written = mf.make_figures(tmp_path / "runs", out)
    names = {p.name for p in written}
    assert "learning_curves.pdf" in names
    assert "rq1_benchmark.pdf" in names
    assert "rq2_quartiles.pdf" not in names         # no *_q* metrics
    assert "rq3_ablation_deltas.pdf" not in names    # no reference + ablation
    assert (out / "learning_curves.pdf").stat().st_size > 0


def test_make_figures_empty_runs_dir(tmp_path):
    # Nonexistent runs/ and no biosnap_dir -> nothing written, all figures skipped.
    written = mf.make_figures(tmp_path / "runs_nonexistent", tmp_path / "out")
    assert written == []


def _write_biosnap_line(a, b, active, flag):
    """One DDI row: drug_a \t drug_b \t 200-dim multi-hot \t flag."""
    vec = ["1" if i in active else "0" for i in range(200)]
    return f"{a}\t{b}\t{','.join(vec)}\t{flag}\n"


def test_make_figures_builds_dataset_frequency(tmp_path):
    # Minimal BioSNAP dir -> dataset_frequency.pdf is emitted from real counts.
    bio = tmp_path / "BioSNAP"
    bio.mkdir()
    (bio / "train.txt").write_text(
        _write_biosnap_line(0, 1, {0, 5}, 1) + _write_biosnap_line(2, 3, {5}, 0))
    (bio / "valid.txt").write_text(_write_biosnap_line(1, 2, {5, 7}, 1))
    (bio / "test.txt").write_text(_write_biosnap_line(0, 3, {5}, 1))
    written = mf.make_figures(tmp_path / "runs_none", tmp_path / "out", bio)
    assert (tmp_path / "out" / "dataset_frequency.pdf") in written


def test_make_figures_skips_dataset_frequency_when_absent(tmp_path):
    written = mf.make_figures(tmp_path / "runs_none", tmp_path / "out",
                              tmp_path / "no_biosnap")
    assert not any(p.name == "dataset_frequency.pdf" for p in written)

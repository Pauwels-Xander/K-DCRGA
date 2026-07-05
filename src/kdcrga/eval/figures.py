"""Vector-PDF figures for the thesis, built from sweep outputs.

Pure plotting functions: each takes already-loaded data plus an output Path,
writes a PDF, and returns that Path. Run-walking/IO lives in
experiments/make_figures.py. The Agg backend is selected at import so no
display is ever needed. Spec: docs/superpowers/specs/2026-06-11-thesis-figures-pipeline-design.md
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito colour-blind-safe qualitative palette (8 colours).
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# RQ3 ablation-delta baseline; matches compile_results.py's REFERENCE.
REFERENCE = "kdcrga_v4"

_TEXT_WIDTH_IN = 6.3  # thesis text width in inches


def _save(fig, out_path):
    """Save fig to out_path as PDF, close it, and return the Path."""
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _apply_style() -> None:
    """Set shared rcParams (serif font, sizes, palette). Idempotent."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })


def _fig_rq1_benchmark(aggregated, metrics=("auroc", "auprc")):
    """Grouped bars: x = config, one bar per metric, height = mean, err = std."""
    _apply_style()
    configs = list(aggregated)
    x = np.arange(len(configs))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(_TEXT_WIDTH_IN, 3.2))
    for j, m in enumerate(metrics):
        means = [aggregated[c].get(m, {}).get("mean", 0.0) for c in configs]
        stds = [aggregated[c].get(m, {}).get("std", 0.0) for c in configs]
        ax.bar(x + j * width - 0.4 + width / 2, means, width,
               yerr=stds, capsize=3, label=m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("RQ1: benchmark (mean ± sd over seeds)")
    fig.tight_layout()
    return fig


def plot_rq1_benchmark(aggregated, out_path, *, metrics=("auroc", "auprc")):
    """Save the RQ1 benchmark figure to out_path; return out_path."""
    return _save(_fig_rq1_benchmark(aggregated, metrics=metrics), out_path)


def detect_n_quartiles(aggregated, metric):
    """Number of `{metric}_q{q}` quartiles present (highest index + 1; 0 if none)."""
    pat = re.compile(rf"{re.escape(metric)}_q(\d+)")
    qs = set()
    for per_metric in aggregated.values():
        for k in per_metric:
            match = pat.fullmatch(k)
            if match:
                qs.add(int(match.group(1)))
    return (max(qs) + 1) if qs else 0


def _fig_rq2_quartiles(aggregated, metric="auprc"):
    """Grouped bars: x = quartile (Q0..), one bar per config."""
    _apply_style()
    n_q = detect_n_quartiles(aggregated, metric)
    if n_q == 0:
        raise ValueError(f"no quartile keys for metric {metric!r} in aggregated")
    configs = list(aggregated)
    x = np.arange(n_q)
    width = 0.8 / max(len(configs), 1)
    fig, ax = plt.subplots(figsize=(_TEXT_WIDTH_IN, 3.2))
    for j, c in enumerate(configs):
        means = [aggregated[c].get(f"{metric}_q{q}", {}).get("mean", 0.0) for q in range(n_q)]
        stds = [aggregated[c].get(f"{metric}_q{q}", {}).get("std", 0.0) for q in range(n_q)]
        ax.bar(x + j * width - 0.4 + width / 2, means, width,
               yerr=stds, capsize=3, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}" for q in range(n_q)])
    ax.set_xlabel("side-effect frequency quartile (ascending)")
    ax.set_ylabel(metric.upper())
    ax.legend()
    ax.set_title(f"RQ2: {metric.upper()} by frequency quartile")
    fig.tight_layout()
    return fig


def plot_rq2_quartiles(aggregated, out_path, *, metric="auprc"):
    """Save the RQ2 per-quartile figure to out_path; return out_path."""
    return _save(_fig_rq2_quartiles(aggregated, metric=metric), out_path)


def _fig_dataset_frequency(counts, n_bins=4):
    """Side-effect frequency distribution: per-side-effect positive-triple counts
    sorted descending vs. rank, with the ``n_bins`` equal-size frequency quartiles
    shaded (q0 = most frequent). Linear y-axis on purpose: the KnowDDI slice keeps
    a truncated mid-frequency band (counts run ~815-1816), so a log axis would
    flatten it and oversell a long tail the benchmark deliberately removed."""
    _apply_style()
    counts = np.sort(np.asarray(counts))[::-1]
    n = len(counts)
    ranks = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(_TEXT_WIDTH_IN, 2.8))
    top = float(counts.max())
    per_bin = n / n_bins
    for q in range(n_bins):
        lo = round(q * per_bin)
        hi = round((q + 1) * per_bin)
        ax.axvspan(lo + 0.5, hi + 0.5, color=OKABE_ITO[(q % 7) + 1], alpha=0.13)
        ax.text((lo + hi) / 2 + 0.5, top * 0.96, f"$q_{q}$",
                ha="center", va="top", fontsize=8)
    ax.plot(ranks, counts, color=OKABE_ITO[0], linewidth=1.6)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, top * 1.05)
    ax.set_xlabel("side effect (rank by frequency)")
    ax.set_ylabel("positive triples")
    ax.set_title("Side-effect frequency (frequency quartiles shaded)")
    fig.tight_layout()
    return fig


def plot_dataset_frequency(counts, out_path, *, n_bins=4):
    """Save the dataset frequency-distribution figure to out_path; return out_path."""
    return _save(_fig_dataset_frequency(counts, n_bins=n_bins), out_path)


def _fig_rq3_ablation_deltas(aggregated, reference=REFERENCE, metric="auprc"):
    """Horizontal bars of mean(ablation) - mean(reference) for `metric`, sorted
    ascending (most-harmful first). Error bar = the ablation's own std (a simple
    honest proxy, not the std of the paired difference)."""
    _apply_style()
    if reference not in aggregated:
        raise ValueError(f"reference {reference!r} not in aggregated configs")
    ref_mean = aggregated[reference].get(metric, {}).get("mean", 0.0)
    items = [(c,
              aggregated[c].get(metric, {}).get("mean", 0.0) - ref_mean,
              aggregated[c].get(metric, {}).get("std", 0.0))
             for c in aggregated if c != reference]
    if not items:
        raise ValueError(f"no ablation configs to compare against {reference!r}")
    items.sort(key=lambda t: t[1])
    labels = [c for c, _, _ in items]
    deltas = [d for _, d, _ in items]
    errs = [e for _, _, e in items]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(_TEXT_WIDTH_IN, 0.5 * len(labels) + 1.5))
    colors = [OKABE_ITO[6] if d < 0 else OKABE_ITO[3] for d in deltas]
    ax.barh(y, deltas, xerr=errs, capsize=3, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ {metric.upper()} vs {reference}")
    ax.set_title(f"RQ3: ablation Δ vs {reference} (mean over seeds)")
    fig.tight_layout()
    return fig


def plot_rq3_ablation_deltas(aggregated, out_path, *, reference=REFERENCE, metric="auprc"):
    """Save the RQ3 ablation-delta figure to out_path; return out_path."""
    return _save(_fig_rq3_ablation_deltas(aggregated, reference=reference, metric=metric), out_path)


def _fig_learning_curves(histories, metrics=("val_auprc", "train_loss")):
    """One stacked panel per metric present in the data. Per config: thin faint
    line per seed (full length) + a bold across-seed mean with a +/-sd band over
    the common (shortest-seed) prefix. `histories` = {config: [history_dict, ...]}."""
    _apply_style()
    present = [m for m in metrics
               if any(m in h and h[m] for hs in histories.values() for h in hs)]
    if not present:
        raise ValueError("no requested metric present in any history")
    fig, axes = plt.subplots(len(present), 1, sharex=True,
                             figsize=(_TEXT_WIDTH_IN, 2.4 * len(present)),
                             squeeze=False)
    axes = axes[:, 0]
    color = {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(histories)}
    for ax, m in zip(axes, present):
        for c, seed_hists in histories.items():
            series = [h[m] for h in seed_hists if m in h and h[m]]
            if not series:
                continue
            for s in series:
                ax.plot(range(1, len(s) + 1), s, color=color[c],
                        alpha=0.25, linewidth=0.8)
            min_len = min(len(s) for s in series)
            arr = np.array([s[:min_len] for s in series])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            epochs = range(1, min_len + 1)
            ax.plot(epochs, mean, color=color[c], linewidth=1.8, label=c)
            ax.fill_between(epochs, mean - std, mean + std, color=color[c], alpha=0.15)
        ax.set_ylabel(m)
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")
    fig.suptitle("Learning curves (mean ± sd over seeds)")
    fig.tight_layout()
    return fig


def plot_learning_curves(histories, out_path, *, metrics=("val_auprc", "train_loss")):
    """Save the learning-curves figure to out_path; return out_path."""
    return _save(_fig_learning_curves(histories, metrics=metrics), out_path)

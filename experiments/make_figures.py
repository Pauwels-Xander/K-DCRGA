"""Build thesis figures from sweep outputs (spec 2026-06-11).

Run: python experiments/make_figures.py [runs_dir] [out_dir]
Defaults: runs/  ->  docs/figures/

Figures whose data is absent are skipped (never faked), mirroring
compile_results.py. Pure CPU + JSON reads, so it is safe to run while a
GPU sweep is still in progress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kdcrga.eval.aggregate import aggregate_runs
from kdcrga.eval.figures import (
    REFERENCE,
    detect_n_quartiles,
    plot_dataset_frequency,
    plot_learning_curves,
    plot_rq1_benchmark,
    plot_rq2_quartiles,
    plot_rq3_ablation_deltas,
)

# Paths are relative to the repo root: like the sibling experiment scripts
# (compile_results.py, the sweeps), this runner is meant to be invoked from the
# project root, e.g. `python experiments/make_figures.py`.
RUNS = Path("runs")
OUT = Path("docs/figures")
ABLATION_DIR = Path("configs/ablations")
BIOSNAP = Path("third_party/knowddi/data/BioSNAP")


def _load_histories(runs_root):
    """Walk runs_root/<config>/seed_*/history.json -> {config: [history, ...]}."""
    runs_root = Path(runs_root)
    out: dict[str, list[dict]] = {}
    if not runs_root.exists():
        return out
    for cfg_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        seeds = []
        for seed_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            f = seed_dir / "history.json"
            if f.exists():
                seeds.append(json.loads(f.read_text()))
        if seeds:
            out[cfg_dir.name] = seeds
    return out


def make_figures(runs_root=RUNS, out_dir=OUT, biosnap_dir=None):
    """Emit every figure with available data; skip (don't fake) the rest.

    ``biosnap_dir`` (opt-in; the CLI passes ``BIOSNAP``) enables the dataset
    frequency-distribution figure, which reads the BioSNAP slice rather than the
    sweep outputs. Left ``None`` by programmatic callers that only have runs/.
    """
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    if biosnap_dir is not None and (Path(biosnap_dir) / "train.txt").exists():
        from kdcrga.data.graph import ddi_side_effect_counts
        counts = ddi_side_effect_counts(biosnap_dir)
        written.append(plot_dataset_frequency(counts, out_dir / "dataset_frequency.pdf"))
    elif biosnap_dir is not None:
        skipped.append("dataset_frequency.pdf (no BioSNAP data)")

    agg = aggregate_runs(runs_root)
    ablation_names = {p.stem for p in ABLATION_DIR.glob("*.yaml")}
    benchmark = {c: v for c, v in agg.items() if c not in ablation_names}
    ablations = {c: v for c, v in agg.items()
                 if c in ablation_names or c == REFERENCE}

    histories = _load_histories(runs_root)
    if histories:
        written.append(plot_learning_curves(histories, out_dir / "learning_curves.pdf"))
    else:
        skipped.append("learning_curves.pdf (no history.json)")

    if benchmark:
        written.append(plot_rq1_benchmark(benchmark, out_dir / "rq1_benchmark.pdf"))
    else:
        skipped.append("rq1_benchmark.pdf (no benchmark runs)")

    # Use the SAME detector plot_rq2_quartiles uses, so a passing guard never
    # trips that function's "no quartile keys" ValueError (and vice versa).
    has_quartiles = detect_n_quartiles(benchmark, "auprc") > 0
    if benchmark and has_quartiles:
        written.append(plot_rq2_quartiles(benchmark, out_dir / "rq2_quartiles.pdf"))
    else:
        skipped.append("rq2_quartiles.pdf (no quartile metrics)")

    if REFERENCE in ablations and len(ablations) > 1:
        written.append(plot_rq3_ablation_deltas(ablations, out_dir / "rq3_ablation_deltas.pdf"))
    else:
        skipped.append("rq3_ablation_deltas.pdf (need reference + >=1 ablation)")

    print("wrote:", ", ".join(p.name for p in written) or "(none)")
    if skipped:
        print("skipped (no data):", "; ".join(skipped))
    return written


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else RUNS
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    make_figures(runs, out, BIOSNAP)


if __name__ == "__main__":
    main()

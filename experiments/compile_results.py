"""Compile sweep outputs into LaTeX-ready result tables for the thesis.

Reads ``runs/<config>/seed_*/`` produced by the sweeps and writes:
  docs/results_rq1.tex          — main benchmark (AUROC/AUPRC/AP@50)
  docs/results_rq2.tex          — per-frequency-quartile breakdown
  docs/results_rq3.tex          — ablation grid (same metrics, ablation configs)
  docs/results_rq5_zero_shot.tex — held-out AUROC/AUPRC (if an RQ5 run exists)
  docs/results_rq1_wilcoxon.tex  — K-DCRGA vs each baseline, paired Wilcoxon p

RQ4 (Spearman rho vs MedDRA distance, per-SOC) is produced separately by the
attention-analysis pipeline once real MedDRA distances are available; it is not
emitted here because it needs the ontology data, not the sweep run metrics.

Run: python experiments/compile_results.py [runs_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kdcrga.eval.aggregate import aggregate_runs, format_latex_table
from kdcrga.eval.significance import wilcoxon_paired

RUNS = Path("runs")
OUT = Path("docs")
REFERENCE = "kdcrga_v4"          # the model RQ1/RQ3 compare against


def per_side_effect_metric(run_dir: Path, metric: str = "auroc") -> dict[int, float]:
    """Load the per-side-effect ``metric`` written by ``save_run_outputs``."""
    f = run_dir / "per_se_metrics.json"
    if not f.exists():
        return {}
    return {int(k): v[metric] for k, v in json.loads(f.read_text()).items()
            if metric in v}


def _wilcoxon_table(runs: Path, agg: dict, metric: str = "auroc") -> str:
    """Paired Wilcoxon of REFERENCE vs every other config on per-SE ``metric``
    (seed 0). Returns a LaTeX tabular."""
    lines = [r"\begin{tabular}{lc}", r"\toprule",
             f"Comparison & Wilcoxon $p$ ({metric}) " + r"\\", r"\midrule"]
    ref_dir = runs / REFERENCE / "seed_0"
    ref = per_side_effect_metric(ref_dir, metric) if ref_dir.exists() else {}
    for cfg in agg:
        if cfg == REFERENCE or not ref:
            continue
        other = per_side_effect_metric(runs / cfg / "seed_0", metric)
        if not other:
            continue
        _, p = wilcoxon_paired(ref, other)
        cfg_esc = cfg.replace("_", r"\_")
        lines.append(f"{REFERENCE.replace('_', chr(92)+'_')} vs {cfg_esc} & "
                     f"{p:.2e} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def compile_results(runs_root: str | Path = RUNS, out_dir: str | Path = OUT) -> list[Path]:
    """Aggregate runs and write all available result tables. Returns the paths
    written (tables with no underlying data are skipped, not faked)."""
    runs_root = Path(runs_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    agg = aggregate_runs(runs_root)
    if not agg:
        print(f"No runs found under {runs_root}/ — nothing to compile.")
        return written

    def _write(name: str, content: str) -> None:
        p = out_dir / name
        p.write_text(content)
        written.append(p)

    # Ablation configs live in configs/ablations/ and must NOT appear in the RQ1
    # main-benchmark table — RQ1 is K-DCRGA vs baselines only. The ablation rows
    # belong in RQ3. (Both sweeps write into the same runs/ tree, so without this
    # split the ablations would contaminate the main benchmark and Wilcoxon table.)
    ablation_names = {p.stem for p in Path("configs/ablations").glob("*.yaml")}
    benchmark = {c: v for c, v in agg.items() if c not in ablation_names}

    # RQ1: main benchmark (baselines + reference, no ablations).
    _write("results_rq1.tex", format_latex_table(benchmark, ["auroc", "auprc", "ap50"]))

    # RQ2: per-quartile (only columns that actually exist in the data).
    n_q = max((int(k.split("_q")[1]) for cfg in benchmark.values() for k in cfg
               if "_q" in k), default=-1) + 1
    if n_q > 0:
        q_metrics = [f"{m}_q{q}" for m in ("auroc", "auprc") for q in range(n_q)]
        _write("results_rq2.tex", format_latex_table(benchmark, q_metrics))

    # RQ3: ablation grid — REFERENCE plus any config that lives in ablations/.
    abl = {c: v for c, v in agg.items() if c in ablation_names or c == REFERENCE}
    if len(abl) > 1:
        _write("results_rq3.tex", format_latex_table(abl, ["auroc", "auprc", "ap50"]))

    # RQ1 Wilcoxon significance — reference vs baselines only (skip ablations).
    wil = _wilcoxon_table(runs_root, benchmark, "auroc")
    _write("results_rq1_wilcoxon.tex", wil)

    # RQ5: zero-shot (a run named with a 'zero_shot' prefix if present).
    zs = {c: v for c, v in agg.items() if "zero_shot" in c}
    if zs:
        _write("results_rq5_zero_shot.tex", format_latex_table(zs, ["auroc", "auprc"]))

    print("DONE; wrote:", ", ".join(p.name for p in written))
    return written


def main() -> None:
    runs = sys.argv[1] if len(sys.argv) > 1 else RUNS
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    compile_results(runs, out)


if __name__ == "__main__":
    main()

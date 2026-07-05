"""Aggregate per-seed RQ4 attention-analysis results into a multi-seed summary.

Reads ``runs/rq4/<model>/seed_*/rq4_results.json`` (written by ``sweep_rq4.py``)
and reports, per z_k arm, the mean / min / max overall Spearman rho across seeds,
the mean attention JSD, and the mean of the per-seed within-SOC median rho. Writes
``runs/rq4/<model>/rq4_aggregate.json`` and prints a small table.

Run: python experiments/aggregate_rq4.py [--model kdcrga_dedicom_v5]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from math import isnan, nan
from pathlib import Path

ARMS = ("meddra_init", "trained", "random")


def within_soc_median(per_soc: dict) -> float:
    """Median over the per-SOC rho values, ignoring NaNs (too-few-pair SOCs)."""
    vals = [v["rho"] for v in per_soc.values() if not isnan(v["rho"])]
    return st.median(vals) if vals else nan


def summarise(per_seed: dict[int, dict]) -> dict[str, dict]:
    """``per_seed``: {seed: arms-dict from a run's JSON}. Returns {arm: stats}."""
    out: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r[arm] for r in per_seed.values() if arm in r]
        if not rows:
            continue
        rhos = [r["rho"] for r in rows]
        out[arm] = {
            "n_seeds": len(rows),
            "rho_mean": st.mean(rhos),
            "rho_min": min(rhos),
            "rho_max": max(rhos),
            "jsd_mean": st.mean(r["mean_jsd_overall"] for r in rows),
            "within_soc_median_mean": st.mean(
                within_soc_median(r["per_soc"]) for r in rows),
        }
    return out


def main(model: str = "kdcrga_dedicom_v5", runs_root: str = "runs") -> None:
    root = Path(runs_root) / "rq4" / model
    paths = sorted(glob.glob(str(root / "seed_*" / "rq4_results.json")))
    if not paths:
        print(f"no RQ4 results under {root}")
        return
    per_seed: dict[int, dict] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        per_seed[int(doc.get("seed", Path(p).parts[-2].split("_")[-1]))] = doc["arms"]

    summary = summarise(per_seed)
    seeds = sorted(per_seed)
    print(f"RQ4 multi-seed summary for {model} (seeds {seeds}):\n")
    print(f"  {'arm':<14} {'rho mean':>9} {'[min, max]':>16} "
          f"{'jsd':>8} {'wSOC med':>9}")
    for arm in ARMS:
        if arm not in summary:
            continue
        s = summary[arm]
        print(f"  {arm:<14} {s['rho_mean']:>+9.3f} "
              f"[{s['rho_min']:+.3f}, {s['rho_max']:+.3f}] "
              f"{s['jsd_mean']:>8.4f} {s['within_soc_median_mean']:>+9.3f}")

    out = {"model": model, "seeds": seeds, "arms": summary}
    (root / "rq4_aggregate.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'rq4_aggregate.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kdcrga_dedicom_v5")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()
    main(args.model, runs_root=args.runs_root)

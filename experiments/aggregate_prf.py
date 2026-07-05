"""Aggregate per-seed operating-point metrics (precision/recall/F1) into a
multi-seed summary table. Reads ``runs/bench/<arm>/seed_*/prf_metrics.json``
(written by ``eval_prf.py``) and prints mean (+ s.d.) per arm, overall and per
frequency quartile.

Run: python experiments/aggregate_prf.py
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ARMS = ["kdcrga_dedicom_v5", "decagon", "rgcn_mlp", "kdcrga_v5",
        "kdcrga_dedicom_v4", "lagat"]


def _mean_sd(xs):
    xs = list(xs)
    if not xs:
        return float("nan"), float("nan")
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def collect(arm: str, runs_root: str = "runs"):
    rows = [json.loads(Path(p).read_text())
            for p in sorted(glob.glob(f"{runs_root}/bench/{arm}/seed_*/prf_metrics.json"))]
    if not rows:
        return None
    keys = ["precision", "recall", "f1"]
    keys += [f"{m}_q{q}" for q in range(4) for m in ("precision", "recall", "f1")]
    out = {"n_seeds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            m, s = _mean_sd(vals)
            out[k] = {"mean": m, "sd": s}
    return out


def main(runs_root: str = "runs") -> None:
    summary = {}
    print(f"{'arm':<20} {'P':>14} {'R':>14} {'F1':>14}  seeds")
    for arm in ARMS:
        s = collect(arm, runs_root)
        if s is None:
            print(f"{arm:<20} (no prf_metrics.json)")
            continue
        summary[arm] = s
        def cell(m):
            return f"{s[m]['mean']:.3f}+/-{s[m]['sd']:.3f}" if m in s else "  -  "
        print(f"{arm:<20} {cell('precision'):>14} {cell('recall'):>14} "
              f"{cell('f1'):>14}  {s['n_seeds']}")
    (Path(runs_root) / "bench" / "prf_aggregate.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {Path(runs_root) / 'bench' / 'prf_aggregate.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()
    main(runs_root=args.runs_root)

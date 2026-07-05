"""Fair RQ1 benchmark sweep: matched-protocol arms for the decoder/encoder study.

All arms share one protocol so differences are attributable to architecture, not
optimisation: effective batch 256 (the batch-128 K-DCRGA arms accumulate over 2
micro-batches; Decagon runs at batch 256) and training to convergence (early stop
on val AUPRC), on the FIXED data split. Four arms give three single-variable
contrasts:

  kdcrga_dedicom_v4 vs decagon          -> the attention encoder (both no ontology, both DEDICOM)
  kdcrga_dedicom_v5 vs kdcrga_dedicom_v4 -> the ontology injection (both attention + DEDICOM)
  kdcrga_dedicom_v5 vs kdcrga_v5        -> the decoder (both v5 encoder; DEDICOM vs shared-MLP)

Runs write to runs/bench/<name>/seed_<N>/. The honest benchmark needs all three
seeds (the gaps are within a few points); select on validation AUPRC.

  python experiments/sweep_bench.py --smoke              # 1 epoch / 2 batches, throwaway dir
  python experiments/sweep_bench.py --resume             # seed 0 only (quick check)
  python experiments/sweep_bench.py --resume --all-seeds # seeds 0,1,2 (the real benchmark)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import run_sweep  # noqa: E402

# (name, config_path, model_kind). The first four are the encoder x decoder 2x2 +
# ontology contrasts; rgcn_mlp completes the 2x2 (R-GCN encoder + shared-MLP, no
# attention); lagat is the matched SOTA-positioning re-run.
CONFIGS = [
    ("decagon", "configs/bench/decagon.yaml", "decagon"),
    ("kdcrga_v5", "configs/bench/kdcrga_v5.yaml", "kdcrga"),
    ("kdcrga_dedicom_v5", "configs/bench/kdcrga_dedicom_v5.yaml", "kdcrga_dedicom"),
    # Same model as kdcrga_dedicom_v5 but trained on the canonical pair-disjoint
    # BioSNAP split (inductive) for the leak-free fusion comparison vs KnowDDI.
    ("kdcrga_dedicom_v5_pd", "configs/bench/kdcrga_dedicom_v5_pd.yaml", "kdcrga_dedicom"),
    # Pair-disjoint CONTROL baselines: confirm our Decagon/LaGAT, scored through the
    # same common evaluator, match the paper (Decagon ~0.918 AUROC) -> proves the
    # pipeline doesn't inflate and K-DCRGA's ~0.954 is a real separation.
    ("decagon_pd", "configs/bench/decagon_pd.yaml", "decagon"),
    ("lagat_pd", "configs/bench/lagat_pd.yaml", "lagat"),
    ("kdcrga_dedicom_v4", "configs/bench/kdcrga_dedicom_v4.yaml", "kdcrga_dedicom"),
    ("rgcn_mlp", "configs/bench/rgcn_mlp.yaml", "kdcrga"),
    ("lagat", "configs/bench/lagat.yaml", "lagat"),
]

DEFAULT_SEEDS = [0]
ALL_SEEDS = [0, 1, 2]


def _parse_seeds(raw: str) -> list[int]:
    """Parse comma-separated seed ids, preserving order and dropping duplicates."""
    seeds: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seed = int(part)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="skip finished runs and resume crashed checkpoints")
    parser.add_argument("--smoke", action="store_true",
                        help="run one seed for one epoch / two batches (throwaway dir)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="run the full seed grid: 0,1,2 (the real benchmark)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds to run, e.g. 1,2")
    parser.add_argument("--only", default=None,
                        help="comma-separated arm name(s) to run (subset of "
                             "CONFIGS), e.g. kdcrga_dedicom_v5. Default: all four.")
    return parser.parse_args(argv)


def _select_configs(only: str | None):
    """Filter CONFIGS to the named arm(s); None/empty -> all. Raises on unknown."""
    if not only:
        return CONFIGS
    by_name = {cfg[0]: cfg for cfg in CONFIGS}
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise ValueError(f"unknown arm(s) {missing}; choose from {list(by_name)}")
    return [by_name[w] for w in wanted]


def _select_seeds(args: argparse.Namespace) -> list[int]:
    if args.smoke:
        return [0]
    if args.seeds is not None and args.all_seeds:
        raise ValueError("Use either --seeds or --all-seeds, not both")
    if args.seeds is not None:
        return _parse_seeds(args.seeds)
    if args.all_seeds:
        return ALL_SEEDS
    return DEFAULT_SEEDS


def main(smoke: bool = False, resume: bool = False,
         seeds: list[int] | None = None, only: str | None = None) -> None:
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    runs_root = "runs/_smoke_bench" if smoke else "runs/bench"
    run_sweep(_select_configs(only), seeds, smoke=smoke, resume=resume,
              runs_root=runs_root)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(smoke=args.smoke, resume=args.resume, seeds=_select_seeds(args),
         only=args.only)

"""RQ1 main benchmark sweep: K-DCRGA vs baselines.

Writes runs/<config>/seed_<N>/ for each (config, seed). Add Plan 3 baseline
configs (decagon, lagat, knowddi) to CONFIGS once they exist.

Deadline mode:
  python experiments/sweep_rq1.py --resume
  -> runs the four RQ1 models on seed 0 only, skipping completed outputs.

Additional seeds later:
  python experiments/sweep_rq1.py --resume --seeds 1,2

Full original 3-seed sweep:
  python experiments/sweep_rq1.py --resume --all-seeds

Smoke run:
  python experiments/sweep_rq1.py --smoke   (1 epoch, 2 batches, 1 seed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import run_sweep  # noqa: E402

# (name, config_path, model_kind). Add KnowDDI once reproduced.
# The headline K-DCRGA is the full ontology model (v5); v4 (ontology-off) is the
# RQ3 ablation arm. Loaded on its own ATC-injected graph by run_sweep.
CONFIGS = [
    ("kdcrga_v5", "configs/kdcrga_v5.yaml", "kdcrga"),
    ("dc_rgcn_v1", "configs/dc_rgcn_v1.yaml", "dc_rgcn"),
    ("decagon_v1", "configs/decagon_v1.yaml", "decagon"),
    ("lagat_v1", "configs/lagat_v1.yaml", "lagat"),
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
                        help="run one seed for one epoch / two batches")
    parser.add_argument("--all-seeds", action="store_true",
                        help="run the full original seed grid: 0,1,2")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds to run, e.g. 1,2")
    return parser.parse_args(argv)


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
         seeds: list[int] | None = None) -> None:
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    run_sweep(CONFIGS, seeds, smoke=smoke, resume=resume)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(smoke=args.smoke, resume=args.resume, seeds=_select_seeds(args))

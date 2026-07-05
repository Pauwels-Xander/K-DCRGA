"""RQ3 ablation sweep: each configs/ablations/*.yaml × 3 seeds.

Writes runs/<ablation>/seed_<N>/. The full model (kdcrga_v5) is the reference
row; kdcrga_v4 is the "no ontology injection" ablation (ATC nodes dropped, z_k
random), so the compiler can show each ablation's delta against the full model.

Run:        python experiments/sweep_rq3_ablations.py
Resume:     python experiments/sweep_rq3_ablations.py --resume
Smoke run:  python experiments/sweep_rq3_ablations.py --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import run_sweep  # noqa: E402

ABLATION_DIR = Path("configs/ablations")


def _configs() -> list[tuple[str, str, str]]:
    cfgs = [
        ("kdcrga_v5", "configs/kdcrga_v5.yaml", "kdcrga"),   # full model (reference)
        ("kdcrga_v4", "configs/kdcrga_v4.yaml", "kdcrga"),   # -ontology injection
    ]
    for yaml_path in sorted(ABLATION_DIR.glob("*.yaml")):
        cfgs.append((yaml_path.stem, str(yaml_path), "kdcrga"))
    return cfgs


def main(smoke: bool = False, resume: bool = False) -> None:
    seeds = [0] if smoke else [0, 1, 2]
    run_sweep(_configs(), seeds, smoke=smoke, resume=resume)


if __name__ == "__main__":
    flags = sys.argv[1:]
    main(smoke="--smoke" in flags, resume="--resume" in flags)

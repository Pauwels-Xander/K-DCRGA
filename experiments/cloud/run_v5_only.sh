#!/usr/bin/env bash
# Finish kdcrga_v5 seeds 1+2 only (resume from checkpoint). No dedicom arms.
set -euo pipefail
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python experiments/sweep_bench.py --resume --only kdcrga_v5 --seeds 1 \
  > logs/v5_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/sweep_bench.py --resume --only kdcrga_v5 --seeds 2 \
  > logs/v5_seed2.log 2>&1 &
wait
echo "== kdcrga_v5 seeds 1,2 done =="
ls runs/bench/kdcrga_v5/seed_1/test_metrics.json runs/bench/kdcrga_v5/seed_2/test_metrics.json

#!/usr/bin/env bash
# kdcrga_dedicom_v4 seeds 1+2 only — one per GPU.
set -euo pipefail
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python experiments/sweep_bench.py --resume --only kdcrga_dedicom_v4 --seeds 1 \
  > logs/dedicom_v4_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/sweep_bench.py --resume --only kdcrga_dedicom_v4 --seeds 2 \
  > logs/dedicom_v4_seed2.log 2>&1 &
wait
echo "== kdcrga_dedicom_v4 seeds 1,2 done =="
ls runs/bench/kdcrga_dedicom_v4/seed_1/test_metrics.json runs/bench/kdcrga_dedicom_v4/seed_2/test_metrics.json
tar -czf ~/phase1b_dedicom_v4_results.tar.gz runs/bench/kdcrga_dedicom_v4
ls -lh ~/phase1b_dedicom_v4_results.tar.gz

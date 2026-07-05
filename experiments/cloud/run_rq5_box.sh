#!/usr/bin/env bash
# RQ5 zero-shot for one seed.
# Usage: REPO=$HOME/K-DCRGA bash experiments/cloud/run_rq5_box.sh <seed>
set -euo pipefail
seed="${1:?seed required (0|1|2)}"
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs
log_file="logs/rq5_seed${seed}.log"
python experiments/sweep_rq5_zero_shot.py --seed "$seed" \
  2>&1 | tee "$log_file"
for arm in random global_mean meddra; do
  test -f "runs/rq5/dedicom_v5/seed_${seed}/${arm}/test_metrics.json"
done

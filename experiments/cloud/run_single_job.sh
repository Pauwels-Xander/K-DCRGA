#!/usr/bin/env bash
# Run a single (arm, seed) benchmark job. Usage: run_single_job.sh <arm> <seed>
set -euo pipefail
arm="${1:?arm required}"
seed="${2:?seed required}"
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs
python experiments/sweep_bench.py --resume --only "$arm" --seeds "$seed" \
  2>&1 | tee "logs/${arm}_seed${seed}.log"
test -f "runs/bench/${arm}/seed_${seed}/test_metrics.json"

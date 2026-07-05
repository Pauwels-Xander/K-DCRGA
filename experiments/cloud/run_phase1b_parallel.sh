#!/usr/bin/env bash
# Phase 1B: kdcrga arms only, seeds 1+2 — fan out across visible GPUs.
# Disjoint from Box 1 (decagon seeds 1,2) and Box 2 (rgcn_mlp, lagat).
set -euo pipefail
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs
NGPU=$(nvidia-smi -L | wc -l)
JOBS=(
  "kdcrga_v5 1" "kdcrga_v5 2"
  "kdcrga_dedicom_v5 1" "kdcrga_dedicom_v5 2"
  "kdcrga_dedicom_v4 1" "kdcrga_dedicom_v4 2"
)

for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!JOBS[@]}"; do
      if [ $((i % NGPU)) -eq "$g" ]; then
        read -r arm seed <<< "${JOBS[$i]}"
        echo ">> [gpu $g] $arm seed $seed"
        CUDA_VISIBLE_DEVICES=$g python experiments/sweep_bench.py \
          --resume --only "$arm" --seeds "$seed"
      fi
    done
  ) > "logs/gpu_${g}.log" 2>&1 &
done
wait

echo "== Phase 1B complete =="
for arm in kdcrga_v5 kdcrga_dedicom_v5 kdcrga_dedicom_v4; do
  for seed in 1 2; do
    f="runs/bench/${arm}/seed_${seed}/test_metrics.json"
    if [ -f "$f" ]; then echo "OK $f"; else echo "MISSING $f"; fi
  done
done

tar -czf ~/phase1b_results.tar.gz \
  runs/bench/kdcrga_v5 \
  runs/bench/kdcrga_dedicom_v5 \
  runs/bench/kdcrga_dedicom_v4
ls -lh ~/phase1b_results.tar.gz

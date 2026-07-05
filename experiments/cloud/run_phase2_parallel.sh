#!/usr/bin/env bash
# Phase 2 only: rgcn_mlp + lagat, seeds 0,1,2 — fan out across visible GPUs.
set -euo pipefail
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs
NGPU=$(nvidia-smi -L | wc -l)
JOBS=("rgcn_mlp 0" "rgcn_mlp 1" "rgcn_mlp 2" "lagat 0" "lagat 1" "lagat 2")

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

echo "== Phase 2 complete =="
for arm in rgcn_mlp lagat; do
  for seed in 0 1 2; do
    f="runs/bench/${arm}/seed_${seed}/test_metrics.json"
    if [ -f "$f" ]; then echo "OK $f"; else echo "MISSING $f"; fi
  done
done

tar -czf ~/phase2_results.tar.gz runs/bench/rgcn_mlp runs/bench/lagat 2>/dev/null \
  || tar -czf ~/phase2_results.tar.gz runs/bench
ls -lh ~/phase2_results.tar.gz

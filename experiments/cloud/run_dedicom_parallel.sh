#!/usr/bin/env bash
# kdcrga dedicom arms only, seeds 1+2 — fan out across visible GPUs.
set -euo pipefail
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs
NGPU=$(nvidia-smi -L | wc -l)
JOBS=(
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
  ) > "logs/dedicom_gpu_${g}.log" 2>&1 &
done
wait

echo "== dedicom seeds 1,2 done =="
for arm in kdcrga_dedicom_v5 kdcrga_dedicom_v4; do
  for seed in 1 2; do
    f="runs/bench/${arm}/seed_${seed}/test_metrics.json"
    if [ -f "$f" ]; then echo "OK $f"; else echo "MISSING $f"; fi
  done
done

tar -czf ~/phase1b_dedicom_results.tar.gz \
  runs/bench/kdcrga_dedicom_v5 \
  runs/bench/kdcrga_dedicom_v4
ls -lh ~/phase1b_dedicom_results.tar.gz

#!/usr/bin/env bash
# Train pair-disjoint baseline arm (decagon_pd or lagat_pd) for given seed(s).
# Usage: REPO=$HOME/K-DCRGA bash experiments/cloud/run_baseline_pd_box.sh <arm> [seeds]
#   seeds default 0; e.g. "1,2" for seeds 1 and 2 sequentially.
set -euo pipefail
arm="${1:?arm required: decagon_pd|lagat_pd}"
seeds="${2:-0}"
case "$arm" in
  decagon_pd|lagat_pd) ;;
  *) echo "unknown arm: $arm" >&2; exit 1 ;;
esac
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs
if [[ "$seeds" == "0" ]]; then
  log_file="logs/${arm}_seed0.log"
else
  log_file="logs/${arm}_s${seeds//,/}.log"
fi
python -c "import sys; sys.path.insert(0,'experiments'); from sweep_bench import CONFIGS; print([c[0] for c in CONFIGS if c[0].endswith('_pd')])"
python experiments/sweep_bench.py --resume --only "$arm" --seeds "$seeds" \
  2>&1 | tee "$log_file"
IFS=',' read -ra seed_arr <<< "$seeds"
for s in "${seed_arr[@]}"; do
  test -f "runs/bench/${arm}/seed_${s}/best.pt"
done

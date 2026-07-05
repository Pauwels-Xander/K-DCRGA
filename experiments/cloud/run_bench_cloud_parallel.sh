#!/usr/bin/env bash
# Parallel version of run_bench_cloud.sh for a MULTI-GPU vast.ai box.
#
# The 14 benchmark runs (8 multi-seed main-arm runs + 6 new-arm runs) are
# independent, so we shard them at the (arm, seed) level and run one stream per
# GPU, pinned with CUDA_VISIBLE_DEVICES. Total GPU-hours are unchanged versus the
# serial script; wall-clock drops by roughly the GPU count. Everything is
# --resume, so a relaunch skips finished (arm,seed) runs and continues crashed
# ones from their last checkpoint.
#
# Setup (data unpack + install) runs ONCE, then the streams fan out.
#
# Prereqs (same as run_bench_cloud.sh):
#   * repo cloned at $REPO (default ~/K-DCRGA), branch feat/plan5-eval-harness
#   * kdcrga_runtime_data.tar.gz uploaded to $HOME
#   * a CUDA PyTorch base image with N GPUs visible
#
# Usage:   bash experiments/cloud/run_bench_cloud_parallel.sh
#   NGPU is auto-detected; override with e.g.  NGPU=4 bash ...
# Run inside tmux so an SSH drop does not kill training.
set -uo pipefail

REPO="${REPO:-$HOME/K-DCRGA}"
DATA_TARBALL="${DATA_TARBALL:-$HOME/kdcrga_runtime_data.tar.gz}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"

cd "$REPO"

echo "== 1. unpack runtime data (once) =="
tar -xzf "$DATA_TARBALL" -C "$REPO"
test -f third_party/knowddi/data/BioSNAP/BKG_file.txt
test -f data/processed/node_features.pt
test -f data/processed/zk_init.pt
test -f data/processed/atc_inject.pt

echo "== 2. install package (once) =="
python -m pip install --upgrade pip
python -m pip install -e .

echo "== 3. CUDA sanity check =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible -- wrong image or GPU"
print("torch", torch.__version__, "| devices:", torch.cuda.device_count())
PY

mkdir -p logs

# (arm, seed) jobs, heaviest first so round-robin balances load. Main arms need
# seeds 1,2 only (seed 0 is the local result); new arms need seeds 0,1,2.
JOBS=(
  "kdcrga_dedicom_v5 1" "kdcrga_dedicom_v5 2"   # heaviest: attention + ATC graph
  "kdcrga_v5 1" "kdcrga_v5 2"                    # heavy: attention + shared view
  "kdcrga_dedicom_v4 1" "kdcrga_dedicom_v4 2"    # medium
  "rgcn_mlp 0" "rgcn_mlp 1" "rgcn_mlp 2"         # medium-light
  "lagat 0" "lagat 1" "lagat 2"                  # light
  "decagon 1" "decagon 2"                        # light
)

echo "== 4. fan out ${#JOBS[@]} runs across ${NGPU} GPU(s) (round-robin, --resume) =="
for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!JOBS[@]}"; do
      if [ $((i % NGPU)) -eq "$g" ]; then
        read -r arm seed <<< "${JOBS[$i]}"
        echo ">> [gpu $g] $arm seed $seed  ($(date +%H:%M:%S))"
        CUDA_VISIBLE_DEVICES="$g" python experiments/sweep_bench.py \
          --resume --only "$arm" --seeds "$seed"
      fi
    done
    echo ">> [gpu $g] stream done"
  ) > "logs/gpu_${g}.log" 2>&1 &
done

wait
echo "== ALL STREAMS DONE. Results in runs/bench/<arm>/seed_<N>/ =="
echo "Per-GPU logs: logs/gpu_0.log .. logs/gpu_$((NGPU-1)).log"
echo "Pull down with:  tar -czf bench_results.tar.gz runs/bench"

#!/usr/bin/env bash
# Run the full RQ1 decomposition benchmark on a fresh vast.ai GPU box.
#
# Produces, on the FIXED data split (seed 0 split shared by all runs):
#   * seeds 1,2 for the four matched arms (decagon, kdcrga_v5,
#     kdcrga_dedicom_v5, kdcrga_dedicom_v4) -> completes the 3-seed set
#     (seed 0 already exists locally in runs/bench/<arm>/seed_0/).
#   * seeds 0,1,2 for the two NEW arms: rgcn_mlp (4th cell of the encoder x
#     decoder 2x2: plain R-GCN + shared-MLP, no attention) and lagat (matched
#     SOTA-positioning re-run).
#
# Runs are seeds-first by VALUE: the multi-seed block is what every headline
# contrast needs, so it goes first -- if you stop early (credit/time), you still
# get the most important results. Everything is --resume, so a re-launch skips
# finished (arm,seed) runs and continues crashed ones from their last checkpoint.
#
# Rough cost: ~14 runs x ~100-130 epochs. On a single 4090/A100 that's tens of
# GPU-hours (order ~$20-60 depending on instance); a multi-GPU box or several
# instances each with a different --only/--seeds shard finishes faster.
#
# Prereqs on the box (done by the runbook before this script is invoked):
#   * repo cloned at $REPO (default ~/K-DCRGA), branch feat/plan5-eval-harness
#   * kdcrga_runtime_data.tar.gz uploaded to $HOME (graph splits + prebuilt .pt artifacts)
#   * a CUDA PyTorch base image (torch + cuda already present)
#
# Usage:  bash experiments/cloud/run_bench_cloud.sh
# Run it inside tmux so an SSH drop does not kill training.
set -euo pipefail

REPO="${REPO:-$HOME/K-DCRGA}"
DATA_TARBALL="${DATA_TARBALL:-$HOME/kdcrga_runtime_data.tar.gz}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAIN_ARMS="decagon,kdcrga_v5,kdcrga_dedicom_v5,kdcrga_dedicom_v4"
NEW_ARMS="rgcn_mlp,lagat"

cd "$REPO"

echo "== 1. unpack runtime data (graph splits + prebuilt embeddings) =="
tar -xzf "$DATA_TARBALL" -C "$REPO"
test -f third_party/knowddi/data/BioSNAP/BKG_file.txt
test -f data/processed/node_features.pt
test -f data/processed/zk_init.pt
test -f data/processed/atc_inject.pt

echo "== 2. install package (torch+CUDA already in image; PyG is pure-python) =="
python -m pip install --upgrade pip
python -m pip install -e .

echo "== 3. CUDA sanity check =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible -- wrong image or GPU"
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
PY

mkdir -p logs

echo "== 4. PRIORITY: multi-seed (1,2) for the four matched arms =="
# seed 0 already exists locally; this completes the 3-seed set for every contrast.
python experiments/sweep_bench.py --resume --only "$MAIN_ARMS" --seeds 1,2 \
  2>&1 | tee logs/bench_main_seeds12.log

echo "== 5. SECONDARY: new arms (rgcn_mlp, lagat), all three seeds =="
python experiments/sweep_bench.py --resume --only "$NEW_ARMS" --seeds 0,1,2 \
  2>&1 | tee logs/bench_newarms_seeds012.log

echo "== DONE. Results in runs/bench/<arm>/seed_<N>/{test_metrics,per_se_metrics,history}.json =="
echo "  main arms: seeds 1,2 (seed 0 is local)"
echo "  rgcn_mlp, lagat: seeds 0,1,2"
echo "Tar them back to pull down:  tar -czf bench_results.tar.gz runs/bench"

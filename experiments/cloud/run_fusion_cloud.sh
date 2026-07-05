#!/usr/bin/env bash
# KnowDDI two-stream fusion cloud run (2080 Ti / V100 class GPU, >=50 GB disk).
# Run inside tmux on the box after syncing code + artifacts from the laptop.
#
# Prereqs on the box (upload before invoking):
#   * repo at ~/K-DCRGA (code tarball + artifacts tarball extracted)
#   * third_party/knowddi/, data/processed/, runs/bench/kdcrga_dedicom_v5/seed_{1,2}/
#   * NO runs/bench/kdcrga_dedicom_v5/seed_0/ (seed 0 is trained fresh here)
#
# Usage:  tmux new -s fusion 'bash experiments/cloud/run_fusion_cloud.sh 2>&1 | tee logs/fusion_cloud.log'
set -euo pipefail

REPO="${REPO:-$HOME/K-DCRGA}"
cd "$REPO"
mkdir -p logs runs/fusion/seed_{0,1,2}

log() { echo "[$(date -Iseconds)] $*"; }

# --- Miniconda (both envs) ---------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  log "Installing Miniconda..."
  apt-get update -qq && apt-get install -y -qq wget bzip2 tmux git
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh

# --- Env A: K-DCRGA (py3.11, torch 2.x + PyG) -------------------------------
if ! conda env list | grep -q '^kdcrga '; then
  log "Creating kdcrga env..."
  conda create -n kdcrga python=3.11 -y
fi
conda activate kdcrga
python -m pip install -U pip
python -m pip install -e ".[viz]"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible — check GPU image / driver"
print("kdcrga env OK:", torch.__version__, torch.cuda.get_device_name(0))
PY

# --- Env B: KnowDDI (py3.7, torch 1.6 + DGL 0.6) ----------------------------
if ! conda env list | grep -q '^knowddi '; then
  log "Creating knowddi env..."
  conda create -n knowddi python=3.7 -y
fi
conda activate knowddi
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
pip install dgl-cu102==0.6.1
pip install -r third_party/knowddi/requirements.txt
python - <<'PY'
import torch, dgl
print("knowddi env OK:", torch.__version__, "dgl", dgl.__version__, "cuda", torch.cuda.is_available())
PY

# KnowDDI dump script (tracked in parent repo, copied into vendored tree)
cp -f experiments/fusion/knowddi_dump_predictions.py third_party/knowddi/pytorch/dump_predictions.py

# --- Phase 1: train K-DCRGA seed 0 only ------------------------------------
log "=== K-DCRGA kdcrga_dedicom_v5 seed 0 ==="
conda activate kdcrga
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python experiments/sweep_bench.py --only kdcrga_dedicom_v5 --seeds 0
test -f runs/bench/kdcrga_dedicom_v5/seed_0/best.pt

# --- Phase 2: KnowDDI x3 ---------------------------------------------------
log "=== KnowDDI BioSNAP seeds 0,1,2 ==="
conda activate knowddi
cd "$REPO/third_party/knowddi/pytorch"
for s in 0 1 2; do
  log "KnowDDI seed $s..."
  python train.py -e "BioSNAP_seed${s}" --dataset=BioSNAP --seed="$s" --gpu 0 \
    --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 \
    --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 \
    --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2
done
cd "$REPO"

# --- Phase 3: dump per-triple predictions -----------------------------------
log "=== Stream F dumps (test + val) ==="
conda activate kdcrga
cd "$REPO"
for s in 0 1 2; do
  mkdir -p "runs/fusion/seed_${s}"
  python -c "
from kdcrga.eval.dump_predictions import dump_stream_f as d
s = ${s}
d(config_path='configs/bench/kdcrga_dedicom_v5.yaml',
  checkpoint_path=f'runs/bench/kdcrga_dedicom_v5/seed_{s}/best.pt',
  eval_txt='third_party/knowddi/data/BioSNAP/test.txt',
  out_path=f'runs/fusion/seed_{s}/f_test.pt', device='cuda')
d(config_path='configs/bench/kdcrga_dedicom_v5.yaml',
  checkpoint_path=f'runs/bench/kdcrga_dedicom_v5/seed_{s}/best.pt',
  eval_txt='third_party/knowddi/data/BioSNAP/valid.txt',
  out_path=f'runs/fusion/seed_{s}/f_val.pt', device='cuda')
print('F dumps seed', s, 'OK')
"
done

log "=== KnowDDI dumps (test + val) ==="
conda activate knowddi
cd "$REPO/third_party/knowddi/pytorch"
for s in 0 1 2; do
  python dump_predictions.py -e "BioSNAP_seed${s}" --dataset=BioSNAP --seed="$s" --gpu 0 --load_model \
    --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2 \
    --out_test "../../../runs/fusion/seed_${s}/knowddi_test.npz" \
    --out_valid "../../../runs/fusion/seed_${s}/knowddi_valid.npz"
done
cd "$REPO"

# --- Phase 4: fuse + aggregate ------------------------------------------------
log "=== Fusion sweep ==="
conda activate kdcrga
python experiments/sweep_fusion.py 0 1 2
cat runs/fusion/aggregate.json

log "=== DONE ==="
tar -czf /root/fusion_results.tar.gz runs/fusion runs/bench/kdcrga_dedicom_v5/seed_0
log "Results tarball: /root/fusion_results.tar.gz"

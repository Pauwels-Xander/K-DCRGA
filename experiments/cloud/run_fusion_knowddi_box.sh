#!/usr/bin/env bash
# Box 2-4: KnowDDI train one seed + dump test/valid npz for fusion.
# Usage: bash run_fusion_knowddi_box.sh <seed>
set -eo pipefail
SEED="${1:?seed required (0|1|2)}"
REPO="${REPO:-$HOME/K-DCRGA}"
cd "$REPO"
mkdir -p "logs" "runs/fusion/seed_${SEED}"

log() { echo "[$(date -Iseconds)] $*"; }

if ! command -v conda >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq wget bzip2 tmux liblmdb-dev build-essential
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q '^knowddi '; then
  conda create -n knowddi python=3.7 -y
fi
conda activate knowddi
# lmdb wheels often fail on minimal images; install system headers once.
if ! dpkg -s liblmdb-dev >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq liblmdb-dev build-essential
fi
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
pip install dgl-cu102==0.6.1
pip install -r third_party/knowddi/requirements.txt
# PyTorch 1.6 needs MKL 2020.x; conda-forge may pull MKL 2026 and break import.
conda install -y mkl=2020.2 mkl-service=2.3.0 -c pytorch
python -c "import torch,dgl; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"

cp -f experiments/fusion/knowddi_dump_predictions.py third_party/knowddi/pytorch/dump_predictions.py

log "KnowDDI train seed ${SEED}"
cd third_party/knowddi/pytorch
python train.py -e "BioSNAP_seed${SEED}" --dataset=BioSNAP --seed="${SEED}" --gpu 0 \
  --eval_every_iter=452 --weight_decay_rate=0.00001 --threshold=0.1 --lamda=0.5 \
  --num_infer_layers=1 --num_dig_layers=3 --gsl_rel_emb_dim=24 \
  --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2

log "KnowDDI dump seed ${SEED}"
python dump_predictions.py -e "BioSNAP_seed${SEED}" --dataset=BioSNAP --seed="${SEED}" --gpu 0 --load_model \
  --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2 \
  --out_test "../../../runs/fusion/seed_${SEED}/knowddi_test.npz" \
  --out_valid "../../../runs/fusion/seed_${SEED}/knowddi_valid.npz"

cd "$REPO"
tar -czf "/root/knowddi_s${SEED}_out.tar.gz" "runs/fusion/seed_${SEED}"
log "DONE -> /root/knowddi_s${SEED}_out.tar.gz"

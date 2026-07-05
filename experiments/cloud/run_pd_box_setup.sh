#!/usr/bin/env bash
# Fresh PD box: unpack artifacts + code, bootstrap kdcrga env, launch one seed.
# Usage: bash run_pd_box_setup.sh <seed>
set -eo pipefail
seed="${1:?seed required (0|1|2)}"
REPO="${REPO:-$HOME/K-DCRGA}"
mkdir -p "$REPO"
cd "$REPO"

log() { echo "[$(date -Iseconds)] $*"; }

if [[ -f /root/fusion_artifacts.tar.gz ]]; then
  log "Extract artifacts"
  tar -xzf /root/fusion_artifacts.tar.gz -C "$REPO"
fi
if [[ -f /root/code_pd.tar.gz ]]; then
  log "Extract code"
  tar -xzf /root/code_pd.tar.gz -C "$REPO"
fi

if ! command -v conda >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq wget bzip2 tmux git
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q '^kdcrga '; then
  conda create -n kdcrga python=3.11 -y
fi
conda activate kdcrga
python -m pip install -U pip
python -m pip install -e ".[viz]"
python -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"

log "Launch pd seed ${seed}"
chmod +x experiments/cloud/run_pd_seed.sh
export REPO
nohup bash experiments/cloud/run_pd_seed.sh "$seed" > ~/pd.out 2>&1 &
echo "PID $! -> tail -f ${REPO}/logs/pd_seed${seed}.log"

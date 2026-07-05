#!/usr/bin/env bash
# Fresh box: unpack artifacts + code, bootstrap kdcrga env, launch decagon_pd or lagat_pd.
# Usage: bash run_baseline_pd_box_setup.sh <decagon_pd|lagat_pd> [seeds]
set -eo pipefail
arm="${1:?arm required: decagon_pd|lagat_pd}"
seeds="${2:-0}"
case "$arm" in
  decagon_pd|lagat_pd) ;;
  *) echo "unknown arm: $arm" >&2; exit 1 ;;
esac
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
# Match pytorch:2.4.1-cuda12.1 image (pip's latest torch needs CUDA 13+ drivers).
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[viz]"
python -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.__version__, torch.cuda.get_device_name(0))"

if [[ "$seeds" == "0" ]]; then
  log_file="logs/${arm}_seed0.log"
else
  log_file="logs/${arm}_s${seeds//,/}.log"
fi
log "Launch ${arm} seeds ${seeds}"
chmod +x experiments/cloud/run_baseline_pd_box.sh
export REPO
nohup bash experiments/cloud/run_baseline_pd_box.sh "$arm" "$seeds" > ~/"${arm}.out" 2>&1 &
echo "PID $! -> tail -f ${REPO}/${log_file}"

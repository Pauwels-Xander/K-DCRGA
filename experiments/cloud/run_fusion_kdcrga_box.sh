#!/usr/bin/env bash
# Box 1: K-DCRGA kdcrga_dedicom_v5 seed 0 train + Stream-F dumps seeds 0,1,2.
set -euo pipefail
REPO="${REPO:-$HOME/K-DCRGA}"
cd "$REPO"
mkdir -p logs runs/fusion/seed_{0,1,2}

log() { echo "[$(date -Iseconds)] $*"; }

if ! command -v conda >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq wget bzip2 tmux
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

log "Train kdcrga_dedicom_v5 seed 0"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python experiments/sweep_bench.py --only kdcrga_dedicom_v5 --seeds 0 --resume
test -f runs/bench/kdcrga_dedicom_v5/seed_0/best.pt

log "Stream-F dumps seeds 0,1,2"
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

tar -czf /root/kdcrga_fusion_out.tar.gz runs/fusion runs/bench/kdcrga_dedicom_v5/seed_0
log "DONE -> /root/kdcrga_fusion_out.tar.gz"

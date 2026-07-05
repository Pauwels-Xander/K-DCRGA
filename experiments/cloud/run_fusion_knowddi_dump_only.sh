#!/usr/bin/env bash
# Resume KnowDDI after train crash: dump test/valid npz from saved best checkpoint.
# Usage: bash run_fusion_knowddi_dump_only.sh <seed>
set -eo pipefail
SEED="${1:?seed required (0|1|2)}"
REPO="${REPO:-$HOME/K-DCRGA}"
cd "$REPO"
mkdir -p "runs/fusion/seed_${SEED}"

log() { echo "[$(date -Iseconds)] $*"; }

# Avoid DataLoader "Too many open files" (train used num_workers=32).
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate knowddi

CKPT="third_party/knowddi/pytorch/experiments/BioSNAP_seed${SEED}/best_graph_classifier.pth"
test -f "$CKPT" || { log "Missing checkpoint $CKPT"; exit 1; }

cp -f experiments/fusion/knowddi_dump_predictions.py third_party/knowddi/pytorch/dump_predictions.py

log "KnowDDI dump-only seed ${SEED} (load ${CKPT})"
cd third_party/knowddi/pytorch
python dump_predictions.py -e "BioSNAP_seed${SEED}" --dataset=BioSNAP --seed="${SEED}" --gpu 0 --load_model \
  --num_workers=4 \
  --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 --MLP_num_layers=3 --MLP_dropout=0.2 \
  --out_test "../../../runs/fusion/seed_${SEED}/knowddi_test.npz" \
  --out_valid "../../../runs/fusion/seed_${SEED}/knowddi_valid.npz"

cd "$REPO"
tar -czf "/root/knowddi_s${SEED}_out.tar.gz" "runs/fusion/seed_${SEED}"
log "DONE -> /root/knowddi_s${SEED}_out.tar.gz"

#!/usr/bin/env bash
# Option A: train K-DCRGA_DEDICOM on the CANONICAL pair-disjoint BioSNAP split for
# ONE seed, then dump leak-free Stream-F predictions on BioSNAP test/valid for the
# KnowDDI fusion comparison. One box per seed. Resume-safe.
# Usage:  REPO=$HOME/K-DCRGA bash experiments/cloud/run_pd_seed.sh <seed>
# Needs the `kdcrga` (torch2.6) conda env active; runs on any modern GPU.
set -euo pipefail
seed="${1:?seed required (0|1|2)}"
cd "${REPO:-$HOME/K-DCRGA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs "runs/fusion_pd/seed_${seed}"

# 1) Train (writes runs/bench/kdcrga_dedicom_v5_pd/seed_<seed>/best.pt)
python experiments/sweep_bench.py --resume --only kdcrga_dedicom_v5_pd --seeds "$seed" \
  2>&1 | tee "logs/pd_seed${seed}.log"
test -f "runs/bench/kdcrga_dedicom_v5_pd/seed_${seed}/best.pt"

# 2) Dump Stream F on the canonical test/valid. Leak-free: this checkpoint never
#    trained on test.txt pairs (pair-disjoint split). The dump is split-agnostic
#    (encodes the BKG, scores the eval triples directly).
python - "$seed" <<'PY'
import sys
from kdcrga.eval.dump_predictions import dump_stream_f as d
s = sys.argv[1]
B = "third_party/knowddi/data/BioSNAP"
cfg = "configs/bench/kdcrga_dedicom_v5.yaml"          # same model; split irrelevant to dump
ckpt = f"runs/bench/kdcrga_dedicom_v5_pd/seed_{s}/best.pt"
d(config_path=cfg, checkpoint_path=ckpt, eval_txt=f"{B}/test.txt",
  out_path=f"runs/fusion_pd/seed_{s}/f_test.pt", device="cuda")
d(config_path=cfg, checkpoint_path=ckpt, eval_txt=f"{B}/valid.txt",
  out_path=f"runs/fusion_pd/seed_{s}/f_val.pt", device="cuda")
print(f"dumped Stream F -> runs/fusion_pd/seed_{s}/")
PY

echo "== pd seed ${seed} done: train + Stream-F dump =="
ls -la "runs/bench/kdcrga_dedicom_v5_pd/seed_${seed}/test_metrics.json" \
       "runs/fusion_pd/seed_${seed}/f_test.pt"

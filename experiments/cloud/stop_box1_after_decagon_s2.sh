#!/usr/bin/env bash
# On Phase-1 Box 1: stop training once decagon seed 2 finishes so the full
# run_bench_cloud.sh does not duplicate kdcrga / rgcn_mlp / lagat on other boxes.
set -euo pipefail
REPO="${REPO:-$HOME/K-DCRGA}"
MARKER="$REPO/runs/bench/decagon/seed_2/test_metrics.json"
LOG="$HOME/box1_stop.log"

echo "$(date -Is) watchdog started; waiting for $MARKER" >> "$LOG"
while [ ! -f "$MARKER" ]; do
  sleep 30
done
echo "$(date -Is) decagon seed 2 done — stopping bench tmux" >> "$LOG"
tmux kill-session -t bench 2>/dev/null || true
pkill -f "run_bench_cloud.sh" 2>/dev/null || true
pkill -f "sweep_bench.py" 2>/dev/null || true
echo "$(date -Is) Box 1 halted (decagon s1+s2 only)" >> "$LOG"

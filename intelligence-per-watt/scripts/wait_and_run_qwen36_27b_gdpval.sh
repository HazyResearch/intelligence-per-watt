#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/lambda-stanford/amir/intelligence-per-watt}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/intelligence-per-watt}"
GPU_COUNT="${GPU_COUNT:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT_DIR"

while true; do
  if "$PROJECT_DIR/scripts/preflight_clean_gpu_attribution.sh" empty; then
    echo "$(date -Is) GPUs clean; launching Qwen servers and GDPval shards"
    "$PROJECT_DIR/scripts/launch_qwen36_27b_per_gpu.sh"
    "$PROJECT_DIR/scripts/run_qwen36_27b_gdpval_shards.sh"
    exit 0
  fi

  echo "$(date -Is) waiting: GPUs have non-IPW compute processes"
  sleep "$POLL_SECONDS"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/lambda-stanford/amir/intelligence-per-watt}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/intelligence-per-watt}"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT_BASE="${PORT_BASE:-8000}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/qwen36_27b_vllm_logs}"
EXTRA_ENV="${EXTRA_ENV:-VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0}"
REQUIRE_CLEAN_GPUS="${REQUIRE_CLEAN_GPUS:-1}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
source "$VENV"

for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  session="ipw_qwen36_gpu${gpu}"
  tmux kill-session -t "$session" 2>/dev/null || true
done

if [[ "$REQUIRE_CLEAN_GPUS" == "1" ]]; then
  "$PROJECT_DIR/scripts/preflight_clean_gpu_attribution.sh" empty
fi

for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  port="$((PORT_BASE + gpu))"
  session="ipw_qwen36_gpu${gpu}"
  log_file="$LOG_DIR/gpu${gpu}_port${port}.log"
  cmd="cd '$ROOT_DIR' && source '$VENV' && CUDA_VISIBLE_DEVICES=$gpu $EXTRA_ENV python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port $port --model '$MODEL' --served-model-name '$MODEL' --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --enable-auto-tool-choice --tool-call-parser '$TOOL_CALL_PARSER' --trust-remote-code > '$log_file' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "launched $session on GPU $gpu port $port -> $log_file"
done

deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-1800}))
for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  port="$((PORT_BASE + gpu))"
  until curl -fsS --max-time 5 "http://localhost:${port}/v1/models" >/dev/null; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for port $port" >&2
      exit 1
    fi
    sleep 5
  done
  echo "ready port $port"
done

if [[ "$REQUIRE_CLEAN_GPUS" == "1" ]]; then
  "$PROJECT_DIR/scripts/preflight_clean_gpu_attribution.sh" vllm
fi

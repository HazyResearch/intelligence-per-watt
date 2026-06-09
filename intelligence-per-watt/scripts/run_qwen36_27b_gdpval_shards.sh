#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/lambda-stanford/amir/intelligence-per-watt}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/intelligence-per-watt}"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT_BASE="${PORT_BASE:-8000}"
GPU_COUNT="${GPU_COUNT:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/stirrup_qwen36_27b_gdpval_8gpu}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-1200}"
MAX_TURNS="${MAX_TURNS:-60}"
TURNS_REMAINING_WARNING_THRESHOLD="${TURNS_REMAINING_WARNING_THRESHOLD:-15}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
CONTEXT_SUMMARIZATION_CUTOFF="${CONTEXT_SUMMARIZATION_CUTOFF:-0.9}"
TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-0.2}"
TELEMETRY_BUFFER_SECONDS="${TELEMETRY_BUFFER_SECONDS:-1500}"
REQUIRE_CLEAN_GPUS="${REQUIRE_CLEAN_GPUS:-1}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"
source "$VENV"

for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  port="$((PORT_BASE + gpu))"
  curl -fsS --max-time 5 "http://localhost:${port}/v1/models" >/dev/null
done

if [[ "$REQUIRE_CLEAN_GPUS" == "1" ]]; then
  "$PROJECT_DIR/scripts/preflight_clean_gpu_attribution.sh" vllm
fi

for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  port="$((PORT_BASE + gpu))"
  session="ipw_qwen36_gdpval_gpu${gpu}"
  shard_dir="$OUTPUT_DIR/gpu${gpu}"
  log_file="$OUTPUT_DIR/gpu${gpu}.log"
  tmux kill-session -t "$session" 2>/dev/null || true
  cmd="cd '$ROOT_DIR' && source '$VENV' && IPW_EVAL_API_KEY=EMPTY ipw run --agent stirrup --model '$MODEL' --dataset gdpval-aa-single --client-base-url http://localhost:$port --api-key EMPTY --query-timeout $QUERY_TIMEOUT --output-dir '$shard_dir' --export-format jsonl --eval-client openai-server --eval-base-url http://localhost:$port/v1 --eval-model '$MODEL' --telemetry-gpu-id $gpu --telemetry-interval $TELEMETRY_INTERVAL --telemetry-buffer-seconds $TELEMETRY_BUFFER_SECONDS --dataset-kwargs '{\"download_files\":true,\"n_shards\":$GPU_COUNT,\"shard_idx\":$gpu}' --agent-kwargs '{\"backend\":\"local\",\"max_turns\":$MAX_TURNS,\"turns_remaining_warning_threshold\":$TURNS_REMAINING_WARNING_THRESHOLD,\"max_tokens\":$MAX_TOKENS,\"context_window\":$CONTEXT_WINDOW,\"include_view_image\":false,\"context_summarization_cutoff\":$CONTEXT_SUMMARIZATION_CUTOFF}' > '$log_file' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "launched $session shard $gpu/$GPU_COUNT on port $port -> $log_file"
done

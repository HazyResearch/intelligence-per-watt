#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-empty}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
GPU_COUNT="${GPU_COUNT:-8}"

if [[ "$MODE" != "empty" && "$MODE" != "vllm" ]]; then
  echo "Usage: $0 [empty|vllm]" >&2
  exit 2
fi

declare -A GPU_BY_BUS=()
while IFS=, read -r index bus_id; do
  index="${index//[[:space:]]/}"
  bus_id="${bus_id//[[:space:]]/}"
  [[ -n "$index" && -n "$bus_id" ]] || continue
  GPU_BY_BUS["$bus_id"]="$index"
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits)

declare -A VLLM_BY_GPU=()
bad=0
seen=0

while IFS=, read -r bus_id pid process_name used_memory; do
  bus_id="${bus_id//[[:space:]]/}"
  pid="${pid//[[:space:]]/}"
  process_name="${process_name#"${process_name%%[![:space:]]*}"}"
  process_name="${process_name%"${process_name##*[![:space:]]}"}"
  used_memory="${used_memory//[[:space:]]/}"
  [[ -n "$pid" && "$pid" != "[Not"*"Found]" ]] || continue

  gpu="${GPU_BY_BUS[$bus_id]:-unknown}"
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  ppid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ' || true)"
  parent_cmd=""
  if [[ -n "$ppid" ]]; then
    parent_cmd="$(ps -p "$ppid" -o args= 2>/dev/null || true)"
  fi
  seen=$((seen + 1))
  echo "gpu=${gpu} pid=${pid} mem=${used_memory}MiB cmd=${cmd:-$process_name}"

  if [[ "$MODE" == "empty" ]]; then
    bad=1
    continue
  fi

  if [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* && "$cmd" == *"$MODEL"* ]] \
    || [[ "$cmd" == "VLLM::EngineCore"* && "$parent_cmd" == *"vllm.entrypoints.openai.api_server"* && "$parent_cmd" == *"$MODEL"* ]]; then
    if [[ "$gpu" =~ ^[0-9]+$ ]]; then
      VLLM_BY_GPU["$gpu"]="${VLLM_BY_GPU[$gpu]:-}${pid} "
    fi
  else
    bad=1
  fi
done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory --format=csv,noheader,nounits)

if [[ "$MODE" == "empty" ]]; then
  if (( seen > 0 )); then
    echo "GPU preflight failed: GPUs are not empty." >&2
    exit 1
  fi
  echo "GPU preflight passed: no active GPU compute processes."
  exit 0
fi

if (( bad != 0 )); then
  echo "GPU preflight failed: found non-vLLM or wrong-model GPU process." >&2
  exit 1
fi

for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
  pids="${VLLM_BY_GPU[$gpu]:-}"
  count="$(wc -w <<< "$pids" | tr -d ' ')"
  if [[ "$count" != "1" ]]; then
    echo "GPU preflight failed: expected exactly one Qwen vLLM process on GPU ${gpu}, found ${count} (${pids})." >&2
    exit 1
  fi
done

echo "GPU preflight passed: exactly one expected Qwen vLLM process per GPU and no other GPU compute processes."

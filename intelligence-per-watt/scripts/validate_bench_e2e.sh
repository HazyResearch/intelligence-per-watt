#!/usr/bin/env bash
# ============================================================================
# validate_bench_e2e.sh — End-to-end validation of the bench command
# ============================================================================
#
# Runs `ipw bench` with various agent/model/dataset combos and validates
# the output artifacts (traces.jsonl, summary.json, results.json).
#
# Usage:
#   # With a running vLLM server:
#   ./scripts/validate_bench_e2e.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B
#
#   # With auto-server:
#   ./scripts/validate_bench_e2e.sh --auto-server --model Qwen/Qwen3-0.6B
#
#   # Quick (1 query, 1 agent, 1 dataset):
#   ./scripts/validate_bench_e2e.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B --quick
#
#   # Full sweep (all agents x datasets):
#   ./scripts/validate_bench_e2e.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B --full
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---- Parse args ------------------------------------------------------------
MODEL=""
PRESET=""
VLLM_URL=""
AUTO_SERVER=false
SWEEP_MODE="standard"
LIMIT=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --preset)      PRESET="$2"; shift 2 ;;
        --vllm-url)    VLLM_URL="$2"; shift 2 ;;
        --auto-server) AUTO_SERVER=true; shift ;;
        --quick)       SWEEP_MODE="quick"; LIMIT=1; shift ;;
        --full)        SWEEP_MODE="full"; shift ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" && -z "$PRESET" ]]; then
    echo "ERROR: Specify --model MODEL or --preset PRESET_NAME"
    exit 1
fi

# ---- Colors & counters -----------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

RESULTS_FILE=$(mktemp /tmp/ipw_bench_results_XXXXXX)
trap "rm -f $RESULTS_FILE" EXIT

log()     { echo -e "${CYAN}[bench]${NC} $*"; }
pass()    { echo -e "  ${GREEN}PASS${NC} $*"; echo "PASS" >> "$RESULTS_FILE"; }
fail()    { echo -e "  ${RED}FAIL${NC} $*"; echo "FAIL" >> "$RESULTS_FILE"; }
skip()    { echo -e "  ${YELLOW}SKIP${NC} $*"; echo "SKIP" >> "$RESULTS_FILE"; }
section() { echo -e "\n${BOLD}=== $* ===${NC}"; }

run_python_checks() {
    local output
    output=$("$PYTHON" -c "$1" 2>&1) || true
    while IFS= read -r line; do
        case "$line" in
            PASS:*)  pass "${line#PASS: }" ;;
            FAIL:*)  fail "${line#FAIL: }" ;;
            SKIP:*)  skip "${line#SKIP: }" ;;
            *)       [[ -n "$line" ]] && log "$line" ;;
        esac
    done <<< "$output"
}

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- Determine sweep matrix ------------------------------------------------
case "$SWEEP_MODE" in
    quick)    SWEEP_PAIRS="react:simpleqa" ;;
    standard) SWEEP_PAIRS="react:simpleqa react:mmlu-pro react:gaia" ;;
    full)
        SWEEP_PAIRS=""
        for ds in simpleqa mmlu-pro supergpqa gaia; do
            SWEEP_PAIRS="$SWEEP_PAIRS react:$ds"
        done
        for ds in simpleqa gaia; do
            SWEEP_PAIRS="$SWEEP_PAIRS openhands:$ds"
        done
        ;;
esac

OUTPUT_BASE=$(mktemp -d /tmp/ipw_validate_bench_XXXXXX)
MATRIX_RESULTS=()

log "Output: $OUTPUT_BASE"
log "Model: ${MODEL:-preset:$PRESET}"
log "Sweep: $SWEEP_MODE (limit=$LIMIT)"
log "Pairs: $SWEEP_PAIRS"

# ============================================================================
section "1. Prereq Checks"
# ============================================================================

run_python_checks "
from ipw.core.registry import AgentRegistry, DatasetRegistry
from ipw.agents import react as _r
from ipw.datasets import ensure_registered
ensure_registered()
agents = [k for k, _ in AgentRegistry.items()]
datasets = [k for k, _ in DatasetRegistry.items()]
print(f'PASS: {len(agents)} agents: {agents}')
print(f'PASS: {len(datasets)} datasets: {sorted(datasets)[:8]}...')
"

if [[ "$AUTO_SERVER" != "true" && -n "$VLLM_URL" ]]; then
    run_python_checks "
import requests
try:
    r = requests.get('${VLLM_URL}/models', timeout=10)
    if r.status_code == 200:
        models = [m['id'] for m in r.json().get('data', [])]
        print(f'PASS: vLLM reachable, serving: {models}')
    else:
        print(f'FAIL: vLLM returned {r.status_code}')
except Exception as e:
    print(f'FAIL: vLLM not reachable: {e}')
"
fi

# ============================================================================
section "2. Bench Command Sweep"
# ============================================================================

run_bench() {
    local agent="$1"
    local dataset="$2"
    local run_name="${agent}_${dataset}"
    local output_dir="$OUTPUT_BASE/$run_name"
    local status_file="$OUTPUT_BASE/${run_name}.status"

    log "Running: agent=$agent dataset=$dataset limit=$LIMIT"
    mkdir -p "$output_dir"

    # Write status to a file to avoid tqdm/logging noise in stdout capture
    "$PYTHON" -c "
import sys, json

from ipw.cli.bench import execute_benchmark

model_name = '${MODEL}'
preset = '${PRESET}'
if preset:
    from ipw.cli.model_presets import resolve_preset
    p = resolve_preset(preset)
    model_name = p['model_id']

vllm_url = '${VLLM_URL}' or None
auto = True if '${AUTO_SERVER}' == 'true' else False

try:
    result = execute_benchmark(
        client_id='vllm',
        model_name=model_name,
        agent_id='${agent}',
        dataset_id='${dataset}',
        max_samples=${LIMIT},
        client_base_url=vllm_url,
        output_dir='${output_dir}',
        enable_telemetry=False,
        skip_warmup=True,
        auto_server=auto,
    )
    q = result.get('queries', 0)
    c = result.get('completed', 0)
    with open('${status_file}', 'w') as f:
        f.write(f'BENCH_OK:{c}/{q}')
except Exception as e:
    with open('${status_file}', 'w') as f:
        f.write(f'BENCH_ERR:{e}')
" 2>&1 || true

    local bench_status
    bench_status=$(cat "$status_file" 2>/dev/null || echo "BENCH_ERR:unknown")

    if [[ "$bench_status" == BENCH_OK:* ]]; then
        local counts="${bench_status#BENCH_OK:}"
        pass "$run_name completed ($counts queries)"
        MATRIX_RESULTS+=("${GREEN}PASS${NC} $run_name ($counts)")
    else
        local err="${bench_status#BENCH_ERR:}"
        fail "$run_name: $err"
        MATRIX_RESULTS+=("${RED}FAIL${NC} $run_name: $err")
        return
    fi

    # execute_benchmark creates a timestamped subdirectory under output_dir
    # Find the actual artifact directory
    local actual_dir
    actual_dir=$(find "$output_dir" -maxdepth 1 -mindepth 1 -type d | head -1 || true)
    if [[ -z "$actual_dir" ]]; then
        actual_dir="$output_dir"
    fi
    log "Artifacts in: $actual_dir"

    # Validate artifacts
    if [[ -f "$actual_dir/results.json" ]]; then
        if "$PYTHON" -c "import json; json.load(open('$actual_dir/results.json'))" 2>/dev/null; then
            pass "$run_name: results.json valid"
        else
            fail "$run_name: results.json invalid JSON"
        fi
    else
        fail "$run_name: results.json missing"
    fi

    if [[ -f "$actual_dir/traces.jsonl" ]]; then
        local trace_count
        trace_count=$(wc -l < "$actual_dir/traces.jsonl")
        pass "$run_name: traces.jsonl ($trace_count lines)"
    else
        skip "$run_name: traces.jsonl not found"
    fi

    if [[ -f "$actual_dir/summary.json" ]]; then
        run_python_checks "
import json
s = json.load(open('$actual_dir/summary.json'))
if all(k in s for k in ['config', 'totals', 'averages']):
    print(f'PASS: $run_name summary.json complete')
else:
    print(f'FAIL: $run_name summary.json incomplete')
"
    else
        skip "$run_name: summary.json not found"
    fi
}

for pair in $SWEEP_PAIRS; do
    agent="${pair%%:*}"
    dataset="${pair##*:}"
    section "Bench: $agent + $dataset"
    run_bench "$agent" "$dataset"
done

# ============================================================================
section "3. Data Completeness (first successful run)"
# ============================================================================

FIRST_TRACE=""
for pair in $SWEEP_PAIRS; do
    run_name="${pair%%:*}_${pair##*:}"
    # Check in timestamped subdirectory first, then direct
    found=$(find "$OUTPUT_BASE/$run_name" -name "traces.jsonl" -type f 2>/dev/null | head -1 || true)
    if [[ -n "$found" ]]; then
        FIRST_TRACE="$found"
        break
    fi
done

if [[ -n "$FIRST_TRACE" ]]; then
    run_python_checks "
import json

traces = []
with open('$FIRST_TRACE') as f:
    for line in f:
        line = line.strip()
        if line:
            traces.append(json.loads(line))

if not traces:
    print('SKIP: No traces to validate')
else:
    t = traces[0]
    turns = t.get('turns', [])

    query_fields = ['query_id', 'workload_type', 'query_text', 'response_text', 'turns', 'total_wall_clock_s', 'completed']
    missing = [f for f in query_fields if f not in t]
    if missing:
        print(f'FAIL: Query missing fields: {missing}')
    else:
        print(f'PASS: All query-level fields present')

    if turns:
        turn_fields = ['turn_index', 'input_tokens', 'output_tokens', 'tools_called', 'wall_clock_s']
        missing = [f for f in turn_fields if f not in turns[0]]
        if missing:
            print(f'FAIL: Turn missing core fields: {missing}')
        else:
            print(f'PASS: All core turn fields present')

    has_tokens = any(turn.get('input_tokens', 0) > 0 for t2 in traces for turn in t2.get('turns', []))
    if has_tokens:
        print('PASS: Token counts populated')
    else:
        print('FAIL: No token counts found')
"
else
    skip "No successful runs for data completeness check"
fi

# ---- Summary ---------------------------------------------------------------
section "Results Matrix"
for r in "${MATRIX_RESULTS[@]+"${MATRIX_RESULTS[@]}"}"; do
    [[ -n "$r" ]] && echo -e "  $r"
done

section "Summary"
PASS_COUNT=$(grep -c '^PASS$' "$RESULTS_FILE") || true
FAIL_COUNT=$(grep -c '^FAIL$' "$RESULTS_FILE") || true
SKIP_COUNT=$(grep -c '^SKIP$' "$RESULTS_FILE") || true
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}, ${YELLOW}$SKIP_COUNT skipped${NC} (${TOTAL} total)"
echo -e "  Output: $OUTPUT_BASE"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "\n${RED}BENCH VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "\n${GREEN}ALL BENCH CHECKS PASSED${NC}"
    exit 0
fi

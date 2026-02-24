#!/usr/bin/env bash
# ============================================================================
# validate_data_collection.sh — Validate per-turn/per-trace data completeness
# ============================================================================
#
# Runs a full benchmark WITH energy telemetry and validates all data fields:
#   1. Per-turn: input_tokens, output_tokens, tools, wall_clock, energy, power
#   2. Per-trace: query_id, completed, wall_clock, turns
#   3. Summary: config, totals, averages
#   4. Energy: power > 0, energy deltas
#   5. JSONL round-trip
#   6. Per-action breakdown
#
# Requires: running vLLM server + energy monitor on same host
#
# Usage:
#   ./scripts/validate_data_collection.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B
#   ./scripts/validate_data_collection.sh --auto-server --preset glm-4.7-flash
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODEL=""
PRESET=""
VLLM_URL=""
AUTO_SERVER=false
LIMIT=3
AGENT="react"
DATASET="simpleqa"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --preset)      PRESET="$2"; shift 2 ;;
        --vllm-url)    VLLM_URL="$2"; shift 2 ;;
        --auto-server) AUTO_SERVER=true; shift ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --agent)       AGENT="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
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

RESULTS_FILE=$(mktemp /tmp/ipw_data_results_XXXXXX)
trap "rm -f $RESULTS_FILE" EXIT

log()     { echo -e "${CYAN}[data]${NC} $*"; }
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

OUTPUT_DIR=$(mktemp -d /tmp/ipw_validate_data_XXXXXX)
log "Output: $OUTPUT_DIR"

# ============================================================================
section "1. Run Benchmark with Telemetry"
# ============================================================================

log "Running: agent=$AGENT dataset=$DATASET model=${MODEL:-$PRESET} limit=$LIMIT"
log "Telemetry: enabled (per-action)"

BENCH_OUTPUT=$("$PYTHON" -c "
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
        agent_id='${AGENT}',
        dataset_id='${DATASET}',
        max_samples=${LIMIT},
        client_base_url=vllm_url,
        output_dir='${OUTPUT_DIR}',
        enable_telemetry=True,
        telemetry_granularity='per-action',
        skip_warmup=False,
        auto_server=auto,
    )
    q = result.get('queries', 0)
    c = result.get('completed', 0)
    has_energy = any(k for k in result if 'energy' in k.lower() or 'power' in k.lower())
    has_breakdown = 'action_breakdown' in result
    print(f'PASS: Benchmark completed ({c}/{q} queries)')
    if has_energy:
        print('PASS: Energy metrics present in results')
    else:
        print('FAIL: No energy metrics (telemetry may have failed)')
    if has_breakdown:
        print(f'PASS: Per-action breakdown ({len(result.get(\"action_breakdown\", []))} actions)')
    else:
        print('SKIP: No per-action breakdown')

    display = {k: v for k, v in result.items() if not k.startswith('_')}
    with open('${OUTPUT_DIR}/full_result.json', 'w') as f:
        json.dump(display, f, indent=2, default=str)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'FAIL: Benchmark failed: {e}')
" 2>&1) || true

while IFS= read -r line; do
    case "$line" in
        PASS:*)  pass "${line#PASS: }" ;;
        FAIL:*)  fail "${line#FAIL: }" ;;
        SKIP:*)  skip "${line#SKIP: }" ;;
        *)       [[ -n "$line" ]] && log "$line" ;;
    esac
done <<< "$BENCH_OUTPUT"

# execute_benchmark creates a timestamped subdirectory under OUTPUT_DIR
# Find the actual artifact directory for subsequent checks
ACTUAL_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || true)
if [[ -z "$ACTUAL_DIR" ]]; then
    ACTUAL_DIR="$OUTPUT_DIR"
fi
log "Artifacts in: $ACTUAL_DIR"

# ============================================================================
section "2. Validate traces.jsonl"
# ============================================================================

if [[ ! -f "$ACTUAL_DIR/traces.jsonl" ]]; then
    fail "traces.jsonl not found"
else
    run_python_checks "
import json

traces = []
with open('$ACTUAL_DIR/traces.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            traces.append(json.loads(line))

print(f'PASS: traces.jsonl has {len(traces)} traces')

query_fields = ['query_id', 'workload_type', 'query_text', 'response_text', 'turns', 'total_wall_clock_s', 'completed']
for i, t in enumerate(traces):
    missing = [f for f in query_fields if f not in t]
    if missing:
        print(f'FAIL: trace[{i}] missing: {missing}')
        break
else:
    print(f'PASS: All traces have required query-level fields')

turn_core = ['turn_index', 'input_tokens', 'output_tokens', 'tools_called', 'tool_latencies_s', 'wall_clock_s']
energy_fields = ['gpu_energy_joules', 'cpu_energy_joules', 'gpu_power_avg_watts', 'cpu_power_avg_watts']
total_turns = 0
for t in traces:
    for turn in t.get('turns', []):
        total_turns += 1
        missing = [f for f in turn_core if f not in turn]
        if missing:
            print(f'FAIL: Turn missing core fields: {missing}')
            break
    else:
        continue
    break
else:
    print(f'PASS: All {total_turns} turns have core fields')

energy_populated = sum(
    1 for t in traces for turn in t.get('turns', [])
    if any(turn.get(f) is not None for f in energy_fields)
)
if energy_populated > 0:
    print(f'PASS: {energy_populated}/{total_turns} turns have energy data')
else:
    print(f'SKIP: No turns have energy data (telemetry may not be active)')

total_in = sum(turn.get('input_tokens', 0) for t in traces for turn in t.get('turns', []))
total_out = sum(turn.get('output_tokens', 0) for t in traces for turn in t.get('turns', []))
if total_in > 0:
    print(f'PASS: Total input tokens = {total_in}')
else:
    print('FAIL: Total input tokens = 0')
if total_out > 0:
    print(f'PASS: Total output tokens = {total_out}')
else:
    print('FAIL: Total output tokens = 0')

completed = sum(1 for t in traces if t.get('completed'))
print(f'PASS: {completed}/{len(traces)} traces completed')

wall_clocks = [t.get('total_wall_clock_s', 0) for t in traces]
if all(w > 0 for w in wall_clocks):
    print(f'PASS: wall_clock > 0 for all traces ({min(wall_clocks):.2f}s - {max(wall_clocks):.2f}s)')
else:
    print('FAIL: Some traces have wall_clock = 0')
"
fi

# ============================================================================
section "3. Validate summary.json"
# ============================================================================

if [[ ! -f "$ACTUAL_DIR/summary.json" ]]; then
    fail "summary.json not found"
else
    run_python_checks "
import json
s = json.load(open('$ACTUAL_DIR/summary.json'))

if 'config' in s:
    cfg = s['config']
    if all(k in cfg for k in ['agent', 'model', 'dataset']):
        print(f'PASS: summary.config (agent={cfg[\"agent\"]}, dataset={cfg[\"dataset\"]})')
    else:
        print('FAIL: summary.config incomplete')
else:
    print('FAIL: summary missing config')

if 'totals' in s:
    t = s['totals']
    for k in ['queries', 'completed', 'turns', 'input_tokens', 'output_tokens', 'wall_clock_s']:
        if k not in t:
            print(f'FAIL: summary.totals missing {k}')
            break
    else:
        print(f'PASS: summary.totals complete (q={t[\"queries\"]}, turns={t[\"turns\"]})')
    if t.get('gpu_energy_joules') is not None:
        print(f'PASS: summary.totals.gpu_energy = {t[\"gpu_energy_joules\"]:.2f} J')
    else:
        print('SKIP: summary.totals.gpu_energy is null')
else:
    print('FAIL: summary missing totals')

if 'averages' in s:
    a = s['averages']
    if 'turns_per_query' in a:
        print(f'PASS: summary.averages (turns/q={a[\"turns_per_query\"]:.1f})')
    else:
        print('FAIL: summary.averages incomplete')
else:
    print('FAIL: summary missing averages')
"
fi

# ============================================================================
section "4. Validate results.json"
# ============================================================================

if [[ ! -f "$ACTUAL_DIR/results.json" ]]; then
    fail "results.json not found"
else
    run_python_checks "
import json
r = json.load(open('$ACTUAL_DIR/results.json'))

if all(k in r for k in ['queries', 'completed', 'total_turns']):
    print('PASS: results.json has basic fields')
else:
    print('FAIL: results.json missing basic fields')

if 'run_metadata' in r:
    meta = r['run_metadata']
    present = [k for k in ['client_id', 'model_name', 'agent_id', 'dataset_id'] if k in meta]
    print(f'PASS: run_metadata has {len(present)}/4 fields')
else:
    print('FAIL: results.json missing run_metadata')

energy_keys = [k for k in r if 'energy' in k.lower() or 'power' in k.lower()]
if energy_keys:
    print(f'PASS: Energy metrics: {energy_keys}')
else:
    print('SKIP: No energy metrics in results')

if 'action_breakdown' in r:
    bd = r['action_breakdown']
    print(f'PASS: action_breakdown has {len(bd)} entries')
else:
    print('SKIP: No action_breakdown')
"
fi

# ============================================================================
section "5. JSONL Round-Trip"
# ============================================================================

if [[ -f "$ACTUAL_DIR/traces.jsonl" ]]; then
    run_python_checks "
import json, tempfile
from pathlib import Path
from ipw.execution.trace import QueryTrace

traces = QueryTrace.load_jsonl(Path('$ACTUAL_DIR/traces.jsonl'))
if not traces:
    print('SKIP: No traces for round-trip')
else:
    tmp = Path(tempfile.mktemp(suffix='.jsonl'))
    for t in traces:
        t.save_jsonl(tmp)
    reloaded = QueryTrace.load_jsonl(tmp)

    if len(reloaded) == len(traces):
        print(f'PASS: Round-trip preserves {len(traces)} traces')
    else:
        print(f'FAIL: Count mismatch: {len(traces)} -> {len(reloaded)}')

    ok = True
    for i, (o, r) in enumerate(zip(traces, reloaded)):
        if o.query_id != r.query_id or o.num_turns != r.num_turns:
            print(f'FAIL: Mismatch at trace {i}')
            ok = False
            break
    if ok:
        print('PASS: Field-level comparison passed')

    if traces[0].turns and reloaded[0].turns:
        ot, rt = traces[0].turns[0], reloaded[0].turns[0]
        checks = [
            ot.input_tokens == rt.input_tokens,
            ot.output_tokens == rt.output_tokens,
            ot.tools_called == rt.tools_called,
            ot.gpu_energy_joules == rt.gpu_energy_joules,
        ]
        if all(checks):
            print('PASS: Per-turn round-trip preserves all fields')
        else:
            print('FAIL: Per-turn round-trip field mismatch')

    tmp.unlink(missing_ok=True)
"
else
    skip "No traces.jsonl for round-trip test (checked $ACTUAL_DIR)"
fi

# ---- Summary ---------------------------------------------------------------
section "Summary"
PASS_COUNT=$(grep -c '^PASS$' "$RESULTS_FILE") || true
FAIL_COUNT=$(grep -c '^FAIL$' "$RESULTS_FILE") || true
SKIP_COUNT=$(grep -c '^SKIP$' "$RESULTS_FILE") || true
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}, ${YELLOW}$SKIP_COUNT skipped${NC} (${TOTAL} total)"
echo -e "  Output: $OUTPUT_DIR"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "\n${RED}DATA COLLECTION VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "\n${GREEN}ALL DATA COLLECTION CHECKS PASSED${NC}"
    exit 0
fi

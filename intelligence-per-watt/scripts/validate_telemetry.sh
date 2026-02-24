#!/usr/bin/env bash
# ============================================================================
# validate_telemetry.sh — Validate energy monitor + GPU telemetry on this host
# ============================================================================
#
# Tests:
#   1. nvidia-smi is accessible and reports GPUs
#   2. Energy monitor binary launches and responds to gRPC health check
#   3. Streaming telemetry produces readings with power_watts > 0
#   4. Energy counter is monotonically increasing
#   5. TelemetrySession integration
#
# Usage:
#   ./scripts/validate_telemetry.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---- Colors & counters -----------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

RESULTS_FILE=$(mktemp /tmp/ipw_telemetry_results_XXXXXX)
trap "rm -f $RESULTS_FILE /tmp/ipw_monitor_pid /tmp/ipw_monitor_target" EXIT

log()     { echo -e "${CYAN}[telemetry]${NC} $*"; }
pass()    { echo -e "  ${GREEN}PASS${NC} $*"; echo "PASS" >> "$RESULTS_FILE"; }
fail()    { echo -e "  ${RED}FAIL${NC} $*"; echo "FAIL" >> "$RESULTS_FILE"; }
skip()    { echo -e "  ${YELLOW}SKIP${NC} $*"; echo "SKIP" >> "$RESULTS_FILE"; }
section() { echo -e "\n${BOLD}=== $* ===${NC}"; }

# Helper: run python, parse PASS/FAIL/SKIP lines from output
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

# Helper: run python that may launch subprocesses (output to temp file, no capture)
run_python_with_subprocess() {
    local tmpout
    tmpout=$(mktemp /tmp/ipw_pyout_XXXXXX)
    # Run Python, redirect energy monitor stderr to /dev/null to avoid blocking
    "$PYTHON" -c "$1" >"$tmpout" 2>/dev/null || true
    while IFS= read -r line; do
        case "$line" in
            PASS:*)  pass "${line#PASS: }" ;;
            FAIL:*)  fail "${line#FAIL: }" ;;
            SKIP:*)  skip "${line#SKIP: }" ;;
            *)       [[ -n "$line" ]] && log "$line" ;;
        esac
    done < "$tmpout"
    rm -f "$tmpout"
}

# ---- Prereq ----------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON" -c "import ipw" 2>/dev/null; then
    echo -e "${RED}ERROR:${NC} Cannot import ipw. Set PYTHON or install deps."
    exit 1
fi

# ============================================================================
section "1. NVIDIA GPU Detection"
# ============================================================================

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ', ' || echo "unknown")
    pass "nvidia-smi found — $GPU_COUNT GPU(s): $GPU_NAMES"
else
    fail "nvidia-smi not found"
fi

POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -n "$POWER" ]]; then
    pass "GPU power reporting works: $POWER"
else
    fail "nvidia-smi power query failed"
fi

MEM=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -n "$MEM" ]]; then
    pass "GPU memory reporting works: $MEM"
else
    fail "nvidia-smi memory query failed"
fi

# ============================================================================
section "2. Energy Monitor Binary"
# ============================================================================

MONITOR_BIN="src/ipw/telemetry/bin/linux-x86_64/energy-monitor"

if [[ -x "$MONITOR_BIN" ]]; then
    pass "Energy monitor binary exists: $MONITOR_BIN"
else
    fail "Energy monitor binary not found at $MONITOR_BIN"
fi

# ============================================================================
section "3. Energy Monitor Launch & Health Check"
# ============================================================================

MONITOR_PID=""
MONITOR_TARGET=""

cleanup_monitor() {
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    rm -f "$RESULTS_FILE" /tmp/ipw_monitor_pid /tmp/ipw_monitor_target
}
trap cleanup_monitor EXIT

# Use run_python_with_subprocess because launch_monitor spawns a child process
# whose inherited stdout/stderr would block $() capture
run_python_with_subprocess "
import sys, os, signal

try:
    from ipw.telemetry.launcher import launch_monitor, wait_for_ready
except ImportError as e:
    print(f'SKIP: Cannot import launcher: {e}')
    sys.exit(0)

try:
    pid, target = launch_monitor(timeout=15.0)
except (RuntimeError, FileNotFoundError) as e:
    print(f'FAIL: Monitor failed to launch: {e}')
    sys.exit(0)

with open('/tmp/ipw_monitor_pid', 'w') as f:
    f.write(str(pid))
with open('/tmp/ipw_monitor_target', 'w') as f:
    f.write(target)

if wait_for_ready(target, timeout=10.0):
    print(f'PASS: Monitor launched (pid={pid}, target={target})')
else:
    print(f'FAIL: Monitor launched but health check failed')
    os.kill(pid, signal.SIGTERM)
"

if [[ -f /tmp/ipw_monitor_pid ]]; then
    MONITOR_PID=$(cat /tmp/ipw_monitor_pid)
    MONITOR_TARGET=$(cat /tmp/ipw_monitor_target)
fi

# ============================================================================
section "4. Streaming Telemetry Readings"
# ============================================================================

if [[ -z "$MONITOR_TARGET" ]]; then
    skip "Monitor not running — skipping streaming tests"
else
    run_python_checks "
import sys, time

from ipw.telemetry import EnergyMonitorCollector

target = '$MONITOR_TARGET'
collector = EnergyMonitorCollector(target=target)

readings = []
try:
    with collector.start():
        deadline = time.monotonic() + 8.0
        for reading in collector.stream_readings():
            readings.append(reading)
            if len(readings) >= 10 or time.monotonic() > deadline:
                break
except Exception as e:
    print(f'FAIL: Streaming error: {e}')
    sys.exit(0)

if len(readings) < 1:
    print('FAIL: No telemetry readings collected')
    sys.exit(0)

print(f'PASS: Collected {len(readings)} readings')

r = readings[0]
if r.power_watts is not None and r.power_watts > 0:
    print(f'PASS: GPU power = {r.power_watts:.1f} W')
else:
    print(f'FAIL: GPU power not populated or zero: {r.power_watts}')

if r.energy_joules is not None:
    print(f'PASS: GPU energy counter = {r.energy_joules:.2f} J')
else:
    print(f'FAIL: GPU energy counter is None')

mem_usage = getattr(r, 'gpu_memory_usage_mb', None)
mem_total = getattr(r, 'gpu_memory_total_mb', None)
if mem_usage is not None or mem_total is not None:
    print(f'PASS: GPU memory metric populated (usage={mem_usage}, total={mem_total})')
else:
    print(f'FAIL: GPU memory metrics not populated')

energies = [rd.energy_joules for rd in readings if rd.energy_joules is not None]
if len(energies) >= 2:
    is_mono = all(energies[i] >= energies[i-1] for i in range(1, len(energies)))
    if is_mono:
        delta = energies[-1] - energies[0]
        print(f'PASS: Energy monotonically increasing (delta={delta:.2f} J over {len(energies)} samples)')
    else:
        print(f'FAIL: Energy counter NOT monotonic: {energies[:5]}...')
else:
    print(f'SKIP: Not enough energy samples for monotonicity ({len(energies)})')
"
fi

# ============================================================================
section "5. TelemetrySession Integration"
# ============================================================================

if [[ -z "$MONITOR_TARGET" ]]; then
    skip "Monitor not running — skipping session tests"
else
    run_python_checks "
import sys, time

from ipw.telemetry import EnergyMonitorCollector
from ipw.execution.telemetry_session import TelemetrySession

target = '$MONITOR_TARGET'
collector = EnergyMonitorCollector(target=target)

try:
    with TelemetrySession(collector, buffer_seconds=60.0, max_samples=1000) as session:
        t_start = time.time()
        time.sleep(3)
        t_end = time.time()

        all_readings = list(session.readings())
        windowed = list(session.window(t_start, t_end))

    if len(all_readings) > 0:
        print(f'PASS: TelemetrySession.readings() returned {len(all_readings)} samples')
    else:
        print('FAIL: TelemetrySession.readings() returned 0 samples')

    if len(windowed) > 0:
        print(f'PASS: TelemetrySession.window() returned {len(windowed)} samples in 3s window')
    else:
        print('FAIL: TelemetrySession.window() returned 0 samples')
except Exception as e:
    print(f'FAIL: TelemetrySession error: {e}')
"
fi

# ---- Cleanup & Summary ----------------------------------------------------
if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
fi

section "Summary"
PASS_COUNT=$(grep -c '^PASS$' "$RESULTS_FILE") || true
FAIL_COUNT=$(grep -c '^FAIL$' "$RESULTS_FILE") || true
SKIP_COUNT=$(grep -c '^SKIP$' "$RESULTS_FILE") || true
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}, ${YELLOW}$SKIP_COUNT skipped${NC} (${TOTAL} total)"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "\n${RED}VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "\n${GREEN}ALL TELEMETRY CHECKS PASSED${NC}"
    exit 0
fi

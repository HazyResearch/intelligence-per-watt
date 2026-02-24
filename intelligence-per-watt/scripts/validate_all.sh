#!/usr/bin/env bash
# ============================================================================
# validate_all.sh — Master validation script for IPW Phase 3
# ============================================================================
#
# Runs all validation scripts in sequence:
#   1. Unit tests (pytest)
#   2. Telemetry validation (energy monitor + GPU)
#   3. Server lifecycle validation (lock files, process detection)
#   4. Bench E2E validation (agent x dataset sweep)
#   5. Data collection validation (per-turn/per-trace completeness)
#
# Usage:
#   # Minimal (unit tests + telemetry + lifecycle only — no vLLM needed):
#   ./scripts/validate_all.sh
#
#   # With running vLLM server:
#   ./scripts/validate_all.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B
#
#   # With auto-server:
#   ./scripts/validate_all.sh --auto-server --preset glm-4.7-flash
#
#   # Full sweep with live vLLM lifecycle test:
#   ./scripts/validate_all.sh --vllm-url http://localhost:8000/v1 --model Qwen/Qwen3-4B --full
#
# Exit codes:
#   0 — all stages passed
#   1 — one or more stages failed
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---- Parse args ------------------------------------------------------------
MODEL=""
PRESET=""
VLLM_URL=""
AUTO_SERVER=""
SWEEP_MODE=""
LIVE_VLLM_MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; EXTRA_ARGS+=(--model "$2"); shift 2 ;;
        --preset)       PRESET="$2"; EXTRA_ARGS+=(--preset "$2"); shift 2 ;;
        --vllm-url)     VLLM_URL="$2"; EXTRA_ARGS+=(--vllm-url "$2"); shift 2 ;;
        --auto-server)  AUTO_SERVER="true"; EXTRA_ARGS+=(--auto-server); shift ;;
        --quick)        SWEEP_MODE="--quick"; shift ;;
        --full)         SWEEP_MODE="--full"; shift ;;
        --with-vllm)    LIVE_VLLM_MODEL="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Colors ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

STAGE_RESULTS=()

stage_pass() { STAGE_RESULTS+=("${GREEN}PASS${NC} $1"); }
stage_fail() { STAGE_RESULTS+=("${RED}FAIL${NC} $1"); }
stage_skip() { STAGE_RESULTS+=("${YELLOW}SKIP${NC} $1"); }

banner() {
    echo ""
    echo -e "${BOLD}##############################################################################${NC}"
    echo -e "${BOLD}#  $*${NC}"
    echo -e "${BOLD}##############################################################################${NC}"
    echo ""
}

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
banner "STAGE 1: Unit Tests (pytest)"
# ============================================================================

if "$PYTHON" -m pytest src/ipw/tests/cli/test_server_manager.py \
                       src/ipw/tests/cli/test_vllm_lifecycle.py \
                       src/ipw/tests/telemetry/test_correlation.py \
                       -v --tb=short 2>&1; then
    stage_pass "Unit tests"
else
    stage_fail "Unit tests"
fi

# ============================================================================
banner "STAGE 2: Telemetry Validation"
# ============================================================================

if bash "$SCRIPT_DIR/validate_telemetry.sh"; then
    stage_pass "Telemetry"
else
    stage_fail "Telemetry"
fi

# ============================================================================
banner "STAGE 3: Server Lifecycle Validation"
# ============================================================================

LIFECYCLE_ARGS=()
if [[ -n "$LIVE_VLLM_MODEL" ]]; then
    LIFECYCLE_ARGS+=(--with-vllm "$LIVE_VLLM_MODEL")
fi

if bash "$SCRIPT_DIR/validate_server_lifecycle.sh" "${LIFECYCLE_ARGS[@]+"${LIFECYCLE_ARGS[@]}"}"; then
    stage_pass "Server lifecycle"
else
    stage_fail "Server lifecycle"
fi

# ============================================================================
banner "STAGE 4: Bench E2E Validation"
# ============================================================================

if [[ -n "$MODEL" || -n "$PRESET" ]]; then
    BENCH_ARGS=("${EXTRA_ARGS[@]}")
    [[ -n "$SWEEP_MODE" ]] && BENCH_ARGS+=("$SWEEP_MODE")

    if bash "$SCRIPT_DIR/validate_bench_e2e.sh" "${BENCH_ARGS[@]}"; then
        stage_pass "Bench E2E"
    else
        stage_fail "Bench E2E"
    fi
else
    stage_skip "Bench E2E (no --model or --preset specified)"
fi

# ============================================================================
banner "STAGE 5: Data Collection Validation"
# ============================================================================

if [[ -n "$MODEL" || -n "$PRESET" ]]; then
    DATA_ARGS=("${EXTRA_ARGS[@]}")

    if bash "$SCRIPT_DIR/validate_data_collection.sh" "${DATA_ARGS[@]}"; then
        stage_pass "Data collection"
    else
        stage_fail "Data collection"
    fi
else
    stage_skip "Data collection (no --model or --preset specified)"
fi

# ============================================================================
banner "FINAL REPORT"
# ============================================================================

echo -e "${BOLD}Stage Results:${NC}"
for r in "${STAGE_RESULTS[@]}"; do
    echo -e "  $r"
done
echo ""

FAILURES=0
for r in "${STAGE_RESULTS[@]}"; do
    if echo -e "$r" | grep -q "FAIL"; then
        ((FAILURES++))
    fi
done

if [[ $FAILURES -gt 0 ]]; then
    echo -e "${RED}${BOLD}$FAILURES stage(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}${BOLD}All stages passed!${NC}"
    exit 0
fi

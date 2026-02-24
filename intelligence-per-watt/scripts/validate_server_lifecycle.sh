#!/usr/bin/env bash
# ============================================================================
# validate_server_lifecycle.sh — Validate vLLM server lifecycle management
# ============================================================================
#
# Tests:
#   1. CLI commands registered (bench, servers)
#   2. Lock file registry (acquire/release/cleanup)
#   3. Process detector (port scanning, kill)
#   4. Server config building with presets
#   5. Error classes
#   6. Live vLLM server lifecycle (optional, with --with-vllm)
#
# Usage:
#   ./scripts/validate_server_lifecycle.sh
#   ./scripts/validate_server_lifecycle.sh --with-vllm Qwen/Qwen3-0.6B
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

LIVE_VLLM_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-vllm) LIVE_VLLM_MODEL="${2:-}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Colors & counters -----------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

RESULTS_FILE=$(mktemp /tmp/ipw_lifecycle_results_XXXXXX)
trap "rm -f $RESULTS_FILE" EXIT

log()     { echo -e "${CYAN}[lifecycle]${NC} $*"; }
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

if ! "$PYTHON" -c "import ipw" 2>/dev/null; then
    echo -e "${RED}ERROR:${NC} Cannot import ipw."
    exit 1
fi

# ============================================================================
section "1. CLI Commands Registered"
# ============================================================================

run_python_checks "
import sys
from click.testing import CliRunner
from ipw.cli import cli

runner = CliRunner()
result = runner.invoke(cli, ['--help'])
if 'servers' in result.output:
    print('PASS: servers command registered')
else:
    print('FAIL: servers command not found in CLI help')
if 'bench' in result.output:
    print('PASS: bench command registered')
else:
    print('FAIL: bench command not found in CLI help')
"

# ============================================================================
section "2. Lock File Registry"
# ============================================================================

run_python_checks "
import json, os, sys, tempfile, shutil
from pathlib import Path
from ipw.cli.vllm_lifecycle import VLLMServerInfo, VLLMServerRegistry

tmpdir = tempfile.mkdtemp(prefix='ipw_test_')
try:
    registry = VLLMServerRegistry(lock_dir=Path(tmpdir))

    info = VLLMServerInfo(pid=os.getpid(), model_id='test/model', port=8000, owner_pid=os.getpid())
    if registry.acquire_lock(8000, info):
        print('PASS: Lock acquired for port 8000')
    else:
        print('FAIL: Lock acquire returned False')

    lock_path = Path(tmpdir) / 'port_8000.lock'
    if lock_path.exists():
        print('PASS: Lock file created on disk')
    else:
        print('FAIL: Lock file not found')

    retrieved = registry.get_lock_info(8000)
    if retrieved and retrieved.model_id == 'test/model':
        print('PASS: Lock info round-trips correctly')
    else:
        print(f'FAIL: Lock info mismatch: {retrieved}')

    info2 = VLLMServerInfo(pid=99999, model_id='other/model', port=8000, owner_pid=os.getpid())
    if not registry.acquire_lock(8000, info2):
        print('PASS: Second acquire correctly rejected')
    else:
        print('FAIL: Second acquire should have been rejected')

    locks = registry.list_locks()
    if 8000 in locks:
        print('PASS: list_locks() shows port 8000')
    else:
        print(f'FAIL: list_locks() missing port 8000')

    registry.release_lock(8000)
    if not lock_path.exists():
        print('PASS: Lock file removed after release')
    else:
        print('FAIL: Lock file still exists after release')

    stale_info = VLLMServerInfo(pid=99999999, model_id='stale/model', port=9000, owner_pid=99999999)
    stale_path = Path(tmpdir) / 'port_9000.lock'
    stale_path.write_text(json.dumps(stale_info.to_dict()))
    cleaned = registry.cleanup_stale_locks()
    if 9000 in cleaned:
        print('PASS: Stale lock cleaned up for dead PID')
    else:
        print(f'FAIL: Stale lock not cleaned: {cleaned}')
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
"

# ============================================================================
section "3. Process Detector"
# ============================================================================

run_python_checks "
import os
from ipw.cli.vllm_lifecycle import VLLMProcessDetector, _is_process_alive

if _is_process_alive(os.getpid()):
    print('PASS: Current process detected as alive')
else:
    print('FAIL: Current process not detected as alive')

if not _is_process_alive(99999999):
    print('PASS: Non-existent PID correctly detected as dead')
else:
    print('FAIL: Non-existent PID detected as alive')

detector = VLLMProcessDetector()
result = detector.find_vllm_on_port(59999)
if result is None:
    print('PASS: No process found on unused port 59999')
else:
    print(f'FAIL: Unexpected process on port 59999')

if detector.kill_process(99999999):
    print('PASS: kill_process(nonexistent) returns True')
else:
    print('FAIL: kill_process(nonexistent) returned False')
"

# ============================================================================
section "4. Server Config Building"
# ============================================================================

run_python_checks "
from ipw.cli.server_manager import ServerConfig, build_server_configs, parse_submodel_spec

configs = build_server_configs('Qwen/Qwen3-4B', 'main', [])
if len(configs) == 1 and configs[0].port == 8000 and configs[0].backend == 'vllm':
    print('PASS: Single vLLM config: port=8000')
else:
    print(f'FAIL: Unexpected config: {[(c.port, c.backend) for c in configs]}')

configs = build_server_configs('Qwen/Qwen3-4B', 'main', ['math:vllm:Qwen/Math'], base_port=8000)
if len(configs) == 2 and configs[1].port == 8001:
    print('PASS: Submodel config: port=8001, alias=math')
else:
    print(f'FAIL: Submodel config wrong')

configs = build_server_configs('Qwen/Qwen3-4B', 'main', ['small:ollama:llama3.2:1b', 'code:vllm:Qwen/Code'], base_port=8000)
if len(configs) == 3 and configs[1].backend == 'ollama' and configs[1].port == 11434:
    print('PASS: Mixed backends handled correctly')
else:
    print(f'FAIL: Mixed backends wrong')

from ipw.cli.model_presets import resolve_preset, list_presets
presets = list_presets()
if len(presets) >= 3:
    print(f'PASS: {len(presets)} presets available: {presets}')
else:
    print(f'FAIL: Too few presets')

p = resolve_preset('glm-4.7-flash')
if p['model_id'] == 'zai-org/GLM-4.7-FP8':
    print('PASS: glm-4.7-flash preset resolves correctly')
else:
    print(f'FAIL: Preset resolution wrong: {p}')
"

# ============================================================================
section "5. Error Classes"
# ============================================================================

run_python_checks "
from ipw.cli.vllm_lifecycle import PortConflictError, ModelMismatchError

try:
    raise PortConflictError(port=8000, existing_model='old/m', requested_model='new/m', owner='ipw')
except PortConflictError as e:
    msg = str(e)
    if '8000' in msg and 'old/m' in msg:
        print('PASS: PortConflictError formats correctly')
    else:
        print(f'FAIL: PortConflictError msg: {msg}')

try:
    raise ModelMismatchError(port=8000, expected_model='exp/m', actual_model='act/m')
except ModelMismatchError as e:
    msg = str(e)
    if '8000' in msg and 'exp/m' in msg:
        print('PASS: ModelMismatchError formats correctly')
    else:
        print(f'FAIL: ModelMismatchError msg: {msg}')
"

# ============================================================================
section "6. Live vLLM Server Lifecycle"
# ============================================================================

if [[ -z "$LIVE_VLLM_MODEL" ]]; then
    skip "No --with-vllm MODEL specified — skipping live server tests"
    skip "(Run with: ./scripts/validate_server_lifecycle.sh --with-vllm Qwen/Qwen3-0.6B)"
else
    HAS_VLLM=$("$PYTHON" -c "
try:
    import vllm; print('yes')
except ImportError:
    print('no')
" 2>/dev/null)

    if [[ "$HAS_VLLM" != "yes" ]]; then
        skip "vllm not installed — skipping live server tests"
    else
        log "Testing live vLLM server with model: $LIVE_VLLM_MODEL"
        run_python_checks "
import sys, time, os, signal, socket
from ipw.cli.server_manager import InferenceServerManager, build_server_configs

PORT = 8199
MODEL = '$LIVE_VLLM_MODEL'

configs = build_server_configs(MODEL, 'test', [], base_port=PORT)
manager = InferenceServerManager(configs)

try:
    urls = manager.start_all()
    if 'test' in urls:
        print(f'PASS: Server started, URL = {urls[\"test\"]}')
    else:
        print(f'FAIL: start_all returned no URL for \"test\"')
        manager.stop_all()
        sys.exit(0)
except Exception as e:
    print(f'FAIL: Server failed to start: {e}')
    sys.exit(0)

try:
    manager.warmup_all()
    print('PASS: Warmup completed')
except Exception as e:
    print(f'FAIL: Warmup failed: {e}')

import requests
try:
    base_url = urls.get('test', f'http://localhost:{PORT}/v1')
    resp = requests.get(f'{base_url}/models', timeout=10)
    if resp.status_code == 200:
        models = resp.json().get('data', [])
        print(f'PASS: /v1/models returns {len(models)} model(s)')
    else:
        print(f'FAIL: /v1/models returned {resp.status_code}')
except Exception as e:
    print(f'FAIL: /v1/models error: {e}')

try:
    manager.stop_all()
    time.sleep(2)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', PORT))
    sock.close()
    if result != 0:
        print('PASS: Server stopped, port is free')
    else:
        print('FAIL: Port still in use after stop')
except Exception as e:
    print(f'FAIL: Server stop error: {e}')
"
    fi
fi

# ---- Summary ---------------------------------------------------------------
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
    echo -e "\n${GREEN}ALL SERVER LIFECYCLE CHECKS PASSED${NC}"
    exit 0
fi

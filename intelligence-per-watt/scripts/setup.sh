#!/usr/bin/env bash
# ============================================================================
# setup.sh — Set up the Intelligence Per Watt development environment
# ============================================================================
#
# Creates a Python 3.13 virtual environment, installs the package with all
# extras, and optionally builds the energy monitor from Rust source.
#
# Usage:
#   ./scripts/setup.sh          # Install with [all] extras
#   ./scripts/setup.sh vllm     # Install with [vllm] only
#   ./scripts/setup.sh agents   # Install with [agents] only
#
# Prerequisites:
#   - uv (will be auto-installed if missing)
#   - Rust + protoc (optional, for energy monitor build)
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT"

info() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

# --- uv ---
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
info "uv $(uv --version)"

# --- venv + Python 3.13 (uv fetches it automatically) ---
if [ ! -d "$PKG/.venv" ]; then
    info "Creating virtual environment (Python 3.13)..."
    uv venv "$PKG/.venv" --python 3.13
fi
source "$PKG/.venv/bin/activate"

# --- Install package ---
EXTRAS="${1:-all}"
info "Installing intelligence-per-watt[${EXTRAS}]..."
uv pip install -e "${PKG}[${EXTRAS}]"

# --- Energy monitor (optional, needs rust + protoc) ---
ENERGY_MONITOR_ROOT="$(cd "$ROOT/.." && pwd)/energy-monitor"
if [ -d "$ENERGY_MONITOR_ROOT" ] && command -v cargo &>/dev/null && command -v protoc &>/dev/null; then
    info "Building energy monitor..."
    python "$ROOT/scripts/build_energy_monitor.py"
elif [ -f "$PKG/src/ipw/telemetry/bin/linux-x86_64/energy-monitor" ]; then
    info "Energy monitor binary already present (pre-built)"
else
    warn "Skipping energy monitor build (install rust and protoc first)"
    warn "  Rust:   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    warn "  Protoc: sudo apt install -y protobuf-compiler"
fi

info "Done! Activate with: source $PKG/.venv/bin/activate"
info "Test with: ipw --help"

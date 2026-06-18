#!/usr/bin/env bash
# scripts/sweep_spec_decode.sh — input-length sweep with spec-decode vs. baseline.
#
# Mirrors sweep_qkv_fuse.sh: runs `ipw profile --client mlx --dataset
# fixed_length` across a list of prompt lengths, with and without a draft
# model. The non-baseline arm passes a smaller vocab-compatible model via
# --client-param draft_model=... so mlx_lm.stream_generate dispatches through
# speculative_generate_step. Energy/TTFT/throughput are captured by IPW's
# existing collector.
#
# Defaults target Qwen3-8B-bf16 (target) + Qwen3-1.7B-bf16 (draft) — same
# Qwen3 tokenizer family, ~5× cheaper draft, expected α≈0.7 → ~1.8× decode.
#
# Usage:
#     bash scripts/sweep_spec_decode.sh
#
# Override via env vars, e.g.:
#     DRAFT="mlx-community/Qwen3-0.6B-bf16" \
#     PROMPT_LENGTHS="50 200 2000" \
#     NUM_DRAFT_TOKENS=6 \
#     bash scripts/sweep_spec_decode.sh
#
# Idempotent: a run dir that already contains a summary.json is skipped, so
# you can resume after Ctrl-C without re-running completed configs.

set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3-8B-bf16}"
DRAFT="${DRAFT:-mlx-community/Qwen3-1.7B-bf16}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-4}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-50 200 500 1000 2000}"
MAX_TOKENS="${MAX_TOKENS:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-6}"
WARMUP_QUERIES="${WARMUP_QUERIES:-1}"
CHARS_PER_TOKEN="${CHARS_PER_TOKEN:-5}"
OUT_ROOT="${OUT_ROOT:-runs/sweep_spec_decode}"

# Re-exec under caffeinate so display / disk / system idle sleep don't
# interrupt the sweep. caffeinate inherits this script's lifetime via exec.
if [[ -z "${UNDER_CAFFEINATE:-}" ]]; then
    export UNDER_CAFFEINATE=1
    exec caffeinate -dims "$0" "$@"
fi

# Locate the IPW repo root (the dir containing .venv/). Script lives in scripts/.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPW_ROOT="$(cd "$HERE/.." && pwd)"
cd "$IPW_ROOT"

if [[ ! -x ".venv/bin/ipw" ]]; then
    echo "[sweep] ERROR: .venv/bin/ipw not found." >&2
    echo "        Run from the intelligence-per-watt repo root with the .venv populated:" >&2
    echo "          uv venv && source .venv/bin/activate" >&2
    echo "          uv pip install -e 'intelligence-per-watt[mlx]'" >&2
    exit 1
fi

# Cache sudo upfront so the Rust energy monitor can launch without prompting.
echo "[sweep] caching sudo for the energy monitor (you will be prompted once)..."
sudo -v

# Background sudo refresher: keeps the auth token warm for the whole sweep.
( while true; do sudo -n true 2>/dev/null || exit; sleep 50; done ) &
SUDO_KEEPER_PID=$!

cleanup() {
    rv=$?
    [[ -n "${SUDO_KEEPER_PID:-}" ]] && kill "$SUDO_KEEPER_PID" 2>/dev/null || true
    exit $rv
}
trap cleanup EXIT INT TERM

# ---- sweep loop -------------------------------------------------------------
mkdir -p "$OUT_ROOT"
echo "[sweep] root=$OUT_ROOT  model=$MODEL  draft=$DRAFT  K=$NUM_DRAFT_TOKENS"
echo "[sweep] Ls={$PROMPT_LENGTHS}  M=$MAX_TOKENS  N=$NUM_SAMPLES  warmup=$WARMUP_QUERIES"

for L in $PROMPT_LENGTHS; do
    for ARM in "baseline" "spec"; do
        if [[ "$ARM" == "baseline" ]]; then
            tag="baseline"
            arm_args=()
        else
            tag="spec_K${NUM_DRAFT_TOKENS}"
            arm_args=(
                --client-param "draft_model=$DRAFT"
                --client-param "num_draft_tokens=$NUM_DRAFT_TOKENS"
            )
        fi
        run_dir="$OUT_ROOT/L${L}_M${MAX_TOKENS}_${tag}"

        if compgen -G "$run_dir/profile_*/summary.json" > /dev/null; then
            echo "[sweep] SKIP $run_dir (already complete)"
            continue
        fi

        echo "[sweep] L=$L  arm=$tag  →  $run_dir"
        HF_HUB_OFFLINE=1 .venv/bin/ipw profile \
            --client mlx \
            --model "$MODEL" \
            --dataset fixed_length \
            --dataset-param "prompt_length=$L" \
            --dataset-param "num_samples=$NUM_SAMPLES" \
            --dataset-param "chars_per_token=$CHARS_PER_TOKEN" \
            --client-param "max_tokens=$MAX_TOKENS" \
            ${arm_args[@]+"${arm_args[@]}"} \
            --warmup-queries "$WARMUP_QUERIES" \
            --output-dir "$run_dir"
    done
done

echo "[sweep] done. Results under $OUT_ROOT/"
echo "[sweep] To plot, adapt scripts/plot_qkv_fuse_sweep.py — the run-dir layout"
echo "[sweep] is identical (per-config HF-dataset rows under profile_*/)."

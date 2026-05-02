#!/usr/bin/env python3
"""Profile multiple Ollama models across multiple batch sizes, then analyze and plot each.

Edit the configuration below, then:

    caffeinate -dims python scripts/run_batch_profiles.py
    caffeinate -dims python scripts/run_batch_profiles.py --resume   # skip completed combos
"""

import json
import os
import shutil
import subprocess
import sys
import traceback
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = [
    # "qwen3:8b",
    # "qwen3.5:9b",
    "granite3.1-dense:8b"
]

BATCH_SIZES = [1, 8, 16]

MAX_QUERIES = 256

CLIENT = "ollama"
CLIENT_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"
STATE_DIR = SCRIPT_DIR / "logs"
STATE_FILE = STATE_DIR / "batch_profile_state.json"


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or "model"


def _dir_name(model: str, max_queries: int, batch_size: int) -> str:
    return f"profile_GPU_{_slugify(model)}_{max_queries}x_batch{batch_size}"


def _combo_key(model: str, batch_size: int) -> str:
    return f"{model}|batch{batch_size}"


# ---------------------------------------------------------------------------
# State persistence (resume support)
# ---------------------------------------------------------------------------
def _load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text())
        return {str(k): str(v).upper() for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        print(f"[WARN] Could not load state from {STATE_FILE}; starting fresh")
        return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def pull_model(model: str) -> bool:
    print(f"  Ensuring model is available: ollama pull {model}")
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        print(f"  [FAIL] ollama pull {model} exited {result.returncode}")
        return False
    return True


def run_profile(model: str, batch_size: int, max_queries: int) -> Path | None:
    """Run ipw profile and return the renamed output directory, or None on failure."""
    target_name = _dir_name(model, max_queries, batch_size)
    target_path = RUNS_DIR / target_name

    if target_path.exists():
        print(f"  Output dir already exists, removing: {target_path.name}")
        shutil.rmtree(target_path)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    before = set(p.name for p in RUNS_DIR.iterdir() if p.is_dir())

    cmd = [
        "ipw", "profile",
        "--client", CLIENT,
        "--model", model,
        "--client-base-url", CLIENT_BASE_URL,
        "--max-queries", str(max_queries),
        "--batch-size", str(batch_size),
        "--output-dir", str(RUNS_DIR),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [FAIL] ipw profile exited {result.returncode}")
        return None

    after = set(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
    new_dirs = after - before
    if len(new_dirs) != 1:
        print(f"  [WARN] Expected 1 new directory in runs/, found {len(new_dirs)}: {new_dirs}")
        if not new_dirs:
            return None
        new_dir = sorted(new_dirs)[0]
    else:
        new_dir = new_dirs.pop()

    src = RUNS_DIR / new_dir
    if src != target_path:
        src.rename(target_path)
        print(f"  Renamed {new_dir} -> {target_name}")
    return target_path


def run_analyze(results_dir: Path) -> bool:
    cmd = ["ipw", "analyze", str(results_dir), "--analysis", "regression"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] ipw analyze exited {result.returncode}")
        return False
    return True


def run_plot(results_dir: Path) -> bool:
    cmd = ["ipw", "plot", str(results_dir)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] ipw plot exited {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=False,
        help="Skip combos already marked SUCCESS in state file.",
    )
    args = parser.parse_args()

    state = _load_state() if args.resume else {}
    results: dict[str, str] = {}

    combos = [(m, b) for m in MODELS for b in BATCH_SIZES]
    print(f"Batch profiling: {len(combos)} combos ({len(MODELS)} models x {len(BATCH_SIZES)} batch sizes)")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Max queries: {MAX_QUERIES}\n")

    pulled: set[str] = set()

    for i, (model, batch_size) in enumerate(combos, 1):
        key = _combo_key(model, batch_size)
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{i}/{len(combos)}] {model}  batch_size={batch_size}")
        print(sep)

        if args.resume and state.get(key) == "SUCCESS":
            print("  Skipping (already succeeded)")
            results[key] = "SUCCESS"
            continue

        start = datetime.now()

        if model not in pulled:
            if not pull_model(model):
                state[key] = "FAILED"
                results[key] = "FAILED"
                _save_state(state)
                continue
            pulled.add(model)

        try:
            out_dir = run_profile(model, batch_size, MAX_QUERIES)
            if out_dir is None:
                raise RuntimeError("profiling produced no output")
            run_analyze(out_dir)
            run_plot(out_dir)
            status = "SUCCESS"
        except Exception:
            traceback.print_exc()
            status = "FAILED"

        elapsed = datetime.now() - start
        state[key] = status
        results[key] = status
        _save_state(state)
        print(f"  [{status}] elapsed {elapsed}")

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)

    ok = sum(1 for v in results.values() if v == "SUCCESS")
    fail = len(results) - ok

    for (model, batch_size) in combos:
        key = _combo_key(model, batch_size)
        st = results.get(key, "SKIPPED")
        tag = "[OK]  " if st == "SUCCESS" else "[FAIL]"
        dn = _dir_name(model, MAX_QUERIES, batch_size)
        print(f"  {tag} {dn}  ({st})")

    print(f"\n{ok}/{len(combos)} succeeded, {fail} failed")
    print(f"State file: {STATE_FILE}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

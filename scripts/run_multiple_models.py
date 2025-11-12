#!/usr/bin/env python3
"""Run Intelligence Per Watt profiling for multiple models sequentially.

This script orchestrates multiple profiling runs, one for each model in the
configured list. Each model runs independently - if one fails, it is logged
and the script continues with the next model.
"""

import json
import subprocess
import sys
import traceback
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent / "logs"
STATE_FILE = STATE_DIR / "run_state.json"

# Configure your models here
MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-small",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# Common arguments for all benchmark runs
COMMON_ARGS = [
    "--client",
    "vllm",
    "--dataset",
    "ipw",
]


def _state_file_path() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_FILE


def _load_run_state() -> dict[str, str]:
    path = _state_file_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("run state is not a dict")
        normalized: dict[str, str] = {}
        for model, status in data.items():
            normalized[str(model)] = str(status).upper()
        return normalized
    except Exception:
        print(f"[ERROR] Failed to load run state from {path}; starting fresh")
        traceback.print_exc()
        return {}


def _save_run_state(state: dict[str, str]) -> None:
    path = _state_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except Exception:
        print(f"[ERROR] Failed to persist run state to {path}")
        traceback.print_exc()


def _parse_args():
    parser = ArgumentParser(
        description="Run Intelligence Per Watt profiling for multiple models sequentially."
    )
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume from previous run state and skip models already marked as SUCCESS.",
    )
    return parser.parse_args()


def run_benchmark(model: str) -> bool:
    """Run benchmark for a single model.

    Args:
        model: Model name/path to benchmark

    Returns:
        Success flag
    """
    cmd = [
        "ipw",
        "profile",
        "--model",
        model,
        *COMMON_ARGS,
    ]

    start_time = datetime.now()

    separator = "=" * 60
    print(separator)
    print(f"Starting benchmark for: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(separator)

    try:
        result = subprocess.run(cmd, check=False)

        end_time = datetime.now()
        elapsed = end_time - start_time

        if result.returncode != 0:
            print(
                f"[FAILED] {model} (exit code: {result.returncode}, elapsed: {elapsed})"
            )
            return False

        print(f"[COMPLETED] {model} (elapsed: {elapsed})")
        return True

    except Exception:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[ERROR] Failed to run {model} (elapsed: {elapsed})")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = _parse_args()
    print("Starting Intelligence Per Watt multi-model profiling run")
    print(f"Configured models: {', '.join(MODELS)}")

    state: dict[str, str] = _load_run_state() if args.resume else {}
    if args.resume:
        print(f"Resume enabled; loaded run state for {len(state)} models")
    else:
        print("Resume disabled; starting with a fresh run state")
        state = {}

    results: dict[str, str] = {}
    print(f"Running benchmarks for {len(MODELS)} models sequentially...")

    for model in MODELS:
        existing = state.get(model)
        if args.resume and existing == "SUCCESS":
            print(f"Skipping {model} (previous run success).")
            results[model] = existing
            continue

        success = run_benchmark(model)
        status = "SUCCESS" if success else "FAILED"
        state[model] = status
        results[model] = status
        _save_run_state(state)

    # Summary
    separator = "=" * 60
    print(separator)
    print("SUMMARY")
    print(separator)

    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    failed_count = len(results) - success_count

    for model, status in results.items():
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        print(f"{prefix} {model}: {status}")

    print(f"Total: {success_count}/{len(MODELS)} succeeded, {failed_count} failed")
    print(f"Run state file: {_state_file_path()}")

    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)

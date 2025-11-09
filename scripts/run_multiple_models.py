#!/usr/bin/env python3
"""Run trafficbench profiling for multiple models sequentially.

This script orchestrates multiple profiling runs, one for each model in the
configured list. Each model runs independently - if one fails, it is logged
and the script continues with the next model.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configure your models here
MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-1.5B",
]

# Common arguments for all benchmark runs
COMMON_ARGS = [
    "--client", "vllm",
    "--dataset", "trafficbench",
    "--max-queries", "100",
]


def run_benchmark(model: str) -> bool:
    """Run benchmark for a single model.
    
    Args:
        model: Model name/path to benchmark
        
    Returns:
        True on success, False on failure
    """
    cmd = [
        "trafficbench", "profile",
        "--model", model,
        *COMMON_ARGS,
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting benchmark for: {model}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        
        if result.returncode != 0:
            print(f"\n[FAILED] {model} (exit code: {result.returncode})", file=sys.stderr)
            return False
        
        print(f"\n[COMPLETED] {model}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to run {model}: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    results = {}
    
    print(f"Running benchmarks for {len(MODELS)} models sequentially...")
    
    for model in MODELS:
        success = run_benchmark(model)
        results[model] = "SUCCESS" if success else "FAILED"
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    failed_count = len(results) - success_count
    
    for model, status in results.items():
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        print(f"{prefix} {model}: {status}")
    
    print(f"\nTotal: {success_count}/{len(MODELS)} succeeded, {failed_count} failed")
    
    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)


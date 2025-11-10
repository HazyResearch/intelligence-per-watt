#!/usr/bin/env python3
"""Run trafficbench profiling for multiple models sequentially.

This script orchestrates multiple profiling runs, one for each model in the
configured list. Each model runs independently - if one fails, it is logged
and the script continues with the next model.
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover - dependency check
    HfApi = None  # type: ignore[assignment]
    HfHubHTTPError = Exception  # type: ignore[assignment]


logger = logging.getLogger("trafficbench.run_multiple_models")


def setup_logging() -> Path:
    """Configure logging to both console and file."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"run_multiple_models_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Avoid duplicate handlers if the script is reloaded.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Output file: %s", log_file)
    return log_file

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

    start_time = datetime.now()

    separator = "=" * 60
    logger.info(separator)
    logger.info("Starting benchmark for: %s", model)
    logger.info("Command: %s", " ".join(cmd))
    logger.info(separator)

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)

        elapsed = datetime.now() - start_time

        if result.returncode != 0:
            logger.error("[FAILED] %s (exit code: %s, elapsed: %s)", model, result.returncode, elapsed)
            return False

        logger.info("[COMPLETED] %s (elapsed: %s)", model, elapsed)
        return True

    except Exception:
        elapsed = datetime.now() - start_time
        logger.exception("[ERROR] Failed to run %s (elapsed: %s)", model, elapsed)
        return False


def ensure_models_available(models: list[str]) -> None:
    """Verify all configured models exist on Hugging Face Hub."""
    if HfApi is None:
        logger.error("huggingface_hub is required to validate models. Install it with `pip install huggingface_hub`.")
        sys.exit(1)

    api = HfApi()
    missing = []
    logger.info("Validating availability for %d models on Hugging Face Hub", len(models))
    for model in models:
        try:
            api.model_info(model)
        except HfHubHTTPError as err:
            if getattr(err, "response", None) and getattr(err.response, "status_code", None) == 404:
                missing.append((model, "not found"))
            else:
                missing.append((model, f"error: {err}"))
        except Exception as err:  # pragma: no cover - unexpected errors
            missing.append((model, f"error {err}"))

    if missing:
        logger.error("One or more models are not available on Hugging Face Hub:")
        for model, reason in missing:
            logger.error("  - %s: %s", model, reason)
        sys.exit(1)

    logger.info("All %d models are available on Hugging Face Hub", len(models))


if __name__ == "__main__":
    setup_logging()
    logger.info("Starting trafficbench multi-model profiling run")
    logger.info("Configured models: %s", ", ".join(MODELS))

    ensure_models_available(MODELS)

    results = {}
    logger.info("Running benchmarks for %d models sequentially...", len(MODELS))

    for model in MODELS:
        success = run_benchmark(model)
        results[model] = "SUCCESS" if success else "FAILED"

    # Summary
    separator = "=" * 60
    logger.info(separator)
    logger.info("SUMMARY")
    logger.info(separator)

    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    failed_count = len(results) - success_count

    for model, status in results.items():
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        logger.info("%s %s: %s", prefix, model, status)

    logger.info("Total: %d/%d succeeded, %d failed", success_count, len(MODELS), failed_count)
    logger.info("Log file located at: %s", Path(logger.handlers[0].baseFilename) if logger.handlers else "unknown")

    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)


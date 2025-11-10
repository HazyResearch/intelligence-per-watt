#!/usr/bin/env python3
"""Run trafficbench profiling for multiple models sequentially.

This script orchestrates multiple profiling runs, one for each model in the
configured list. Each model runs independently - if one fails, it is logged
and the script continues with the next model.
"""

import json
import logging
import subprocess
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover - dependency check
    HfApi = None  # type: ignore[assignment]
    HfHubHTTPError = Exception  # type: ignore[assignment]


logger = logging.getLogger("trafficbench.run_multiple_models")
MAIN_LOG_FILE: Path | None = None
RUN_LOG_DIR: Path | None = None
STATE_FILE: Path | None = None


def _slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return slug.strip("_") or "model"


def _state_file_path() -> Path:
    path = STATE_FILE
    if path is None:
        base = Path(__file__).resolve().parent / "logs"
        base.mkdir(parents=True, exist_ok=True)
        path = base / "run_state.json"
    return path


def _load_run_state() -> dict[str, dict[str, str]]:
    path = _state_file_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("run state is not a dict")
        normalized: dict[str, dict[str, str]] = {}
        for model, info in data.items():
            if not isinstance(info, dict):
                continue
            status = str(info.get("status", "")).upper()
            log_path = str(info.get("log", "")).strip()
            normalized[model] = {"status": status, "log": log_path}
        return normalized
    except Exception:
        logger.exception("Failed to load run state from %s; starting fresh", path)
        return {}


def _save_run_state(state: dict[str, dict[str, str]]) -> None:
    path = _state_file_path()
    serializable = {
        model: {"status": info.get("status", ""), "log": str(info.get("log", ""))}
        for model, info in state.items()
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
    except Exception:
        logger.exception("Failed to persist run state to %s", path)


def _parse_args():
    parser = ArgumentParser(description="Run trafficbench profiling for multiple models sequentially.")
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume from previous run state and skip models already marked as SUCCESS.",
    )
    return parser.parse_args()


def setup_logging() -> Path:
    """Configure logging to both console and file."""
    global MAIN_LOG_FILE, RUN_LOG_DIR, STATE_FILE

    log_dir = Path(__file__).resolve().parent / "logs"
    run_logs_dir = log_dir / "runs"

    log_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir.mkdir(parents=True, exist_ok=True)

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
    MAIN_LOG_FILE = log_file
    RUN_LOG_DIR = run_logs_dir
    STATE_FILE = log_dir / "run_state.json"

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
]


def run_benchmark(model: str) -> tuple[bool, Path]:
    """Run benchmark for a single model.

    Args:
        model: Model name/path to benchmark

    Returns:
        Tuple of (success flag, per-run log path)
    """
    cmd = [
        "trafficbench", "profile",
        "--model", model,
        *COMMON_ARGS,
    ]

    start_time = datetime.now()

    run_log_dir = RUN_LOG_DIR or (Path(__file__).resolve().parent / "logs" / "runs")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_log_dir / f"{_slugify(model)}_{start_time.strftime('%Y%m%d-%H%M%S')}.log"

    separator = "=" * 60
    logger.info(separator)
    logger.info("Starting benchmark for: %s", model)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("Per-run log: %s", run_log_path)
    logger.info(separator)

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        end_time = datetime.now()
        elapsed = end_time - start_time

        with run_log_path.open("w", encoding="utf-8") as run_log:
            run_log.write(f"{separator}\n")
            run_log.write(f"Model: {model}\n")
            run_log.write(f"Command: {' '.join(cmd)}\n")
            run_log.write(f"Started: {start_time.isoformat()}\n")
            run_log.write(f"Completed: {end_time.isoformat()}\n")
            run_log.write(f"Exit code: {result.returncode}\n")
            if result.stdout:
                run_log.write("\n[STDOUT]\n")
                run_log.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    run_log.write("\n")
            if result.stderr:
                run_log.write("\n[STDERR]\n")
                run_log.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    run_log.write("\n")

        if result.stdout:
            logger.info("[trafficbench stdout]\n%s", result.stdout.rstrip())
        if result.stderr:
            logger.error("[trafficbench stderr]\n%s", result.stderr.rstrip())

        if result.returncode != 0:
            logger.error("[FAILED] %s (exit code: %s, elapsed: %s)", model, result.returncode, elapsed)
            logger.error("See per-run log for details: %s", run_log_path)
            return False, run_log_path

        logger.info("[COMPLETED] %s (elapsed: %s)", model, elapsed)
        logger.info("Per-run log saved to: %s", run_log_path)
        return True, run_log_path

    except Exception:
        end_time = datetime.now()
        elapsed = end_time - start_time
        logger.exception("[ERROR] Failed to run %s (elapsed: %s)", model, elapsed)
        try:
            with run_log_path.open("a", encoding="utf-8") as run_log:
                run_log.write(f"{separator}\n")
                run_log.write(f"Model: {model}\n")
                run_log.write(f"Command: {' '.join(cmd)}\n")
                run_log.write(f"Started: {start_time.isoformat()}\n")
                run_log.write(f"Errored: {end_time.isoformat()}\n")
                run_log.write("Exception encountered. See main log for traceback.\n")
        except Exception:
            logger.debug("Failed to write exception details to per-run log: %s", run_log_path, exc_info=True)
        return False, run_log_path


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
    args = _parse_args()
    setup_logging()
    logger.info("Starting trafficbench multi-model profiling run")
    logger.info("Configured models: %s", ", ".join(MODELS))

    ensure_models_available(MODELS)

    state = _load_run_state() if args.resume else {}
    if args.resume:
        logger.info("Resume enabled; loaded run state for %d models", len(state))
    else:
        logger.info("Resume disabled; starting with a fresh run state")
        state = {}

    results: dict[str, dict[str, str]] = {}
    logger.info("Running benchmarks for %d models sequentially...", len(MODELS))

    for model in MODELS:
        existing = state.get(model)
        if args.resume and existing and existing.get("status") == "SUCCESS":
            logger.info(
                "Skipping %s (previous run success). Log: %s",
                model,
                existing.get("log", "unknown"),
            )
            results[model] = existing
            continue

        success, run_log = run_benchmark(model)
        status = "SUCCESS" if success else "FAILED"
        record = {"status": status, "log": str(run_log)}
        state[model] = record
        results[model] = record
        _save_run_state(state)

    # Summary
    separator = "=" * 60
    logger.info(separator)
    logger.info("SUMMARY")
    logger.info(separator)

    success_count = sum(1 for info in results.values() if info["status"] == "SUCCESS")
    failed_count = len(results) - success_count

    for model, info in results.items():
        status = info["status"]
        run_log = info.get("log") or "unknown"
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        logger.info("%s %s: %s (log: %s)", prefix, model, status, run_log)

    logger.info("Total: %d/%d succeeded, %d failed", success_count, len(MODELS), failed_count)
    per_run_dir = RUN_LOG_DIR or (Path(__file__).resolve().parent / "logs" / "runs")
    logger.info("Per-run logs directory: %s", per_run_dir)
    logger.info("Run state file: %s", _state_file_path())
    if MAIN_LOG_FILE:
        logger.info("Main log file located at: %s", MAIN_LOG_FILE)
    else:
        logger.info("Main log file path unavailable from logger configuration.")

    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)


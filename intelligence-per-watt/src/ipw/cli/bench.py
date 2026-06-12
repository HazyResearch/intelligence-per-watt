"""Run agent benchmarks with energy telemetry for Intelligence Per Watt measurement.

This command provides the full profiling pipeline:
1. Auto-start/stop inference servers (optional, excluded from profiling)
2. Warmup queries (excluded from profiling)
3. Benchmark execution with energy telemetry
4. Per-action energy breakdown (optional)
5. Results export (JSONL, HF dataset, summary)

Usage:
    # Simple benchmark
    ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --limit 5

    # With auto-server management (startup/shutdown excluded from profiling)
    ipw bench --agent react --preset glm-4.7-flash --dataset gaia --auto-server --limit 5

    # With per-action energy breakdown
    ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --per-action

    # Without energy telemetry
    ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --no-telemetry
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import shlex
import statistics
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import click

from ipw.cli._console import error, info, success, warning
from ipw.cli._display import (
    compute_trace_metrics,
    print_banner,
    print_config_summary,
    print_efficiency_panel,
    print_metrics_table,
    print_output_path,
)
from ipw.cli.server_manager import (
    InferenceServerManager,
    build_server_configs,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("./outputs")


# ---------------------------------------------------------------------------
# Model creation helpers
# ---------------------------------------------------------------------------

def _create_vllm_model(model: str, base_url: str | None = None):
    from agno.models.openai import OpenAIChat
    return OpenAIChat(
        id=model,
        base_url=base_url or "http://localhost:8000/v1",
        collect_metrics_on_completion=True,
    )


def _create_openai_model(model: str, base_url: str | None = None):
    from agno.models.openai import OpenAIChat
    if base_url:
        return OpenAIChat(id=model, base_url=base_url, collect_metrics_on_completion=True)
    return OpenAIChat(id=model, collect_metrics_on_completion=True)


def _create_ollama_model(model: str, base_url: str | None = None):
    from agno.models.ollama import Ollama
    return Ollama(id=model, host=base_url or "http://localhost:11434")


MODEL_FACTORIES: Dict[str, Callable] = {
    "vllm": _create_vllm_model,
    "openai": _create_openai_model,
    "ollama": _create_ollama_model,
}


def create_model(client_id: str, model: str, base_url: str | None = None, agent_id: str | None = None):
    """Create a model instance based on client type.

    When *agent_id* is ``"openhands"``, returns an OpenHands ``LLM`` object
    instead of an Agno model because the OpenHands SDK expects its own type.
    """
    if agent_id == "openhands":
        return _create_openhands_llm(model, base_url, client_id)
    if agent_id in ("dspy-rlm", "forgecode"):
        return {
            "model": model,
            "base_url": base_url,
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "client": client_id,
            "cloud": client_id == "openai" and not base_url,
        }
    if client_id not in MODEL_FACTORIES:
        raise ValueError(f"Unknown client: {client_id}. Supported: {list(MODEL_FACTORIES.keys())}")
    return MODEL_FACTORIES[client_id](model, base_url)


def _create_openhands_llm(model: str, base_url: str | None, client_id: str):
    """Create an OpenHands SDK ``LLM`` instance."""
    from openhands.sdk import LLM

    kwargs: Dict[str, Any] = {"model": model, "api_key": "EMPTY"}
    if base_url:
        # litellm needs /v1 in the base_url (it appends /chat/completions)
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/v1"):
            clean_url = clean_url + "/v1"
        kwargs["base_url"] = clean_url
    # Prefix model name for litellm provider routing when using local vLLM
    if client_id == "vllm" and not model.startswith("openai/"):
        kwargs["model"] = f"openai/{model}"
    return LLM(**kwargs)


def get_model_alias(model_id: str) -> str:
    """Get a clean alias from a model ID."""
    if "/" in model_id:
        model_id = model_id.split("/")[-1]
    return model_id.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

def _compute_energy_delta(values: List[Optional[float]]) -> Optional[float]:
    """Compute energy delta from cumulative values."""
    filtered = [v for v in values if v is not None and math.isfinite(v) and v >= 0]
    if len(filtered) < 2:
        return None
    delta = filtered[-1] - filtered[0]
    return delta if delta >= 0 else None


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Compute mean, returning None if no valid values."""
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(filtered) if filtered else None


def _safe_max(values: List[Optional[float]]) -> Optional[float]:
    """Compute max, returning None if no valid values."""
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    return max(filtered) if filtered else None


def _extract_hardware_info(samples) -> Dict[str, Any]:
    """Extract hardware configuration from telemetry samples."""
    hardware_info: Dict[str, Any] = {
        "gpu_count": None,
        "cpu_count": None,
        "hardware_stack": None,
    }

    if not samples:
        return hardware_info

    first_reading = samples[0].reading

    if first_reading.platform:
        hardware_info["hardware_stack"] = first_reading.platform

    if first_reading.system_info and first_reading.system_info.cpu_count:
        hardware_info["cpu_count"] = first_reading.system_info.cpu_count

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        gpu_ids = [g.strip() for g in cuda_visible.split(",") if g.strip()]
        hardware_info["gpu_count"] = len(gpu_ids)
    elif first_reading.gpu_info and first_reading.gpu_info.name:
        hardware_info["gpu_count"] = 1

    return hardware_info


def _compute_energy_metrics(samples, start_time: float, end_time: float) -> Dict[str, Any]:
    """Compute energy metrics from telemetry samples."""
    readings = [s.reading for s in samples]

    gpu_energy = _compute_energy_delta([r.energy_joules for r in readings])
    cpu_energy = _compute_energy_delta([r.cpu_energy_joules for r in readings])

    gpu_power_samples = [r.power_watts for r in readings if r.power_watts is not None]
    cpu_power_samples = [r.cpu_power_watts for r in readings if r.cpu_power_watts is not None]

    mbu_samples = [
        r.gpu_memory_bandwidth_utilization_pct for r in readings
        if getattr(r, 'gpu_memory_bandwidth_utilization_pct', None) is not None
        and r.gpu_memory_bandwidth_utilization_pct >= 0
    ]

    duration = max(end_time - start_time, 0.0)
    total_energy = (gpu_energy or 0) + (cpu_energy or 0)

    return {
        "duration_seconds": duration,
        "gpu_energy_joules": gpu_energy,
        "cpu_energy_joules": cpu_energy,
        "total_energy_joules": total_energy if total_energy > 0 else None,
        "avg_gpu_power_watts": _safe_mean(gpu_power_samples),
        "max_gpu_power_watts": _safe_max(gpu_power_samples),
        "avg_cpu_power_watts": _safe_mean(cpu_power_samples),
        "avg_mbu_pct": statistics.mean(mbu_samples) if mbu_samples else None,
        "max_mbu_pct": max(mbu_samples) if mbu_samples else None,
        "telemetry_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Server warmup helpers
# ---------------------------------------------------------------------------

def _wait_for_server_ready(client_id: str, base_url: str | None = None, timeout: float = 60.0) -> bool:
    """Wait for inference server to be ready."""
    import urllib.error
    import urllib.request

    if client_id == "ollama":
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/version"
    elif client_id == "vllm":
        base = (base_url or "http://localhost:8000").rstrip("/")
        if base.endswith("/v1"):
            url = base + "/models"
        else:
            url = base + "/v1/models"
    elif client_id == "openai":
        return True
    else:
        return True

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, Exception):
            time.sleep(0.5)

    return False


def _run_warmup_query(model, warmup_prompt: str = "Hello") -> None:
    """Run a warmup query to initialize model and exclude cold-start costs."""
    try:
        response = model.response(warmup_prompt)
        _ = response.content if hasattr(response, 'content') else str(response)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _build_run_metadata() -> Dict[str, Any]:
    """Capture CLI invocation details and version information."""
    try:
        ipw_version = importlib_metadata.version("ipw")
    except importlib_metadata.PackageNotFoundError:
        ipw_version = "unknown"

    import platform

    return {
        "cli_invocation": {
            "argv": list(sys.argv),
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        },
        "versions": {
            "ipw": ipw_version,
            "python": platform.python_version(),
        },
        "timestamp": datetime.now().isoformat(),
    }


def _get_output_path(dataset_id: str, model: str, output_dir: str | None = None) -> Path:
    """Generate organized output path for benchmark results."""
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "bench"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    return base / f"{dataset_id}_{safe_model}_{timestamp}"


def execute_benchmark(
    client_id: str,
    model_name: str,
    agent_id: str,
    dataset_id: str,
    max_samples: int | None = None,
    client_base_url: str | None = None,
    output_dir: str | None = None,
    enable_telemetry: bool = True,
    telemetry_granularity: str = "benchmark",
    skip_warmup: bool = False,
    auto_server: bool = False,
    submodels: Sequence[str] | None = None,
    base_port: int = 8000,
    seed: int | None = None,
    api_key: str = "EMPTY",
    preset_vllm_args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute a benchmark run with energy telemetry.

    Args:
        client_id: Model provider identifier (vllm, openai, ollama)
        model_name: Model identifier (HuggingFace ID or model name)
        agent_id: Agent type identifier (react, openhands, terminus)
        dataset_id: Dataset identifier
        max_samples: Maximum number of queries to run
        client_base_url: Override default API endpoint
        output_dir: Directory to save results
        enable_telemetry: Whether to collect energy telemetry
        telemetry_granularity: "benchmark" or "per-action"
        skip_warmup: Skip server warmup phase
        auto_server: Auto-start/stop inference servers
        submodels: Submodel specs (alias:backend:model_id)
        base_port: Base port for vLLM servers
        seed: Random seed for reproducibility
        api_key: API key for inference server

    Returns:
        Dictionary with benchmark metrics and energy metrics.
    """
    # Set API key for model providers that check environment variables
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    import ipw.datasets
    ipw.datasets.ensure_registered()

    # Register the OpenAI client for LLM-judge scoring.
    # We avoid ipw.clients.ensure_registered() because it eagerly imports
    # all client backends (vllm, ollama), which may fail if their native
    # libraries are unavailable.  The openai client has no native deps.
    try:
        import ipw.clients.openai  # noqa: F401
    except ImportError:
        pass

    from ipw.agents import dspy_rlm as _dspy_rlm  # noqa: F401
    from ipw.agents import forgecode as _forgecode  # noqa: F401
    from ipw.agents import react as _react  # noqa: F401
    try:
        from ipw.agents import openhands as _openhands  # noqa: F401
    except ImportError:
        pass
    try:
        from ipw.agents import terminus as _terminus  # noqa: F401
    except ImportError:
        pass
    try:
        from ipw.agents import terminus_tb as _terminus_tb  # noqa: F401
    except ImportError:
        pass
    from ipw.core.registry import AgentRegistry, DatasetRegistry
    from ipw.execution.agentic_runner import AgenticRunner
    from ipw.execution.exporters import export_jsonl, export_summary_json
    from ipw.telemetry.events import EventRecorder

    server_manager: Optional[InferenceServerManager] = None
    managed_urls: Dict[str, str] = {}

    if auto_server:
        all_submodels = list(submodels or [])

        model_alias = get_model_alias(model_name)
        configs = build_server_configs(
            main_model=model_name,
            main_alias=model_alias,
            submodel_specs=all_submodels,
            base_port=base_port,
            main_backend=client_id,
            extra_args=preset_vllm_args,
        )

        server_manager = InferenceServerManager(configs)

        info("Starting inference servers (excluded from profiling)...")
        managed_urls = server_manager.start_all()

        if model_alias in managed_urls:
            client_base_url = managed_urls[model_alias]
            info(f"Using managed server at {client_base_url}")

        if not skip_warmup:
            info("Running warmup queries (excluded from profiling)...")
            server_manager.warmup_all()

    try:
        if not auto_server and not skip_warmup:
            info("Waiting for inference server...")
            if not _wait_for_server_ready(client_id, client_base_url, timeout=120.0):
                warning("Server not responding, proceeding anyway...")

        # Resolve agent and dataset via registries
        try:
            agent_cls = AgentRegistry.get(agent_id)
        except KeyError:
            available = [k for k, _ in AgentRegistry.items()]
            raise ValueError(f"Unknown agent '{agent_id}'. Available: {', '.join(available)}")

        try:
            dataset_cls = DatasetRegistry.get(dataset_id)
        except KeyError:
            available = [k for k, _ in DatasetRegistry.items()]
            raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {', '.join(available)}")

        # Create model using appropriate factory
        base_url = client_base_url or "http://localhost:8000/v1"
        # Ensure URL has /v1 suffix for vLLM
        if client_id == "vllm" and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        resolved_model = create_model(client_id, model_name, base_url, agent_id=agent_id)

        # Warmup query (excluded from profiling)
        # Skip agno warmup for openhands — it uses its own LLM type
        if not skip_warmup and not auto_server and agent_id != "openhands":
            info("Running warmup query (excluded from measurements)...")
            _run_warmup_query(resolved_model)

        # Create event recorder
        event_recorder = EventRecorder()

        # Create dataset and agent
        dataset_instance = dataset_cls()
        agent_instance = agent_cls(
            model=resolved_model,
            event_recorder=event_recorder,
        )

        # Prepare output directory
        actual_output_dir = _get_output_path(dataset_id, model_name, output_dir)
        actual_output_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "agent": agent_id,
            "model": model_name,
            "dataset": dataset_id,
            "client": client_id,
            "client_base_url": client_base_url,
            "max_samples": max_samples,
            "telemetry_granularity": telemetry_granularity,
            "auto_server": auto_server,
            "seed": seed,
        }

        # Create runner
        runner = AgenticRunner(
            agent=agent_instance,
            dataset=dataset_instance,
            telemetry_session=None,
            config=run_config,
            event_recorder=event_recorder,
            run_dir=actual_output_dir,
        )

        # === TELEMETRY STARTS HERE ===
        result = _execute_with_telemetry(
            runner=runner,
            max_queries=max_samples,
            enable_telemetry=enable_telemetry,
            telemetry_granularity=telemetry_granularity,
            event_recorder=event_recorder,
        )
        # === TELEMETRY ENDS HERE ===

        # Add run metadata
        result["run_metadata"] = {
            **_build_run_metadata(),
            "client_id": client_id,
            "model_name": model_name,
            "agent_id": agent_id,
            "dataset_id": dataset_id,
            "max_samples": max_samples,
            "telemetry_granularity": telemetry_granularity,
            "warmup_excluded": not skip_warmup,
            "auto_server": auto_server,
            "submodels": list(submodels) if submodels else [],
            "managed_server_urls": managed_urls,
            "seed": seed,
        }

        # Save results
        results_path = actual_output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        info(f"Results saved to: {actual_output_dir}")

        # Export traces if available
        traces = result.get("_traces")
        if traces:
            export_jsonl(traces, actual_output_dir / "traces.jsonl")

            # Pass benchmark-level energy metrics so the summary can
            # include aggregate telemetry even when per-query energy is
            # unavailable (i.e. telemetry_granularity == "benchmark").
            bench_energy = {}
            for key in (
                "gpu_energy_joules", "cpu_energy_joules",
                "avg_gpu_power_watts", "max_gpu_power_watts",
                "avg_cpu_power_watts",
                "avg_mbu_pct", "max_mbu_pct",
                "duration_seconds", "telemetry_samples",
            ):
                if key in result:
                    bench_energy[key] = result[key]

            export_summary_json(
                traces, run_config,
                actual_output_dir / "summary.json",
                bench_energy=bench_energy,
            )

        return result

    finally:
        if server_manager:
            info("Stopping inference servers (excluded from profiling)...")
            server_manager.stop_all()


def _execute_with_telemetry(
    runner,
    max_queries: int | None,
    enable_telemetry: bool,
    telemetry_granularity: str,
    event_recorder,
) -> Dict[str, Any]:
    """Execute benchmark with optional energy telemetry collection."""
    if not enable_telemetry:
        info("Running without energy telemetry (--no-telemetry)")
        start_time = time.time()
        traces = asyncio.run(runner.run(max_queries=max_queries))
        end_time = time.time()
        return {
            "duration_seconds": end_time - start_time,
            "queries": len(traces),
            "completed": sum(1 for t in traces if t.completed),
            "total_turns": sum(t.num_turns for t in traces),
            "_traces": traces,
        }

    try:
        from ipw.execution.telemetry_session import TelemetrySession
        from ipw.telemetry import EnergyMonitorCollector

        collector = EnergyMonitorCollector()
        with TelemetrySession(
            collector, buffer_seconds=3600.0, max_samples=100_000
        ) as telemetry:
            start_time = time.time()
            traces = asyncio.run(runner.run(max_queries=max_queries))
            end_time = time.time()

            samples = list(telemetry.window(start_time, end_time))

        result: Dict[str, Any] = {
            "queries": len(traces),
            "completed": sum(1 for t in traces if t.completed),
            "total_turns": sum(t.num_turns for t in traces),
            "_traces": traces,
        }

        if samples:
            energy_metrics = _compute_energy_metrics(samples, start_time, end_time)

            hardware_info = _extract_hardware_info(samples)
            energy_metrics.update(hardware_info)

            result.update(energy_metrics)

            # Per-action energy breakdown
            if telemetry_granularity == "per-action" and event_recorder:
                from ipw.telemetry.correlation import (
                    compute_analysis,
                    correlate_energy_to_events,
                )

                events = event_recorder.get_events()
                if events:
                    breakdowns = correlate_energy_to_events(samples, events)
                    analysis = compute_analysis(breakdowns)

                    turns = analysis.get("action_counts", {}).get("lm_inference", 0)
                    tool_call_count = analysis.get("action_counts", {}).get("tool_call", 0)

                    tools_used = []
                    for event in events:
                        if event.event_type == "tool_call_start":
                            tool_name = event.metadata.get("tool")
                            if tool_name and tool_name not in tools_used:
                                tools_used.append(tool_name)

                    result["turns"] = turns
                    result["tools_used"] = tools_used
                    result["tools_used_count"] = tool_call_count

                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    missing_token_metrics = False
                    for event in events:
                        if event.event_type == "lm_inference_end":
                            prompt_tokens = event.metadata.get("prompt_tokens")
                            completion_tokens = event.metadata.get("completion_tokens")
                            if prompt_tokens is None or completion_tokens is None:
                                missing_token_metrics = True
                                continue
                            total_prompt_tokens += prompt_tokens
                            total_completion_tokens += completion_tokens

                    result["total_prompt_tokens"] = (
                        None if missing_token_metrics else total_prompt_tokens
                    )
                    result["total_completion_tokens"] = (
                        None if missing_token_metrics else total_completion_tokens
                    )
                    result["total_tokens"] = (
                        None
                        if missing_token_metrics
                        else total_prompt_tokens + total_completion_tokens
                    )

                    result["action_breakdown"] = [
                        {
                            "action_type": b.action_type,
                            "step_number": b.step_number,
                            "gpu_energy_joules": b.gpu_energy_joules,
                            "cpu_energy_joules": b.cpu_energy_joules,
                            "total_energy_joules": b.total_energy_joules,
                            "duration_ms": b.duration_ms,
                            "max_power_watts": b.max_power_watts,
                            "avg_power_watts": b.avg_power_watts,
                            "memory_bandwidth_gbps": b.memory_bandwidth_gbps,
                            "metadata": b.metadata,
                        }
                        for b in breakdowns
                    ]
                    result["energy_analysis"] = analysis
        else:
            warning("No telemetry samples collected during benchmark")
            result["duration_seconds"] = end_time - start_time

        return result

    except Exception as e:
        warning(f"Telemetry unavailable: {e}. Running without energy measurement.")
        start_time = time.time()
        traces = asyncio.run(runner.run(max_queries=max_queries))
        end_time = time.time()
        return {
            "duration_seconds": end_time - start_time,
            "queries": len(traces),
            "completed": sum(1 for t in traces if t.completed),
            "total_turns": sum(t.num_turns for t in traces),
            "_traces": traces,
        }


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command(help="Run agent benchmarks with energy telemetry for IPW measurement.")
@click.option(
    "--agent",
    "agent_id",
    required=True,
    help="Agent type (react, dspy-rlm, forgecode, openhands, terminus)",
)
@click.option(
    "--model",
    required=False,
    help="Model name (HuggingFace model ID or preset name)",
)
@click.option(
    "--preset",
    "preset_name",
    default=None,
    help="Model preset name (e.g. glm-4.7-flash, qwen3-30b-a3b)",
)
@click.option(
    "--dataset",
    "dataset_id",
    required=True,
    help="Dataset to benchmark (gaia, hle, simpleqa, etc.)",
)
@click.option(
    "--limit",
    "max_samples",
    type=int,
    default=None,
    help="Maximum number of queries to evaluate",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--client",
    "client_id",
    default="vllm",
    show_default=True,
    help="Model provider (vllm, openai, ollama)",
)
@click.option(
    "--vllm-url",
    help="vLLM server URL (default: http://localhost:8000/v1)",
)
@click.option(
    "--api-key",
    default="EMPTY",
    show_default=True,
    help="API key for inference server",
)
@click.option(
    "--per-action",
    is_flag=True,
    default=False,
    help="Record per-action energy breakdown (tool calls, LM inference)",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    default=False,
    help="Disable energy telemetry collection",
)
@click.option(
    "--skip-warmup",
    is_flag=True,
    default=False,
    help="Skip server warmup phase (includes cold-start costs in measurements)",
)
@click.option(
    "--auto-server",
    is_flag=True,
    default=False,
    help="Auto-start/stop inference servers (excludes startup/shutdown from profiling)",
)
@click.option(
    "--submodel",
    "submodels",
    multiple=True,
    help="Submodel spec: alias:backend:model_id (e.g., math:vllm:Qwen/Qwen2.5-Math-72B)",
)
@click.option(
    "--base-port",
    type=int,
    default=8000,
    help="Base port for vLLM servers when using --auto-server (default: 8000)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible benchmark sampling",
)
def bench(
    agent_id: str,
    model: str | None,
    preset_name: str | None,
    dataset_id: str,
    max_samples: int | None,
    output_dir: str | None,
    client_id: str,
    vllm_url: str | None,
    api_key: str,
    per_action: bool,
    no_telemetry: bool,
    skip_warmup: bool,
    auto_server: bool,
    submodels: tuple[str, ...],
    base_port: int,
    seed: int | None,
) -> None:
    """Run agent benchmarks with energy telemetry for IPW measurement."""
    # Resolve preset
    if preset_name and model:
        error("Specify either --model or --preset, not both.")
        raise click.Abort()

    preset_vllm_args: Dict[str, Any] | None = None
    if preset_name:
        from ipw.cli.model_presets import resolve_preset
        try:
            preset_config = resolve_preset(preset_name)
        except KeyError as exc:
            error(str(exc))
            raise click.Abort()
        model = preset_config["model_id"]
        preset_vllm_args = preset_config.get("vllm_args")
        info(f"Preset: {preset_name} -> {model}")

    if not model:
        error("--model or --preset is required.")
        raise click.Abort()

    # Set random seed
    if seed is not None:
        random.seed(seed)
        info(f"  Random seed: {seed}")

    # Determine base URL
    base_url = vllm_url
    if not base_url and client_id == "vllm":
        base_url = "http://localhost:8000/v1"

    telemetry_granularity = "per-action" if per_action else "benchmark"

    print_banner()
    config_info: dict[str, object] = {
        "Dataset": dataset_id,
        "Agent": agent_id,
        "Model": model,
        "Client": client_id,
    }
    if not auto_server:
        config_info["Server URL"] = base_url
    if max_samples:
        config_info["Limit"] = max_samples
    config_info["Telemetry"] = "disabled" if no_telemetry else f"enabled ({telemetry_granularity})"
    if auto_server:
        config_info["Auto-server"] = f"enabled (base port: {base_port})"
    config_info["Warmup"] = "skipped" if skip_warmup else "enabled"
    print_config_summary(config=config_info)

    try:
        metrics = execute_benchmark(
            client_id=client_id,
            model_name=model,
            agent_id=agent_id,
            dataset_id=dataset_id,
            max_samples=max_samples,
            client_base_url=base_url,
            output_dir=output_dir,
            enable_telemetry=not no_telemetry,
            telemetry_granularity=telemetry_granularity,
            skip_warmup=skip_warmup,
            auto_server=auto_server,
            submodels=submodels,
            base_port=base_port,
            seed=seed,
            api_key=api_key,
            preset_vllm_args=preset_vllm_args,
        )

        success("Benchmark completed!")

        # Extract traces for rich display
        traces = metrics.get("_traces")
        if traces:
            metric_rows = compute_trace_metrics(traces)
            print_metrics_table(rows=metric_rows, title="Benchmark Metrics")
            # Pass benchmark-level energy for the efficiency panel
            bench_energy_display = {}
            for key in ("gpu_energy_joules", "avg_gpu_power_watts"):
                if key in metrics:
                    bench_energy_display[key] = metrics[key]
            print_efficiency_panel(
                traces=traces,
                bench_energy=bench_energy_display if bench_energy_display else None,
            )

        # Display output path
        out_path = metrics.get("_output_dir") or metrics.get("output_dir")
        if out_path:
            from pathlib import Path as _Path
            print_output_path(path=_Path(str(out_path)))

    except Exception as e:
        error(f"Benchmark failed: {e}")
        raise click.Abort()


__all__ = ["bench"]

"""Run agentic benchmarks against a dataset."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from ._console import error, info, success, warning
from ._display import (
    compute_trace_metrics,
    print_banner,
    print_config_summary,
    print_efficiency_panel,
    print_metrics_table,
    print_output_path,
)


def _create_model_for_agent(agent_id: str, model: str, base_url: str, api_key: str):
    """Create the framework-specific model object for the given agent type."""
    if agent_id == "react":
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=model, api_key=api_key, base_url=f"{base_url}/v1")
    elif agent_id == "openhands":
        from openhands.sdk import LLM

        # LiteLLM requires provider prefix: ollama/ for Ollama, openai/ for OpenAI-compatible
        is_ollama = "11434" in base_url or "ollama" in base_url.lower()
        if "/" in model and not model.startswith(("openai/", "ollama/")):
            litellm_model = model
        else:
            prefix = "ollama" if is_ollama else "openai"
            litellm_model = f"{prefix}/{model}"
        # Ollama native API doesn't use /v1; OpenAI-compatible servers do
        llm_base_url = base_url if is_ollama else f"{base_url}/v1"
        return LLM(model=litellm_model, api_key=api_key, base_url=llm_base_url)
    elif agent_id == "terminus":
        return model  # Terminus takes a string
    else:
        return model  # Fallback: pass string


@click.command(help="Run an agentic benchmark against a dataset.")
@click.option("--agent", "agent_id", required=True, help="Agent identifier")
@click.option("--model", default=None, help="Model name for the agent (or use --preset)")
@click.option("--preset", "preset_name", default=None, help="Model preset name (e.g. glm-4.7-flash)")
@click.option("--dataset", "dataset_id", required=True, help="Dataset identifier")
@click.option(
    "--client-base-url",
    default="http://localhost:8000",
    show_default=True,
    help="Inference server base URL",
)
@click.option(
    "--api-key",
    default="EMPTY",
    show_default=True,
    help="API key for the inference server",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Max queries to run (default: all)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./runs/",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--export-format",
    default="jsonl,hf",
    show_default=True,
    help="Comma-separated export formats (jsonl, hf)",
)
@click.option(
    "--estimate-flops",
    is_flag=True,
    default=False,
    help="Enable FLOPs estimation",
)
@click.option(
    "--agent-kwargs",
    default=None,
    help="JSON string of extra agent keyword arguments",
)
@click.option(
    "--dataset-kwargs",
    default=None,
    help="JSON string of extra dataset keyword arguments",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    show_default=True,
    help="Number of tasks to run in parallel",
)
@click.option(
    "--query-timeout",
    type=float,
    default=None,
    help="Wall-clock timeout in seconds per query (default: no limit)",
)
@click.option(
    "--eval-client",
    default="openai",
    show_default=True,
    help="Client for evaluation",
)
@click.option(
    "--eval-model",
    default="gpt-5-nano-2025-08-07",
    show_default=True,
    help="Model for evaluation",
)
def run_cmd(
    agent_id: str,
    model: str | None,
    preset_name: str | None,
    dataset_id: str,
    client_base_url: str,
    api_key: str,
    max_queries: int | None,
    output_dir: str,
    export_format: str,
    estimate_flops: bool,
    agent_kwargs: str | None,
    dataset_kwargs: str | None,
    concurrency: int,
    query_timeout: float | None,
    eval_client: str,
    eval_model: str,
) -> None:
    """Execute an agentic benchmark run."""
    from .model_presets import resolve_preset

    # Resolve model from --preset if provided
    if preset_name and model:
        raise click.ClickException("Specify either --model or --preset, not both.")
    if preset_name:
        try:
            preset = resolve_preset(preset_name)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        model = preset["model_id"]
        info(f"Preset: {preset_name} → {model}")
    if not model:
        raise click.ClickException("Either --model or --preset is required.")

    import ipw.datasets

    ipw.datasets.ensure_registered()

    # Ensure agent modules are imported for registry population
    from ipw.agents import react as _react, openhands as _openhands, terminus as _terminus, terminus_tb as _terminus_tb  # noqa: F401
    from ipw.core.registry import AgentRegistry, DatasetRegistry
    from ipw.execution.agentic_runner import AgenticRunner
    from ipw.execution.exporters import export_artifacts_manifest, export_hf_dataset, export_jsonl, export_summary_json
    from ipw.execution.telemetry_session import TelemetrySession
    from ipw.telemetry import EnergyMonitorCollector
    from ipw.telemetry.events import EventRecorder

    # Parse agent kwargs
    extra_kwargs: dict = {}
    if agent_kwargs:
        try:
            extra_kwargs = json.loads(agent_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --agent-kwargs: {exc}"
            ) from exc

    # Parse dataset kwargs
    extra_dataset_kwargs: dict = {}
    if dataset_kwargs:
        try:
            extra_dataset_kwargs = json.loads(dataset_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --dataset-kwargs: {exc}"
            ) from exc

    # Resolve agent
    try:
        agent_cls = AgentRegistry.get(agent_id)
    except KeyError:
        available = [k for k, _ in AgentRegistry.items()]
        raise click.ClickException(
            f"Unknown agent '{agent_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Resolve dataset
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
    except KeyError:
        available = [k for k, _ in DatasetRegistry.items()]
        raise click.ClickException(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Preflight: dataset requirements
    try:
        dataset_instance = dataset_cls(**extra_dataset_kwargs)
        issues = dataset_instance.verify_requirements()
        if issues:
            raise click.ClickException(
                "Dataset requirements not satisfied:\n- " + "\n- ".join(issues)
            )
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to initialize dataset: {exc}") from exc

    # Create event recorder and agent
    event_recorder = EventRecorder()
    resolved_model = _create_model_for_agent(agent_id, model, client_base_url, api_key)

    # Terminus-based agents need api_base so LiteLLM can reach the local server
    if agent_id in ("terminus", "terminus-tb") and "api_base" not in extra_kwargs:
        extra_kwargs["api_base"] = f"{client_base_url}/v1"

    try:
        agent_instance = agent_cls(
            model=resolved_model,
            event_recorder=event_recorder,
            **extra_kwargs,
        )
    except TypeError as exc:
        raise click.ClickException(
            f"Failed to initialize agent '{agent_id}': {exc}"
        ) from exc

    # Prepare output directory
    model_slug = "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"
    run_dir = Path(output_dir) / f"run_{agent_id}_{model_slug}_{dataset_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build config for summary
    run_config = {
        "agent": agent_id,
        "model": model,
        "dataset": dataset_id,
        "client_base_url": client_base_url,
        "max_queries": max_queries,
        "concurrency": concurrency,
        "query_timeout": query_timeout,
        "export_format": export_format,
        "estimate_flops": estimate_flops,
        "eval_client": eval_client,
        "eval_model": eval_model,
    }

    print_banner()
    run_display_config: dict[str, object] = {
        "Agent": agent_id,
        "Model": model,
        "Dataset": dataset_id,
        "Base URL": client_base_url,
        "Max Queries": max_queries or "(all)",
    }
    if concurrency > 1:
        run_display_config["Concurrency"] = concurrency
    if query_timeout:
        run_display_config["Query Timeout"] = f"{query_timeout:.0f}s"
    run_display_config["Output"] = str(run_dir)
    print_config_summary(config=run_display_config)

    # Build an agent factory for concurrent execution so each thread gets
    # its own agent instance with independent state.
    _model_ref = resolved_model
    _agent_cls_ref = agent_cls
    _extra_kwargs_ref = dict(extra_kwargs)

    def _make_agent() -> "BaseAgent":
        rec = EventRecorder()
        return _agent_cls_ref(model=_model_ref, event_recorder=rec, **_extra_kwargs_ref)

    # Run the agentic benchmark with energy telemetry
    collector = EnergyMonitorCollector()
    with TelemetrySession(collector) as telemetry:
        runner = AgenticRunner(
            agent=agent_instance,
            dataset=dataset_instance,
            telemetry_session=telemetry,
            config=run_config,
            event_recorder=event_recorder,
            run_dir=run_dir,
            concurrency=concurrency,
            agent_factory=_make_agent if concurrency > 1 else None,
            query_timeout=query_timeout,
        )

        try:
            traces = asyncio.run(runner.run(max_queries=max_queries))
        except Exception as exc:
            error(f"Run failed: {exc}")
            sys.exit(1)

    if not traces:
        warning("No traces collected.")
        return

    # FLOPs estimation
    if estimate_flops:
        try:
            from ipw.compute.flops import estimate_flops as do_estimate_flops

            for trace in traces:
                total_flops, flops_per_token = do_estimate_flops(
                    model,
                    trace.total_input_tokens,
                    trace.total_output_tokens,
                )
                if total_flops > 0:
                    info(
                        f"  {trace.query_id}: {total_flops:.2e} FLOPs "
                        f"({flops_per_token:.2e} per token)"
                    )
        except Exception as exc:
            warning(f"FLOPs estimation failed: {exc}")

    # Export results
    formats = [f.strip().lower() for f in export_format.split(",") if f.strip()]

    if "jsonl" in formats:
        jsonl_path = export_jsonl(traces, run_dir / "traces.jsonl")
        info(f"Exported JSONL: {jsonl_path}")

    if "hf" in formats:
        hf_path = export_hf_dataset(traces, run_dir / "hf_dataset")
        info(f"Exported HF dataset: {hf_path}")

    summary_path = export_summary_json(traces, run_config, run_dir / "summary.json")
    info(f"Exported summary: {summary_path}")

    manifest_path = export_artifacts_manifest(run_dir)
    if manifest_path:
        info(f"Exported artifacts manifest: {manifest_path}")

    # Rich summary display
    total_completed = sum(1 for t in traces if t.completed)
    success(f"Run complete: {total_completed}/{len(traces)} queries completed")

    metric_rows = compute_trace_metrics(traces)
    print_metrics_table(rows=metric_rows, title="Run Metrics")
    print_efficiency_panel(traces=traces)
    print_output_path(path=run_dir)


__all__ = ["run_cmd"]

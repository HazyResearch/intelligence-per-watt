"""Run profiling against an inference client."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import click

from ipw.core.types import ProfilerConfig

from ._console import _print_result, info, success, warning
from ._display import (
    MetricRow,
    compute_profile_metrics,
    print_banner,
    print_config_summary,
    print_efficiency_panel,
    print_metrics_table,
    print_output_path,
)


def _collect_params(ctx, param, values):
    collected: Dict[str, str] = {}
    for item in values:
        for piece in item.split(","):
            if not piece:
                continue
            key, _, raw = piece.partition("=")
            key = key.strip()
            if not key:
                continue
            collected[key] = raw.strip()
    return collected


@click.command(help="Run profiling against an inference client.")
@click.option("--client", "client_id", required=True, help="Client identifier")
@click.option("--model", required=True, help="Model name to invoke")
@click.option("--dataset", "dataset_id", default="ipw", help="Dataset identifier")
@click.option("--client-base-url", help="Client base URL")
@click.option(
    "--eval-client",
    help="Evaluation client identifier (judge)",
    default="openai",
    show_default=True,
)
@click.option(
    "--eval-base-url",
    help="Evaluation client base URL",
    default="https://api.openai.com/v1",
    show_default=True,
)
@click.option(
    "--eval-model",
    help="Evaluation model to use for scoring",
    default="gpt-5-nano-2025-08-07",
    show_default=True,
)
@click.option(
    "--dataset-param",
    multiple=True,
    callback=_collect_params,
    help="Dataset params key=value",
)
@click.option(
    "--client-param",
    multiple=True,
    callback=_collect_params,
    help="Client params key=value",
)
@click.option("--output-dir", type=click.Path())
@click.option("--max-queries", type=int)
@click.option(
    "--warmup-queries",
    type=int,
    default=3,
    show_default=True,
    help="Number of warmup queries to discard before measurement (0 to disable)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Number of prompts to send concurrently (batch inference)",
)
def profile(
    dataset_id: str,
    client_id: str,
    client_base_url: str | None,
    model: str,
    dataset_param,
    client_param,
    output_dir: str | None,
    max_queries: int | None,
    warmup_queries: int,
    batch_size: int,
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
) -> None:
    """Execute profiling run with the execution pipeline."""
    import ipw.analysis
    import ipw.clients
    import ipw.datasets

    ipw.clients.ensure_registered()
    missing_reason = getattr(ipw.clients, "MISSING_CLIENTS", {}).get(client_id)
    if missing_reason:
        raise click.ClickException(
            f"Inference client '{client_id}' is unavailable: {missing_reason}"
        )

    ipw.datasets.ensure_registered()
    ipw.analysis.ensure_registered()

    from ipw.analysis.base import AnalysisContext
    from ipw.core.registry import AnalysisRegistry, DatasetRegistry
    from ipw.execution import ProfilerRunner  # Deferred import for heavy dependencies

    config = ProfilerConfig(
        dataset_id=dataset_id,
        client_id=client_id,
        client_base_url=client_base_url,
        dataset_params=dataset_param,
        client_params=client_param,
        model=model,
        max_queries=max_queries,
        output_dir=Path(output_dir) if output_dir else None,
        warmup_queries=warmup_queries,
        max_concurrency=batch_size,
    )

    # Preflight: dataset requirements (api keys, etc)
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
        dataset_instance = dataset_cls(**dataset_param)
        issues = dataset_instance.verify_requirements()
        if issues:
            raise click.ClickException(
                "Dataset requirements not satisfied:\n- " + "\n- ".join(issues)
            )
        _warn_on_custom_eval(dataset_instance, eval_client, eval_base_url, eval_model)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    print_banner()
    print_config_summary(config={
        "Model": model,
        "Dataset": dataset_id,
        "Client": client_id,
        "Base URL": client_base_url or "(default)",
        "Warmup Queries": warmup_queries,
        "Max Queries": max_queries or "(all)",
        "Batch Size": batch_size,
    })

    runner = ProfilerRunner(config)
    runner.run()
    success("Profiling run completed")

    # Display aggregate metrics table
    records = runner.records
    if records:
        metric_rows = compute_profile_metrics(records, model)
        print_metrics_table(rows=metric_rows, title="Profile Metrics")

    # Post-run analysis
    accuracy_value: float | None = None
    results_dir = runner._output_path
    if results_dir and results_dir.exists():
        info("Running post-profile analysis...")
        context = AnalysisContext(
            results_dir=results_dir,
            options={
                "model": model,
                "eval_client": eval_client,
                "eval_base_url": eval_base_url,
                "eval_model": eval_model,
            },
        )
        try:
            analysis = AnalysisRegistry.create("accuracy")
            result = analysis.run(context)
            _print_result(result, verbose=False)
            # Extract accuracy for efficiency panel
            if result.summary and "accuracy" in result.summary:
                try:
                    accuracy_value = float(result.summary["accuracy"])
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            warning(f"Warning: Analysis failed: {e}")

    # Efficiency panel & output path
    if records:
        print_efficiency_panel(records=records, model=model, accuracy=accuracy_value)
    print_output_path(path=results_dir)

    # Persist aggregate metrics and efficiency data into summary.json
    if results_dir and records:
        _save_metrics_to_summary(results_dir, metric_rows, records, model, accuracy_value)


__all__ = ["profile"]


def _save_metrics_to_summary(
    results_dir: Path,
    metric_rows: List[MetricRow],
    records: list,
    model: str,
    accuracy: Optional[float],
) -> None:
    """Append profile_metrics and efficiency data to the run's summary.json."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return

    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    summary["profile_metrics"] = [
        {
            "label": row.label,
            "avg": row.avg,
            "median": row.median,
            "min": row.min,
            "max": row.max,
            "std": row.std,
            "unit": row.unit,
        }
        for row in metric_rows
        if not all(v is None for v in (row.avg, row.median, row.min, row.max, row.std))
    ]

    energies: list[float] = []
    powers: list[float] = []
    for rec in records:
        mm = rec.model_metrics.get(model)
        if mm is None and len(rec.model_metrics) == 1:
            mm = next(iter(rec.model_metrics.values()))
        if mm is None:
            continue
        e = mm.energy_metrics.per_query_joules
        if e is not None:
            energies.append(e)
        p = mm.power_metrics.gpu.per_query_watts.avg
        if p is not None:
            powers.append(p)

    total_energy = sum(energies) if energies else None
    avg_power = statistics.mean(powers) if powers else None

    efficiency: dict = {}
    if accuracy is not None:
        efficiency["accuracy"] = accuracy
    if total_energy is not None:
        efficiency["total_energy_j"] = total_energy
    if avg_power is not None:
        efficiency["avg_power_w"] = avg_power
    if accuracy is not None and total_energy and total_energy > 0:
        efficiency["ipj"] = accuracy / total_energy
    if accuracy is not None and avg_power and avg_power > 0:
        efficiency["ipw"] = accuracy / avg_power

    summary["efficiency"] = efficiency

    summary_path.write_text(json.dumps(summary, indent=2, default=str))


def _warn_on_custom_eval(
    dataset, eval_client: str | None, eval_base_url: str | None, eval_model: str | None
) -> None:
    client_default = getattr(dataset, "eval_client", None)
    base_default = getattr(dataset, "eval_base_url", None)
    model_default = getattr(dataset, "eval_model", None)

    provided_client = (eval_client or "").strip().lower()
    expected_client = (client_default or "").strip().lower()
    client_mismatch = bool(client_default and eval_client and provided_client != expected_client)

    provided_base = (eval_base_url or "").strip()
    expected_base = (base_default or "").strip()
    base_mismatch = bool(base_default and eval_base_url and provided_base != expected_base)

    provided_model = (eval_model or "").strip()
    expected_model = (model_default or "").strip()
    model_mismatch = bool(model_default and eval_model and provided_model != expected_model)

    if not (client_mismatch or base_mismatch or model_mismatch):
        return

    warning(
        "Using custom evaluation settings for %s. Defaults: client=%s, base_url=%s, model=%s."
        % (
            getattr(dataset, "dataset_name", dataset.__class__.__name__),
            client_default or "(unspecified)",
            base_default or "(unspecified)",
            model_default or "(unspecified)",
        )
    )

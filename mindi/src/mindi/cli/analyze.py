"""Analyze profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ..analysis import build_regression_report


@click.command(help="Analyze profiling results and compute metrics.")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--skip-zeroes",
    is_flag=True,
    help="Skip regression rows with null statistics when printing results.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit results as JSON.",
)
def analyze(
    directory: Path,
    skip_zeroes: bool,
    json_output: bool,
) -> None:
    """Compute regression statistics for a profiling run."""

    try:
        report = build_regression_report(
            directory,
            skip_zeroes=skip_zeroes,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        raise click.ClickException(f"Failed to analyze profiling data: {exc}") from exc

    if json_output:
        payload = {
            "model": report.model,
            "hardware_label": report.hardware_label,
            "warnings": report.warnings,
            "total_samples": report.total_samples,
            "summary_path": str(report.summary_path),
            "regressions": report.regressions,
            "zero_counts": report.zero_counts,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo()
    click.echo(f"Model: {report.model}")
    click.echo(f"Hardware: {report.hardware_label}")

    if report.warnings:
        click.echo()
        for warning in report.warnings:
            click.echo(f"Warning: {warning}")

    click.echo()
    click.echo("Linear regression results:")

    ordered_keys = [
        "input_tokens_vs_ttft",
        "total_tokens_vs_power",
        "total_tokens_vs_latency",
        "total_tokens_vs_energy",
        "total_tokens_vs_power_log",
    ]

    for name in ordered_keys:
        stats = report.regressions.get(name)
        if not stats:
            continue
        click.echo(
            f"  {name}: slope={_sig(stats.get('slope'))} "
            f"intercept={_sig(stats.get('intercept'))} "
            f"r2={_sig(stats.get('r2'))} "
            f"avg={_sig(stats.get('avg_y'))} "
            f"(n={stats.get('count', 0)})"
        )

    click.echo()
    click.echo(f"Summary written to: {report.summary_path}")


def _sig(value: Optional[float]) -> str:
    if value is None:
        return "None"
    try:
        return f"{value:.5g}"
    except Exception:  # pragma: no cover
        return "None"


__all__ = ["analyze"]

"""Analyze profiling results."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict

import click

from ..analysis.base import AnalysisContext, AnalysisResult
from ..core.registry import AnalysisRegistry
from ._console import info, warning


def _collect_options(ctx, param, values):
    """Parse key=value options into a dictionary."""
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


@click.command(help="Analyze profiling results and compute metrics.")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--analysis",
    "analysis_name",
    default="regression",
    show_default=True,
    help="Which registered analysis to execute.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show data and metadata fields in addition to summary and artifacts.",
)
@click.option(
    "--option",
    "options",
    multiple=True,
    callback=_collect_options,
    help="Analysis-specific options (e.g., --option model=llama3.2:1b).",
)
def analyze(
    directory: Path,
    analysis_name: str,
    verbose: bool,
    options: Dict[str, Any],
) -> None:
    """Compute analysis results for a profiling run."""
    import trafficbench.analysis

    trafficbench.analysis.ensure_registered()

    context = AnalysisContext(
        results_dir=directory,
        options=options,
    )

    try:
        analysis = AnalysisRegistry.create(analysis_name)
        result = analysis.run(context)
    except KeyError as exc:
        available = ", ".join(sorted(name for name, _ in AnalysisRegistry.items()))
        raise click.ClickException(
            f"Unknown analysis '{analysis_name}'. Available analyses: {available}."
        ) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    _print_result(result, verbose=verbose)


def _print_result(result: AnalysisResult, *, verbose: bool) -> None:
    info(f"Analysis: {result.analysis}")

    if result.summary:
        info("")
        info("Summary:")
        for key, value in result.summary.items():
            info(f"  {key}: {value}")

    if result.warnings:
        info("")
        info("Warnings:")
        for warn_msg in result.warnings:
            warning(f"  {warn_msg}")

    if result.artifacts:
        info("")
        info("Artifacts:")
        for name, path in result.artifacts.items():
            info(f"  {name}: {path}")

    if verbose and result.data:
        info("")
        info("Data:")
        info(textwrap.indent(json.dumps(result.data, indent=2, default=str), "  "))

    if verbose and result.metadata:
        info("")
        info("Metadata:")
        info(textwrap.indent(json.dumps(result.metadata, indent=2, default=str), "  "))


__all__ = ["analyze"]

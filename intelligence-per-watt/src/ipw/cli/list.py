"""List registered components (clients, datasets, analyses, and visualizations)."""

from __future__ import annotations

import click
from ipw.core.registry import (
    AnalysisRegistry,
    ClientRegistry,
    DatasetRegistry,
    VisualizationRegistry,
)

from ._console import error, info


@click.group(help="List available components")
def list_cmd() -> None:
    """List available components in the registry."""


@list_cmd.command("clients", help="List available inference clients")
def list_clients() -> None:
    """List all registered inference clients."""
    import ipw.clients

    ipw.clients.ensure_registered()

    items = ClientRegistry.items()
    missing = getattr(ipw.clients, "MISSING_CLIENTS", {})

    if not items and not missing:
        error("No clients registered")
        return

    if items:
        info("Clients:")
        for client_id, client_cls in items:
            info(f"  {client_id:20}")
    else:
        error("No clients registered")

    if missing:
        info("")
        info("Unavailable clients:")
        for client_id, reason in sorted(missing.items()):
            info(f"  {client_id:20} {reason}")


@list_cmd.command("datasets", help="List available datasets")
def list_datasets() -> None:
    """List all registered dataset providers."""
    import ipw.datasets

    ipw.datasets.ensure_registered()

    items = DatasetRegistry.items()

    if not items:
        error("No datasets registered")
        return

    info("Datasets:")
    for dataset_id, dataset_cls in items:
        info(f"  {dataset_id}")


@list_cmd.command("analyses", help="List available analysis providers")
def list_analyses() -> None:
    """List all registered analysis providers."""
    import ipw.analysis

    ipw.analysis.ensure_registered()

    items = AnalysisRegistry.items()

    if not items:
        error("No analyses registered")
        return

    info("Analyses:")
    for analysis_id, analysis_cls in items:
        info(f"  {analysis_id}")


@list_cmd.command("visualizations", help="List available visualization providers")
def list_visualizations() -> None:
    """List all registered visualization providers."""
    import ipw.visualization

    ipw.visualization.ensure_registered()

    items = VisualizationRegistry.items()

    if not items:
        error("No visualizations registered")
        return

    info("Visualizations:")
    for visualization_id, visualization_cls in items:
        info(f"  {visualization_id}")


@list_cmd.command("all", help="List all available components")
def list_all() -> None:
    """List all registered components (clients, datasets, analyses, and visualizations)."""
    ctx = click.get_current_context()

    ctx.invoke(list_clients)
    info("")
    ctx.invoke(list_datasets)
    info("")
    ctx.invoke(list_analyses)
    info("")
    ctx.invoke(list_visualizations)
    info("")


__all__ = ["list_cmd"]

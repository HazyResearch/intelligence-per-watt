"""List registered components (clients and datasets)."""

from __future__ import annotations

import click

from mindi.core.registry import ClientRegistry, DatasetRegistry

from ._console import error, info


@click.group(help="List available components")
def list_cmd() -> None:
    """List available components in the registry."""


@list_cmd.command("clients", help="List available inference clients")
def list_clients() -> None:
    """List all registered inference clients."""
    items = ClientRegistry.items()

    if not items:
        error("No clients registered")
        return

    info("Clients:")
    for client_id, client_cls in items:
        info(f"  {client_id:20}")


@list_cmd.command("datasets", help="List available datasets")
def list_datasets() -> None:
    """List all registered dataset providers."""
    items = DatasetRegistry.items()

    if not items:
        error("No datasets registered")
        return

    info("Datasets:")
    for dataset_id, dataset_cls in items:
        info(f"  {dataset_id}")


@list_cmd.command("all", help="List all available components")
def list_all() -> None:
    """List all registered components (clients and datasets)."""
    ctx = click.get_current_context()

    ctx.invoke(list_clients)
    info("")
    ctx.invoke(list_datasets)
    info("")


__all__ = ["list_cmd"]

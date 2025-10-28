"""Helpers to resolve registered components from identifiers."""

from __future__ import annotations

from typing import Any, Mapping

from ..core.client import InferenceClient
from ..core.collector import HardwareCollector
from ..core.dataset import DatasetProvider
from ..core.registry import ClientRegistry, CollectorRegistry, DatasetRegistry


class ResolutionError(RuntimeError):
    """Raised when a configured component cannot be resolved."""


def resolve_dataset(dataset_id: str, params: Mapping[str, Any]) -> DatasetProvider:
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
    except KeyError as exc:
        raise ResolutionError(f"Unknown dataset '{dataset_id}'") from exc

    try:
        return dataset_cls(**params)
    except TypeError as exc:
        raise ResolutionError(
            f"Failed to instantiate dataset '{dataset_id}' with params {params!r}: {exc}"
        ) from exc


def resolve_client(
    client_id: str,
    base_url: str | None,
    params: Mapping[str, Any],
) -> InferenceClient:
    try:
        client_cls = ClientRegistry.get(client_id)
    except KeyError as exc:
        raise ResolutionError(f"Unknown client '{client_id}'") from exc

    try:
        return client_cls(base_url, **params)
    except TypeError as exc:
        raise ResolutionError(
            f"Failed to instantiate client '{client_id}' with params {params!r}: {exc}"
        ) from exc


def resolve_collector(
    collector_id: str,
    params: Mapping[str, Any],
) -> HardwareCollector:
    try:
        collector_cls = CollectorRegistry.get(collector_id)
    except KeyError as exc:
        raise ResolutionError(f"Unknown collector '{collector_id}'") from exc

    try:
        return collector_cls(**params)
    except TypeError as exc:
        raise ResolutionError(
            f"Failed to instantiate collector '{collector_id}' with params {params!r}: {exc}"
        ) from exc


def ensure_client_ready(client: InferenceClient) -> None:
    if not client.health():
        raise ResolutionError(
            f"Client '{client.client_name}' at {getattr(client, 'base_url', '')} is unavailable"
        )


def ensure_collector_available(collector_cls: type[HardwareCollector]) -> None:
    if not collector_cls.is_available():
        raise ResolutionError(
            f"Collector '{collector_cls.collector_name}' is not available on this system"
        )


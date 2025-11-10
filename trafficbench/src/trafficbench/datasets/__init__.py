"""Dataset implementations bundled with TrafficBench.

Datasets register themselves with ``trafficbench.core.DatasetRegistry``.
"""

from .base import DatasetProvider


def ensure_registered() -> None:
    """Import built-in dataset providers to populate the registry."""
    from . import trafficbench  # noqa: F401


__all__ = ["DatasetProvider", "ensure_registered"]

"""Visualization implementations bundled with TrafficBench.

Visualizations register themselves with ``trafficbench.core.VisualizationRegistry``.
"""

from .base import VisualizationContext, VisualizationProvider, VisualizationResult


def ensure_registered() -> None:
    """Import built-in visualization providers to populate the registry."""
    from . import output_kde  # noqa: F401  (registers on import)
    from . import regression  # noqa: F401  (registers on import)


__all__ = [
    "VisualizationProvider",
    "VisualizationContext",
    "VisualizationResult",
    "ensure_registered",
]

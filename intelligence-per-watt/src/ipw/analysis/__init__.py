"""Analysis implementations bundled with Intelligence Per Watt.

Analyses register themselves with ``ipw.core.AnalysisRegistry``.
"""

from .base import AnalysisContext, AnalysisProvider, AnalysisResult


def ensure_registered() -> None:
    """Import built-in analysis providers to populate the registry."""
    from . import regression  # noqa: F401  (registers on import)


__all__ = ["AnalysisProvider", "AnalysisContext", "AnalysisResult", "ensure_registered"]

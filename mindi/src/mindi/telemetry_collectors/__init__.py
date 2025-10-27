"""Telemetry collector implementations bundled with Mindi.

Collectors register themselves with ``mindi.core.CollectorRegistry``.
"""

from . import mindi

__all__ = ["mindi"]

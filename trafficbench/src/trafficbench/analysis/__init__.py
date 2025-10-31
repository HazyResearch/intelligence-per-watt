"""Analysis implementations bundled with TrafficBench.

Analyses register themselves with ``trafficbench.core.AnalysisRegistry``.
"""

from . import regression
from .base import AnalysisContext, AnalysisProvider, AnalysisResult

__all__ = ["AnalysisProvider", "AnalysisContext", "AnalysisResult", "regression"]

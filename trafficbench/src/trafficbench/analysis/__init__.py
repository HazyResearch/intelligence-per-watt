"""Analysis implementations bundled with TrafficBench.

Analyses register themselves with ``trafficbench.core.AnalysisRegistry``.
"""

from .base import AnalysisProvider, AnalysisContext, AnalysisResult
from . import regression

__all__ = ["AnalysisProvider", "AnalysisContext", "AnalysisResult", "regression"]

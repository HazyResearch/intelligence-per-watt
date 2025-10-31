"""Analysis implementations bundled with Mindi.

Analyses register themselves with ``mindi.core.AnalysisRegistry``.
"""

from .base import AnalysisProvider, AnalysisContext, AnalysisResult
from . import regression

__all__ = ["AnalysisProvider", "AnalysisContext", "AnalysisResult", "regression"]

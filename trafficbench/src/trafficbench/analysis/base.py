"""Core contracts for analysis plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if mapping is None:
        return MappingProxyType({})
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


@dataclass(slots=True, frozen=True)
class AnalysisContext:
    """Common inputs made available to analysis plugins."""

    results_dir: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    """Standardized payload returned by analysis plugins."""

    analysis: str
    summary: Mapping[str, Any]
    data: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))
        object.__setattr__(self, "data", _freeze_mapping(self.data))
        object.__setattr__(self, "artifacts", _freeze_mapping(self.artifacts))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


class AnalysisProvider(ABC):
    """Base class for extendible analysis plugins."""

    #: Registry key for this analysis. Subclasses should override this constant.
    analysis_id: str

    @abstractmethod
    def run(self, context: AnalysisContext) -> AnalysisResult:
        """Execute the analysis and return a structured result."""

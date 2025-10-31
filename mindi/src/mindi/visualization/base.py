"""Core contracts for visualization plugins."""

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
class VisualizationContext:
    """Common inputs made available to visualization plugins."""

    results_dir: Path
    output_dir: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(slots=True, frozen=True)
class VisualizationResult:
    """Standardized payload returned by visualization plugins."""

    visualization: str
    artifacts: Mapping[str, Path]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _freeze_mapping(self.artifacts))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


class VisualizationProvider(ABC):
    """Base class for extendible visualization plugins."""

    #: Registry key for this visualization. Subclasses should override this constant.
    visualization_id: str

    @abstractmethod
    def render(self, context: VisualizationContext) -> VisualizationResult:
        """Execute the visualization and return paths to generated artifacts."""


__all__ = ["VisualizationProvider", "VisualizationContext", "VisualizationResult"]


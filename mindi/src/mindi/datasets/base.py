from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from ..core.types import DatasetRecord


class DatasetProvider(ABC):
    """Base interface for providing prompts to the profiler."""

    dataset_id: str
    dataset_name: str

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    @abstractmethod
    def iter_records(self) -> Iterable[DatasetRecord]:
        """Yield dataset records in the order they should be executed."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of records."""

__all__ = ["DatasetProvider"]

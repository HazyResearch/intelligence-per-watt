from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from .types import TelemetryReading


class HardwareCollector(ABC):
    """Base class for hardware telemetry collectors."""

    collector_id: str
    collector_name: str

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when the energy monitor is reachable."""

    @abstractmethod
    def stream_readings(self) -> Iterable[TelemetryReading]:
        """Stream telemetry readings provided by the energy monitor."""


__all__ = ["HardwareCollector"]

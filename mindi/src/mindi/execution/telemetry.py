"""Telemetry session helpers for profiling runs."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Optional

from ..core.collector import HardwareCollector
from ..core.types import TelemetryReading


@dataclass
class TelemetrySample:
    timestamp: float
    reading: TelemetryReading


class TelemetrySession(AbstractContextManager["TelemetrySession"]):
    """Capture telemetry readings in a background thread."""

    def __init__(
        self,
        collector: HardwareCollector,
        *,
        buffer_seconds: float = 30.0,
        max_samples: int = 10_000,
    ) -> None:
        self._collector = collector
        self._buffer_seconds = buffer_seconds
        self._max_samples = max_samples
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._collector_ctx = None

    def __enter__(self) -> "TelemetrySession":
        self._collector_ctx = self._collector.start()
        self._collector_ctx.__enter__()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._collector_ctx is not None:
            self._collector_ctx.__exit__(None, None, None)

    def _run(self) -> None:
        try:
            for reading in self._collector.stream_readings():
                timestamp = time.time()
                self._samples.append(TelemetrySample(timestamp=timestamp, reading=reading))
                self._trim(timestamp)
                if self._stop_event.is_set():
                    break
        except Exception:  # pragma: no cover - surface to caller on access
            self._stop_event.set()
            raise

    def _trim(self, current_time: float) -> None:
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        return list(self._samples)

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        for sample in list(self._samples):
            if start_time <= sample.timestamp <= end_time:
                yield sample


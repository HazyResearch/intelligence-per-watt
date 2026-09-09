"""Telemetry session helpers for profiling runs."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Optional

from ..core.types import TelemetryReading
from ..telemetry import EnergyMonitorCollector

LOGGER = logging.getLogger(__name__)

# How far after a requested start the first retained sample may land before the
# window counts as truncated. The energy monitor streams at roughly 50 ms, so a
# second of slack absorbs ordinary cadence jitter and a late-starting sampler
# while still catching real eviction, which runs to tens of seconds or more.
DEFAULT_COVERAGE_TOLERANCE_SECONDS = 1.0


@dataclass
class TelemetrySample:
    timestamp: float
    reading: TelemetryReading


@dataclass(frozen=True)
class WindowCoverage:
    """How much of a requested interval ``window()`` was able to return.

    ``window()`` filters whatever retention has left it; it cannot report what
    was already trimmed. Integrating a short window yields a *plausible but too
    small* energy figure rather than an error, so any caller turning a window
    into a per-query total needs a way to ask whether the window was whole.
    """

    start_time: float
    end_time: float
    sample_count: int
    earliest_retained: Optional[float]
    tolerance_seconds: float

    @property
    def requested_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def missing_seconds(self) -> float:
        """Seconds at the head of the interval that retention had already dropped."""
        if self.earliest_retained is None:
            # Nothing retained at all -- absence of telemetry, not truncation of
            # it. `sample_count` is the field that distinguishes the two.
            return 0.0
        return max(0.0, self.earliest_retained - self.start_time)

    @property
    def truncated(self) -> bool:
        return self.missing_seconds > self.tolerance_seconds


class TelemetrySession(AbstractContextManager["TelemetrySession"]):
    """Capture telemetry readings in a background thread."""

    def __init__(
        self,
        collector: EnergyMonitorCollector,
        *,
        buffer_seconds: Optional[float],
        max_samples: Optional[int],
    ) -> None:
        """Capture telemetry readings in a background thread.

        Both retention bounds are deliberately required. They cap how much
        history ``window()`` can return, and a window shorter than the interval
        it is asked for yields a *plausible but too small* energy figure rather
        than an error -- so a shared default is a trap. The previous 30 s
        default silently truncated every ``ipw profile`` query longer than 30 s
        from the first commit onward. Size these to the longest query the
        caller can produce; pass ``None`` to disable a bound entirely.
        """
        self._collector = collector
        self._buffer_seconds = buffer_seconds
        self._max_samples = max_samples
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._collector_ctx = None
        self._gpu_device_id = self._resolve_gpu_device_id()
        self._single_visible_gpu = self._has_single_visible_gpu()

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
                if not self._include_reading(reading):
                    continue
                timestamp = (
                    float(reading.timestamp_nanos) / 1_000_000_000.0
                    if reading.timestamp_nanos is not None
                    else time.time()
                )
                self._samples.append(
                    TelemetrySample(timestamp=timestamp, reading=reading)
                )
                self._trim(timestamp)
                if self._stop_event.is_set():
                    break
        except Exception:  # pragma: no cover - surface to caller on access
            self._stop_event.set()
            raise

    def _visible_gpu_parts(self) -> list[str]:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        return [part.strip() for part in visible.split(",") if part.strip()]

    def _has_single_visible_gpu(self) -> bool:
        return len(self._visible_gpu_parts()) == 1

    def _resolve_gpu_device_id(self) -> Optional[int]:
        explicit = os.getenv("IPW_GPU_DEVICE_ID")
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                return None

        parts = self._visible_gpu_parts()
        if len(parts) != 1:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    def _include_reading(self, reading: TelemetryReading) -> bool:
        if self._gpu_device_id is None:
            return True
        gpu_info = reading.gpu_info
        if gpu_info is None:
            return True
        device_id = int(gpu_info.device_id)
        if device_id == self._gpu_device_id:
            return True
        # NVML reports device 0 when CUDA_VISIBLE_DEVICES exposes exactly one
        # physical GPU. In that mode the monitor is already isolated to one H100,
        # so keep the visible device-0 samples instead of dropping all telemetry.
        return self._single_visible_gpu and device_id == 0

    def _trim(self, current_time: float) -> None:
        if self._buffer_seconds is None:
            return
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        return list(self._samples)

    def coverage(
        self,
        start_time: float,
        end_time: float,
        *,
        tolerance_seconds: float = DEFAULT_COVERAGE_TOLERANCE_SECONDS,
    ) -> WindowCoverage:
        """Report whether ``window(start_time, end_time)`` sees the whole interval.

        Truncation is inferred from the oldest sample still retained: if the
        buffer holds nothing from before ``start_time``, then everything
        preceding its earliest sample was trimmed away and the window is short
        at its head by that much.
        """
        samples = list(self._samples)
        earliest = samples[0].timestamp if samples else None
        return WindowCoverage(
            start_time=start_time,
            end_time=end_time,
            sample_count=sum(
                1 for s in samples if start_time <= s.timestamp <= end_time
            ),
            earliest_retained=earliest,
            tolerance_seconds=tolerance_seconds,
        )

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        # Warn here rather than leaving it to the caller: a truncated window is
        # indistinguishable from a short query downstream, and going quiet about
        # it is what let the 30 s default under-report energy unnoticed. Callers
        # that want to record the shortfall rather than just log it should use
        # `coverage()`.
        shortfall = self.coverage(start_time, end_time)
        if shortfall.truncated:
            LOGGER.warning(
                "Telemetry window truncated: %.1fs of the requested %.1fs interval "
                "was already evicted from the sample buffer. Energy integrated over "
                "this window is under-reported. Raise buffer_seconds/max_samples on "
                "this TelemetrySession.",
                shortfall.missing_seconds,
                shortfall.requested_seconds,
            )
        for sample in list(self._samples):
            if start_time <= sample.timestamp <= end_time:
                yield sample

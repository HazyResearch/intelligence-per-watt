"""Raw telemetry time series written alongside a run's results.

Per-query records only keep summary stats (min/max/avg), which can't show how
a quantity such as die temperature evolves across a run. These writers keep
the full series on disk so it can be plotted against time afterwards:

- ``telemetry_timeline.csv``: one row per telemetry sample.
- ``query_windows.csv``: one row per query, giving the interval it ran over.

Both use Unix timestamps in seconds, so query windows line up with samples
without any shared state between the two files.
"""

from __future__ import annotations

import csv
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional

from ..core.types import TelemetryReading

TIMELINE_FILENAME = "telemetry_timeline.csv"
QUERY_WINDOWS_FILENAME = "query_windows.csv"

# Reading fields recorded per sample, in column order. ``temperature_celsius``
# is the GPU temperature (its proto name predates CPU temperature).
TIMELINE_FIELDS = (
    "power_watts",
    "cpu_power_watts",
    "ane_power_watts",
    "energy_joules",
    "cpu_energy_joules",
    "ane_energy_joules",
    "temperature_celsius",
    "cpu_temperature_celsius",
    "gpu_compute_utilization_pct",
    "gpu_memory_usage_mb",
    "cpu_memory_usage_mb",
)

TIMELINE_COLUMNS = ("timestamp", "time_s", "platform", *TIMELINE_FIELDS)
QUERY_WINDOW_COLUMNS = ("query_index", "warmup", "start_timestamp", "end_timestamp")

# The monitor streams at ~20 Hz; flushing about once a second bounds what a
# crash can lose without a syscall per sample.
_FLUSH_EVERY_ROWS = 20


def _cell(value: Optional[float]) -> str:
    return "" if value is None else repr(float(value))


class TimelineWriter:
    """Append telemetry samples to a CSV as they arrive.

    Thread-safe: the telemetry thread writes while the main thread may close.
    Writes after ``close()`` are dropped rather than raising, since the
    telemetry thread can deliver one last sample during shutdown.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TIMELINE_COLUMNS)
        self._first_timestamp: Optional[float] = None
        self._pending_rows = 0

    def write(self, timestamp: float, reading: TelemetryReading) -> None:
        with self._lock:
            if self._file.closed:
                return
            if self._first_timestamp is None:
                self._first_timestamp = timestamp
            self._writer.writerow(
                (
                    repr(timestamp),
                    f"{timestamp - self._first_timestamp:.3f}",
                    reading.platform or "",
                    *(_cell(getattr(reading, name)) for name in TIMELINE_FIELDS),
                )
            )
            self._pending_rows += 1
            if self._pending_rows >= _FLUSH_EVERY_ROWS:
                self._file.flush()
                self._pending_rows = 0

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class QueryWindowWriter(AbstractContextManager["QueryWindowWriter"]):
    """Append each query's start/end time to a CSV, flushed per row."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(QUERY_WINDOW_COLUMNS)
        self._file.flush()

    def write(self, query_index: int, start: float, end: float, *, warmup: bool) -> None:
        self._writer.writerow((query_index, int(warmup), repr(start), repr(end)))
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "QUERY_WINDOWS_FILENAME",
    "QUERY_WINDOW_COLUMNS",
    "QueryWindowWriter",
    "TIMELINE_COLUMNS",
    "TIMELINE_FIELDS",
    "TIMELINE_FILENAME",
    "TimelineWriter",
]

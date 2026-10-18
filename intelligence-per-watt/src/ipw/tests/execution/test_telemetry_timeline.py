"""Tests for the raw telemetry timeline written alongside run results."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from unittest.mock import Mock

from ipw.core.types import TelemetryReading
from ipw.execution.telemetry_session import TelemetrySample, TelemetrySession
from ipw.execution.telemetry_timeline import (
    QUERY_WINDOW_COLUMNS,
    TIMELINE_COLUMNS,
    QueryWindowWriter,
    TimelineWriter,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _collector(readings, release: threading.Event | None = None) -> Mock:
    """A collector whose stream yields ``readings``, then optionally blocks."""

    def stream():
        yield from readings
        if release is not None:
            release.wait(timeout=5.0)

    collector = Mock()
    ctx = Mock()
    ctx.__enter__ = Mock(return_value=ctx)
    ctx.__exit__ = Mock(return_value=None)
    collector.start.return_value = ctx
    collector.stream_readings.side_effect = stream
    return collector


class TestTimelineWriter:
    def test_writes_header_and_relative_time(self, tmp_path: Path) -> None:
        path = tmp_path / "timeline.csv"
        writer = TimelineWriter(path)
        writer.write(
            100.0,
            TelemetryReading(
                temperature_celsius=45.0,
                cpu_temperature_celsius=60.5,
                platform="macos",
            ),
        )
        writer.write(100.25, TelemetryReading(cpu_temperature_celsius=61.0))
        writer.close()

        with path.open(newline="") as handle:
            assert tuple(next(csv.reader(handle))) == TIMELINE_COLUMNS
        rows = _rows(path)
        assert [row["time_s"] for row in rows] == ["0.000", "0.250"]
        assert float(rows[0]["timestamp"]) == 100.0
        assert rows[0]["platform"] == "macos"
        assert float(rows[0]["cpu_temperature_celsius"]) == 60.5
        assert float(rows[0]["temperature_celsius"]) == 45.0

    def test_missing_values_are_empty_cells(self, tmp_path: Path) -> None:
        path = tmp_path / "timeline.csv"
        writer = TimelineWriter(path)
        writer.write(1.0, TelemetryReading())
        writer.close()

        row = _rows(path)[0]
        assert row["cpu_temperature_celsius"] == ""
        assert row["power_watts"] == ""

    def test_write_after_close_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "timeline.csv"
        writer = TimelineWriter(path)
        writer.close()
        writer.write(1.0, TelemetryReading())  # must not raise

        assert _rows(path) == []


class TestQueryWindowWriter:
    def test_records_each_query(self, tmp_path: Path) -> None:
        path = tmp_path / "windows.csv"
        with QueryWindowWriter(path) as windows:
            windows.write(0, 10.0, 12.5, warmup=True)
            windows.write(1, 13.0, 20.0, warmup=False)

        with path.open(newline="") as handle:
            assert tuple(next(csv.reader(handle))) == QUERY_WINDOW_COLUMNS
        rows = _rows(path)
        assert [row["warmup"] for row in rows] == ["1", "0"]
        assert float(rows[1]["start_timestamp"]) == 13.0
        assert float(rows[1]["end_timestamp"]) == 20.0


class TestSessionTimeline:
    def test_timeline_path_streams_every_sample(self, tmp_path: Path) -> None:
        path = tmp_path / "timeline.csv"
        readings = [
            TelemetryReading(cpu_temperature_celsius=50.0 + i, timestamp_nanos=i * 10**9)
            for i in range(3)
        ]
        release = threading.Event()
        session = TelemetrySession(
            _collector(readings, release),
            buffer_seconds=None,
            max_samples=None,
            timeline_path=path,
        )
        with session:
            for _ in range(100):
                if len(session.readings()) == 3:
                    break
                threading.Event().wait(0.01)
            release.set()

        rows = _rows(path)
        assert [float(row["cpu_temperature_celsius"]) for row in rows] == [50.0, 51.0, 52.0]
        assert [row["time_s"] for row in rows] == ["0.000", "1.000", "2.000"]

    def test_start_timeline_backfills_buffered_samples_once(self, tmp_path: Path) -> None:
        session = TelemetrySession(Mock(), buffer_seconds=None, max_samples=None)
        for i in range(2):
            session._samples.append(
                TelemetrySample(
                    timestamp=float(i), reading=TelemetryReading(cpu_temperature_celsius=40.0 + i)
                )
            )

        path = session.start_timeline(tmp_path / "timeline.csv")
        session._timeline.close()

        rows = _rows(path)
        assert [float(row["cpu_temperature_celsius"]) for row in rows] == [40.0, 41.0]

    def test_buffer_trimming_does_not_truncate_timeline(self, tmp_path: Path) -> None:
        # The in-memory buffer is bounded; the file must keep everything.
        path = tmp_path / "timeline.csv"
        readings = [
            TelemetryReading(power_watts=float(i), timestamp_nanos=i * 10**9)
            for i in range(10)
        ]
        release = threading.Event()
        session = TelemetrySession(
            _collector(readings, release),
            buffer_seconds=None,
            max_samples=3,
            timeline_path=path,
        )
        with session:
            for _ in range(100):
                if session.readings() and list(session.readings())[-1].reading.power_watts == 9.0:
                    break
                threading.Event().wait(0.01)
            release.set()

        assert len(session.readings()) == 3
        assert len(_rows(path)) == 10

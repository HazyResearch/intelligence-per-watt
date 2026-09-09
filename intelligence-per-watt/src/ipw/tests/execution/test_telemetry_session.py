"""Tests for telemetry session management."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from ipw.core.types import ProfilerConfig, TelemetryReading
from ipw.execution.telemetry_session import TelemetrySample, TelemetrySession


class TestTelemetrySession:
    """Test TelemetrySession context manager."""

    def test_initializes_with_collector(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)
        assert session._collector is collector

    def test_context_manager_starts_collector(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)

        # Create an empty iterator for stream_readings
        collector.stream_readings.return_value = iter([])

        with TelemetrySession(collector, buffer_seconds=None, max_samples=None) as session:
            collector.start.assert_called_once()
            assert session is not None

    def test_context_manager_stops_thread(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)
        collector.stream_readings.return_value = iter([])

        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)
        with session:
            pass

        # Thread should be stopped
        assert session._stop_event.is_set()

    def test_window_filters_by_time(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)

        # Manually add samples
        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=3.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=4.0, reading=TelemetryReading())
        )

        windowed = list(session.window(1.5, 3.5))
        assert len(windowed) == 2
        assert windowed[0].timestamp == 2.0
        assert windowed[1].timestamp == 3.0

    def test_window_includes_boundaries(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=3.0, reading=TelemetryReading())
        )

        windowed = list(session.window(1.0, 3.0))
        assert len(windowed) == 3

    def test_window_returns_empty_when_no_overlap(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )

        windowed = list(session.window(5.0, 10.0))
        assert len(windowed) == 0

    def test_readings_returns_all_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=None)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )

        readings = list(session.readings())
        assert len(readings) == 2

    def test_trim_removes_old_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=5.0, max_samples=None)

        session._samples.append(
            TelemetrySample(timestamp=1.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=2.0, reading=TelemetryReading())
        )
        session._samples.append(
            TelemetrySample(timestamp=10.0, reading=TelemetryReading())
        )

        session._trim(10.0)

        # Only samples within buffer_seconds (5.0) should remain
        # cutoff = 10.0 - 5.0 = 5.0
        # So samples at timestamp >= 5.0 remain
        # That means only the sample at 10.0 should remain
        assert len(session._samples) == 1
        assert session._samples[0].timestamp == 10.0

    def test_respects_max_samples(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector, buffer_seconds=None, max_samples=3)

        for i in range(5):
            session._samples.append(
                TelemetrySample(timestamp=float(i), reading=TelemetryReading())
            )

        # deque with maxlen should keep only last 3
        assert len(session._samples) == 3
        assert session._samples[0].timestamp == 2.0
        assert session._samples[-1].timestamp == 4.0

    def test_integration_with_real_collector(self) -> None:
        """Integration test with actual streaming (if available)."""
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)

        # Simulate a few readings
        readings = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
            TelemetryReading(energy_joules=200.0),
        ]

        def reading_generator():
            for r in readings:
                yield r
                time.sleep(0.01)  # Small delay

        collector.stream_readings.return_value = reading_generator()

        with TelemetrySession(collector, buffer_seconds=10.0, max_samples=None) as session:
            # Give thread time to collect samples
            time.sleep(0.1)

        # Should have collected some samples
        assert len(session._samples) > 0


class TestRetentionCoversLongQueries:
    """Regression guards for the 30 s truncation bug.

    `window()` returns whatever samples survive trimming, without signalling
    that it returned less than the caller asked for. These pin the two
    properties that made the truncation possible.
    """

    def test_retention_must_be_explicit(self) -> None:
        """No shared default -- every call site states its own retention."""
        with pytest.raises(TypeError):
            TelemetrySession(Mock())

    def test_profiler_default_outlasts_a_long_query(self) -> None:
        cfg = ProfilerConfig(dataset_id="d", client_id="c")
        assert cfg.telemetry_buffer_seconds >= 3600.0
        # A time buffer is only real if the sample cap can hold it; the energy
        # monitor streams at ~50 ms, so both bounds have to agree.
        assert cfg.telemetry_max_samples >= cfg.telemetry_buffer_seconds / 0.05

    def test_window_keeps_a_two_minute_query_whole(self) -> None:
        """A 120 s query -- 4x the old default -- must survive intact."""
        session = TelemetrySession(
            Mock(), buffer_seconds=7200.0, max_samples=150_000
        )
        now = time.time()
        n = 2400  # 120 s at 50 ms
        for i in range(n):
            session._samples.append(
                TelemetrySample(
                    timestamp=now - 120.0 + i * 0.05,
                    reading=TelemetryReading(),
                )
            )
        session._trim(now)

        covered = list(session.window(now - 120.0, now))
        assert len(covered) == n, "samples from the start of the query were evicted"
        assert covered[0].timestamp <= now - 119.9


class TestWindowCoverage:
    """Truncation must announce itself.

    A window cut short by retention still yields a per-query energy figure, and
    that figure is plausible -- just too small. These pin the signal that makes
    the difference visible instead of silent.
    """

    @staticmethod
    def _session_with_history(
        seconds: float, *, buffer_seconds: float | None
    ) -> TelemetrySession:
        session = TelemetrySession(
            Mock(), buffer_seconds=buffer_seconds, max_samples=None
        )
        now = time.time()
        for i in range(int(seconds / 0.05)):
            session._samples.append(
                TelemetrySample(
                    timestamp=now - seconds + i * 0.05, reading=TelemetryReading()
                )
            )
        session._trim(now)
        return session

    def test_intact_window_reports_full_coverage(self) -> None:
        session = self._session_with_history(120.0, buffer_seconds=7200.0)
        now = time.time()

        coverage = session.coverage(now - 60.0, now)

        assert not coverage.truncated
        assert coverage.missing_seconds == 0.0
        assert coverage.sample_count > 0

    def test_trimmed_window_reports_truncation(self) -> None:
        """The exact shape of the bug: 120 s query, 30 s of retention."""
        session = self._session_with_history(120.0, buffer_seconds=30.0)
        now = time.time()

        coverage = session.coverage(now - 120.0, now)

        assert coverage.truncated
        # ~90 s of the query was evicted before anyone asked for it.
        assert coverage.missing_seconds == pytest.approx(90.0, abs=2.0)
        assert coverage.requested_seconds == pytest.approx(120.0, abs=0.1)

    def test_window_warns_when_truncated(self, caplog) -> None:
        session = self._session_with_history(120.0, buffer_seconds=30.0)
        now = time.time()

        with caplog.at_level("WARNING"):
            list(session.window(now - 120.0, now))

        assert "truncated" in caplog.text.lower()

    def test_window_is_quiet_when_intact(self, caplog) -> None:
        session = self._session_with_history(120.0, buffer_seconds=7200.0)
        now = time.time()

        with caplog.at_level("WARNING"):
            list(session.window(now - 60.0, now))

        assert caplog.text == ""

    def test_empty_buffer_is_absence_not_truncation(self) -> None:
        """No telemetry at all is a different failure, and must not be conflated."""
        session = TelemetrySession(Mock(), buffer_seconds=30.0, max_samples=None)
        now = time.time()

        coverage = session.coverage(now - 120.0, now)

        assert coverage.sample_count == 0
        assert not coverage.truncated

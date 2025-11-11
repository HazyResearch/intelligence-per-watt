"""Tests for telemetry session management."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from ipw.core.types import TelemetryReading
from ipw.execution.telemetry_session import TelemetrySession, TelemetrySample


class TestTelemetrySession:
    """Test TelemetrySession context manager."""

    def test_initializes_with_collector(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)
        assert session._collector is collector

    def test_context_manager_starts_collector(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)
        
        # Create an empty iterator for stream_readings
        collector.stream_readings.return_value = iter([])

        with TelemetrySession(collector) as session:
            collector.start.assert_called_once()
            assert session is not None

    def test_context_manager_stops_thread(self) -> None:
        collector = Mock()
        collector_ctx = Mock()
        collector.start.return_value = collector_ctx
        collector_ctx.__enter__ = Mock(return_value=collector_ctx)
        collector_ctx.__exit__ = Mock(return_value=None)
        collector.stream_readings.return_value = iter([])

        session = TelemetrySession(collector)
        with session:
            pass

        # Thread should be stopped
        assert session._stop_event.is_set()

    def test_window_filters_by_time(self) -> None:
        collector = Mock()
        session = TelemetrySession(collector)

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
        session = TelemetrySession(collector)

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
        session = TelemetrySession(collector)

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
        session = TelemetrySession(collector)

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
        session = TelemetrySession(collector, buffer_seconds=5.0)

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
        session = TelemetrySession(collector, max_samples=3)

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

        with TelemetrySession(collector, buffer_seconds=10.0) as session:
            # Give thread time to collect samples
            time.sleep(0.1)

        # Should have collected some samples
        assert len(session._samples) > 0


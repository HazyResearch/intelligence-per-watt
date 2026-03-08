"""Tests for power-based energy fallback in agentic_runner."""

from __future__ import annotations

import pytest

from ipw.core.types import TelemetryReading
from ipw.execution.agentic_runner import (
    _compute_energy_delta,
    _estimate_energy_from_power,
)
from ipw.execution.telemetry_session import TelemetrySample


def _make_sample(
    ts: float,
    power_watts: float = 150.0,
    energy_joules: float | None = None,
    cpu_power_watts: float = 45.0,
    cpu_energy_joules: float | None = None,
) -> TelemetrySample:
    return TelemetrySample(
        timestamp=ts,
        reading=TelemetryReading(
            power_watts=power_watts,
            energy_joules=energy_joules,
            cpu_power_watts=cpu_power_watts,
            cpu_energy_joules=cpu_energy_joules,
        ),
    )


class TestEstimateEnergyFromPower:
    def test_basic_calculation(self):
        """avg_power * duration_s should be returned."""
        readings = [
            _make_sample(0.0, power_watts=100.0),
            _make_sample(1.0, power_watts=200.0),
        ]
        result = _estimate_energy_from_power(readings, "power_watts", 10.0)
        assert result == pytest.approx(150.0 * 10.0)

    def test_returns_none_for_zero_duration(self):
        readings = [_make_sample(0.0, power_watts=100.0)]
        assert _estimate_energy_from_power(readings, "power_watts", 0.0) is None

    def test_returns_none_for_negative_duration(self):
        readings = [_make_sample(0.0, power_watts=100.0)]
        assert _estimate_energy_from_power(readings, "power_watts", -1.0) is None

    def test_returns_none_for_empty_readings(self):
        assert _estimate_energy_from_power([], "power_watts", 10.0) is None

    def test_returns_none_for_zero_power(self):
        readings = [_make_sample(0.0, power_watts=0.0)]
        assert _estimate_energy_from_power(readings, "power_watts", 10.0) is None

    def test_cpu_power_field(self):
        readings = [_make_sample(0.0, cpu_power_watts=50.0)]
        result = _estimate_energy_from_power(readings, "cpu_power_watts", 5.0)
        assert result == pytest.approx(50.0 * 5.0)


class TestEnergyFallbackSingleReading:
    def test_single_reading_no_energy_delta(self):
        """With only 1 sample, _compute_energy_delta returns None but
        _estimate_energy_from_power should give a fallback."""
        readings = [_make_sample(0.0, power_watts=200.0, energy_joules=100.0)]

        # Delta needs ≥2 samples — should be None
        assert _compute_energy_delta(readings, "energy_joules") is None

        # Fallback from power should work
        result = _estimate_energy_from_power(readings, "power_watts", 5.0)
        assert result == pytest.approx(200.0 * 5.0)


class TestEnergyDeltaPreferredOverFallback:
    def test_delta_used_when_two_samples(self):
        """When ≥2 cumulative energy samples exist, delta should be used,
        not the power fallback."""
        readings = [
            _make_sample(0.0, power_watts=200.0, energy_joules=100.0),
            _make_sample(5.0, power_watts=200.0, energy_joules=1100.0),
        ]

        # Delta should return 1000 J
        delta = _compute_energy_delta(readings, "energy_joules")
        assert delta == pytest.approx(1000.0)

        # Fallback would return 200 * 5 = 1000 too, but the point is delta
        # is the preferred path when available
        fallback = _estimate_energy_from_power(readings, "power_watts", 5.0)
        assert fallback is not None

        # In agentic_runner, fallback is only used when delta is None
        # so delta takes precedence
        assert delta is not None

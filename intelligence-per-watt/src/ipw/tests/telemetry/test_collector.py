"""Tests for telemetry/collector.py — EnergyMonitorCollector._convert."""

from __future__ import annotations

from types import SimpleNamespace

from ipw.telemetry.collector import EnergyMonitorCollector


class TestConvertMethod:
    """Test that _convert extracts all fields from protobuf messages."""

    def _make_message(self, **kwargs) -> SimpleNamespace:
        """Create a fake protobuf message with the given fields."""
        return SimpleNamespace(**kwargs)

    def test_cpu_fields_extracted(self) -> None:
        """cpu_power_watts and cpu_energy_joules are extracted from protobuf."""
        msg = self._make_message(
            power_watts=200.0,
            energy_joules=100.0,
            temperature_celsius=65.0,
            gpu_memory_usage_mb=4096.0,
            cpu_memory_usage_mb=8192.0,
            cpu_power_watts=85.0,
            cpu_energy_joules=42.5,
            platform="linux",
            timestamp_nanos=1000000000,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts == 85.0
        assert reading.cpu_energy_joules == 42.5

    def test_cpu_fields_none_when_absent(self) -> None:
        """cpu_power_watts and cpu_energy_joules are None when not in message."""
        msg = self._make_message(
            power_watts=200.0,
            energy_joules=100.0,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts is None
        assert reading.cpu_energy_joules is None

    def test_negative_cpu_values_become_none(self) -> None:
        """Negative values (sentinel for unavailable) become None via _safe_float."""
        msg = self._make_message(
            cpu_power_watts=-1.0,
            cpu_energy_joules=-1.0,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.cpu_power_watts is None
        assert reading.cpu_energy_joules is None

    def test_all_standard_fields_extracted(self) -> None:
        """All standard telemetry fields are properly extracted."""
        msg = self._make_message(
            power_watts=250.0,
            energy_joules=500.0,
            temperature_celsius=72.0,
            gpu_memory_usage_mb=6144.0,
            cpu_memory_usage_mb=16384.0,
            cpu_power_watts=95.0,
            cpu_energy_joules=200.0,
            platform="linux",
            timestamp_nanos=2000000000,
        )

        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        reading = collector._convert(msg)

        assert reading.power_watts == 250.0
        assert reading.energy_joules == 500.0
        assert reading.temperature_celsius == 72.0
        assert reading.gpu_memory_usage_mb == 6144.0
        assert reading.cpu_memory_usage_mb == 16384.0
        assert reading.cpu_power_watts == 95.0
        assert reading.cpu_energy_joules == 200.0
        assert reading.platform == "linux"
        assert reading.timestamp_nanos == 2000000000

"""Tests for telemetry/collector.py — EnergyMonitorCollector._convert."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipw.telemetry.collector import EnergyMonitorCollector
from ipw.telemetry.proto import get_stub_bundle

# repo root / energy-monitor/proto/energy.proto, from src/ipw/tests/telemetry/
_PROTO_PATH = (
    Path(__file__).resolve().parents[5] / "energy-monitor" / "proto" / "energy.proto"
)


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


class TestAppleNeuralEngineFields:
    """Regression: these proto fields existed but were never mapped.

    The macOS collector samples ``ane_power``/``ane_energy`` via powermetrics and
    the proto carries them, but ``_convert`` dropped them -- so
    ``EnergyMetrics.ane_*`` was always None in real runs while unit tests passed
    on a hand-populated fixture. Any ANE-resident model (Apple Foundation Models)
    appeared to consume no energy at all.
    """

    def _make_message(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def _convert(self, **kwargs):
        collector = EnergyMonitorCollector.__new__(EnergyMonitorCollector)
        return collector._convert(self._make_message(**kwargs))

    def test_ane_fields_extracted(self) -> None:
        reading = self._convert(
            power_watts=12.0,
            energy_joules=30.0,
            cpu_power_watts=4.0,
            cpu_energy_joules=9.0,
            ane_power_watts=6.5,
            ane_energy_joules=15.25,
            platform="macos",
        )

        assert reading.ane_power_watts == 6.5
        assert reading.ane_energy_joules == 15.25

    def test_ane_unavailable_sentinel_becomes_none(self) -> None:
        # The Rust service emits -1.0 when a rail is unreadable.
        reading = self._convert(ane_power_watts=-1.0, ane_energy_joules=-1.0)

        assert reading.ane_power_watts is None
        assert reading.ane_energy_joules is None

    def test_ane_fields_none_when_absent(self) -> None:
        # Non-Apple platforms omit them entirely.
        reading = self._convert(power_watts=200.0, energy_joules=100.0)

        assert reading.ane_power_watts is None
        assert reading.ane_energy_joules is None

    def test_gpu_utilization_and_total_memory_extracted(self) -> None:
        """Dropped alongside the ANE fields; these populate on NVIDIA hosts."""
        reading = self._convert(
            gpu_memory_total_mb=81920.0,
            gpu_compute_utilization_pct=88.0,
            gpu_tensor_core_utilization_pct=41.0,
            gpu_memory_bandwidth_utilization_pct=55.0,
            platform="linux",
        )

        assert reading.gpu_memory_total_mb == 81920.0
        assert reading.gpu_compute_utilization_pct == 88.0
        assert reading.gpu_tensor_core_utilization_pct == 41.0
        assert reading.gpu_memory_bandwidth_utilization_pct == 55.0


class TestProtoDescriptorCoverage:
    """The dynamic descriptor must declare every field the service sends.

    A descriptor missing a field still deserializes the stream -- the tag lands
    in unknown fields -- so ``_convert``'s ``getattr(msg, name, None)`` silently
    read back None. The CPU, ANE and GPU-utilization rails were dark end to end
    while the ``_convert`` tests above passed on hand-built namespaces.
    """

    def _proto_field_names(self) -> set[str]:
        source = _PROTO_PATH.read_text()
        body = source.split("message TelemetryReading {", 1)[1].split("\n}", 1)[0]
        return set(re.findall(r"^\s*[\w.]+\s+(\w+)\s*=\s*\d+;", body, re.MULTILINE))

    def test_descriptor_declares_every_proto_field(self) -> None:
        if not _PROTO_PATH.exists():
            pytest.skip("energy.proto not available (installed package)")

        declared = {
            field.name
            for field in get_stub_bundle().TelemetryReadingCls.DESCRIPTOR.fields
        }

        assert self._proto_field_names() - declared == set()

    def test_rail_fields_survive_a_serialization_round_trip(self) -> None:
        message_cls = get_stub_bundle().TelemetryReadingCls
        message = message_cls(
            power_watts=1.0,
            cpu_power_watts=8.5,
            cpu_energy_joules=42.0,
            ane_power_watts=6.5,
            ane_energy_joules=15.25,
            platform="macos",
        )

        decoded = message_cls.FromString(message.SerializeToString())
        reading = EnergyMonitorCollector.__new__(EnergyMonitorCollector)._convert(
            decoded
        )

        assert reading.cpu_power_watts == 8.5
        assert reading.cpu_energy_joules == 42.0
        assert reading.ane_power_watts == 6.5
        assert reading.ane_energy_joules == 15.25

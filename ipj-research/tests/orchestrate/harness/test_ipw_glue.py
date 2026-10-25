from __future__ import annotations

from dataclasses import dataclass

from ipj.metrics import MEASURED_DEDICATED_JOULES_PROVENANCE
from ipj.orchestrate.harness.ipw_glue import _ShellBackend, telemetry_energy_probe


class FakeResult:
    """Mimics an IPW MCPToolResult without importing real IPW."""

    def __init__(self, content, *, metadata, cost_usd=0.0, latency_seconds=0.1):
        self.content = content
        self.metadata = metadata
        self.cost_usd = cost_usd
        self.latency_seconds = latency_seconds


class FakeServer:
    def __init__(self, result):
        self._result = result
        self.seen = None

    def execute(self, prompt, **params):
        self.seen = prompt
        return self._result


def test_shell_backend_maps_content_and_derives_success_true():
    res = FakeResult("Exit code: 0\n\nSTDOUT:\nhi", metadata={"tool": "bash", "return_code": 0})
    backend = _ShellBackend(FakeServer(res))
    out = backend.run(command="echo hi")
    assert out.content == "Exit code: 0\n\nSTDOUT:\nhi"
    assert out.success is True
    assert out.latency_s == 0.1


def test_shell_backend_derives_success_false_on_nonzero_exit():
    res = FakeResult("Exit code: 2", metadata={"tool": "bash", "return_code": 2})
    out = _ShellBackend(FakeServer(res)).run(command="false")
    assert out.success is False


def test_shell_backend_derives_success_false_on_error_signal():
    res = FakeResult("Error: timed out", metadata={"tool": "bash", "error": "timeout"})
    out = _ShellBackend(FakeServer(res)).run(command="sleep 999")
    assert out.success is False


def test_shell_backend_defaults_success_true_without_signal():
    res = FakeResult("ok", metadata={})
    out = _ShellBackend(FakeServer(res)).run(command="noop")
    assert out.success is True


@dataclass
class _Reading:
    energy_joules: float | None


@dataclass
class _Sample:
    timestamp: float
    reading: _Reading


class _TelemetrySession:
    def __init__(self, samples):
        self.samples = samples

    def window(self, start, end):
        return [sample for sample in self.samples if start <= sample.timestamp <= end]


def test_telemetry_probe_differences_cumulative_epoch_readings():
    session = _TelemetrySession(
        [
            _Sample(1_000.0, _Reading(10.0)),
            _Sample(1_000.05, _Reading(12.0)),
            _Sample(1_000.1, _Reading(15.0)),
        ]
    )
    measurement = telemetry_energy_probe(
        session,
        provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
        accelerator_device_coverage_complete=True,
    )(1_000.0, 1_000.1)
    assert measurement.joules == 5.0
    assert measurement.complete is True


def test_telemetry_probe_fails_closed_on_missing_or_reset_counter():
    missing = _TelemetrySession([_Sample(1.0, _Reading(None)), _Sample(1.1, _Reading(2.0))])
    reset = _TelemetrySession([_Sample(1.0, _Reading(10.0)), _Sample(1.1, _Reading(2.0))])
    assert telemetry_energy_probe(missing)(1.0, 1.1).joules is None
    assert telemetry_energy_probe(reset)(1.0, 1.1).joules is None


def test_telemetry_probe_fails_closed_on_internal_counter_reset():
    session = _TelemetrySession(
        [
            _Sample(1.0, _Reading(10.0)),
            _Sample(1.05, _Reading(2.0)),
            _Sample(1.1, _Reading(12.0)),
        ]
    )
    assert telemetry_energy_probe(session)(1.0, 1.1).joules is None


def test_telemetry_probe_needs_explicit_provenance_for_completeness():
    session = _TelemetrySession(
        [_Sample(1.0, _Reading(0.0)), _Sample(1.1, _Reading(0.0))]
    )
    measurement = telemetry_energy_probe(session)(1.0, 1.1)
    assert measurement.joules == 0.0
    assert measurement.complete is False


def test_telemetry_probe_needs_explicit_device_coverage_for_completeness():
    session = _TelemetrySession(
        [_Sample(1.0, _Reading(0.0)), _Sample(1.1, _Reading(0.0))]
    )
    measurement = telemetry_energy_probe(
        session,
        provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
    )(1.0, 1.1)

    assert measurement.joules == 0.0
    assert measurement.accelerator_device_coverage_complete is False
    assert measurement.complete is False

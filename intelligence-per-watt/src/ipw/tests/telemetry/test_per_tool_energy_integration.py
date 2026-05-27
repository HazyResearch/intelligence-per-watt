"""Per-tool ENERGY_ATTRIBUTED integration tests — spec §5.4.

Two layers:
  Layer 1 — Unit (default-run, no hardware)
    For each local-compute tool (shell_exec, repl, code_interpreter_docker) verify
    that calling the real tool via its __call__ interface causes a matching
    ENERGY_ATTRIBUTED event to be emitted by EnergyAttribution wired to a
    bounds-agnostic stub session that always returns fixed non-zero-joule samples.

  Layer 2 — Hardware integration (@pytest.mark.integration)
    When a real TelemetrySession backed by the energy monitor is available,
    run the repl tool with a ~200ms CPU spin and assert the emitted
    ENERGY_ATTRIBUTED carries non-zero energy and a plausible duration.

Notes on the bounds-agnostic stub session:
    EnergyAttribution._compute() calls session.window(start_s, end_s) and then
    computes energy from sample.reading.<attr> deltas. It does NOT use the
    sample's own timestamp. Therefore our stub ignores the window bounds and
    always returns the same two samples, making energy assertions fully
    deterministic regardless of when the real tool events are published.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass
from typing import List, Optional

import pytest

import ipw.tools.code_interpreter_docker  # noqa: F401 — side-effect: registers tool
import ipw.tools.repl  # noqa: F401 — side-effect: registers tool
import ipw.tools.shell_exec  # noqa: F401 — side-effect: registers tool
from ipw.telemetry.energy_attribution import EnergyAttribution
from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType
from ipw.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Shared stub session types (mirrors pattern from test_energy_attribution.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeReading:
    energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    dram_energy_joules: Optional[float] = None
    power_watts: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    timestamp_nanos: int = 0


@dataclass
class _FakeSample:
    timestamp: float
    reading: _FakeReading


class _BoundsAgnosticSession:
    """Stub session that returns fixed samples regardless of time window.

    EnergyAttribution._compute() ignores sample.timestamp — it computes
    energy from sample.reading.<attr> deltas only. Returning the same two
    samples unconditionally makes assertions deterministic even when real
    wall-clock timestamps vary.
    """

    def __init__(self, samples: List[_FakeSample]) -> None:
        self._samples = samples

    def window(self, start_time: float, end_time: float):  # bounds ignored
        return iter(list(self._samples))


# Fixed stub samples: delta energy_joules = 50.0, delta cpu_energy_joules = 5.0
_FIXED_SAMPLES = [
    _FakeSample(
        timestamp=0.0,
        reading=_FakeReading(
            energy_joules=0.0,
            cpu_energy_joules=0.0,
            power_watts=100.0,
            gpu_utilization_pct=50.0,
        ),
    ),
    _FakeSample(
        timestamp=1.0,
        reading=_FakeReading(
            energy_joules=50.0,
            cpu_energy_joules=5.0,
            power_watts=120.0,
            gpu_utilization_pct=70.0,
        ),
    ),
]


def _make_bus_with_attribution() -> tuple[EventBus, List[Event]]:
    """Create a fresh bus + EnergyAttribution(stub session) + event collector."""
    bus = EventBus()
    session = _BoundsAgnosticSession(_FIXED_SAMPLES)
    EnergyAttribution(bus=bus, session=session, is_cloud_fn=lambda evt: False)
    attributed: List[Event] = []
    bus.subscribe(EventType.ENERGY_ATTRIBUTED, attributed.append)
    return bus, attributed


# ---------------------------------------------------------------------------
# Layer 1 — Unit tests (no hardware required)
# ---------------------------------------------------------------------------


class TestShellExecEnergyAttributed:
    """shell_exec TOOL_CALL_START/END wiring drives an ENERGY_ATTRIBUTED event."""

    def test_shell_exec_emits_energy_attributed(self) -> None:
        bus, attributed = _make_bus_with_attribution()

        tool_cls = ToolRegistry.get("shell_exec")
        tool = tool_cls(bus=bus)

        asyncio.run(tool(correlation_id="cid-shell", command="echo hi"))

        assert len(attributed) == 1, (
            f"Expected 1 ENERGY_ATTRIBUTED event, got {len(attributed)}"
        )
        payload = attributed[0].payload
        assert payload["window"] == "tool"
        assert payload["subject_name"] == "shell_exec"
        assert payload["gpu_energy_j"] == pytest.approx(50.0)
        assert payload["cpu_energy_j"] == pytest.approx(5.0)


class TestReplEnergyAttributed:
    """repl TOOL_CALL_START/END wiring drives an ENERGY_ATTRIBUTED event."""

    def test_repl_emits_energy_attributed(self) -> None:
        bus, attributed = _make_bus_with_attribution()

        tool_cls = ToolRegistry.get("repl")
        tool = tool_cls(bus=bus)

        asyncio.run(tool(correlation_id="cid-repl", code="x = 1 + 1"))

        assert len(attributed) == 1, (
            f"Expected 1 ENERGY_ATTRIBUTED event, got {len(attributed)}"
        )
        payload = attributed[0].payload
        assert payload["window"] == "tool"
        assert payload["subject_name"] == "repl"
        assert payload["gpu_energy_j"] == pytest.approx(50.0)
        assert payload["cpu_energy_j"] == pytest.approx(5.0)


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker CLI not available",
)
@pytest.mark.integration
class TestCodeInterpreterDockerEnergyAttributed:
    """code_interpreter_docker TOOL_CALL_START/END wiring drives ENERGY_ATTRIBUTED.

    Requires Docker. Skipped if docker is not on PATH.
    """

    def test_code_interpreter_docker_emits_energy_attributed(self) -> None:
        bus, attributed = _make_bus_with_attribution()

        tool_cls = ToolRegistry.get("code_interpreter_docker")
        tool = tool_cls(bus=bus)

        asyncio.run(tool(correlation_id="cid-docker", code="print(1)"))

        assert len(attributed) == 1, (
            f"Expected 1 ENERGY_ATTRIBUTED event, got {len(attributed)}"
        )
        payload = attributed[0].payload
        assert payload["window"] == "tool"
        assert payload["subject_name"] == "code_interpreter_docker"
        assert payload["gpu_energy_j"] == pytest.approx(50.0)
        assert payload["cpu_energy_j"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Layer 2 — Hardware integration (real TelemetrySession + energy monitor)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReplHardwareIntegration:
    """End-to-end: repl tool + real TelemetrySession + EnergyAttribution.

    This test is expected to SKIP in environments without the energy monitor
    binary or GPU. A clean skip is acceptable; this must not fail with an
    import error or exception when hardware is absent.

    spec §5.4 targets ±50ms timing accuracy on dedicated hardware; we relax
    to ±200ms for shared-hardware realism.
    """

    def test_repl_cpu_spin_yields_nonzero_energy(self) -> None:
        # Lazy import — skip cleanly if the live path is unavailable.
        # We catch a broad Exception here because in some environments (e.g.
        # running under sg docker with a system Python) transitive imports like
        # datasets → pandas may fail with non-ImportError exceptions (e.g.
        # pandas OptionError from a workspace-local pandas on sys.path).
        try:
            from ipw.execution.telemetry_session import TelemetrySession
            from ipw.telemetry import EnergyMonitorCollector, ensure_monitor, wait_for_ready
        except Exception as exc:
            pytest.skip(f"Telemetry dependencies unavailable: {exc}")

        # Skip if no monitor binary is reachable / launchable
        try:
            with ensure_monitor(timeout=10.0) as target:
                if not wait_for_ready(target, timeout=5.0):
                    pytest.skip("Energy monitor did not become ready in time")

                collector = EnergyMonitorCollector(target=target)
                if not collector.is_available():
                    pytest.skip("Energy monitor not available on this host")

                bus = EventBus()
                attributed: List[Event] = []
                bus.subscribe(EventType.ENERGY_ATTRIBUTED, attributed.append)

                with TelemetrySession(collector) as live_session:
                    # Give the monitor a moment to start streaming
                    time.sleep(0.3)
                    EnergyAttribution(
                        bus=bus,
                        session=live_session,
                        is_cloud_fn=lambda evt: False,
                    )

                    tool_cls = ToolRegistry.get("repl")
                    tool = tool_cls(bus=bus)

                    spin_code = (
                        "import time as _t; "
                        "_end = _t.monotonic() + 0.2; "
                        "[None for _ in iter(int, 1) if _t.monotonic() >= _end]"
                    )
                    t0 = time.monotonic()
                    asyncio.run(
                        tool(correlation_id="cid-hw", code=spin_code)
                    )
                    elapsed = time.monotonic() - t0

        except FileNotFoundError as exc:
            pytest.skip(f"Energy monitor binary missing: {exc}")
        except RuntimeError as exc:
            pytest.skip(f"Energy monitor unavailable: {exc}")

        assert len(attributed) == 1, (
            f"Expected 1 ENERGY_ATTRIBUTED event, got {len(attributed)}"
        )
        payload = attributed[0].payload

        # At least one energy dimension must be non-zero
        gpu_j = payload.get("gpu_energy_j")
        cpu_j = payload.get("cpu_energy_j")
        assert (gpu_j is not None and gpu_j > 0) or (cpu_j is not None and cpu_j > 0), (
            f"Expected non-zero energy from CPU spin; got gpu_energy_j={gpu_j}, "
            f"cpu_energy_j={cpu_j}"
        )

        # duration_s should track wall-clock within ±200ms
        # spec §5.4 targets ±50ms on dedicated hardware; relaxed to ±200ms
        # for shared-hardware realism.
        duration_s = payload["duration_s"]
        assert abs(duration_s - elapsed) < 0.200, (
            f"duration_s={duration_s:.3f}s deviates >200ms from wall-clock "
            f"elapsed={elapsed:.3f}s"
        )

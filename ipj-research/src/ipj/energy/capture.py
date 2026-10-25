"""Capture GPU power for the wall-clock window of an external run, via IPW."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .integrate import PowerSample, energy_joules

try:
    from ipw.execution.telemetry_session import TelemetrySession
    from ipw.telemetry import EnergyMonitorCollector

    _IPW_OK = True
except Exception:  # noqa: BLE001 - ipw's agentic stack may not be installed in the ipj venv
    # Degrade to a no-op capture; energy is then captured externally (see runs/orch/*power*).
    TelemetrySession = None  # type: ignore[assignment,misc]
    EnergyMonitorCollector = None  # type: ignore[assignment,misc]
    _IPW_OK = False


ENERGY_MONITOR_DEFAULT_SAMPLE_INTERVAL_S = 0.05
DEFAULT_BOUNDARY_GRACE_INTERVALS = 3.0
DEFAULT_MAX_SAMPLE_GAP_INTERVALS = 3.0


@dataclass(frozen=True, slots=True)
class EnergyCoveragePolicy:
    """Eligibility thresholds for integrating a run's power samples.

    The policy is expressed in seconds so callers using a monitor configured
    with a non-default collection interval can supply matching thresholds.
    Boundary gaps allow normal sampling jitter; an internal gap beyond the
    configured maximum invalidates the complete run measurement.
    """

    max_boundary_gap_s: float
    max_sample_gap_s: float
    minimum_distinct_samples: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_boundary_gap_s) or self.max_boundary_gap_s < 0:
            raise ValueError("max_boundary_gap_s must be finite and non-negative")
        if not math.isfinite(self.max_sample_gap_s) or self.max_sample_gap_s <= 0:
            raise ValueError("max_sample_gap_s must be finite and positive")
        if (
            isinstance(self.minimum_distinct_samples, bool)
            or not isinstance(self.minimum_distinct_samples, int)
            or self.minimum_distinct_samples < 2
        ):
            raise ValueError("minimum_distinct_samples must be an integer of at least two")

    def covers(self, samples: Iterable[PowerSample], start: float, end: float) -> bool:
        """Return whether samples continuously cover the complete run window."""
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            return False

        points = tuple(samples)
        if any(
            not math.isfinite(point.t)
            or not math.isfinite(point.watts)
            or point.watts < 0
            or point.t < start
            or point.t > end
            for point in points
        ):
            return False

        times = sorted({point.t for point in points})
        if len(times) < self.minimum_distinct_samples:
            return False
        if times[0] - start > self.max_boundary_gap_s:
            return False
        if end - times[-1] > self.max_boundary_gap_s:
            return False
        return all(
            later - earlier <= self.max_sample_gap_s
            for earlier, later in zip(times, times[1:])
        )


DEFAULT_ENERGY_COVERAGE_POLICY = EnergyCoveragePolicy(
    max_boundary_gap_s=(
        ENERGY_MONITOR_DEFAULT_SAMPLE_INTERVAL_S * DEFAULT_BOUNDARY_GRACE_INTERVALS
    ),
    max_sample_gap_s=(
        ENERGY_MONITOR_DEFAULT_SAMPLE_INTERVAL_S * DEFAULT_MAX_SAMPLE_GAP_INTERVALS
    ),
)


def samples_from_session(session, start: float, end: float) -> list[PowerSample]:
    """Map IPW TelemetrySamples in [start, end] to PowerSamples.

    Drops unavailable readings: ``None`` and the energy monitor's negative
    sentinel (the Rust service publishes -1 for metrics it can't read).
    """
    out: list[PowerSample] = []
    for s in session.window(start, end):
        watts = s.reading.power_watts
        if watts is None:
            continue
        try:
            timestamp = float(s.timestamp)
            watts = float(watts)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(timestamp)
            and math.isfinite(watts)
            and start <= timestamp <= end
            and watts >= 0
        ):
            out.append(PowerSample(timestamp, watts))
    return out


class RunEnergyCapture:
    """Start the IPW energy monitor, retain every sample, expose windowed energy."""

    def __init__(
        self,
        target: str | None = None,
        *,
        coverage_policy: EnergyCoveragePolicy = DEFAULT_ENERGY_COVERAGE_POLICY,
    ) -> None:
        self._coverage_policy = coverage_policy
        if not _IPW_OK:  # ipw telemetry unavailable -> no-op; energy captured externally
            self._session = None
            self._controller_clock_aligned = False
            return
        # EnergyMonitorCollector resolves None via env/visible-GPU default; pass it straight through.
        self._collector = EnergyMonitorCollector(target=target)
        self._controller_clock_aligned = self._collector.shares_controller_clock
        # buffer_seconds=None + max_samples=None → keep every sample for the whole run.
        self._session = TelemetrySession(self._collector, buffer_seconds=None, max_samples=None)

    def __enter__(self) -> "RunEnergyCapture":
        if self._session is not None:
            self._session.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._session is not None:
            self._session.__exit__(*exc)

    def power_samples(self, start: float, end: float) -> list[PowerSample]:
        if self._session is None:
            return []
        return samples_from_session(self._session, start, end)

    def joules(self, start: float, end: float) -> float | None:
        """Return measured energy, or ``None`` when it cannot be measured.

        Unknown energy must not be represented as zero. The run window must have
        enough distinct samples, both boundaries must be covered, and no internal
        gap may exceed :attr:`EnergyCoveragePolicy.max_sample_gap_s`. Valid
        zero-watt samples still integrate to the legitimate value ``0.0``.
        """
        if self._session is None:
            return None
        if not self._controller_clock_aligned:
            # Remote agent timestamps are not comparable to controller run
            # boundaries until an offset/RTT uncertainty handshake is added.
            return None
        samples = self.power_samples(start, end)
        if not self._coverage_policy.covers(samples, start, end):
            return None
        return energy_joules(samples, start, end)

    def accelerator_device_coverage_complete(
        self,
        start: float,
        end: float,
    ) -> bool:
        """Whether every expected accelerator sensor covered the run window.

        The current telemetry wire format has only an aggregate reading and
        cannot prove expected-versus-observed device coverage. Keep measured
        Joules available for descriptive use, but fail rigorous accounting
        closed until a topology-aware agent capability is negotiated.
        """
        del start, end
        return False

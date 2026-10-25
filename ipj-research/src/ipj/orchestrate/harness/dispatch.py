"""Worker dispatch: call_worker (subtask, keep control) vs delegate_task (hand off the whole task)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Protocol

from ipj.metrics import (
    UNKNOWN_JOULES_PROVENANCE,
    accelerator_joules_are_complete,
    resolve_joules_provenance,
)

from .workers import WorkerBinding, WorkerPool


@dataclass(slots=True, frozen=True)
class AcceleratorEnergyMeasurement:
    """One accelerator-energy observation and its headline-eligibility metadata.

    A bare numeric probe result remains accepted for two-release compatibility,
    but it has unknown provenance and is therefore never considered complete.
    New probes should return this type and explicitly declare provenance.
    """

    joules: float | None
    provenance: str = UNKNOWN_JOULES_PROVENANCE
    complete: bool = False
    accelerator_device_coverage_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be a boolean")
        if not isinstance(self.accelerator_device_coverage_complete, bool):
            raise ValueError("accelerator_device_coverage_complete must be a boolean")
        if not isinstance(self.provenance, str):
            raise ValueError("provenance must be a string")
        if self.joules is not None:
            try:
                valid_joules = (
                    not isinstance(self.joules, bool)
                    and isinstance(self.joules, Real)
                    and math.isfinite(float(self.joules))
                    and self.joules >= 0
                )
            except (OverflowError, TypeError, ValueError):
                valid_joules = False
            if not valid_joules:
                raise ValueError("joules must be finite and non-negative when present")
        object.__setattr__(
            self,
            "provenance",
            resolve_joules_provenance(self.joules, self.provenance),
        )
        if self.complete and not accelerator_joules_are_complete(
            self.joules,
            self.provenance,
            self.accelerator_device_coverage_complete,
        ):
            raise ValueError(
                "complete accelerator energy requires finite non-negative Joules and "
                "measured_dedicated provenance plus complete device coverage"
            )


def normalize_energy_measurement(
    value: AcceleratorEnergyMeasurement | float | None,
) -> AcceleratorEnergyMeasurement:
    """Normalize legacy numeric probes without inventing provenance or completeness."""
    if isinstance(value, AcceleratorEnergyMeasurement):
        return value
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return AcceleratorEnergyMeasurement(None)
    try:
        return AcceleratorEnergyMeasurement(float(value))
    except (OverflowError, ValueError):
        return AcceleratorEnergyMeasurement(None)


@dataclass(slots=True)
class WorkerResult:
    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    dollars: float | None = None
    latency_s: float | None = None
    gpu_energy_j: float | None = None
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


class WorkerServer(Protocol):
    def execute(self, prompt: str, **params: Any) -> WorkerResult: ...


ServerFactory = Callable[[WorkerBinding], WorkerServer]
EnergyProbe = Callable[
    [float, float],
    "AcceleratorEnergyMeasurement | float | None",
]
DelegateRunner = Callable[[WorkerBinding, str], WorkerResult]


class WorkerDispatch:
    """Routes worker actions to provider backends, attributing per-call latency and (local-only) GPU energy."""

    def __init__(
        self,
        pool: WorkerPool,
        server_factory: ServerFactory,
        *,
        energy_probe: EnergyProbe | None = None,
        delegate_runner: DelegateRunner | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.pool = pool
        self.factory = server_factory
        self.energy_probe = energy_probe
        self.delegate_runner = delegate_runner
        self.clock = clock

    def _attribute_energy(self, binding: WorkerBinding, result: WorkerResult, t0: float, t1: float) -> WorkerResult:
        if binding.is_local:
            if self.energy_probe is not None:
                measurement = normalize_energy_measurement(self.energy_probe(t0, t1))
            else:
                try:
                    measurement = AcceleratorEnergyMeasurement(
                        result.gpu_energy_j,
                        provenance=result.gpu_energy_provenance,
                        complete=result.gpu_energy_complete,
                        accelerator_device_coverage_complete=(
                            result.accelerator_device_coverage_complete
                        ),
                    )
                except ValueError:
                    measurement = AcceleratorEnergyMeasurement(None)
            result.gpu_energy_j = measurement.joules
            result.gpu_energy_provenance = measurement.provenance
            result.gpu_energy_complete = measurement.complete
            result.accelerator_device_coverage_complete = (
                measurement.accelerator_device_coverage_complete
            )
        else:
            result.gpu_energy_j = None
            result.gpu_energy_provenance = UNKNOWN_JOULES_PROVENANCE
            result.gpu_energy_complete = False
            result.accelerator_device_coverage_complete = False
        if result.latency_s is None:
            result.latency_s = t1 - t0
        return result

    def call_worker(self, alias: str, sub_prompt: str, context: str | None) -> WorkerResult:
        """Run a self-contained subtask on a worker; the orchestrator keeps control of the task."""
        binding = self.pool.binding(alias)
        server = self.factory(binding)
        prompt = sub_prompt if not context else f"{context}\n\n{sub_prompt}"
        t0 = self.clock()
        result = server.execute(prompt)
        t1 = self.clock()
        return self._attribute_energy(binding, result, t0, t1)

    def delegate_task(self, alias: str, task_prompt: str) -> WorkerResult:
        """Hand the whole task to a worker for a clean restart (no live state handoff)."""
        binding = self.pool.binding(alias)
        t0 = self.clock()
        if self.delegate_runner is not None:
            result = self.delegate_runner(binding, task_prompt)
        else:
            result = self.factory(binding).execute(task_prompt)
        t1 = self.clock()
        return self._attribute_energy(binding, result, t0, t1)

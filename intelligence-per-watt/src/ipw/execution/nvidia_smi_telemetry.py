"""Per-GPU telemetry sampling backed by ``nvidia-smi``."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Sequence

from ..core.types import GpuInfo, TelemetryReading
from .telemetry_session import TelemetrySample


@dataclass(frozen=True)
class _GpuSample:
    gpu_id: int
    power_watts: float
    memory_used_mb: float | None
    memory_total_mb: float | None
    utilization_pct: float | None


class NvidiaSmiTelemetrySession(AbstractContextManager["NvidiaSmiTelemetrySession"]):
    """Capture cumulative energy for selected NVIDIA GPU ids.

    The bundled energy monitor can expose multi-GPU hosts as one aggregate
    device. This session is intentionally narrow: it polls ``nvidia-smi`` for
    explicit GPU ids and integrates power over wall-clock time so concurrent
    one-GPU benchmark shards can keep independent per-prompt energy windows.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        *,
        interval_seconds: float = 0.2,
        buffer_seconds: float = 30.0,
        max_samples: int = 10_000,
    ) -> None:
        if not gpu_ids:
            raise ValueError("At least one GPU id is required")
        self._gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._buffer_seconds = float(buffer_seconds)
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_timestamp: float | None = None
        self._last_power_watts: dict[int, float] = {}
        self._energy_joules: dict[int, float] = {gpu_id: 0.0 for gpu_id in self._gpu_ids}

    def __enter__(self) -> "NvidiaSmiTelemetrySession":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                # A transient nvidia-smi failure should not kill the benchmark.
                pass
            self._stop_event.wait(self._interval_seconds)

    def _sample_once(self) -> None:
        now = time.time()
        gpu_samples = _query_nvidia_smi(self._gpu_ids)
        if not gpu_samples:
            return

        current_power = {sample.gpu_id: sample.power_watts for sample in gpu_samples}
        if self._last_timestamp is not None:
            dt = max(0.0, now - self._last_timestamp)
            for gpu_id, power in current_power.items():
                previous = self._last_power_watts.get(gpu_id, power)
                self._energy_joules[gpu_id] += ((previous + power) / 2.0) * dt

        self._last_timestamp = now
        self._last_power_watts = current_power

        total_power = sum(current_power.values())
        total_energy = sum(self._energy_joules.get(gpu_id, 0.0) for gpu_id in self._gpu_ids)
        memory_used = _sum_optional(sample.memory_used_mb for sample in gpu_samples)
        memory_total = _sum_optional(sample.memory_total_mb for sample in gpu_samples)
        utilization = _mean_optional(sample.utilization_pct for sample in gpu_samples)

        device_id = self._gpu_ids[0] if len(self._gpu_ids) == 1 else -1
        reading = TelemetryReading(
            power_watts=total_power,
            energy_joules=total_energy,
            gpu_memory_usage_mb=memory_used,
            gpu_memory_total_mb=memory_total,
            gpu_compute_utilization_pct=utilization,
            timestamp_nanos=int(now * 1_000_000_000),
            gpu_info=GpuInfo(
                name=f"NVIDIA GPU {device_id}" if device_id >= 0 else "NVIDIA GPU group",
                vendor="NVIDIA",
                device_id=device_id,
                device_type="GPU",
                backend="nvidia-smi",
            ),
        )
        sample = TelemetrySample(timestamp=now, reading=reading)
        with self._lock:
            self._samples.append(sample)
            self._trim(now)

    def _trim(self, current_time: float) -> None:
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        with self._lock:
            return list(self._samples)

    def window(self, start_time: float, end_time: float) -> Iterator[TelemetrySample]:
        with self._lock:
            samples = list(self._samples)
        for sample in samples:
            if start_time <= sample.timestamp <= end_time:
                yield sample


def _query_nvidia_smi(gpu_ids: Sequence[int]) -> list[_GpuSample]:
    gpu_arg = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_arg}",
            "--query-gpu=index,power.draw,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5.0,
    )

    samples: list[_GpuSample] = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue
        gpu_id = _to_int(fields[0])
        power = _to_float(fields[1])
        if gpu_id is None or power is None:
            continue
        samples.append(
            _GpuSample(
                gpu_id=gpu_id,
                power_watts=power,
                memory_used_mb=_to_float(fields[2]),
                memory_total_mb=_to_float(fields[3]),
                utilization_pct=_to_float(fields[4]),
            )
        )
    return samples


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_optional(values) -> float | None:
    filtered = [value for value in values if value is not None]
    return sum(filtered) if filtered else None


def _mean_optional(values) -> float | None:
    filtered = [value for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


__all__ = ["NvidiaSmiTelemetrySession"]

"""Hardware collector implementation backed by the energy monitor."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

import grpc

from trafficbench.core.types import GpuInfo, SystemInfo, TelemetryReading

from .launcher import (DEFAULT_TARGET, ensure_monitor, normalize_target,
                       wait_for_ready)
from .proto import get_stub_bundle


class EnergyMonitorCollector:
    collector_name = "Energy Monitor"

    def __init__(
        self,
        target: str = DEFAULT_TARGET,
        *,
        channel_options: Optional[Tuple[Tuple[str, str], ...]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._target = normalize_target(target or DEFAULT_TARGET)
        self._channel_options = channel_options or ()
        self._timeout = timeout
        self._bundle = get_stub_bundle()

    @contextmanager
    def start(self):
        with ensure_monitor(self._target, timeout=self._timeout, launch=True):
            yield

    @classmethod
    def is_available(cls) -> bool:
        return wait_for_ready(DEFAULT_TARGET, timeout=1.0)

    def stream_readings(self) -> Iterable[TelemetryReading]:
        channel = grpc.insecure_channel(self._target, options=self._channel_options)
        stub = self._bundle.stub_factory(channel)
        stream = stub.StreamTelemetry(self._bundle.StreamRequestCls())
        try:
            for raw in stream:
                yield self._convert(raw)
        except grpc.RpcError as exc:
            code_name = "UNKNOWN"
            details = ""
            if isinstance(exc, grpc.Call):
                status = exc.code()
                if status is not None:
                    code_name = getattr(status, "name", str(status))
                detail_text = exc.details()
                if detail_text:
                    details = detail_text
            message = f"Energy monitor stream failed: {code_name}"
            if details:
                message = f"{message} {details}"
            raise RuntimeError(message) from exc
        finally:
            channel.close()

    def _convert(self, message) -> TelemetryReading:
        system_info = getattr(message, "system_info", None)
        gpu_info = getattr(message, "gpu_info", None)

        system = None
        if system_info is not None:
            system = SystemInfo(
                os_name=getattr(system_info, "os_name", ""),
                os_version=getattr(system_info, "os_version", ""),
                kernel_version=getattr(system_info, "kernel_version", ""),
                host_name=getattr(system_info, "host_name", ""),
                cpu_count=getattr(system_info, "cpu_count", 0),
                cpu_brand=getattr(system_info, "cpu_brand", ""),
            )

        gpu = None
        if gpu_info is not None:
            gpu = GpuInfo(
                name=getattr(gpu_info, "name", ""),
                vendor=getattr(gpu_info, "vendor", ""),
                device_id=getattr(gpu_info, "device_id", 0),
                device_type=getattr(gpu_info, "device_type", ""),
                backend=getattr(gpu_info, "backend", ""),
            )

        return TelemetryReading(
            power_watts=_safe_float(getattr(message, "power_watts", None)),
            energy_joules=_safe_float(getattr(message, "energy_joules", None)),
            temperature_celsius=_safe_float(
                getattr(message, "temperature_celsius", None)
            ),
            gpu_memory_usage_mb=_safe_float(
                getattr(message, "gpu_memory_usage_mb", None)
            ),
            cpu_memory_usage_mb=_safe_float(
                getattr(message, "cpu_memory_usage_mb", None)
            ),
            platform=getattr(message, "platform", None),
            timestamp_nanos=getattr(message, "timestamp_nanos", None),
            system_info=system,
            gpu_info=gpu,
        )


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


__all__ = ["EnergyMonitorCollector"]

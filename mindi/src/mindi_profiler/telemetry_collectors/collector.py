from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from ..core.collector import HardwareCollector
from ..core.types import GpuInfo, SystemInfo, TelemetryReading

_DEFAULT_TARGET = "127.0.0.1:50052"


@dataclass
class _StubBundle:
    stub_factory: type
    TelemetryReadingCls: type
    StreamRequestCls: type
    HealthRequestCls: type


def _normalize_target(target: str) -> str:
    if target.startswith("grpc://"):
        target = target[len("grpc://") :]
    if target.startswith("http://"):
        target = target[len("http://") :]
    if target.startswith("https://"):
        target = target[len("https://") :]
    if "/" in target:
        target = target.split("/", 1)[0]
    if ":" not in target:
        target += ":50052"
    return target


def _ensure_stub_bundle() -> _StubBundle:
    pool = descriptor_pool.Default()
    try:
        pool.FindFileByName("energy.proto")
    except KeyError:
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_proto.name = "energy.proto"
        file_proto.package = "mindi.energy"
        file_proto.syntax = "proto3"

        system_info = file_proto.message_type.add()
        system_info.name = "SystemInfo"
        _add_field(system_info, "os_name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(system_info, "os_version", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(system_info, "kernel_version", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(system_info, "host_name", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(system_info, "cpu_count", 5, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
        _add_field(system_info, "cpu_brand", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        gpu_info = file_proto.message_type.add()
        gpu_info.name = "GpuInfo"
        _add_field(gpu_info, "name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(gpu_info, "vendor", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(gpu_info, "device_id", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
        _add_field(gpu_info, "device_type", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(gpu_info, "backend", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        telemetry = file_proto.message_type.add()
        telemetry.name = "TelemetryReading"
        _add_field(telemetry, "power_watts", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        _add_field(telemetry, "energy_joules", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        _add_field(telemetry, "temperature_celsius", 3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        _add_field(telemetry, "gpu_memory_usage_mb", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        _add_field(telemetry, "cpu_memory_usage_mb", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        _add_field(telemetry, "platform", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        _add_field(telemetry, "timestamp_nanos", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
        _add_field(
            telemetry,
            "system_info",
            8,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".mindi.energy.SystemInfo",
        )
        _add_field(
            telemetry,
            "gpu_info",
            9,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".mindi.energy.GpuInfo",
        )

        stream_req = file_proto.message_type.add()
        stream_req.name = "StreamRequest"

        health_req = file_proto.message_type.add()
        health_req.name = "HealthRequest"

        health_res = file_proto.message_type.add()
        health_res.name = "HealthResponse"
        _add_field(health_res, "healthy", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
        _add_field(health_res, "platform", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        service = file_proto.service.add()
        service.name = "EnergyMonitor"
        method = service.method.add()
        method.name = "Health"
        method.input_type = ".mindi.energy.HealthRequest"
        method.output_type = ".mindi.energy.HealthResponse"

        method = service.method.add()
        method.name = "StreamTelemetry"
        method.input_type = ".mindi.energy.StreamRequest"
        method.output_type = ".mindi.energy.TelemetryReading"
        method.server_streaming = True

        pool.Add(file_proto)

    TelemetryReadingCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.TelemetryReading")
    )
    StreamRequestCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.StreamRequest")
    )
    HealthRequestCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.HealthRequest")
    )
    HealthResponseCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.HealthResponse")
    )

    class EnergyMonitorStub:
        def __init__(self, channel: grpc.Channel) -> None:
            self.Health = channel.unary_unary(
                "/mindi.energy.EnergyMonitor/Health",
                request_serializer=HealthRequestCls.SerializeToString,
                response_deserializer=HealthResponseCls.FromString,
            )
            self.StreamTelemetry = channel.unary_stream(
                "/mindi.energy.EnergyMonitor/StreamTelemetry",
                request_serializer=StreamRequestCls.SerializeToString,
                response_deserializer=TelemetryReadingCls.FromString,
            )

    return _StubBundle(
        stub_factory=EnergyMonitorStub,
        TelemetryReadingCls=TelemetryReadingCls,
        StreamRequestCls=StreamRequestCls,
        HealthRequestCls=HealthRequestCls,
    )


def _add_field(message, name, number, field_type, *, type_name=None, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name


class EnergyMonitorCollector(HardwareCollector):
    collector_id = "energy-monitor"
    collector_name = "Mindi Energy Monitor"

    def __init__(self, target: str = _DEFAULT_TARGET, *, channel_options: Optional[Tuple[Tuple[str, str], ...]] = None) -> None:
        self._target = _normalize_target(target)
        self._channel_options = channel_options or ()
        self._bundle = _ensure_stub_bundle()

    @classmethod
    def is_available(cls) -> bool:
        bundle = _ensure_stub_bundle()
        channel = grpc.insecure_channel(_normalize_target(_DEFAULT_TARGET))
        stub = bundle.stub_factory(channel)
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            stub.Health(bundle.HealthRequestCls(), timeout=1)
            return True
        except Exception:
            return False
        finally:
            channel.close()

    def stream_readings(self) -> Iterable[TelemetryReading]:
        channel = grpc.insecure_channel(self._target, options=self._channel_options)
        stub = self._bundle.stub_factory(channel)
        stream = stub.StreamTelemetry(self._bundle.StreamRequestCls())
        try:
            for raw in stream:
                yield self._convert(raw)
        except grpc.RpcError as exc:
            raise RuntimeError(
                f"Energy monitor stream failed: {exc.code().name} {exc.details()}"
            ) from exc
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
            temperature_celsius=_safe_float(getattr(message, "temperature_celsius", None)),
            gpu_memory_usage_mb=_safe_float(getattr(message, "gpu_memory_usage_mb", None)),
            cpu_memory_usage_mb=_safe_float(getattr(message, "cpu_memory_usage_mb", None)),
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

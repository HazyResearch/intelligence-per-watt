"""Dynamic protobuf helpers for the energy monitor API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type, cast

import grpc
from google.protobuf import descriptor_pb2 as _descriptor_pb2, descriptor_pool, message_factory

descriptor_pb2 = cast(Any, _descriptor_pb2)


@dataclass(frozen=True)
class StubBundle:
    """Container for lazily constructed gRPC stub callables and message classes."""

    stub_factory: Type[Any]
    TelemetryReadingCls: Type[Any]
    StreamRequestCls: Type[Any]
    HealthRequestCls: Type[Any]


_STUB_BUNDLE: StubBundle | None = None


def get_stub_bundle() -> StubBundle:
    """Return the cached dynamic gRPC stub bundle, creating it on first use."""

    global _STUB_BUNDLE
    if _STUB_BUNDLE is not None:
        return _STUB_BUNDLE

    pool = descriptor_pool.Default()
    try:
        pool.FindFileByName("energy.proto")
    except KeyError:
        _register_proto_descriptors(pool)

    TelemetryReadingCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("energy.TelemetryReading")
    )
    StreamRequestCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("energy.StreamRequest")
    )
    HealthRequestCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("energy.HealthRequest")
    )
    HealthResponseCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("energy.HealthResponse")
    )

    class EnergyMonitorStub:
        def __init__(self, channel: grpc.Channel) -> None:
            self.Health = channel.unary_unary(
                "/energy.EnergyMonitor/Health",
                request_serializer=HealthRequestCls.SerializeToString,
                response_deserializer=HealthResponseCls.FromString,
            )
            self.StreamTelemetry = channel.unary_stream(
                "/energy.EnergyMonitor/StreamTelemetry",
                request_serializer=StreamRequestCls.SerializeToString,
                response_deserializer=TelemetryReadingCls.FromString,
            )

    _STUB_BUNDLE = StubBundle(
        stub_factory=EnergyMonitorStub,
        TelemetryReadingCls=TelemetryReadingCls,
        StreamRequestCls=StreamRequestCls,
        HealthRequestCls=HealthRequestCls,
    )
    return _STUB_BUNDLE


def _register_proto_descriptors(pool: descriptor_pool.DescriptorPool) -> None:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "energy.proto"
    file_proto.package = "energy"
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
        type_name=".energy.SystemInfo",
    )
    _add_field(
        telemetry,
        "gpu_info",
        9,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".energy.GpuInfo",
    )

    file_proto.message_type.add().name = "StreamRequest"

    health_req = file_proto.message_type.add()
    health_req.name = "HealthRequest"

    health_res = file_proto.message_type.add()
    health_res.name = "HealthResponse"
    _add_field(health_res, "healthy", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    _add_field(health_res, "platform", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    service = file_proto.service.add()
    service.name = "EnergyMonitor"

    health_method = service.method.add()
    health_method.name = "Health"
    health_method.input_type = ".energy.HealthRequest"
    health_method.output_type = ".energy.HealthResponse"

    stream_method = service.method.add()
    stream_method.name = "StreamTelemetry"
    stream_method.input_type = ".energy.StreamRequest"
    stream_method.output_type = ".energy.TelemetryReading"
    stream_method.server_streaming = True

    pool.Add(file_proto)


def _add_field(message, name, number, field_type, *, type_name=None, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name


__all__ = ["StubBundle", "get_stub_bundle"]


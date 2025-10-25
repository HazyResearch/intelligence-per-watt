"""Developer-oriented CLI commands implemented with Click."""

from __future__ import annotations

import time
from typing import Tuple

import click
import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from ._binaries import run_attached
from ._console import console, success
from ._group import OrderedGroup


# ---------------------------------------------------------------------------
# Energy subgroup
# ---------------------------------------------------------------------------


@click.group(cls=OrderedGroup, help="Energy monitoring and efficiency tools")
def energy() -> None:
    """Energy subgroup."""


@energy.command(help="Start the energy monitor service in the foreground.")
@click.argument("args", nargs=-1)
def start(args: Tuple[str, ...]) -> None:
    exit_code = run_attached("mindi-energy-monitor", list(args))
    raise click.exceptions.Exit(exit_code)


@energy.command("monitor", help="Stream telemetry directly from mindi-energy-monitor (gRPC).")
@click.option(
    "--url",
    type=str,
    default="127.0.0.1:50052",
    show_default=True,
    help="Energy monitor gRPC target (host:port)",
)
@click.option(
    "-i",
    "--interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between printed samples",
)
def energy_monitor(url: str, interval: float) -> None:
    target = _normalize_grpc_target(url)
    stub_cls, TelemetryReadingCls, StreamRequestCls, HealthRequestCls = _build_energy_stub()

    channel = grpc.insecure_channel(target)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError as exc:
        raise click.ClickException(
            f"Unable to reach energy monitor at {target}\n"
            f"Start the energy monitor with: mindi energy start"
        ) from exc

    stub = stub_cls(channel)

    try:
        health = stub.Health(HealthRequestCls(), timeout=5)
        status = "healthy" if getattr(health, "healthy", False) else "unhealthy"
        platform = getattr(health, "platform", "unknown")
        success(f"Connected to energy monitor ({status}, platform: {platform})")
    except grpc.RpcError as exc:
        raise click.ClickException(f"Health check failed: {exc.code().name} {exc.details()}") from exc

    console.print("Energy Monitoring Dashboard (Ctrl+C to stop)")
    console.print(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
            "Time", "Energy(J)", "Power(W)", "Temp(°C)", "GPU MB", "CPU MB"
        )
    )
    console.print("-" * 68)

    start = time.time()
    last_emit = start - interval  # Ensure first reading prints immediately

    stream = stub.StreamTelemetry(StreamRequestCls())
    try:
        for reading in stream:
            now = time.time()
            if now - last_emit < max(interval, 0.05):
                continue
            last_emit = now
            elapsed = now - start
            energy_val = getattr(reading, "energy_joules", float("nan"))
            power_val = getattr(reading, "power_watts", float("nan"))
            temp_val = getattr(reading, "temperature_celsius", float("nan"))
            gpu_mb_val = getattr(reading, "gpu_memory_usage_mb", float("nan"))
            cpu_mb_val = getattr(reading, "cpu_memory_usage_mb", float("nan"))
            console.print(
                "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
                    _format_elapsed(elapsed),
                    _format_metric(energy_val, width=10, precision=3),
                    _format_metric(power_val, width=10, precision=2),
                    _format_metric(temp_val, width=10, precision=1),
                    _format_metric(gpu_mb_val, width=10, precision=1),
                    _format_metric(cpu_mb_val, width=8, precision=1),
                )
            )
    except KeyboardInterrupt:
        console.print("\nStopping monitor")
    except grpc.RpcError as exc:
        raise click.ClickException(f"Telemetry stream failed: {exc.code().name} {exc.details()}") from exc
    finally:
        channel.close()


def _normalize_grpc_target(target: str) -> str:
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


def _format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs}s"


_ENERGY_STUB_FACTORY = None
_TELEMETRY_READING_CLS = None
_STREAM_REQUEST_CLS = None
_HEALTH_REQUEST_CLS = None


def _build_energy_stub():
    global _ENERGY_STUB_FACTORY, _TELEMETRY_READING_CLS, _STREAM_REQUEST_CLS, _HEALTH_REQUEST_CLS

    if _ENERGY_STUB_FACTORY is not None:
        return (
            _ENERGY_STUB_FACTORY,
            _TELEMETRY_READING_CLS,
            _STREAM_REQUEST_CLS,
            _HEALTH_REQUEST_CLS,
        )

    pool = descriptor_pool.Default()
    try:
        pool.FindFileByName("energy.proto")
    except KeyError:
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_proto.name = "energy.proto"
        file_proto.package = "mindi.energy"
        file_proto.syntax = "proto3"

        # SystemInfo message
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

        reset_req = file_proto.message_type.add()
        reset_req.name = "ResetRequest"

        reset_res = file_proto.message_type.add()
        reset_res.name = "ResetResponse"
        _add_field(reset_res, "success", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)

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

        method = service.method.add()
        method.name = "ResetEnergyBaseline"
        method.input_type = ".mindi.energy.ResetRequest"
        method.output_type = ".mindi.energy.ResetResponse"

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
    ResetRequestCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.ResetRequest")
    )
    ResetResponseCls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("mindi.energy.ResetResponse")
    )

    class EnergyMonitorStub:
        def __init__(self, channel: grpc.Channel) -> None:
            self._channel = channel
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
            self.ResetEnergyBaseline = channel.unary_unary(
                "/mindi.energy.EnergyMonitor/ResetEnergyBaseline",
                request_serializer=ResetRequestCls.SerializeToString,
                response_deserializer=ResetResponseCls.FromString,
            )

    _ENERGY_STUB_FACTORY = EnergyMonitorStub
    _TELEMETRY_READING_CLS = TelemetryReadingCls
    _STREAM_REQUEST_CLS = StreamRequestCls
    _HEALTH_REQUEST_CLS = HealthRequestCls

    return EnergyMonitorStub, TelemetryReadingCls, StreamRequestCls, HealthRequestCls


def _add_field(message, name, number, field_type, *, type_name=None, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name


def _format_metric(value: float, *, width: int, precision: int) -> str:
    if value is None or value < 0:
        return f"{'-':>{width}}"
    try:
        return f"{value:>{width}.{precision}f}"
    except (ValueError, TypeError):
        return f"{'-':>{width}}"

__all__ = ["energy"]

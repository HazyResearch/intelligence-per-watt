from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


class Platform(str, Enum):
    """Supported hardware platforms."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "macos"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SystemInfo:
    """System metadata mirrored from the energy monitor proto."""

    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""
    host_name: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""


@dataclass(slots=True)
class GpuInfo:
    """GPU metadata mirrored from the energy monitor proto."""

    name: str = ""
    vendor: str = ""
    device_id: int = 0
    device_type: str = ""
    backend: str = ""

@dataclass(slots=True)
class HardwareInfo:
    """Wrapper around system and GPU metadata as returned by the energy monitor."""

    system_info: Optional[SystemInfo] = None
    gpu_info: Optional[GpuInfo] = None


@dataclass(slots=True)
class TelemetryReading:
    """
    Field names and semantics match `mindi-energy-monitor/proto/energy.proto`.
    The Rust service publishes -1 or 0 for unavailable metrics; Python callers
    may additionally use ``None`` to indicate a missing reading.
    """

    power_watts: Optional[float] = None
    energy_joules: Optional[float] = None
    temperature_celsius: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    cpu_memory_usage_mb: Optional[float] = None
    platform: Optional[str] = None
    timestamp_nanos: Optional[int] = None
    system_info: Optional[SystemInfo] = None
    gpu_info: Optional[GpuInfo] = None

@dataclass(slots=True)
class ChatUsage:
    """Token accounting for a chat exchange."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class Response:
    """Canonical chat completion response."""

    content: str
    usage: ChatUsage
    time_to_first_token_ms: float


@dataclass(slots=True)
class DatasetRecord:
    """Normalized dataset entry consumed by the profiling runner."""

    problem: str
    answer: str
    subject: str
    dataset_metadata: MutableMapping[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ProfilerConfig:
    dataset_id: str
    collector_id: str
    client_id: str
    client_base_url: str | None = None

    dataset_params: Mapping[str, Any] = field(default_factory=dict)
    collector_params: Mapping[str, Any] = field(default_factory=dict)
    client_params: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    run_id: str = ""
    output_dir: Path | None = None
    batch_size: int = 1
    max_queries: int | None = None
    additional_parameters: Mapping[str, Any] = field(default_factory=dict)

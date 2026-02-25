from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


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
    first_token_time: Optional[float] = None
    request_start_time: float = 0.0
    request_end_time: float = 0.0


@dataclass(slots=True)
class DatasetRecord:
    """Normalized dataset entry consumed by the profiling runner."""

    problem: str
    answer: str
    subject: str
    dataset_metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfilerConfig:
    """Configuration payload for the profiling runner."""

    dataset_id: str
    client_id: str
    client_base_url: str | None = None

    dataset_params: Mapping[str, Any] = field(default_factory=dict)
    client_params: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    output_dir: Path | None = None
    max_concurrency: int = 1
    max_queries: int | None = None
    phased_profiling: bool = False
    run_hardware_benchmarks: bool = True
    additional_parameters: Mapping[str, Any] = field(default_factory=dict)


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
class TelemetryReading:
    """
    Field names and semantics match `energy-monitor/proto/energy.proto`.
    The Rust service publishes -1 or 0 for unavailable metrics; Python callers
    may additionally use ``None`` to indicate a missing reading.
    """

    power_watts: Optional[float] = None
    energy_joules: Optional[float] = None
    temperature_celsius: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    cpu_memory_usage_mb: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    ane_power_watts: Optional[float] = None
    ane_energy_joules: Optional[float] = None
    gpu_compute_utilization_pct: Optional[float] = None
    gpu_memory_bandwidth_utilization_pct: Optional[float] = None
    gpu_tensor_core_utilization_pct: Optional[float] = None
    platform: Optional[str] = None
    timestamp_nanos: Optional[int] = None
    system_info: Optional[SystemInfo] = None
    gpu_info: Optional[GpuInfo] = None


@dataclass(slots=True)
class AgentRunResult:
    """Result from an agent run, capturing cost and tool-use telemetry."""

    content: str
    tool_calls_attempted: int = 0
    tool_calls_succeeded: int = 0
    tool_names_used: list[str] = field(default_factory=list)
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    trace: "QueryTrace | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentRunResult",
    "ChatUsage",
    "Response",
    "DatasetRecord",
    "ProfilerConfig",
    "SystemInfo",
    "GpuInfo",
    "TelemetryReading",
]

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
    client_id: str
    client_base_url: str | None = None

    dataset_params: Mapping[str, Any] = field(default_factory=dict)
    client_params: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    run_id: str = ""
    output_dir: Path | None = None
    batch_size: int = 1
    max_queries: int | None = None
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
class MetricStats:
    avg: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None


@dataclass(slots=True)
class ComputeMetrics:
    flops_per_request: Optional[float] = None
    macs_per_request: Optional[float] = None


@dataclass(slots=True)
class EnergyMetrics:
    per_query_joules: Optional[float] = None
    total_joules: Optional[float] = None


@dataclass(slots=True)
class LatencyMetrics:
    per_token_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    time_to_first_token_seconds: Optional[float] = None
    total_query_seconds: Optional[float] = None


@dataclass(slots=True)
class MemoryMetrics:
    cpu_mb: MetricStats = field(default_factory=MetricStats)
    gpu_mb: MetricStats = field(default_factory=MetricStats)


@dataclass(slots=True)
class PowerComponentMetrics:
    per_query_watts: MetricStats = field(default_factory=MetricStats)
    total_watts: MetricStats = field(default_factory=MetricStats)


@dataclass(slots=True)
class PowerMetrics:
    gpu: PowerComponentMetrics = field(default_factory=PowerComponentMetrics)


@dataclass(slots=True)
class TokenMetrics:
    input: Optional[int] = None
    output: Optional[int] = None


@dataclass(slots=True)
class ModelMetrics:
    compute_metrics: ComputeMetrics = field(default_factory=ComputeMetrics)
    energy_metrics: EnergyMetrics = field(default_factory=EnergyMetrics)
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory_metrics: MemoryMetrics = field(default_factory=MemoryMetrics)
    power_metrics: PowerMetrics = field(default_factory=PowerMetrics)
    temperature_metrics: MetricStats = field(default_factory=MetricStats)
    token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    gpu_info: Optional[GpuInfo] = None
    system_info: Optional[SystemInfo] = None
    lm_correctness: bool = False
    lm_response: str = ""


@dataclass(slots=True)
class ProfilingRecord:
    problem: str
    answer: str
    dataset_metadata: MutableMapping[str, Any] = field(default_factory=dict)
    subject: str = ""
    model_answers: MutableMapping[str, str] = field(default_factory=dict)
    model_metrics: MutableMapping[str, ModelMetrics] = field(default_factory=dict)

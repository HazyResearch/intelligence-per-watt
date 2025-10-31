from __future__ import annotations

from dataclasses import dataclass, field
from typing import MutableMapping, Optional

from ..core.types import GpuInfo, SystemInfo


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
    total: Optional[int] = None


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
    dataset_metadata: MutableMapping[str, object] = field(default_factory=dict)
    subject: str = ""
    model_answers: MutableMapping[str, str] = field(default_factory=dict)
    model_metrics: MutableMapping[str, ModelMetrics] = field(default_factory=dict)


__all__ = [
    "MetricStats",
    "ComputeMetrics",
    "EnergyMetrics",
    "LatencyMetrics",
    "MemoryMetrics",
    "PowerComponentMetrics",
    "PowerMetrics",
    "TokenMetrics",
    "ModelMetrics",
    "ProfilingRecord",
]

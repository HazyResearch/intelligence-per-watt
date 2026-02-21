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
    total_flops: Optional[int] = None
    flops_per_token: Optional[float] = None


@dataclass(slots=True)
class EnergyMetrics:
    # GPU energy (backward compatible naming)
    per_query_joules: Optional[float] = None
    total_joules: Optional[float] = None
    # CPU energy
    cpu_per_query_joules: Optional[float] = None
    cpu_total_joules: Optional[float] = None
    # ANE energy (macOS only)
    ane_per_query_joules: Optional[float] = None
    ane_total_joules: Optional[float] = None


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
class HardwareUtilizationGpu:
    compute_utilization_pct: Optional[float] = None
    memory_bandwidth_utilization_pct: Optional[float] = None
    tensor_core_utilization_pct: Optional[float] = None
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None


@dataclass(slots=True)
class HardwareUtilizationDerived:
    mfu: Optional[float] = None
    mbu: Optional[float] = None
    arithmetic_intensity: Optional[float] = None


@dataclass(slots=True)
class HardwareUtilization:
    gpu: HardwareUtilizationGpu = field(default_factory=HardwareUtilizationGpu)
    derived: HardwareUtilizationDerived = field(default_factory=HardwareUtilizationDerived)


@dataclass(slots=True)
class PowerComponentMetrics:
    per_query_watts: MetricStats = field(default_factory=MetricStats)
    total_watts: MetricStats = field(default_factory=MetricStats)


@dataclass(slots=True)
class PowerMetrics:
    gpu: PowerComponentMetrics = field(default_factory=PowerComponentMetrics)
    cpu: PowerComponentMetrics = field(default_factory=PowerComponentMetrics)


@dataclass(slots=True)
class TokenMetrics:
    input: Optional[int] = None
    output: Optional[int] = None
    total: Optional[int] = None


@dataclass(slots=True)
class PhaseMetrics:
    prefill_energy_j: Optional[float] = None
    decode_energy_j: Optional[float] = None
    prefill_duration_ms: Optional[float] = None
    decode_duration_ms: Optional[float] = None
    prefill_power_avg_w: Optional[float] = None
    decode_power_avg_w: Optional[float] = None
    prefill_energy_per_input_token_j: Optional[float] = None
    decode_energy_per_output_token_j: Optional[float] = None
    prefill_energy_components_j: Optional[dict[str, float]] = None
    decode_energy_components_j: Optional[dict[str, float]] = None


@dataclass(slots=True)
class CostMetrics:
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    tool_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None


@dataclass(slots=True)
class ModelMetrics:
    compute_metrics: ComputeMetrics = field(default_factory=ComputeMetrics)
    energy_metrics: EnergyMetrics = field(default_factory=EnergyMetrics)
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory_metrics: MemoryMetrics = field(default_factory=MemoryMetrics)
    power_metrics: PowerMetrics = field(default_factory=PowerMetrics)
    temperature_metrics: MetricStats = field(default_factory=MetricStats)
    token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    phase_metrics: PhaseMetrics = field(default_factory=PhaseMetrics)
    hardware_utilization: HardwareUtilization = field(default_factory=HardwareUtilization)
    cost: CostMetrics = field(default_factory=CostMetrics)
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
    "CostMetrics",
    "EnergyMetrics",
    "LatencyMetrics",
    "MemoryMetrics",
    "HardwareUtilizationGpu",
    "HardwareUtilizationDerived",
    "HardwareUtilization",
    "PowerComponentMetrics",
    "PowerMetrics",
    "TokenMetrics",
    "PhaseMetrics",
    "ModelMetrics",
    "ProfilingRecord",
]

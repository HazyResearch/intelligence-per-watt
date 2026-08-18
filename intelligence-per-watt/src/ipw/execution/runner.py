"""Profiler runner orchestration."""

from __future__ import annotations

import json
import logging
import math
import platform
import shutil
import statistics
import time
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import click
from tqdm.auto import tqdm

from datasets import Dataset

from ..clients.base import InferenceClient
from ..compute.flops import estimate_flops
from ..core.registry import ClientRegistry, DatasetRegistry
from ..core.types import DatasetRecord, GpuInfo, ProfilerConfig, Response, SystemInfo, TelemetryReading
from ..telemetry import EnergyMonitorCollector
from .hardware import derive_hardware_label
from .telemetry_session import TelemetrySample, TelemetrySession
from .types import (
    ComputeMetrics,
    DerivedEfficiencyMetrics,
    EnergyMetrics,
    HardwareUtilization,
    HardwareUtilizationDerived,
    HardwareUtilizationGpu,
    LatencyMetrics,
    MemoryMetrics,
    MetricStats,
    ModelMetrics,
    PhaseMetrics,
    PowerComponentMetrics,
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)

LOGGER = logging.getLogger(__name__)

# Theoretical peak BF16 TFLOPS for common GPUs (used for MFU calculation).
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    # NVIDIA Hopper
    "H100": 989.5,
    "H100 SXM": 989.5,
    "H100 PCIe": 756.0,
    "H200": 989.5,
    # NVIDIA Blackwell
    "B200": 2250.0,
    "B100": 1750.0,
    "GB200": 2250.0,
    # NVIDIA Ada Lovelace
    "RTX 4090": 165.2,
    "RTX 4080": 97.5,
    "RTX 4070 Ti": 73.4,
    "RTX 4070": 58.6,
    "L40S": 183.0,
    "L40": 181.0,
    "L4": 30.3,
    # NVIDIA Ampere
    "A100": 312.0,
    "A100 SXM": 312.0,
    "A100 PCIe": 312.0,
    "A6000": 77.4,
    "A10": 62.5,
    "RTX 3090": 71.0,
    "RTX 3080": 47.2,
    # NVIDIA Volta
    "V100": 28.3,
    "V100 SXM2": 28.3,
    # AMD
    "MI300X": 1307.4,
    "MI250X": 383.0,
    "MI210": 181.0,
}


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile value from a sorted-on-the-fly list."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _lookup_gpu_peak_tflops(gpu_name: str | None) -> float | None:
    """Look up theoretical peak BF16 TFLOPS for a GPU by name."""
    if not gpu_name:
        return None
    name_upper = gpu_name.upper().strip()
    for key, val in GPU_PEAK_TFLOPS_BF16.items():
        if key.upper() in name_upper:
            return val
    return None


class ProfilerRunner:
    """Coordinate dataset iteration, inference calls, telemetry capture, and persistence."""

    _FLUSH_INTERVAL = 100
    _HARDWARE_PRIME_TIMEOUT_SECONDS = 2.0
    _HARDWARE_PRIME_POLL_INTERVAL_SECONDS = 0.05

    # The runner is intentionally a slim orchestrator, but it still handles a
    # fair amount of coordination work:
    #
    # 1. Resolve dataset / client implementations from the registries so that we
    #    only depend on the registry surface, not the old resolution helpers.
    # 2. Spin up the `TelemetrySession`, which hides the threaded sampling loop
    #    that continuously pulls energy/power/memory readings into a rolling
    #    buffer while the run executes.
    # 3. For each dataset record, send the request to the client, collect the
    #    telemetry samples that overlap the query window, and transform the raw
    #    response + telemetry into the strongly typed `ProfilingRecord` payload
    #    defined in `ipw.execution.types`.
    # 4. Accumulate all records in-memory and write a HuggingFace dataset to the
    #    configured output directory once the run completes, along with a
    #    `summary.json` containing run metadata and aggregate energy totals.
    #
    # The actual measurements and conversions stay localized to helper methods
    # (`_compute_energy_metrics`, `_stat_summary`, etc.) so that the control flow
    # remains readable. Any future refactor (e.g., streaming writes or different
    # telemetry aggregation) should only need to touch the helpers and the final
    # persistence step.

    def __init__(self, config: ProfilerConfig) -> None:
        self._config = config
        self._records: list[ProfilingRecord] = []
        self._output_path: Optional[Path] = None
        self._output_prepared: bool = False
        self._hardware_label: Optional[str] = None
        self._system_info: Optional[SystemInfo] = None
        self._gpu_info: Optional[GpuInfo] = None
        self._baseline_energy: Optional[float] = None
        self._last_energy_total: Optional[float] = None
        self._overwrite_confirmed: bool = False
        self._client_info: Optional[Mapping[str, Any]] = None

    @property
    def records(self) -> list[ProfilingRecord]:
        """Read-only access to collected profiling records."""
        return list(self._records)

    def run(self) -> None:
        dataset = self._resolve_dataset(
            self._config.dataset_id, self._config.dataset_params
        )
        client: InferenceClient | None = None
        collector = EnergyMonitorCollector()

        try:
            client = self._resolve_client(
                self._config.client_id,
                self._config.client_base_url,
                self._config.client_params,
            )

            self._ensure_client_ready(client)

            with TelemetrySession(collector) as telemetry:
                self._process_records(dataset, client, telemetry)

            # After the run, not before: `describe` reports per-run tallies that
            # only exist once inference has happened (the AFM client counts the
            # queries it skipped on context overflow). Still before
            # `_close_client`, since closing may release the backend handles the
            # metadata is read from.
            self._client_info = self._describe_client(client)

            if not self._records:
                return

            self._persist_records(dataset)
        finally:
            self._close_client(client)

    def _process_records(
        self,
        dataset,
        client,
        telemetry: TelemetrySession,
    ) -> None:
        warmup = self._config.warmup_queries
        total_queries = self._config.max_queries or dataset.size()
        total_with_warmup = total_queries + warmup
        iterator = enumerate(dataset)
        # Prime hardware metadata early so the output directory label is accurate.
        self._prime_hardware_metadata(telemetry)
        # Prepare output directory (and confirm overwrite) before any inference.
        self._ensure_output_prepared(dataset)
        with tqdm(total=total_queries, desc="Profiling", unit="query") as progress:
            for index, record in iterator:
                if index >= total_with_warmup:
                    break
                start = time.time()
                response = self._invoke_client(client, record)
                end = time.time()

                if index < warmup:
                    progress.set_postfix_str(f"warmup {index + 1}/{warmup}")
                    continue  # Discard warmup queries

                samples = list(telemetry.window(start, end))
                built = self._build_record(index, record, response, samples, start, end)
                if built is not None:
                    self._records.append(built)
                    if len(self._records) % self._FLUSH_INTERVAL == 0:
                        self._persist_records(dataset)
                progress.update(1)

    def _build_record(
        self,
        index: int,
        record: DatasetRecord,
        response: Response,
        samples: Sequence[TelemetrySample],
        start_time: float,
        end_time: float,
    ) -> Optional[ProfilingRecord]:
        self._update_hardware_metadata(samples)
        telemetry_readings = [sample.reading for sample in samples]

        energy_metrics = self._compute_energy_metrics(telemetry_readings)
        power_stats = _stat_summary(
            [reading.power_watts for reading in telemetry_readings]
        )
        cpu_power_stats = _stat_summary(
            [reading.cpu_power_watts for reading in telemetry_readings]
        )
        ane_power_stats = _stat_summary(
            [reading.ane_power_watts for reading in telemetry_readings]
        )
        soc_power_stats = _stat_summary(
            [
                _sum_optional(
                    reading.power_watts,
                    reading.cpu_power_watts,
                    reading.ane_power_watts,
                )
                for reading in telemetry_readings
            ]
        )
        temperature_stats = _stat_summary(
            [reading.temperature_celsius for reading in telemetry_readings]
        )
        cpu_memory_stats = _stat_summary(
            [reading.cpu_memory_usage_mb for reading in telemetry_readings]
        )
        gpu_memory_stats = _stat_summary(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )
        compute_util_stats = _stat_summary(
            [reading.gpu_compute_utilization_pct for reading in telemetry_readings]
        )
        memory_bw_util_stats = _stat_summary(
            [reading.gpu_memory_bandwidth_utilization_pct for reading in telemetry_readings]
        )
        tensor_util_stats = _stat_summary(
            [reading.gpu_tensor_core_utilization_pct for reading in telemetry_readings]
        )

        memory_used_gb = _max_gb(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )
        memory_total_gb = _max_gb(
            [reading.gpu_memory_total_mb for reading in telemetry_readings]
        )

        usage = response.usage
        total_seconds = max(end_time - start_time, 0.0)

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        per_token_ms = None
        throughput_tokens = None
        if completion_tokens is not None and completion_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / completion_tokens
            throughput_tokens = completion_tokens / total_seconds

        # Choose the basis for the derived efficiency metrics below.
        #
        # On Apple Silicon `power_watts` / `energy_joules` carry the GPU rail
        # only (see energy-monitor/src/collectors/macos.rs), with CPU and ANE in
        # separate fields. A model executing on the Neural Engine -- as Apple
        # Foundation Models does -- would therefore look nearly free. Use the
        # whole-SoC sum there instead. Discrete-GPU hosts keep the established
        # GPU-only basis so historical numbers stay comparable.
        is_apple_soc = any(
            reading.platform == "macos" for reading in telemetry_readings
        )
        energy_basis = (
            energy_metrics.soc_per_query_joules
            if is_apple_soc
            else energy_metrics.per_query_joules
        )
        power_basis = soc_power_stats.avg if is_apple_soc else power_stats.avg
        basis_name = "soc" if is_apple_soc else "gpu"
        energy_metrics.basis = basis_name

        # --- Tier 1.1: Per-token energy normalization ---
        if energy_basis is not None:
            if completion_tokens is not None and completion_tokens > 0:
                energy_metrics.energy_per_output_token_joules = (
                    energy_basis / completion_tokens
                )
            if total_tokens is not None and total_tokens > 0:
                energy_metrics.energy_per_total_token_joules = (
                    energy_basis / total_tokens
                )

        # --- Tier 3.2: ITL percentiles from token timestamps ---
        latency_metrics = LatencyMetrics(
            per_token_ms=per_token_ms,
            throughput_tokens_per_sec=throughput_tokens,
            time_to_first_token_seconds=(
                response.time_to_first_token_ms / 1000.0
                if response.time_to_first_token_ms is not None
                else None
            ),
            total_query_seconds=total_seconds,
        )
        if response.token_timestamps and len(response.token_timestamps) > 1:
            itls = [
                (response.token_timestamps[i] - response.token_timestamps[i - 1]) * 1000
                for i in range(1, len(response.token_timestamps))
            ]
            latency_metrics.median_itl_ms = statistics.median(itls)
            latency_metrics.p90_itl_ms = _percentile(itls, 90)
            latency_metrics.p95_itl_ms = _percentile(itls, 95)
            latency_metrics.p99_itl_ms = _percentile(itls, 99)
            latency_metrics.itl_std_ms = statistics.stdev(itls) if len(itls) > 1 else 0.0

        # --- Tier 1.3: FLOPs estimation ---
        total_flops = 0.0
        flops_per_token = 0.0
        if prompt_tokens is not None and completion_tokens is not None:
            total_flops, flops_per_token = estimate_flops(
                self._config.model, prompt_tokens, completion_tokens
            )
        compute_metrics = ComputeMetrics()
        if total_flops > 0:
            compute_metrics.total_flops = int(total_flops)
            compute_metrics.flops_per_token = flops_per_token
            compute_metrics.flops_per_request = total_flops

        # --- Tier 1.2: Throughput per watt + FLOPs efficiency ---
        efficiency = DerivedEfficiencyMetrics()
        if throughput_tokens is not None and power_basis and power_basis > 0:
            efficiency.throughput_per_watt = throughput_tokens / power_basis
        if total_flops > 0 and energy_basis and energy_basis > 0:
            efficiency.flops_per_joule = total_flops / energy_basis
        if total_flops > 0 and power_basis and power_basis > 0 and total_seconds > 0:
            actual_flops_per_sec = total_flops / total_seconds
            efficiency.flops_per_watt = actual_flops_per_sec / power_basis

        # --- Tier 2.2: MFU ---
        gpu_name = self._gpu_info.name if self._gpu_info else None
        peak_tflops = _lookup_gpu_peak_tflops(gpu_name)
        mfu = None
        if total_flops > 0 and total_seconds > 0 and peak_tflops is not None:
            actual_tflops = (total_flops / total_seconds) / 1e12
            mfu = actual_tflops / peak_tflops

        # --- Tier 2.1: Phase separation at TTFT ---
        phase_metrics = self._compute_phase_metrics(
            response, samples, telemetry_readings, prompt_tokens, completion_tokens
        )

        model_name = self._config.model

        hardware_utilization = HardwareUtilization(
            gpu=HardwareUtilizationGpu(
                compute_utilization_pct=compute_util_stats.avg,
                memory_bandwidth_utilization_pct=memory_bw_util_stats.avg,
                tensor_core_utilization_pct=tensor_util_stats.avg,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
            ),
            derived=HardwareUtilizationDerived(mfu=mfu),
        )

        model_metrics = ModelMetrics(
            compute_metrics=compute_metrics,
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(
                cpu_mb=cpu_memory_stats,
                gpu_mb=gpu_memory_stats,
            ),
            power_metrics=PowerMetrics(
                gpu=PowerComponentMetrics(
                    per_query_watts=power_stats,
                    total_watts=MetricStats(
                        avg=power_stats.avg,
                        max=power_stats.max,
                        median=power_stats.median,
                        min=power_stats.min,
                    ),
                ),
                cpu=PowerComponentMetrics(
                    per_query_watts=cpu_power_stats,
                    total_watts=MetricStats(
                        avg=cpu_power_stats.avg,
                        max=cpu_power_stats.max,
                        median=cpu_power_stats.median,
                        min=cpu_power_stats.min,
                    ),
                ),
                ane=PowerComponentMetrics(
                    per_query_watts=ane_power_stats,
                    total_watts=MetricStats(
                        avg=ane_power_stats.avg,
                        max=ane_power_stats.max,
                        median=ane_power_stats.median,
                        min=ane_power_stats.min,
                    ),
                ),
                soc=PowerComponentMetrics(
                    per_query_watts=soc_power_stats,
                    total_watts=MetricStats(
                        avg=soc_power_stats.avg,
                        max=soc_power_stats.max,
                        median=soc_power_stats.median,
                        min=soc_power_stats.min,
                    ),
                ),
                basis=basis_name,
            ),
            temperature_metrics=temperature_stats,
            token_metrics=TokenMetrics(
                input=prompt_tokens,
                output=completion_tokens,
                total=total_tokens,
            ),
            phase_metrics=phase_metrics,
            hardware_utilization=hardware_utilization,
            efficiency=efficiency,
            gpu_info=self._gpu_info,
            system_info=self._system_info,
            lm_response=response.content,
        )

        record_payload = ProfilingRecord(
            problem=record.problem,
            answer=record.answer,
            dataset_metadata=dict(record.dataset_metadata),
            subject=record.subject,
            model_answers={model_name: response.content},
            model_metrics={model_name: model_metrics},
        )

        return record_payload

    def _compute_phase_metrics(
        self,
        response: Response,
        samples: Sequence[TelemetrySample],
        telemetry_readings: Sequence[TelemetryReading],
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> PhaseMetrics:
        """Split telemetry at TTFT boundary to populate prefill/decode phase metrics."""
        if (
            response.time_to_first_token_ms is None
            or response.time_to_first_token_ms <= 0
            or not samples
        ):
            return PhaseMetrics()

        # Compute the epoch timestamp of the first token
        # request_start_time is epoch-based (time.time()), but if unavailable
        # we can estimate from the first sample's timestamp
        if response.request_start_time and response.request_start_time > 0:
            ttft_epoch = response.request_start_time + (response.time_to_first_token_ms / 1000.0)
        elif samples:
            # Fallback: use the first sample timestamp as a reference point
            first_sample_ts = samples[0].timestamp
            ttft_epoch = first_sample_ts + (response.time_to_first_token_ms / 1000.0)
        else:
            return PhaseMetrics()

        prefill_readings = [s.reading for s in samples if s.timestamp <= ttft_epoch]
        decode_readings = [s.reading for s in samples if s.timestamp > ttft_epoch]

        # Need at least 2 readings in a phase to compute an energy delta
        prefill_energy = self._compute_phase_energy(prefill_readings)
        decode_energy = self._compute_phase_energy(decode_readings)

        # Duration
        prefill_duration_ms = response.time_to_first_token_ms
        decode_duration_ms = None
        if samples:
            last_ts = samples[-1].timestamp
            decode_duration_ms = max((last_ts - ttft_epoch) * 1000.0, 0.0) if last_ts > ttft_epoch else None

        # Power averages, on the same basis as the phase energy above so the
        # prefill/decode split is consistent with the per-query metrics.
        prefill_power = _stat_summary(
            [self._phase_power(r) for r in prefill_readings]
        )
        decode_power = _stat_summary([self._phase_power(r) for r in decode_readings])

        # Per-token energy
        prefill_energy_per_input = None
        if prefill_energy is not None and prompt_tokens is not None and prompt_tokens > 0:
            prefill_energy_per_input = prefill_energy / prompt_tokens

        decode_energy_per_output = None
        if decode_energy is not None and completion_tokens is not None and completion_tokens > 0:
            decode_energy_per_output = decode_energy / completion_tokens

        return PhaseMetrics(
            prefill_energy_j=prefill_energy,
            decode_energy_j=decode_energy,
            prefill_duration_ms=prefill_duration_ms,
            decode_duration_ms=decode_duration_ms,
            prefill_power_avg_w=prefill_power.avg,
            decode_power_avg_w=decode_power.avg,
            prefill_energy_per_input_token_j=prefill_energy_per_input,
            decode_energy_per_output_token_j=decode_energy_per_output,
        )

    @staticmethod
    def _phase_energy_value(reading: TelemetryReading) -> Optional[float]:
        """Energy counter for a phase, on the platform's appropriate basis.

        Mirrors the per-query choice: whole-SoC on Apple Silicon (where
        `energy_joules` is the GPU rail alone), GPU elsewhere.
        """
        if reading.platform == "macos":
            return _sum_optional(
                reading.energy_joules,
                reading.cpu_energy_joules,
                reading.ane_energy_joules,
            )
        return reading.energy_joules

    @staticmethod
    def _phase_power(reading: TelemetryReading) -> Optional[float]:
        """Power sample for a phase, on the same basis as the phase energy."""
        if reading.platform == "macos":
            return _sum_optional(
                reading.power_watts,
                reading.cpu_power_watts,
                reading.ane_power_watts,
            )
        return reading.power_watts

    def _compute_phase_energy(
        self, readings: Sequence[TelemetryReading]
    ) -> Optional[float]:
        """Compute the energy delta for a phase from its readings."""
        values = [
            value
            for value in (self._phase_energy_value(r) for r in readings)
            if value is not None
        ]
        return self._compute_energy_delta(values) if len(values) >= 2 else None

    def _compute_energy_metrics(
        self, readings: Sequence[TelemetryReading]
    ) -> EnergyMetrics:
        """Compute energy metrics from telemetry readings.

        Energy values should be monotonically increasing cumulative counters.
        Negative deltas indicate counter reset or data anomaly and are treated as None.
        """
        # GPU energy
        gpu_energy_values = [
            reading.energy_joules
            for reading in readings
            if reading.energy_joules is not None
        ]
        gpu_per_query = self._compute_energy_delta(gpu_energy_values)

        # CPU energy
        cpu_energy_values = [
            reading.cpu_energy_joules
            for reading in readings
            if reading.cpu_energy_joules is not None
        ]
        cpu_per_query = self._compute_energy_delta(cpu_energy_values)

        # ANE energy (macOS only)
        ane_energy_values = [
            reading.ane_energy_joules
            for reading in readings
            if reading.ane_energy_joules is not None
        ]
        ane_per_query = self._compute_energy_delta(ane_energy_values)

        # Whole-SoC energy, summed per reading before differencing so a rail
        # that is unavailable on this platform simply drops out. Yields None
        # only when no rail reported anything.
        soc_energy_values = [
            value
            for value in (
                _sum_optional(
                    reading.energy_joules,
                    reading.cpu_energy_joules,
                    reading.ane_energy_joules,
                )
                for reading in readings
            )
            if value is not None
        ]
        soc_per_query = self._compute_energy_delta(soc_energy_values)

        # Maintain baseline tracking for GPU (backward compat)
        if gpu_energy_values:
            start_value = gpu_energy_values[0]
            end_value = gpu_energy_values[-1]
            if self._baseline_energy is None:
                self._baseline_energy = start_value
            if (
                self._last_energy_total is not None
                and start_value < self._last_energy_total
            ):
                self._baseline_energy = start_value
            self._last_energy_total = end_value

        return EnergyMetrics(
            per_query_joules=gpu_per_query,
            total_joules=gpu_per_query,
            cpu_per_query_joules=cpu_per_query,
            cpu_total_joules=cpu_per_query,
            ane_per_query_joules=ane_per_query,
            ane_total_joules=ane_per_query,
            soc_per_query_joules=soc_per_query,
            soc_total_joules=soc_per_query,
        )

    def _compute_energy_delta(
        self, energy_values: list[float]
    ) -> Optional[float]:
        """Compute energy delta from a list of cumulative energy values."""
        if not energy_values:
            return None

        start_value = energy_values[0]
        end_value = energy_values[-1]

        # Validate energy values are finite and non-negative
        if not (
            math.isfinite(start_value)
            and math.isfinite(end_value)
            and start_value >= 0
            and end_value >= 0
        ):
            return None

        per_query_delta = end_value - start_value
        return per_query_delta if per_query_delta >= 0 else None

    def _update_hardware_metadata(self, readings: Sequence[TelemetrySample]) -> None:
        for sample in readings:
            reading = sample.reading
            if reading.system_info is not None:
                self._system_info = reading.system_info
            if reading.gpu_info is not None:
                self._gpu_info = reading.gpu_info

        candidate = derive_hardware_label(self._system_info, self._gpu_info)
        if candidate and (self._hardware_label in (None, "UNKNOWN_HW")):
            self._hardware_label = candidate

    def _get_output_path(self, dataset_label: str | None = None) -> Path:
        if self._output_path is not None:
            return self._output_path

        hardware_label = self._hardware_label or "UNKNOWN_HW"
        model_slug = _slugify_model(self._config.model)
        dataset_segment = dataset_label or self._config.dataset_id or "dataset"
        dataset_segment = str(dataset_segment).strip() or "dataset"
        default_runs_dir = Path(__file__).resolve().parents[4] / "runs"
        base_dir = self._config.output_dir or default_runs_dir
        profile_dir = f"profile_{hardware_label}_{model_slug}_{dataset_segment}".strip("_")

        output_path = Path(base_dir) / profile_dir

        self._hardware_label = hardware_label
        self._output_path = output_path
        return output_path

    def _invoke_client(self, client, record: DatasetRecord) -> Response:
        payload: MutableMapping[str, object] = dict(self._config.additional_parameters)
        return client.stream_chat_completion(
            self._config.model, record.problem, **payload
        )

    def _resolve_dataset(self, dataset_id: str, params: Mapping[str, Any]):
        try:
            dataset_cls = DatasetRegistry.get(dataset_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown dataset '{dataset_id}'") from exc

        try:
            return dataset_cls(**params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate dataset '{dataset_id}' with params {params!r}: {exc}"
            ) from exc

    def _resolve_client(
        self,
        client_id: str,
        base_url: str | None,
        params: Mapping[str, Any],
    ) -> InferenceClient:
        try:
            client_cls = ClientRegistry.get(client_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown client '{client_id}'") from exc

        try:
            return client_cls(base_url, **params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate client '{client_id}' with params {params!r}: {exc}"
            ) from exc

    def _ensure_client_ready(self, client: InferenceClient) -> None:
        if not client.health():
            raise RuntimeError(
                f"Client '{client.client_name}' at {getattr(client, 'base_url', '')} is unavailable"
            )
        client.prepare(self._config.model)

    def _describe_client(self, client: InferenceClient) -> Optional[Mapping[str, Any]]:
        """Collect optional backend metadata for ``summary.json``.

        ``describe`` is not part of the :class:`InferenceClient` contract; a
        client implements it when it knows things about the run that cannot be
        recovered from the records (the AFM client reports its SDK version,
        context size and host chip, since the on-device model variant it used is
        chosen by the framework and is not otherwise observable).
        """
        describe_fn = getattr(client, "describe", None)
        if not callable(describe_fn):
            return None
        try:
            return describe_fn()
        except Exception:
            LOGGER.warning("Failed to collect client metadata", exc_info=True)
            return None

    def _close_client(self, client: InferenceClient | None) -> None:
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                LOGGER.warning("Failed to close inference client cleanly", exc_info=True)

    def _prime_hardware_metadata(self, telemetry: TelemetrySession) -> None:
        """Wait briefly for telemetry samples so hardware labels are stable."""
        try:
            initial_samples = list(telemetry.readings() or [])
        except TypeError:
            initial_samples = []
        if initial_samples:
            self._update_hardware_metadata(initial_samples)
        if self._hardware_label not in (None, "UNKNOWN_HW"):
            return

        session_type = TelemetrySession
        if not isinstance(session_type, type):
            return
        if not isinstance(telemetry, session_type):
            return

        deadline = time.time() + self._HARDWARE_PRIME_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                samples = list(telemetry.readings() or [])
            except TypeError:
                samples = []
            if samples:
                self._update_hardware_metadata(samples)
                if self._hardware_label not in (None, "UNKNOWN_HW"):
                    return
            time.sleep(self._HARDWARE_PRIME_POLL_INTERVAL_SECONDS)

    def _ensure_output_prepared(self, dataset) -> Path:
        """Resolve and prepare the output directory once per run."""
        if self._output_prepared:
            return self._get_output_path()

        dataset_label = (
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "dataset_id", None)
            or self._config.dataset_id
        )
        output_path = self._get_output_path(
            str(dataset_label).strip() or self._config.dataset_id
        )
        if output_path.exists():
            self._confirm_overwrite(output_path)
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_prepared = True
        return output_path

    def _persist_records(self, dataset) -> None:
        if not self._records:
            return
        output_path = self._ensure_output_prepared(dataset)

        dataset_obj = Dataset.from_list([asdict(record) for record in self._records])
        dataset_obj.save_to_disk(str(output_path))
        output_path.mkdir(parents=True, exist_ok=True)

        summary = {
            "model": self._config.model,
            "profiler_config": _jsonify(asdict(self._config)),
            "dataset": getattr(dataset, "dataset_id", self._config.dataset_id),
            "dataset_name": getattr(dataset, "dataset_name", None),
            "hardware_label": self._hardware_label,
            "generated_at": time.time(),
            "total_queries": len(self._records),
            "system_info": asdict(self._system_info) if self._system_info else None,
            "gpu_info": asdict(self._gpu_info) if self._gpu_info else None,
            "output_dir": str(output_path),
            "versions": _get_versions(),
            "client_info": _jsonify(self._client_info) if self._client_info else None,
        }
        summary_path = output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def _confirm_overwrite(self, output_path: Path) -> None:
        """Prompt before overwriting an existing output directory."""
        if self._overwrite_confirmed:
            return

        prompt = (
            f"Output directory already exists at {output_path}. "
            "Overwrite it? This will remove existing run data."
        )
        proceed = click.confirm(prompt, default=False)

        if not proceed:
            raise RuntimeError(
                f"Profiling aborted to avoid overwriting existing output at {output_path}."
            )
        self._overwrite_confirmed = True


def _sum_optional(*values: Optional[float]) -> Optional[float]:
    """Sum the values that are present, or return None when none are.

    Used to combine the CPU/GPU/ANE power and energy rails, which are reported
    separately and are each unavailable on some platforms.
    """
    present = [float(v) for v in values if v is not None]
    return sum(present) if present else None


def _stat_summary(values: Iterable[Optional[float]]) -> MetricStats:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return MetricStats()
    return MetricStats(
        avg=sum(filtered) / len(filtered),
        max=max(filtered),
        median=statistics.median(filtered),
        min=min(filtered),
    )


def _max_gb(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return the maximum value converted from MB to GB, or None."""
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return max(filtered) / 1024.0


def _slugify_model(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"


def _jsonify(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonify(item) for item in value]
    return value


def _get_versions() -> dict[str, str]:
    try:
        ipw_version = importlib_metadata.version("ipw")
    except importlib_metadata.PackageNotFoundError:
        ipw_version = "unknown"

    return {
        "ipw": ipw_version,
        "python": platform.python_version(),
    }

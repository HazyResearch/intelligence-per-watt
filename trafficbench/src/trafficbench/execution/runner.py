"""Profiler runner orchestration."""

from __future__ import annotations

import json
import math
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from datasets import Dataset
from tqdm.auto import tqdm

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry
from ..core.types import (DatasetRecord, GpuInfo, ProfilerConfig, Response,
                          SystemInfo, TelemetryReading)
from ..telemetry import EnergyMonitorCollector
from .hardware import derive_hardware_label
from .telemetry_session import TelemetrySample, TelemetrySession
from .types import (ComputeMetrics, EnergyMetrics, LatencyMetrics,
                    MemoryMetrics, MetricStats, ModelMetrics,
                    PowerComponentMetrics, PowerMetrics, ProfilingRecord,
                    TokenMetrics)


class ProfilerRunner:
    """Coordinate dataset iteration, inference calls, telemetry capture, and persistence."""

    _FLUSH_INTERVAL = 100

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
    #    defined in `trafficbench.execution.types`.
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
        self._hardware_label: Optional[str] = None
        self._system_info: Optional[SystemInfo] = None
        self._gpu_info: Optional[GpuInfo] = None
        self._baseline_energy: Optional[float] = None
        self._last_energy_total: Optional[float] = None

    def run(self) -> None:
        dataset = self._resolve_dataset(
            self._config.dataset_id, self._config.dataset_params
        )
        client = self._resolve_client(
            self._config.client_id,
            self._config.client_base_url,
            self._config.client_params,
        )

        collector = EnergyMonitorCollector()

        self._ensure_client_ready(client)

        with TelemetrySession(collector) as telemetry:
            self._process_records(dataset, client, telemetry)

        if not self._records:
            return

        self._persist_records(dataset)

    def _process_records(
        self,
        dataset,
        client,
        telemetry: TelemetrySession,
    ) -> None:
        total_queries = self._config.max_queries or dataset.size()
        iterator = enumerate(dataset)
        with tqdm(total=total_queries, desc="Profiling", unit="query") as progress:
            for index, record in iterator:
                if index >= total_queries:
                    break
                start = time.time()
                response = self._invoke_client(client, record)
                end = time.time()
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
        temperature_stats = _stat_summary(
            [reading.temperature_celsius for reading in telemetry_readings]
        )
        cpu_memory_stats = _stat_summary(
            [reading.cpu_memory_usage_mb for reading in telemetry_readings]
        )
        gpu_memory_stats = _stat_summary(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )

        usage = response.usage
        total_seconds = max(end_time - start_time, 0.0)

        # Defensive: ensure token counts are valid integers
        prompt_tokens = usage.prompt_tokens if usage.prompt_tokens is not None else 0
        completion_tokens = (
            usage.completion_tokens if usage.completion_tokens is not None else 0
        )

        per_token_ms = None
        throughput_tokens = None
        if completion_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / completion_tokens
            throughput_tokens = completion_tokens / total_seconds

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

        model_name = self._config.model

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
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
                )
            ),
            temperature_metrics=temperature_stats,
            token_metrics=TokenMetrics(
                input=prompt_tokens,
                output=completion_tokens,
                total=prompt_tokens + completion_tokens,
            ),
            gpu_info=self._gpu_info,
            system_info=self._system_info,
            lm_correctness=False,
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

    def _compute_energy_metrics(
        self, readings: Sequence[TelemetryReading]
    ) -> EnergyMetrics:
        """Compute energy metrics from telemetry readings.

        Energy values should be monotonically increasing cumulative counters.
        Negative deltas indicate counter reset or data anomaly and are treated as None.
        """
        energy_values = [
            reading.energy_joules
            for reading in readings
            if reading.energy_joules is not None
        ]
        if not energy_values:
            return EnergyMetrics()

        start_value = energy_values[0]
        end_value = energy_values[-1]

        # Validate energy values are finite and non-negative
        if not (
            math.isfinite(start_value)
            and math.isfinite(end_value)
            and start_value >= 0
            and end_value >= 0
        ):
            return EnergyMetrics()

        if self._baseline_energy is None:
            self._baseline_energy = start_value

        per_query = None
        if self._last_energy_total is None:
            # First query - use delta within this query
            per_query = max(end_value - start_value, 0.0)
        else:
            # Subsequent queries - delta from last query's end
            delta = end_value - self._last_energy_total
            if delta < 0:
                # Energy counter decreased - likely reset or data corruption
                # Reset baseline and start fresh
                self._baseline_energy = end_value
                per_query = None
            else:
                per_query = delta

        self._last_energy_total = end_value

        return EnergyMetrics(
            per_query_joules=per_query,
            total_joules=per_query,
        )

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

    def _get_output_path(self) -> Path:
        if self._output_path is not None:
            return self._output_path

        hardware_label = self._hardware_label or "UNKNOWN_HW"
        model_slug = _slugify_model(self._config.model)
        default_runs_dir = Path(__file__).resolve().parents[4] / "runs"
        base_dir = self._config.output_dir or default_runs_dir
        profile_dir = f"profile_{hardware_label}_{model_slug}".strip("_")

        output_path = Path(base_dir) / profile_dir

        self._hardware_label = hardware_label
        self._output_path = output_path
        return output_path

    def _compute_total_energy(self) -> Optional[float]:
        if self._baseline_energy is None or self._last_energy_total is None:
            return None
        total = self._last_energy_total - self._baseline_energy
        return total if total >= 0 else None

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

    def _persist_records(self, dataset) -> None:
        if not self._records:
            return

        output_path = self._get_output_path()
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_obj = Dataset.from_list([asdict(record) for record in self._records])
        dataset_obj.save_to_disk(str(output_path))
        output_path.mkdir(parents=True, exist_ok=True)

        summary = {
            "model": self._config.model,
            "dataset": getattr(dataset, "dataset_id", self._config.dataset_id),
            "dataset_name": getattr(dataset, "dataset_name", None),
            "hardware_label": self._hardware_label,
            "generated_at": time.time(),
            "total_queries": len(self._records),
            "system_info": asdict(self._system_info) if self._system_info else None,
            "gpu_info": asdict(self._gpu_info) if self._gpu_info else None,
            "output_dir": str(output_path),
        }
        summary_path = output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))


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


def _slugify_model(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"

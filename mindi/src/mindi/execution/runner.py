"""Profiler runner orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Optional

from ..core.types import DatasetRecord, ProfilerConfig, Response
from ..telemetry import MindiEnergyMonitorCollector
from .resolution import (
    ResolutionError,
    ensure_client_ready,
    resolve_client,
    resolve_dataset,
)
from .sink import JsonlResultSink, ResultSink
from .telemetry import TelemetrySession


@dataclass
class ProfilerEvent:
    """Single profiling observation."""

    run_id: str
    record_index: int
    dataset_record: DatasetRecord
    response: Response
    telemetry: List[Mapping[str, object]]
    metadata: Mapping[str, object]


class ProfilerRunner:
    """Coordinate dataset iteration, inference calls, and telemetry capture."""

    def __init__(
        self,
        config: ProfilerConfig,
        *,
        sink: Optional[ResultSink] = None,
    ) -> None:
        self._config = config
        self._sink = sink or self._default_sink(config)

    def _default_sink(self, config: ProfilerConfig) -> ResultSink:
        base_dir = config.output_dir or Path.cwd() / "runs"
        run_label = config.run_id or f"run-{int(time.time())}"
        path = Path(base_dir) / run_label / "events.jsonl"
        return JsonlResultSink(path)

    def run(self) -> None:
        dataset = resolve_dataset(self._config.dataset_id, self._config.dataset_params)
        client = resolve_client(
            self._config.client_id,
            self._config.client_base_url,
            self._config.client_params,
        )

        collector = MindiEnergyMonitorCollector()

        ensure_client_ready(client)

        with TelemetrySession(collector) as telemetry:
            self._emit_config_snapshot()
            self._process_records(dataset, client, telemetry)

        self._sink.close()

    def _emit_config_snapshot(self) -> None:
        event = {
            "type": "config",
            "config": _snapshot_config(self._config),
            "timestamp": time.time(),
        }
        self._sink.append(event)

    def _process_records(
        self,
        dataset,
        client,
        telemetry: TelemetrySession,
    ) -> None:
        total = self._config.max_queries or dataset.size()
        for index, record in enumerate(dataset):
            if index >= total:
                break
            start = time.time()
            response = self._invoke_client(client, record)
            end = time.time()
            samples = list(telemetry.window(start, end))
            self._sink.append(
                {
                    "type": "event",
                    "run_id": self._config.run_id,
                    "record_index": index,
                    "problem": record.problem,
                    "answer": record.answer,
                    "subject": record.subject,
                    "usage": asdict(response.usage),
                    "response": response.content,
                    "ttft_ms": response.time_to_first_token_ms,
                    "start_time": start,
                    "end_time": end,
                    "telemetry": [
                        {
                            "timestamp": sample.timestamp,
                            "reading": asdict(sample.reading),
                        }
                        for sample in samples
                    ],
                    "metadata": dict(record.dataset_metadata),
                }
            )

    def _invoke_client(self, client, record: DatasetRecord) -> Response:
        payload: MutableMapping[str, object] = dict(self._config.additional_parameters)
        return client.stream_chat_completion(self._config.model, record.problem, **payload)


def _snapshot_config(config: ProfilerConfig) -> Mapping[str, object]:
    data = asdict(config)
    serialized: MutableMapping[str, object] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return dict(serialized)

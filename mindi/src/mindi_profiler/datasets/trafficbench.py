from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from ..core.dataset import DatasetProvider, register_dataset
from ..core.types import DatasetRecord


@register_dataset("trafficbench")
class TrafficBenchDataset(DatasetProvider):
    """Dataset provider for the bundled TrafficBench benchmark."""

    dataset_name = "TrafficBench"
    dataset_id = "trafficbench"

    def __init__(
        self,
        *,
        path: str,
        max_records: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self._path}")
        self._max_records = max_records
        self._shuffle = shuffle
        self._seed = seed

    def iter_records(self) -> Iterable[DatasetRecord]:
        records = list(self._load_records())
        if self._shuffle and len(records) > 1:
            rng = random.Random(self._seed)
            rng.shuffle(records)

        limit = self._max_records if self._max_records is not None else len(records)
        emitted = 0
        for record in records:
            if not self._is_valid(record):
                continue
            yield record
            emitted += 1
            if emitted >= limit:
                break

    def _load_records(self) -> Iterator[DatasetRecord]:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw: Dict[str, Any] = json.loads(stripped)
                yield self._parse_record(raw)

    def _parse_record(self, raw: Dict[str, Any]) -> DatasetRecord:
        problem = str(raw.get("problem") or raw.get("prompt") or "").strip()
        answer = str(raw.get("answer") or raw.get("expected_answer") or "").strip()
        subject = str(raw.get("subject") or "general").strip() or "general"

        dataset_metadata = dict(raw)
        model_metrics = dict(raw.get("model_metrics") or {})

        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=dataset_metadata,
            model_metrics=model_metrics,
        )

    def _is_valid(self, record: DatasetRecord) -> bool:
        return bool(record.problem and record.answer and record.subject and record.dataset_metadata)


__all__ = ["TrafficBenchDataset"]

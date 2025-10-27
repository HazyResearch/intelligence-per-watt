from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, MutableMapping

from datasets import load_from_disk

from ..core.dataset import DatasetProvider
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord


_DEFAULT_DATASET_DIR = "mixed_1k_seed1_base"


def _default_dataset_path() -> Path:
    base = resources.files("mindi.datasets") / "data" / _DEFAULT_DATASET_DIR
    return Path(base)


@DatasetRegistry.register("trafficbench")
class TrafficBenchDataset(DatasetProvider):
    """Dataset provider for the bundled TrafficBench benchmark."""

    dataset_name = "TrafficBench"
    dataset_id = "trafficbench"

    def __init__(self) -> None:
        self._path = _default_dataset_path()
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset location not found: {self._path}")
        self._records = tuple(self._load_all_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def _load_all_records(self) -> Iterable[DatasetRecord]:
        if self._path.is_dir():
            yield from self._load_from_dataset_dir(self._path)
        else:
            yield from (
                record
                for record in self._load_from_jsonl(self._path)
                if self._is_valid(record)
            )

    def _load_from_dataset_dir(self, directory: Path) -> Iterable[DatasetRecord]:
        dataset = load_from_disk(str(directory))
        if isinstance(dataset, dict):
            hf_dataset = next(iter(dataset.values()))
        else:
            hf_dataset = dataset

        raw_records: Iterable[MutableMapping[str, Any]] = (
            hf_dataset if isinstance(hf_dataset, list) else hf_dataset.to_list()
        )
        for raw in raw_records:
            record = self._parse_record(raw)
            if self._is_valid(record):
                yield record

    def _load_from_jsonl(self, file_path: Path) -> Iterator[DatasetRecord]:
        with file_path.open("r", encoding="utf-8") as handle:
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
        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=dataset_metadata,
        )

    def _is_valid(self, record: DatasetRecord) -> bool:
        return bool(record.problem and record.answer and record.subject and record.dataset_metadata)

    def size(self) -> int:
        return len(self._records)


__all__ = ["TrafficBenchDataset"]

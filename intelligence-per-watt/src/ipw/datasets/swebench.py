from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DATASET_PATHS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "verified_mini": "MariusHobbhahn/swe-bench-verified-mini",
}


def _parse_test_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
    if isinstance(value, list):
        return value
    return []


@DatasetRegistry.register("swebench")
class SWEBenchDataset(DatasetProvider):
    """SWE-bench dataset (princeton-nlp/SWE-bench_Verified).

    Supports two variants:
    - ``verified``: Full 500-task dataset
    - ``verified_mini``: 50-task subset
    """

    dataset_id = "swebench"
    dataset_name = "SWE-bench"
    evaluation_method = "swebench"

    _default_split = "test"
    _default_variant = "verified_mini"

    # SWE-bench correctness is determined by test execution, not LLM judge.
    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        variant: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._variant = variant or self._default_variant
        if self._variant not in _DATASET_PATHS:
            raise ValueError(
                f"Unknown SWE-bench variant '{self._variant}'. "
                f"Choose from: {list(_DATASET_PATHS)}"
            )
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _build_records(self) -> List[DatasetRecord]:
        rows = self._load_raw_rows()
        records: List[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(raw)
            if record is not None:
                records.append(record)
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        hf_path = _DATASET_PATHS[self._variant]
        dataset = load_dataset(hf_path, split=self._split)
        rows: Sequence[MutableMapping[str, object]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]
        normalized: list[MutableMapping[str, object]] = []
        for row in rows:
            if isinstance(row, MutableMapping):
                normalized.append(row)
            else:
                normalized.append(dict(row))
        return normalized

    def _convert_row(self, raw: MutableMapping[str, object]) -> Optional[DatasetRecord]:
        instance_id = str(raw.get("instance_id") or "")
        repo = str(raw.get("repo") or "")
        problem_statement = str(raw.get("problem_statement") or "").strip()

        if not instance_id or not problem_statement:
            return None

        # The problem is the issue description
        problem = problem_statement

        # The "answer" is the ground-truth patch
        patch = str(raw.get("patch") or "")

        fail_to_pass = _parse_test_list(raw.get("FAIL_TO_PASS", ""))
        pass_to_pass = _parse_test_list(raw.get("PASS_TO_PASS", ""))

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": raw.get("base_commit"),
            "hints_text": raw.get("hints_text"),
            "version": raw.get("version"),
            "test_patch": raw.get("test_patch"),
            "created_at": raw.get("created_at"),
            "environment_setup_commit": raw.get("environment_setup_commit"),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "difficulty": raw.get("difficulty"),
            "variant": self._variant,
        }

        return DatasetRecord(
            problem=problem,
            answer=patch,
            subject=repo,
            dataset_metadata=metadata,
        )


__all__ = ["SWEBenchDataset"]

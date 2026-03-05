from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_INPUT_PROMPT = """You are a software performance engineer. Your task is to optimize the code in the repository to improve performance.

Repository: {repo}

## Problem Statement
{problem_statement}

## Workload Description
{workload}

## Expected Speedup
The optimization should achieve approximately {expected_speedup:.1f}x speedup.

## Instructions
1. Analyze the codebase to identify performance bottlenecks
2. Implement optimizations that improve performance
3. Ensure all existing tests still pass
4. Generate a git patch with your changes

Please provide your optimization patch in unified diff format."""


def _parse_test_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value] if value else []
    return []


@DatasetRegistry.register("swefficiency")
class SWEfficiencyDataset(DatasetProvider):
    """SWEfficiency benchmark dataset (swefficiency/swefficiency).

    Software performance optimization benchmark (SWE-bench style).
    """

    dataset_id = "swefficiency"
    dataset_name = "SWEfficiency"
    evaluation_method = "swefficiency"

    _hf_path = "swefficiency/swefficiency"
    _default_split = "test"

    # SWEfficiency does not use LLM judge by default -- correctness is
    # determined by running test suites, so we leave eval settings at None.
    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """Structural validation only — true correctness requires test execution."""
        if not response or not response.strip():
            return False, {"reason": "empty_response"}
        has_patch = any(
            m in response for m in ("diff --git", "---", "+++", "@@")
        )
        return None, {
            "reason": "requires_test_execution",
            "has_patch": has_patch,
            "instance_id": record.dataset_metadata.get("instance_id", ""),
        }

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
        dataset = load_dataset(self._hf_path, split=self._split)
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
        workload = str(raw.get("workload") or "")
        speedup = float(raw.get("speedup", raw.get("expected_speedup", 1.0)) or 1.0)

        if not instance_id or not problem_statement:
            return None

        problem = _DEFAULT_INPUT_PROMPT.format(
            repo=repo,
            problem_statement=problem_statement,
            workload=workload,
            expected_speedup=speedup,
        )

        # The "answer" is the ground-truth patch
        patch = str(raw.get("patch") or "")

        covering_tests = _parse_test_list(
            raw.get("covering_tests", raw.get("COVERING_TESTS", []))
        )
        pass_to_pass = _parse_test_list(
            raw.get("pass_to_pass", raw.get("PASS_TO_PASS", []))
        )

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": raw.get("base_commit"),
            "test_patch": raw.get("test_patch"),
            "test_cmd": raw.get("test_cmd"),
            "rebuild_cmd": raw.get("rebuild_cmd"),
            "image_name": raw.get("image_name"),
            "speedup": speedup,
            "covering_tests": covering_tests,
            "pass_to_pass": pass_to_pass,
        }

        return DatasetRecord(
            problem=problem,
            answer=patch,
            subject=repo,
            dataset_metadata=metadata,
        )


__all__ = ["SWEfficiencyDataset"]

from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

try:
    from datasets import load_dataset

    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


@DatasetRegistry.register("terminalbench")
class TerminalBenchDataset(DatasetProvider):
    """TerminalBench dataset for evaluating terminal/CLI task completion.

    Loads from HuggingFace ``terminal-bench/terminal-bench``.
    """

    dataset_id = "terminalbench"
    dataset_name = "TerminalBench"
    evaluation_method = "terminalbench"

    _hf_path = "terminal-bench/terminal-bench"
    _default_split = "test"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if not _HAS_DATASETS:
            raise ImportError(
                "The 'datasets' package is required for TerminalBench. "
                "Install with: pip install datasets"
            )
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not _HAS_DATASETS:
            issues.append(
                "Missing 'datasets' package. Install with: pip install datasets"
            )
        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY (preferred) or OPENAI_API_KEY for scoring."
            )
        return issues

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        handler = self._resolve_handler(eval_client)
        return handler.evaluate(
            problem=record.problem,
            reference=record.answer,
            model_answer=response,
            metadata=record.dataset_metadata,
        )

    def _resolve_handler(self, eval_client: Optional[InferenceClient]):
        judge_client = eval_client or ClientRegistry.create(
            self.eval_client or "openai",
            base_url=self.eval_base_url or "https://api.openai.com/v1",
            model=self.eval_model or "gpt-5-nano-2025-08-07",
        )
        return EvaluationRegistry.create(self.evaluation_method, client=judge_client)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _build_records(self) -> List[DatasetRecord]:
        dataset = load_dataset(self._hf_path, split=self._split)
        rows: list[MutableMapping[str, object]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        records: List[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            if not isinstance(raw, MutableMapping):
                raw = dict(raw)
            record = self._convert_row(raw, idx)
            if record is not None:
                records.append(record)
        return records

    def _convert_row(
        self, raw: MutableMapping[str, object], idx: int
    ) -> Optional[DatasetRecord]:
        question = str(
            raw.get("prompt") or raw.get("question") or raw.get("instruction") or ""
        ).strip()
        answer = str(
            raw.get("answer") or raw.get("expected_output") or raw.get("gold_answer") or ""
        ).strip()

        if not question or not answer:
            return None

        category = str(raw.get("category") or raw.get("type") or "terminal")
        task_id = str(raw.get("id") or raw.get("task_id") or f"tb_{idx}")

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "task_id": task_id,
            "category": category,
        }

        return DatasetRecord(
            problem=question,
            answer=answer,
            subject=category,
            dataset_metadata=metadata,
        )


__all__ = ["TerminalBenchDataset"]

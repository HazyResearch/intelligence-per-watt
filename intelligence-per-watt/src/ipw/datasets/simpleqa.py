from __future__ import annotations

import ast
import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_INPUT_PROMPT = """Please answer the question below with a short, factual response.

- Return only your answer, which should be a word, phrase, name, number, or date.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a name, use the full name without abbreviations.
- Do not include any explanations or additional context.

Question: {question}"""


@DatasetRegistry.register("simpleqa")
class SimpleQADataset(DatasetProvider):
    """SimpleQA benchmark dataset (basicv8vc/SimpleQA).

    Short-form factual QA testing parametric knowledge.
    """

    dataset_id = "simpleqa"
    dataset_name = "SimpleQA"
    evaluation_method = "simpleqa"

    _hf_path = "basicv8vc/SimpleQA"
    _default_split = "test"

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

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
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
        rows = self._load_raw_rows()
        records: List[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            record = self._convert_row(raw, idx)
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

    def _convert_row(
        self, raw: MutableMapping[str, object], idx: int
    ) -> Optional[DatasetRecord]:
        # Parse metadata field which may be a JSON string or dict
        raw_metadata = raw.get("metadata", {})
        if isinstance(raw_metadata, str):
            try:
                parsed_metadata = ast.literal_eval(raw_metadata)
            except (ValueError, SyntaxError):
                parsed_metadata = {}
        else:
            parsed_metadata = raw_metadata or {}

        topic = str(
            parsed_metadata.get("topic", raw.get("topic", "General"))
        )

        question = str(raw.get("problem") or raw.get("question") or "").strip()
        answer = str(raw.get("answer") or raw.get("gold_answer") or "").strip()

        if not question or not answer:
            return None

        problem = _DEFAULT_INPUT_PROMPT.format(question=question)

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "original_index": idx,
            "topic": topic,
            "answer_type": parsed_metadata.get(
                "answer_type", raw.get("answer_type", "Other")
            ),
        }

        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=topic,
            dataset_metadata=metadata,
        )


__all__ = ["SimpleQADataset"]

"""GPQA benchmark dataset (Idavidrein/gpqa)."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_INPUT_PROMPT = """Please answer the following graduate-level question. Choose the correct answer from the options provided.

Question: {question}

Options:
{options}

Provide only the letter of the correct answer (A, B, C, or D)."""


@DatasetRegistry.register("gpqa")
class GPQADataset(DatasetProvider):
    """GPQA (Graduate-Level Google-Proof Q&A) benchmark dataset.

    Expert-level multiple-choice questions across STEM domains.
    """

    dataset_id = "gpqa"
    dataset_name = "GPQA"
    evaluation_method = "natural_reasoning"

    _hf_path = "Idavidrein/gpqa"
    _default_subset = "gpqa_diamond"
    _default_split = "train"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._split = split or self._default_split
        self._subset = subset or self._default_subset
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
        dataset = load_dataset(self._hf_path, self._subset, split=self._split)
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
        question = str(raw.get("Question") or raw.get("question") or "").strip()
        correct_answer = str(
            raw.get("Correct Answer") or raw.get("correct_answer") or ""
        ).strip()

        if not question or not correct_answer:
            return None

        # Build options from available choice columns
        choices = []
        for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"):
            val = raw.get(key)
            if val:
                choices.append(str(val).strip())

        # Insert correct answer at a consistent position for the prompt
        all_options = [correct_answer] + choices
        option_labels = ["A", "B", "C", "D"]
        options_text = "\n".join(
            f"{label}. {opt}"
            for label, opt in zip(option_labels, all_options)
            if opt
        )

        problem = _DEFAULT_INPUT_PROMPT.format(question=question, options=options_text)

        domain = str(raw.get("Subdomain") or raw.get("subdomain") or raw.get("High-level domain") or "STEM")

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "original_index": idx,
            "domain": domain,
            "correct_option": "A",
        }

        return DatasetRecord(
            problem=problem,
            answer=f"A. {correct_answer}",
            subject=domain,
            dataset_metadata=metadata,
        )


__all__ = ["GPQADataset"]

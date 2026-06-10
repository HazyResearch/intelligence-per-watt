from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider


def _format_options(options: list[str]) -> str:
    rendered = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        rendered.append(f"{letter}. {option}")
    return "\n".join(rendered)


@DatasetRegistry.register("mmlu-pro")
class MMLUProDataset(DatasetProvider):
    dataset_id = "mmlu-pro"
    dataset_name = "MMLU-Pro"
    evaluation_method = "mmlu-pro"
    _hf_path = "TIGER-Lab/MMLU-Pro"
    _default_split = "test"

    def __init__(self, *, split: Optional[str] = None, max_samples: Optional[int] = None) -> None:
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

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
        if self._max_samples is not None:
            dataset = dataset.select(range(min(self._max_samples, len(dataset))))
        rows: Sequence[MutableMapping[str, object]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        normalized: list[MutableMapping[str, object]] = []
        for row in rows:
            if isinstance(row, MutableMapping):
                normalized.append(row)
            else:
                normalized.append(dict(row))
        return normalized

    def _convert_row(self, raw: MutableMapping[str, object]) -> Optional[DatasetRecord]:
        question = str(raw.get("question") or "").strip()
        options = [str(option) for option in raw.get("options", []) or []]
        answer = str(raw.get("answer") or "").strip().upper()
        subject = str(raw.get("category") or "general").strip() or "general"

        if not question or not answer:
            return None

        prompt_parts = [question]
        if options:
            prompt_parts.append("")
            prompt_parts.append("Options:")
            prompt_parts.append(_format_options(options))
        prompt_parts.append("")
        prompt_parts.append("Respond with the correct letter.")
        problem = "\n".join(part for part in prompt_parts if part is not None).strip()

        metadata = {
            "dataset_name": self.dataset_name,
            "question_id": raw.get("question_id"),
            "options": options,
            "answer_index": raw.get("answer_index"),
            "source": raw.get("src"),
        }

        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=metadata,
        )

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


__all__ = ["MMLUProDataset"]

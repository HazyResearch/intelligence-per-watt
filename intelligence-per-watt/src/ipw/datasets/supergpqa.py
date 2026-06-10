from __future__ import annotations

import os
from typing import Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider


def _format_options(options: Iterable[str]) -> str:
    rendered = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        rendered.append(f"{letter}. {option}")
    return "\n".join(rendered)


@DatasetRegistry.register("supergpqa")
class SuperGPQADataset(DatasetProvider):
    dataset_id = "supergpqa"
    dataset_name = "SuperGPQA"
    evaluation_method = "supergpqa"
    _hf_path = "m-a-p/SuperGPQA"
    _default_split = "train"

    def __init__(self, *, split: Optional[str] = None, max_samples: Optional[int] = None) -> None:
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

    def _build_records(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for raw in self._load_raw_rows():
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
        options_raw = raw.get("options") or []
        options = [str(option).strip() for option in options_raw if str(option).strip()]
        answer_letter = str(raw.get("answer_letter") or "").strip().upper()
        answer_text = str(raw.get("answer") or "").strip()

        subject = str(
            raw.get("subfield")
            or raw.get("field")
            or raw.get("discipline")
            or "general"
        ).strip() or "general"

        if not question or not options or not answer_letter:
            return None

        prompt_sections = [question, "", "Options:", _format_options(options), "", "Respond with the correct letter only."]
        problem = "\n".join(section for section in prompt_sections if section).strip()

        metadata = {
            "dataset_name": self.dataset_name,
            "uuid": raw.get("uuid"),
            "discipline": raw.get("discipline"),
            "field": raw.get("field"),
            "subfield": raw.get("subfield"),
            "difficulty": raw.get("difficulty"),
            "is_calculation": raw.get("is_calculation"),
            "answer_text": answer_text,
            "options": options,
        }

        return DatasetRecord(
            problem=problem,
            answer=answer_letter,
            subject=subject,
            dataset_metadata=metadata,
        )


__all__ = ["SuperGPQADataset"]

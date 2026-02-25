from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider


@DatasetRegistry.register("hle")
class HLEDataset(DatasetProvider):
    """Humanity's Last Exam (HLE) benchmark dataset (cais/hle).

    Expert-level knowledge across many academic disciplines from the
    Center for AI Safety (CAIS).  Only text-only samples are loaded by
    default; set ``text_only=False`` to include multimodal items.
    """

    dataset_id = "hle"
    dataset_name = "HLE"
    evaluation_method = "hle"

    _hf_path = "cais/hle"
    _default_split = "test"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        text_only: bool = True,
    ) -> None:
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._text_only = text_only
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
            if self._max_samples is not None and len(records) >= self._max_samples:
                break
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        dataset = load_dataset(self._hf_path, split=self._split)
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

    def _convert_row(
        self, raw: MutableMapping[str, object], idx: int
    ) -> Optional[DatasetRecord]:
        question = str(
            raw.get("question") or raw.get("instruction") or raw.get("prompt") or ""
        ).strip()
        answer = str(
            raw.get("answer") or raw.get("gold_answer") or raw.get("response") or ""
        ).strip()

        if not question or not answer:
            return None

        # Detect multimodal content
        has_image = bool(
            raw.get("image") or raw.get("image_path") or raw.get("images")
        )
        has_audio = bool(
            raw.get("audio") or raw.get("audio_path") or raw.get("audios")
        )

        if self._text_only and (has_image or has_audio):
            return None

        category = str(
            raw.get("category") or raw.get("subject") or raw.get("type") or "general"
        )
        difficulty = raw.get("difficulty") or raw.get("level")
        task_id = str(raw.get("id") or raw.get("task_id") or f"hle_{idx}")

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "task_id": task_id,
            "category": category,
            "difficulty": str(difficulty) if difficulty else None,
            "has_image": has_image,
            "has_audio": has_audio,
        }

        return DatasetRecord(
            problem=question,
            answer=answer,
            subject=category,
            dataset_metadata=metadata,
        )


__all__ = ["HLEDataset"]

"""WildChat benchmark dataset (allenai/WildChat)."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_INPUT_PROMPT = """Please respond to the following user message thoughtfully and helpfully.

{message}"""


@DatasetRegistry.register("wildchat")
class WildChatDataset(DatasetProvider):
    """WildChat benchmark dataset (allenai/WildChat).

    Real user-LLM conversations for open-ended chat evaluation.
    Uses LLM-as-judge pairwise comparison for scoring.
    """

    dataset_id = "wildchat"
    dataset_name = "WildChat"
    evaluation_method = "wildchat"

    _hf_path = "allenai/WildChat"
    _default_split = "train"

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

    def _convert_row(
        self, raw: MutableMapping[str, object], idx: int
    ) -> Optional[DatasetRecord]:
        # WildChat stores conversations as a list of {role, content} turns.
        # Extract the first user message as the problem and the first
        # assistant response as the reference answer.
        conversation = raw.get("conversation") or raw.get("messages") or []
        if not isinstance(conversation, list) or len(conversation) < 2:
            return None

        user_message: Optional[str] = None
        assistant_response: Optional[str] = None
        for turn in conversation:
            role = str(turn.get("role", "")).lower() if isinstance(turn, dict) else ""
            content = str(turn.get("content", "")).strip() if isinstance(turn, dict) else ""
            if role == "user" and user_message is None:
                user_message = content
            elif role == "assistant" and user_message is not None and assistant_response is None:
                assistant_response = content

        if not user_message or not assistant_response:
            return None

        problem = _DEFAULT_INPUT_PROMPT.format(message=user_message)

        language = str(raw.get("language", raw.get("lang", "unknown")))
        model_name = str(raw.get("model", raw.get("model_name", "unknown")))

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "original_index": idx,
            "language": language,
            "source_model": model_name,
        }

        return DatasetRecord(
            problem=problem,
            answer=assistant_response,
            subject=language,
            dataset_metadata=metadata,
        )


__all__ = ["WildChatDataset"]

from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.
- This question may require multi-hop reasoning across multiple Wikipedia articles.
{wiki_context}

Here is the question:

{question}"""


@DatasetRegistry.register("frames")
class FRAMESDataset(DatasetProvider):
    """FRAMES benchmark dataset (google/frames-benchmark).

    Multi-hop factual retrieval requiring 2-15 Wikipedia articles.
    """

    dataset_id = "frames"
    dataset_name = "FRAMES"
    evaluation_method = "frames"

    _hf_path = "google/frames-benchmark"
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
        question = str(
            raw.get("Prompt") or raw.get("prompt") or raw.get("question") or ""
        ).strip()
        answer = str(
            raw.get("Answer") or raw.get("answer") or raw.get("gold_answer") or ""
        ).strip()

        if not question or not answer:
            return None

        # Extract reasoning types
        reasoning = raw.get("reasoning_types", raw.get("reasoning_type", ""))
        if isinstance(reasoning, list):
            reasoning = ", ".join(str(r) for r in reasoning)
        reasoning = str(reasoning)

        # Extract wiki links
        wiki_links_raw = raw.get("wiki_links", raw.get("wikipedia_links", []))
        if isinstance(wiki_links_raw, str):
            wiki_links = [link.strip() for link in wiki_links_raw.split(",") if link.strip()]
        elif isinstance(wiki_links_raw, list):
            wiki_links = [str(link) for link in wiki_links_raw]
        else:
            wiki_links = []

        # Build wiki context
        wiki_context = ""
        if wiki_links:
            wiki_context = (
                "\n\nRelevant Wikipedia articles that may help answer this question:\n"
                + "\n".join(f"- {link}" for link in wiki_links)
            )

        problem = _DEFAULT_INPUT_PROMPT.format(
            question=question, wiki_context=wiki_context
        )

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "index": idx,
            "reasoning_types": reasoning,
            "wiki_links": wiki_links,
        }

        subject = reasoning if reasoning else "general"

        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=metadata,
        )


__all__ = ["FRAMESDataset"]

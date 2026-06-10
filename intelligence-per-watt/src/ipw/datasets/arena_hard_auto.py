"""Arena-Hard-Auto v2 dataset provider."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from huggingface_hub import hf_hub_download

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

LOGGER = logging.getLogger(__name__)

_HF_REPO = "lmarena-ai/arena-hard-auto"
_QUESTION_FILE = "data/arena-hard-v2.0/question.jsonl"
_REFERENCE_MODEL = "gpt-4.1"
_REFERENCE_FILE_TEMPLATE = "data/arena-hard-v2.0/model_answer/{model}.jsonl"

_PROMPT_TEMPLATE = """Please answer the following user request as helpfully and accurately as possible.

{prompt}"""

_JUDGE_TEMPLATE = """You are judging two assistant answers to the same user request.

User request:
{prompt}

Reference answer:
{reference}

Candidate answer:
{candidate}

Judge overall helpfulness, correctness, completeness, and instruction following.
Return exactly one of these lines and nothing else:
verdict: candidate
verdict: reference
verdict: tie
"""


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@DatasetRegistry.register("arena-hard-v2")
@DatasetRegistry.register("arena-hard-auto-v2")
class ArenaHardAutoV2Dataset(DatasetProvider):
    """Arena-Hard-Auto v2 open-ended chat benchmark.

    The official metric is pairwise LLM judging against other model answers.
    This provider normalizes prompts for smoke/full execution and marks rows
    explicitly unscorable until a pairwise comparison target is supplied.
    """

    dataset_id = "arena-hard-auto-v2"
    dataset_name = "Arena-Hard-Auto V2"
    evaluation_method = "pairwise_judge"

    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        max_samples: Optional[int] = None,
        question_path: Optional[str] = None,
        reference_model: str = _REFERENCE_MODEL,
        reference_answer_path: Optional[str] = None,
    ) -> None:
        self._max_samples = max_samples
        self._question_path = Path(question_path) if question_path else None
        self._reference_model = reference_model
        self._reference_answer_path = (
            Path(reference_answer_path) if reference_answer_path else None
        )
        self._reference_answers = self._load_reference_answers()
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
        if not response or not response.strip():
            return False, {"reason": "empty_response"}
        if not record.answer or not record.answer.strip():
            return None, {
                "reason": "missing_reference_answer",
                "question_id": record.dataset_metadata.get("question_id"),
                "reference_model": self._reference_model,
            }

        judge_client = eval_client
        if judge_client is None:
            judge_client = ClientRegistry.create(
                self.eval_client or "openai",
                base_url=self.eval_base_url or "https://api.openai.com/v1",
                model=self.eval_model or "gpt-5-nano-2025-08-07",
            )
        if not hasattr(judge_client, "chat"):
            return None, {"reason": "no_llm_client"}

        prompt = _JUDGE_TEMPLATE.format(
            prompt=record.problem,
            reference=record.answer,
            candidate=response,
        )
        try:
            raw = judge_client.chat(
                system_prompt="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=256,
            )
        except Exception as exc:
            LOGGER.error("Arena-Hard scoring failed: %s", exc)
            return None, {"reason": "judge_error", "error": str(exc)}

        verdict = _parse_verdict(raw)
        if verdict is None:
            return None, {"reason": "missing_verdict", "raw_judge_output": raw}

        return verdict in {"candidate", "tie"}, {
            "match_type": "pairwise_judge",
            "verdict": verdict,
            "reference_model": self._reference_model,
            "raw_judge_output": raw,
        }

    def _resolve_question_path(self) -> Path:
        if self._question_path is not None:
            return self._question_path
        return Path(
            hf_hub_download(
                repo_id=_HF_REPO,
                repo_type="dataset",
                filename=_QUESTION_FILE,
            )
        )

    def _resolve_reference_answer_path(self) -> Path:
        if self._reference_answer_path is not None:
            return self._reference_answer_path
        return Path(
            hf_hub_download(
                repo_id=_HF_REPO,
                repo_type="dataset",
                filename=_REFERENCE_FILE_TEMPLATE.format(model=self._reference_model),
            )
        )

    def _load_reference_answers(self) -> dict[str, str]:
        try:
            rows = _load_jsonl(self._resolve_reference_answer_path())
        except Exception as exc:
            LOGGER.warning("Arena-Hard reference answers unavailable: %s", exc)
            return {}

        answers: dict[str, str] = {}
        for row in rows:
            uid = str(row.get("uid") or row.get("question_id") or row.get("id") or "")
            if not uid:
                continue
            answer = _extract_answer(row)
            if answer:
                answers[uid] = answer
        return answers

    def _build_records(self) -> List[DatasetRecord]:
        rows = _load_jsonl(self._resolve_question_path())
        if self._max_samples is not None:
            rows = rows[: self._max_samples]
        records: list[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            record = self._convert_row(raw, idx)
            if record is not None:
                records.append(record)
        return records

    def _convert_row(
        self,
        raw: MutableMapping[str, object],
        idx: int,
    ) -> Optional[DatasetRecord]:
        turns = raw.get("turns")
        if isinstance(turns, list) and turns:
            prompt = "\n\n".join(str(turn).strip() for turn in turns if str(turn).strip())
        else:
            prompt = str(raw.get("prompt") or raw.get("question") or "").strip()
        if not prompt:
            return None

        question_id = str(raw.get("uid") or raw.get("question_id") or raw.get("id") or idx)
        category = str(raw.get("category") or "chat")
        reference_answer = self._reference_answers.get(question_id, "")
        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "question_id": question_id,
            "category": category,
            "turns": turns if isinstance(turns, list) else [prompt],
            "workload_type": "chat",
            "reference_model": self._reference_model,
        }
        if not reference_answer:
            metadata["unscorable_reason"] = "missing_reference_answer"
            metadata["score_metadata"] = {
                "reason": "missing_reference_answer",
                "reference_model": self._reference_model,
            }
        return DatasetRecord(
            problem=_PROMPT_TEMPLATE.format(prompt=prompt),
            answer=reference_answer,
            subject=category,
            dataset_metadata=metadata,
        )


def _extract_answer(row: MutableMapping[str, object]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, MutableMapping):
                continue
            if message.get("role") != "assistant":
                continue
            content: Any = message.get("content")
            if isinstance(content, MutableMapping):
                return str(content.get("answer") or content.get("content") or "").strip()
            return str(content or "").strip()
    return str(row.get("answer") or row.get("response") or "").strip()


def _parse_verdict(raw: str) -> str | None:
    for line in (raw or "").splitlines():
        match = re.fullmatch(r"\s*verdict:\s*(candidate|reference|tie)\s*", line, re.I)
        if match:
            return match.group(1).lower()
    return None


__all__ = ["ArenaHardAutoV2Dataset"]

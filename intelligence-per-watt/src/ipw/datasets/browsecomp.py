"""BrowseComp browsing benchmark dataset."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import requests

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

_QUERY_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


@DatasetRegistry.register("browsecomp")
class BrowseCompDataset(DatasetProvider):
    """OpenAI BrowseComp benchmark for hard-to-find short-answer browsing tasks."""

    dataset_id = "browsecomp"
    dataset_name = "BrowseComp"
    evaluation_method = "browsecomp"

    def __init__(
        self,
        *,
        max_samples: Optional[int] = None,
        csv_path: Optional[str] = None,
        csv_url: str = _CSV_URL,
    ) -> None:
        self._max_samples = max_samples
        self._csv_path = csv_path
        self._csv_url = csv_url
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY or OPENAI_API_KEY for BrowseComp judging."
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

    def _build_records(self) -> List[DatasetRecord]:
        rows = self._load_raw_rows()
        records: list[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            record = self._convert_row(raw, idx)
            if record is not None:
                records.append(record)
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        if self._csv_path:
            text = open(self._csv_path, encoding="utf-8").read()
        else:
            response = requests.get(self._csv_url, timeout=60)
            response.raise_for_status()
            text = response.text
        rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
        if self._max_samples is not None:
            rows = rows[: self._max_samples]
        return rows

    def _convert_row(
        self,
        raw: MutableMapping[str, object],
        idx: int,
    ) -> Optional[DatasetRecord]:
        canary = str(raw.get("canary") or "")
        encrypted_problem = str(raw.get("problem") or "")
        encrypted_answer = str(raw.get("answer") or "")
        if not canary or not encrypted_problem or not encrypted_answer:
            return None

        question = _decrypt(encrypted_problem, canary)
        answer = _decrypt(encrypted_answer, canary)
        problem = _QUERY_TEMPLATE.format(question=question)

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "original_index": idx,
            "canary": canary,
            "workload_type": "deepresearch",
        }
        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject="browsing",
            dataset_metadata=metadata,
        )


__all__ = ["BrowseCompDataset"]

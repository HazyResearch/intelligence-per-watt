"""DeepResearch Bench dataset provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

from huggingface_hub import hf_hub_download

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_HF_REPO = "muset-ai/DeepResearch-Bench-Dataset"
_DEFAULT_REPORT = "generated_reports/openai-deepresearch.jsonl"

_PROMPT_TEMPLATE = """You are a deep research assistant. Conduct thorough research and produce a comprehensive, well-structured report with citations and analysis.

## Research Task
{prompt}"""


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@DatasetRegistry.register("deepresearchbench")
class DeepResearchBenchDataset(DatasetProvider):
    """DeepResearch Bench long-form research benchmark."""

    dataset_id = "deepresearchbench"
    dataset_name = "DeepResearch Bench"
    evaluation_method = "deepresearchbench"

    def __init__(
        self,
        *,
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
        reference_report: str = _DEFAULT_REPORT,
    ) -> None:
        self._max_samples = max_samples
        self._data_dir = Path(data_dir) if data_dir else None
        self._reference_report = reference_report
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY or OPENAI_API_KEY for research-report judging."
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

    def _resolve_file(self, filename: str) -> Path:
        if self._data_dir is not None:
            path = self._data_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        return Path(
            hf_hub_download(
                repo_id=_HF_REPO,
                repo_type="dataset",
                filename=filename,
            )
        )

    def _build_records(self) -> List[DatasetRecord]:
        report_path = self._resolve_file(self._reference_report)
        rows = _load_jsonl(report_path)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        records: list[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            prompt = str(raw.get("prompt") or "").strip()
            article = str(raw.get("article") or "").strip()
            task_id = raw.get("id", idx)
            if not prompt:
                continue
            problem = _PROMPT_TEMPLATE.format(prompt=prompt)
            metadata: MutableMapping[str, object] = {
                "dataset_name": self.dataset_name,
                "task_id": task_id,
                "reference_report": self._reference_report,
                "workload_type": "deepresearch",
            }
            records.append(
                DatasetRecord(
                    problem=problem,
                    answer=article,
                    subject="deepresearch",
                    dataset_metadata=metadata,
                )
            )
        return records


__all__ = ["DeepResearchBenchDataset"]

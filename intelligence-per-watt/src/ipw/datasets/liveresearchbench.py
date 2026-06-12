"""LiveResearchBench dataset provider."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_REPO_URL = "https://github.com/Ayanami0730/deep_research_bench.git"
_CACHE_DIR = Path.home() / ".cache" / "liveresearch_bench"

_PROMPT_TEMPLATE = """You are a deep research assistant. Conduct thorough research on the following topic and produce a comprehensive, well-structured research report with citations and analysis.

## Research Task
{prompt}"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_criteria_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if row_id is not None:
            index[int(row_id)] = row
    return index


@DatasetRegistry.register("liveresearch")
@DatasetRegistry.register("liveresearchbench")
class LiveResearchBenchDataset(DatasetProvider):
    """LiveResearchBench deep-research task set from deep_research_bench."""

    dataset_id = "liveresearchbench"
    dataset_name = "LiveResearchBench"
    evaluation_method = "liveresearchbench"

    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        path: Optional[str] = None,
    ) -> None:
        self._max_samples = max_samples
        self._split = split
        self._seed = seed
        self._local_path = Path(path) if path else None
        self._repo_dir = self._local_path or _CACHE_DIR
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if self._local_path is None and shutil.which("git") is None:
            issues.append("git binary not found. Install git or pass a local LiveResearchBench path.")
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

    def _ensure_repo(self) -> Path:
        if self._local_path is not None:
            if not self._local_path.exists():
                raise FileNotFoundError(self._local_path)
            return self._local_path
        if not self._repo_dir.exists():
            self._repo_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", _REPO_URL, str(self._repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        return self._repo_dir

    def _build_records(self) -> List[DatasetRecord]:
        repo_dir = self._ensure_repo()
        query_path = repo_dir / "data" / "prompt_data" / "query.jsonl"
        if not query_path.exists():
            raise FileNotFoundError(query_path)
        queries = _load_jsonl(query_path)

        criteria_index: dict[int, dict[str, Any]] = {}
        criteria_path = repo_dir / "data" / "criteria_data" / "criteria.jsonl"
        if criteria_path.exists():
            criteria_index = _build_criteria_index(_load_jsonl(criteria_path))

        if self._split in ("en", "zh"):
            queries = [q for q in queries if q.get("language") == self._split]
        if self._seed is not None:
            random.Random(self._seed).shuffle(queries)
        if self._max_samples is not None:
            queries = queries[: self._max_samples]

        records: list[DatasetRecord] = []
        for idx, query in enumerate(queries):
            prompt = str(query.get("prompt") or "").strip()
            if not prompt:
                continue
            q_id = query.get("id", idx)
            criteria = criteria_index.get(int(q_id)) if q_id is not None else None
            metadata: MutableMapping[str, object] = {
                "dataset_name": self.dataset_name,
                "task_id": q_id,
                "topic": query.get("topic", ""),
                "language": query.get("language", "en"),
                "workload_type": "deepresearch",
            }
            if criteria:
                metadata["dimension_weight"] = criteria.get("dimension_weight", {})
                metadata["criterions"] = criteria.get("criterions", {})
                metadata["rubric"] = json.dumps(criteria.get("criterions", {}), ensure_ascii=False)
            records.append(
                DatasetRecord(
                    problem=_PROMPT_TEMPLATE.format(prompt=prompt),
                    answer="__research_report_rubric_judge__",
                    subject=str(query.get("topic") or "deepresearch"),
                    dataset_metadata=metadata,
                )
            )

        return records


__all__ = ["LiveResearchBenchDataset"]

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from huggingface_hub import snapshot_download

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gaia_benchmark"

_DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

{file}

Here is the question:

{question}"""


@DatasetRegistry.register("gaia")
class GAIADataset(DatasetProvider):
    """GAIA benchmark dataset (gaia-benchmark/GAIA).

    Loads from HuggingFace, downloads artifacts (attached files) on first use,
    and yields ``DatasetRecord`` instances ready for profiling.
    """

    dataset_id = "gaia"
    dataset_name = "GAIA"
    evaluation_method = "gaia"

    _hf_path = "gaia-benchmark/GAIA"
    _default_subset = "2023_all"
    _default_split = "validation"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._split = split or self._default_split
        self._subset = subset or self._default_subset
        self._max_samples = max_samples
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
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

    def _ensure_downloaded(self) -> Path:
        dataset_location = self._cache_dir / "GAIA"
        if not dataset_location.exists():
            dataset_location.mkdir(parents=True, exist_ok=True)
            try:
                snapshot_download(
                    repo_id=self._hf_path,
                    repo_type="dataset",
                    local_dir=str(dataset_location),
                )
            except Exception:
                shutil.rmtree(dataset_location, ignore_errors=True)
                raise
        return dataset_location

    def _build_records(self) -> List[DatasetRecord]:
        dataset_location = self._ensure_downloaded()
        dataset = load_dataset(
            str(dataset_location),
            name=self._subset,
            split=self._split,
        )

        rows: Sequence[MutableMapping[str, object]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        files_location = dataset_location / "2023" / self._split

        records: List[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(raw, files_location)
            if record is not None:
                records.append(record)
        return records

    def _convert_row(
        self,
        raw: MutableMapping[str, object],
        files_location: Path,
    ) -> Optional[DatasetRecord]:
        task_id = str(raw.get("task_id") or "")
        question = str(raw.get("Question") or "").strip()
        answer = str(raw.get("Final answer") or "").strip()
        level = raw.get("Level")

        if not question or not answer:
            return None

        # Discover associated files
        file_name: Optional[str] = None
        file_path: Optional[Path] = None
        if files_location.exists():
            files = [f for f in os.listdir(files_location) if task_id in f]
            if files:
                file_name = files[0]
                file_path = files_location / file_name

        # Format the prompt
        if file_name and file_path:
            file_info = (
                f"The following file is referenced in the question below and you will "
                f"likely need to use it in order to find the correct answer.\n"
                f"File name: {file_name}\n"
                f"File path: {file_path}\n"
                f"Use the file reading tools to access this file."
            )
        elif file_name:
            file_info = (
                f"The following file is referenced in the question: {file_name}\n"
                f"(Note: File path not available)"
            )
        else:
            file_info = ""

        problem = _DEFAULT_INPUT_PROMPT.format(file=file_info, question=question)

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "task_id": task_id,
            "level": level,
            "file_name": file_name,
            "file_path": str(file_path) if file_path else None,
        }

        subject = f"level_{level}" if level else "general"

        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=metadata,
        )


__all__ = ["GAIADataset"]

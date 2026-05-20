"""GDPval dataset (openai/gdpval).

Real-world professional tasks across 44 occupations. Each task has:
- A long-form prompt
- A set of reference files (Excel/PDF/Word) the agent should read
- A set of deliverable files the model is expected to produce
- A machine-readable rubric (rubric_json) for grading

This provider downloads reference files into a per-task cache, augments the
prompt with on-disk paths, and routes scoring to the ``gdpval`` evaluation
handler (rubric-based LLM-as-judge).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import unquote

from huggingface_hub import hf_hub_download

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gdpval"

_PROMPT_TEMPLATE = """You are a domain expert in: {occupation} ({sector}).

Complete the following professional task. Use any reference files provided \
and produce a high-quality answer that a professional in this role would \
accept as a deliverable.

{file_section}

## Task

{task_prompt}

## Tools and workflow

You have `terminal` (bash) and `file_editor`. The reference files for this \
task are at the absolute paths listed above — read them directly.

This task asks you to read the reference files (if any), produce one or more \
output files in the current directory, and call `finish` describing what you \
produced.

Available system tools and Python libraries (installed and ready):

- `web_search "<query>" [N]` — DuckDuckGo search; prints JSON list of \
  {{title, url, snippet}}. Use for tasks that need internet research. After \
  finding a URL, fetch with `curl -L`.
- `tesseract` + `pdftoppm`/`pdftotext`/`pdfinfo` — for OCR of scanned PDFs.
- `ffmpeg` — for audio/video manipulation (e.g. extract audio from mp4).
- Python: `pdfplumber`, `openpyxl`, `xlsxwriter`, `python-docx`, \
  `python-pptx`, `reportlab`, `fpdf2`, `weasyprint`, `Pillow`, \
  `pdf2image`, `psd_tools`, `whisper`, `pandas`.

## Important: produce-first, verify-before-finish

1. PRODUCE FIRST: As soon as you have any plausible understanding of the \
   task, write a first draft of the output file with `file_editor:create`. \
   A partial deliverable on disk is far better than no deliverable.

2. VERIFY AFTER WRITING SCRIPTS: After writing a generator script (e.g. \
   `build_workbook.py`), you MUST run it AND then `ls -la` to confirm the \
   deliverable file actually appears. A saved script with no output file \
   on disk is worth zero. If the script errors, read the traceback, fix \
   it, re-run — iterate until the actual deliverable file exists.

3. AVOID THE EXPLORATION LOOP: You CAN read different sections of an input \
   file you've already opened, and you SHOULD re-read your own output files \
   to verify them. But do NOT repeat the same high-level survey: don't \
   `ls inputs/` twice, don't "examine the template structure" twice, don't \
   "dump all source files" twice. After your first sweep through the \
   inputs, you have enough to write a draft — start writing.

4. Before calling `finish`, run `ls -la *.xlsx *.pdf *.docx *.pptx` \
   (whichever extension the task asked for) and confirm the deliverable \
   file exists on disk. Do not ask the user questions; make reasonable \
   assumptions when information is ambiguous.
"""


def _hf_uri_to_repo_path(
    hf_uri: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse a HuggingFace dataset URI into ``(repo, file_path, revision)``.

    Accepts both forms:
        hf://datasets/<org>/<name>/<file_path>
        hf://datasets/<org>/<name>@<revision>/<file_path>

    The file_path is URL-decoded (``%20`` → space, etc.).
    """
    if not hf_uri or not isinstance(hf_uri, str):
        return None
    prefix = "hf://datasets/"
    if not hf_uri.startswith(prefix):
        return None
    remainder = hf_uri[len(prefix):]
    parts = remainder.split("/", 2)
    if len(parts) < 3:
        return None
    org, name_with_rev, file_path = parts
    if "@" in name_with_rev:
        name, revision = name_with_rev.split("@", 1)
    else:
        name, revision = name_with_rev, None
    repo = f"{org}/{name}"
    return repo, unquote(file_path), revision


@DatasetRegistry.register("gdpval")
class GDPvalDataset(DatasetProvider):
    """OpenAI GDPval dataset (openai/gdpval).

    Loads 220 professional tasks from HuggingFace, downloads associated
    reference files on demand, and yields one ``DatasetRecord`` per task.
    Scoring uses the rubric-based ``gdpval`` evaluation handler.
    """

    dataset_id = "gdpval"
    dataset_name = "GDPval"
    evaluation_method = "gdpval"

    _hf_path = "openai/gdpval"
    _default_split = "train"
    _default_subset = "default"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        download_files: bool = True,
        shard_idx: Optional[int] = None,
        n_shards: Optional[int] = None,
    ) -> None:
        self._split = split or self._default_split
        self._subset = subset or self._default_subset
        self._max_samples = max_samples
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_files = download_files
        if (shard_idx is None) != (n_shards is None):
            raise ValueError("shard_idx and n_shards must be specified together")
        if n_shards is not None and n_shards <= 0:
            raise ValueError("n_shards must be > 0")
        if shard_idx is not None and not (0 <= shard_idx < n_shards):
            raise ValueError(f"shard_idx must be in [0, {n_shards})")
        self._shard_idx = shard_idx
        self._n_shards = n_shards
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY (preferred) or "
                "OPENAI_API_KEY — GDPval scoring uses LLM-as-judge over the rubric."
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
        dataset = load_dataset(self._hf_path, name=self._subset, split=self._split)

        rows: Sequence[MutableMapping[str, Any]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        # Shard by global task index, modulo n_shards. Round-robin distribution
        # gives each shard a similar mix of task difficulty (rather than e.g.
        # shard 0 getting all the bank-audit tasks).
        if self._n_shards is not None:
            rows = [r for i, r in enumerate(rows) if i % self._n_shards == self._shard_idx]

        records: list[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(dict(raw))
            if record is not None:
                records.append(record)
        return records

    def _convert_row(self, raw: MutableMapping[str, Any]) -> Optional[DatasetRecord]:
        task_id = str(raw.get("task_id") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        if not task_id or not prompt:
            return None

        occupation = str(raw.get("occupation") or "").strip() or "Professional"
        sector = str(raw.get("sector") or "").strip() or "General"

        reference_hf_uris = list(raw.get("reference_file_hf_uris") or [])
        reference_urls = list(raw.get("reference_file_urls") or [])
        deliverable_hf_uris = list(raw.get("deliverable_file_hf_uris") or [])

        rubric_json_raw = raw.get("rubric_json") or ""
        rubric_pretty = raw.get("rubric_pretty") or ""

        # Materialize reference files on disk (downloaded lazily on first use)
        inputs_dir = self._cache_dir / task_id / "inputs"
        if self._download_files:
            local_paths = self._materialize_files(reference_hf_uris, inputs_dir)
        else:
            local_paths = []

        # Build a human-readable file listing for the prompt.
        # Files are copied into the agent's working environment under
        # ./inputs/ (OpenHands LocalConversation) or /workspace/inputs/
        # (Terminus Docker) — see ipw/agents/{openhands,terminus}.py.
        if local_paths:
            file_lines = "\n".join(f"- {p.name}" for p in local_paths)
            file_section = (
                "## Reference files\n\n"
                "The following files are available in the working directory "
                "under `inputs/` (or `/workspace/inputs/` if you are in a "
                "sandboxed shell). Read them with whatever file tools you "
                "have available.\n\n"
                f"{file_lines}\n"
            )
        elif reference_urls:
            url_lines = "\n".join(f"- {u}" for u in reference_urls)
            file_section = (
                "## Reference files (download from URL)\n\n"
                f"{url_lines}\n"
            )
        else:
            file_section = ""

        task_prompt = _PROMPT_TEMPLATE.format(
            occupation=occupation,
            sector=sector,
            file_section=file_section,
            task_prompt=prompt,
        )

        metadata: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "task_id": task_id,
            "instance_id": task_id,
            "occupation": occupation,
            "sector": sector,
            "reference_file_hf_uris": reference_hf_uris,
            "reference_file_urls": reference_urls,
            "deliverable_file_hf_uris": deliverable_hf_uris,
            "gdpval_inputs_dir": str(inputs_dir) if local_paths else None,
            "rubric_pretty": rubric_pretty,
            "rubric_json": rubric_json_raw,
        }

        # The "answer" slot is unused for GDPval (no reference answers); we
        # store rubric_pretty there so it shows up in artifact dumps.
        return DatasetRecord(
            problem=task_prompt,
            answer=rubric_pretty if isinstance(rubric_pretty, str) else "",
            subject=occupation,
            dataset_metadata=metadata,
        )

    def _materialize_files(self, hf_uris: List[str], dest_dir: Path) -> List[Path]:
        """Download each HF URI into ``dest_dir`` (idempotent, cached)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[Path] = []
        for uri in hf_uris:
            parsed = _hf_uri_to_repo_path(uri)
            if parsed is None:
                LOGGER.warning("Skipping non-HF reference URI: %s", uri)
                continue
            repo, path_in_repo, revision = parsed
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=path_in_repo,
                    revision=revision,
                    cache_dir=str(self._cache_dir / "_hf_cache"),
                )
            except Exception as exc:
                LOGGER.warning("Failed to download %s: %s", uri, exc)
                continue
            # Copy into dest_dir with a flat filename so agents see a clean tree
            target = dest_dir / Path(path_in_repo).name
            if not target.exists():
                shutil.copy2(local, target)
            local_paths.append(target)
        return local_paths


__all__ = ["GDPvalDataset"]

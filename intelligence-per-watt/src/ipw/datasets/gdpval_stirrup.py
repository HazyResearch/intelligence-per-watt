"""GDPval dataset variant for Stirrup-style file submission.

This preserves IPW's per-prompt telemetry flow while using a prompt and
submission contract close to Artificial Analysis' GDPval-AA task-submission
setup. Scoring is single-submission deliverable judging, not pairwise Elo.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .gdpval import GDPvalDataset

_ENVIRONMENT_SUMMARY = """Your environment comes preinstalled with Python and common data, document, media, and graphics tooling.
Expect packages and system tools such as pandas, numpy, scipy, matplotlib,
Pillow, openpyxl, python-docx, python-pptx, PyMuPDF/pdfplumber, reportlab,
weasyprint, ffmpeg, tesseract, poppler utilities, LibreOffice, pandoc,
graphviz, and standard build tools to be available when supported by the
selected execution backend.
"""

_STIRRUP_PROMPT_TEMPLATE = """You are tasked with completing a specific assignment.

## Environment

The shell execution tool provides access to a Linux-based execution environment
with a filesystem where you can create, read, and modify files.

{environment_summary}

## Reference Files Location

The reference files for the task are available in your environment's file
system. The agent system prompt may also list uploaded file paths.

Here are the reference file names for this task:

<reference_files>
{reference_files}
</reference_files>

## Completing Your Work

In order to complete the task you must use the finish tool to submit your work.
If you do not use the finish tool you will fail this task.

Work discipline:
1. Use the provided reference files in the execution environment. Do not browse
   or re-download a reference file unless it is genuinely missing from the
   current directory.
2. The shell tool starts in the task working directory. Use relative paths for
   shell commands. Do not use `/mnt/data`, `/tmp`, `~/`, or other absolute paths
   in shell commands.
3. Keep exploratory command output small. Inspect only the rows, columns, and
   metadata needed to produce the deliverable.
4. Decide the output file name and extension before broad exploration.
5. Within your first five shell/tool actions, create a draft deliverable file
   with the correct extension. Improve that file iteratively. A partial but
   valid deliverable is better than no submitted file.
6. Produce a first draft output file early, then verify it exists with `ls -la`.
   Saved scripts without a produced output file are not deliverables.
7. If a turn-limit warning appears, stop exploring and immediately create or
   update, verify, and submit the best deliverable file you have.
8. To submit, pass only the relative output file path shown by `ls -la` to the
   finish tool.

Tool-call discipline:
- For shell work, call `code_exec` with a compact JSON object like
  `{{"cmd":"ls -la"}}`.
- Do not put large multi-line scripts directly in the tool JSON if you can avoid
  it. Prefer short commands that create or run a script in the current directory.
- Never run `cd /home/user`, `cd /home/ubuntu`, `cd /tmp`, or `cd ~/...`; the
  shell already starts in the task working directory.

Required in your finish call:
1. A brief summary of what you accomplished
2. A list of relative file paths for all files you want to submit. Do not submit folders.

Finish path rules:
- Submit only files you created in the current working directory.
- Use paths such as `analysis.xlsx`, `report.pdf`, or `outputs/chart.png`.
- Never submit `/home/user/...`, `/home/ubuntu/...`, `/tmp/...`, `~/...`, or a directory.
- Before calling finish, run `ls -la` and submit exactly the file path shown there.

## Task

Here is the task you need to complete:

<task>
{task}
</task>

Please begin working on the task now.
"""


@DatasetRegistry.register("gdpval-stirrup")
@DatasetRegistry.register("gdpval-aa-single")
class GDPvalStirrupDataset(GDPvalDataset):
    """GDPval task provider for Stirrup-backed file submissions."""

    dataset_id = "gdpval-stirrup"
    dataset_name = "GDPval Stirrup Single-Submission"
    evaluation_method = "gdpval-deliverable"
    requires_serial_telemetry = True

    def __init__(
        self,
        *,
        environment_summary: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._environment_summary = environment_summary or _ENVIRONMENT_SUMMARY
        super().__init__(**kwargs)

    def verify_requirements(self) -> list[str]:
        if self.eval_client and self.eval_client != "openai":
            return []
        issues = super().verify_requirements()
        # The dataset itself can still be inspected without Stirrup. The agent
        # dependency is checked by the stirrup agent at construction time.
        return [
            issue.replace("GDPval scoring uses LLM-as-judge over the rubric", "GDPval-Stirrup scoring uses deliverable-aware LLM judging")
            for issue in issues
        ]

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
        deliverable_files = list(raw.get("deliverable_files") or [])

        rubric_json_raw = raw.get("rubric_json") or ""
        rubric_pretty = raw.get("rubric_pretty") or ""

        inputs_dir = self._cache_dir / task_id / "inputs"
        if self._download_files:
            local_paths = self._materialize_files(reference_hf_uris, inputs_dir)
        else:
            local_paths = []

        if local_paths:
            reference_files = "\n".join(f"- {p.name}" for p in local_paths)
        elif reference_urls:
            reference_files = "\n".join(f"- {u}" for u in reference_urls)
        else:
            reference_files = "(none)"

        task_prompt = _STIRRUP_PROMPT_TEMPLATE.format(
            environment_summary=self._environment_summary,
            reference_files=reference_files,
            task=prompt,
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
            "deliverable_files": deliverable_files,
            "gdpval_inputs_dir": str(inputs_dir) if local_paths else None,
            "rubric_pretty": rubric_pretty,
            "rubric_json": rubric_json_raw,
            "workload_type": "gdpval-stirrup",
        }

        return DatasetRecord(
            problem=task_prompt,
            answer=rubric_pretty if isinstance(rubric_pretty, str) else "",
            subject=occupation,
            dataset_metadata=metadata,
        )


__all__ = ["GDPvalStirrupDataset"]

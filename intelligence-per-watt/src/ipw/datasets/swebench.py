from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider
from ._git_workspace import prepare_git_workspace

_DATASET_PATHS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "verified_mini": "MariusHobbhahn/swe-bench-verified-mini",
}


def _parse_test_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
    if isinstance(value, list):
        return value
    return []


_FENCED_DIFF_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_PATCH_MARKERS = ("diff --git", "--- a/", "+++ b/", "@@ ")


def _extract_patch(response: str) -> str:
    fenced = _FENCED_DIFF_RE.findall(response or "")
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)

    lines = (response or "").splitlines()
    patch_lines: list[str] = []
    in_patch = False
    for line in lines:
        if any(line.startswith(marker) for marker in _PATCH_MARKERS):
            in_patch = True
        if in_patch:
            patch_lines.append(line)
    return "\n".join(patch_lines).strip()


def _run_cmd(cmd: list[str], *, cwd: Path, timeout_s: int) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode, result.stdout[-12000:]


def _git_diff(workspace: Path) -> str:
    try:
        rc, out = _run_cmd(["git", "diff", "--binary"], cwd=workspace, timeout_s=60)
    except Exception:
        return ""
    return out if rc == 0 else ""


def _apply_patch_if_needed(workspace: Path, patch: str, timeout_s: int) -> tuple[bool, str]:
    if not patch:
        return True, "no_patch_to_apply"
    if not patch.endswith("\n"):
        patch += "\n"
    existing_diff = _git_diff(workspace)
    if existing_diff.strip():
        return True, "workspace_already_modified"
    return _apply_patch(workspace, patch, timeout_s)


def _apply_patch(workspace: Path, patch: str, timeout_s: int) -> tuple[bool, str]:
    if not patch:
        return True, "no_patch_to_apply"
    if not patch.endswith("\n"):
        patch += "\n"
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if check.returncode != 0:
        return False, check.stdout[-4000:]
    applied = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return applied.returncode == 0, applied.stdout[-4000:]


def _test_command(metadata: MutableMapping[str, object]) -> list[str]:
    explicit = str(
        metadata.get("test_cmd") or os.getenv("IPW_SWEBENCH_TEST_CMD") or ""
    ).strip()
    if explicit:
        return ["bash", "-lc", explicit]
    tests = list(metadata.get("fail_to_pass") or []) + list(metadata.get("pass_to_pass") or [])
    if tests:
        return ["python", "-m", "pytest", *[str(test) for test in tests]]
    return []


@DatasetRegistry.register("swebench")
class SWEBenchDataset(DatasetProvider):
    """SWE-bench dataset (princeton-nlp/SWE-bench_Verified).

    Supports two variants:
    - ``verified``: Full 500-task dataset
    - ``verified_mini``: 50-task subset
    """

    dataset_id = "swebench"
    dataset_name = "SWE-bench"
    evaluation_method = "swebench"

    _default_split = "test"
    _default_variant = "verified_mini"

    # SWE-bench correctness is determined by test execution, not LLM judge.
    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        variant: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._variant = variant or self._default_variant
        if self._variant not in _DATASET_PATHS:
            raise ValueError(
                f"Unknown SWE-bench variant '{self._variant}'. "
                f"Choose from: {list(_DATASET_PATHS)}"
            )
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def prepare_workspace(self, record: DatasetRecord, workspace: Path) -> None:
        prepare_git_workspace(record.dataset_metadata, workspace)

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """Score by applying the produced patch and running available tests."""
        if not response or not response.strip():
            return False, {"reason": "empty_response"}
        metadata = record.dataset_metadata
        workspace_raw = metadata.get("workspace_path")
        workspace = Path(str(workspace_raw)) if workspace_raw else None
        if workspace is None or not workspace.exists():
            return False, {
                "reason": "workspace_unavailable",
                "instance_id": metadata.get("instance_id", ""),
            }

        patch = _extract_patch(response)
        existing_diff = _git_diff(workspace)
        if not patch and not existing_diff.strip():
            return False, {
                "reason": "no_patch_or_workspace_diff",
                "instance_id": metadata.get("instance_id", ""),
            }

        timeout_s = int(os.getenv("IPW_SWEBENCH_TEST_TIMEOUT", "600"))
        ok, detail = _apply_patch_if_needed(workspace, patch, timeout_s)
        if not ok:
            return False, {
                "reason": "patch_apply_failed",
                "instance_id": metadata.get("instance_id", ""),
                "apply_output": detail,
                "has_patch": bool(patch),
            }

        test_patch = str(metadata.get("test_patch") or "").strip()
        test_patch_applied = False
        test_patch_output = ""
        if test_patch:
            test_patch_ok, test_patch_output = _apply_patch(
                workspace,
                test_patch,
                timeout_s,
            )
            if not test_patch_ok:
                return False, {
                    "reason": "test_patch_apply_failed",
                    "instance_id": metadata.get("instance_id", ""),
                    "apply_output": test_patch_output,
                    "has_patch": bool(patch or existing_diff.strip()),
                }
            test_patch_applied = True

        cmd = _test_command(metadata)
        if not cmd:
            return False, {
                "reason": "no_test_command",
                "instance_id": metadata.get("instance_id", ""),
                "has_patch": bool(patch or existing_diff.strip()),
            }

        try:
            rc, output = _run_cmd(cmd, cwd=workspace, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as exc:
            return False, {
                "reason": "test_timeout",
                "instance_id": metadata.get("instance_id", ""),
                "timeout_s": timeout_s,
                "test_output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            }
        except Exception as exc:
            return False, {
                "reason": "test_execution_error",
                "instance_id": metadata.get("instance_id", ""),
                "error": str(exc),
            }

        return rc == 0, {
            "match_type": "test_execution",
            "instance_id": metadata.get("instance_id", ""),
            "test_command": shlex.join(cmd),
            "test_returncode": rc,
            "test_output": output,
            "has_patch": bool(patch or existing_diff.strip()),
            "test_patch_applied": test_patch_applied,
            "test_patch_output": test_patch_output,
        }

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _build_records(self) -> List[DatasetRecord]:
        rows = self._load_raw_rows()
        records: List[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(raw)
            if record is not None:
                records.append(record)
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        hf_path = _DATASET_PATHS[self._variant]
        dataset = load_dataset(hf_path, split=self._split)
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

    def _convert_row(self, raw: MutableMapping[str, object]) -> Optional[DatasetRecord]:
        instance_id = str(raw.get("instance_id") or "")
        repo = str(raw.get("repo") or "")
        problem_statement = str(raw.get("problem_statement") or "").strip()

        if not instance_id or not problem_statement:
            return None

        # The problem is the issue description
        problem = problem_statement

        # The "answer" is the ground-truth patch
        patch = str(raw.get("patch") or "")

        fail_to_pass = _parse_test_list(raw.get("FAIL_TO_PASS", ""))
        pass_to_pass = _parse_test_list(raw.get("PASS_TO_PASS", ""))

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": raw.get("base_commit"),
            "hints_text": raw.get("hints_text"),
            "version": raw.get("version"),
            "test_patch": raw.get("test_patch"),
            "created_at": raw.get("created_at"),
            "environment_setup_commit": raw.get("environment_setup_commit"),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "difficulty": raw.get("difficulty"),
            "variant": self._variant,
        }

        return DatasetRecord(
            problem=problem,
            answer=patch,
            subject=repo,
            dataset_metadata=metadata,
        )


__all__ = ["SWEBenchDataset"]

from __future__ import annotations

import json
import os
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
from .swebench import _apply_patch, _apply_patch_if_needed, _extract_patch, _git_diff, _run_cmd

_DEFAULT_INPUT_PROMPT = """You are a software performance engineer. Your task is to optimize the code in the repository to improve performance.

Repository: {repo}

## Problem Statement
{problem_statement}

## Workload Description
{workload}

## Expected Speedup
The optimization should achieve approximately {expected_speedup:.1f}x speedup.

## Instructions
1. Analyze the codebase to identify performance bottlenecks
2. Implement optimizations that improve performance
3. Ensure all existing tests still pass
4. Generate a git patch with your changes

Please provide your optimization patch in unified diff format."""


def _parse_test_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value] if value else []
    return []


def _test_command(metadata: MutableMapping[str, object]) -> list[str]:
    explicit = str(
        metadata.get("test_cmd") or os.getenv("IPW_SWEFFICIENCY_TEST_CMD") or ""
    ).strip()
    if explicit:
        return ["bash", "-lc", explicit]
    tests = list(metadata.get("covering_tests") or []) + list(metadata.get("pass_to_pass") or [])
    if tests:
        return ["python", "-m", "pytest", *[str(test) for test in tests]]
    return []


@DatasetRegistry.register("swefficiency")
class SWEfficiencyDataset(DatasetProvider):
    """SWEfficiency benchmark dataset (swefficiency/swefficiency).

    Software performance optimization benchmark (SWE-bench style).
    """

    dataset_id = "swefficiency"
    dataset_name = "SWEfficiency"
    evaluation_method = "swefficiency"

    _hf_path = "swefficiency/swefficiency"
    _default_split = "test"

    # SWEfficiency does not use LLM judge by default -- correctness is
    # determined by running test suites, so we leave eval settings at None.
    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

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

    def prepare_workspace(self, record: DatasetRecord, workspace: Path) -> None:
        prepare_git_workspace(record.dataset_metadata, workspace)

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """Score by applying the produced optimization and running available tests."""
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

        timeout_s = int(os.getenv("IPW_SWEFFICIENCY_TEST_TIMEOUT", "600"))
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

        rebuild_cmd = str(metadata.get("rebuild_cmd") or "").strip()
        rebuild_output = ""
        if rebuild_cmd:
            try:
                rebuild_rc, rebuild_output = _run_cmd(
                    ["bash", "-lc", rebuild_cmd],
                    cwd=workspace,
                    timeout_s=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return False, {
                    "reason": "rebuild_timeout",
                    "instance_id": metadata.get("instance_id", ""),
                    "timeout_s": timeout_s,
                    "rebuild_output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                }
            except Exception as exc:
                return False, {
                    "reason": "rebuild_error",
                    "instance_id": metadata.get("instance_id", ""),
                    "error": str(exc),
                }
            if rebuild_rc != 0:
                return False, {
                    "reason": "rebuild_failed",
                    "instance_id": metadata.get("instance_id", ""),
                    "rebuild_command": rebuild_cmd,
                    "rebuild_output": rebuild_output,
                    "has_patch": bool(patch or existing_diff.strip()),
                }

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
            "rebuild_command": rebuild_cmd,
            "rebuild_output": rebuild_output,
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
                if self._max_samples is not None and len(records) >= self._max_samples:
                    break
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        dataset = load_dataset(self._hf_path, split=self._split)
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
        workload = str(raw.get("workload") or "")
        speedup = float(raw.get("speedup", raw.get("expected_speedup", 1.0)) or 1.0)

        if not instance_id or not problem_statement:
            return None

        problem = _DEFAULT_INPUT_PROMPT.format(
            repo=repo,
            problem_statement=problem_statement,
            workload=workload,
            expected_speedup=speedup,
        )

        # The "answer" is the ground-truth patch
        patch = str(raw.get("patch") or "")

        covering_tests = _parse_test_list(
            raw.get("covering_tests", raw.get("COVERING_TESTS", []))
        )
        pass_to_pass = _parse_test_list(
            raw.get("pass_to_pass", raw.get("PASS_TO_PASS", []))
        )

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": raw.get("base_commit"),
            "test_patch": raw.get("test_patch"),
            "test_cmd": raw.get("test_cmd"),
            "rebuild_cmd": raw.get("rebuild_cmd"),
            "image_name": raw.get("image_name"),
            "speedup": speedup,
            "covering_tests": covering_tests,
            "pass_to_pass": pass_to_pass,
        }

        return DatasetRecord(
            problem=problem,
            answer=patch,
            subject=repo,
            dataset_metadata=metadata,
        )


__all__ = ["SWEfficiencyDataset"]

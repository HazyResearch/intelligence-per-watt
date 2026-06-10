"""LiveCodeBench code-generation dataset with local test execution scoring."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

_HF_DATASET = "livecodebench/code_generation_lite"
_HF_DATASET_FULL = "livecodebench/code_generation"
_DEFAULT_SPLIT = "test"
_TIMEOUT_SECONDS = 30

_PROMPT_TEMPLATE = """Solve the following competitive programming problem. Write a complete Python solution that reads from stdin and writes to stdout.

## Problem
{problem_statement}

## Input Format
{input_format}

## Output Format
{output_format}

## Constraints
{constraints}

## Examples
{examples}

Write ONLY the Python code solution. Do not include explanations."""


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _format_examples(example_inputs: list[str], example_outputs: list[str]) -> str:
    parts = []
    for idx, (inp, out) in enumerate(zip(example_inputs, example_outputs), 1):
        parts.append(f"Input {idx}:\n{inp}\n\nOutput {idx}:\n{out}")
    return "\n\n".join(parts) if parts else "No examples provided."


def _extract_code(answer: str) -> str:
    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    heredoc = _extract_heredoc_code(answer)
    if heredoc:
        return heredoc

    lines = answer.strip().splitlines()
    code_lines: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(
            (
                "def ",
                "class ",
                "from ",
                "import ",
                "if __name__",
                "for ",
                "while ",
                "input(",
                "print(",
                "sys.stdin",
            )
        ):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else answer.strip()


def _extract_heredoc_code(answer: str) -> str:
    """Extract code written through shell here-docs from terminal transcripts."""
    lines = (answer or "").splitlines()
    for idx, line in enumerate(lines):
        match = re.search(r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_-]*)['\"]?", line)
        if not match:
            continue
        delimiter = match.group(1)
        captured: list[str] = []
        for raw_line in lines[idx + 1 :]:
            cleaned = _clean_heredoc_line(raw_line)
            if cleaned.strip() == delimiter:
                return "\n".join(captured).strip()
            captured.append(cleaned)
    return ""


def _clean_heredoc_line(raw_line: str) -> str:
    stripped = raw_line.lstrip()
    if stripped.startswith("> "):
        return stripped[2:]
    if stripped == ">":
        return ""
    # Some terminal transcripts include short continuation-prompt fragments
    # such as "xe>" or "y>" as standalone lines. They are not code.
    if re.fullmatch(r"[A-Za-z0-9_./:@-]{1,24}>\s*", stripped):
        return ""
    prompt_prefix = re.match(r"\s*[A-Za-z0-9_./:@-]{1,24}>\s?(.*)$", raw_line)
    if prompt_prefix:
        return prompt_prefix.group(1)
    return raw_line


def _numeric_match(actual: str, expected: str, rel_tol: float = 1e-6) -> bool:
    actual_parts = actual.split()
    expected_parts = expected.split()
    if len(actual_parts) != len(expected_parts):
        return False
    for actual_part, expected_part in zip(actual_parts, expected_parts):
        try:
            actual_value = float(actual_part)
            expected_value = float(expected_part)
        except ValueError:
            if actual_part != expected_part:
                return False
            continue
        if expected_value == 0.0:
            if abs(actual_value) > rel_tol:
                return False
        elif abs(actual_value - expected_value) / max(abs(expected_value), 1e-15) > rel_tol:
            return False
    return True


def _run_single_test(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: int,
) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        actual = result.stdout.strip()
        expected = expected_output.strip()
        if result.returncode != 0:
            return False, f"runtime_error:{result.returncode}:{result.stderr[:500]}"
        if actual == expected:
            return True, "exact_match"
        if [line.rstrip() for line in actual.splitlines()] == [
            line.rstrip() for line in expected.splitlines()
        ]:
            return True, "match_after_strip"
        if _numeric_match(actual, expected):
            return True, "numeric_match"
        return False, f"wrong_answer: expected={expected[:200]!r}, got={actual[:200]!r}"
    except subprocess.TimeoutExpired:
        return False, f"timeout:{timeout}s"
    except Exception as exc:
        return False, f"execution_error:{exc}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


@DatasetRegistry.register("livecodebench")
class LiveCodeBenchDataset(DatasetProvider):
    """LiveCodeBench competitive-programming code generation benchmark."""

    dataset_id = "livecodebench"
    dataset_name = "LiveCodeBench"
    evaluation_method = "test_execution"

    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        full: bool = False,
    ) -> None:
        self._split = split or _DEFAULT_SPLIT
        self._max_samples = max_samples
        self._full = full
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

        test_inputs = list(record.dataset_metadata.get("test_inputs") or [])
        test_outputs = list(record.dataset_metadata.get("test_outputs") or [])
        if not test_inputs or not test_outputs:
            return None, {"reason": "no_test_cases"}
        if len(test_inputs) != len(test_outputs):
            count = min(len(test_inputs), len(test_outputs))
            test_inputs = test_inputs[:count]
            test_outputs = test_outputs[:count]

        code = _extract_code(response)
        if not code:
            return False, {"reason": "no_code_extracted"}

        timeout = _TIMEOUT_SECONDS
        if record.dataset_metadata.get("time_limit") is not None:
            try:
                timeout = min(max(int(float(str(record.dataset_metadata["time_limit"]))) * 2, 5), 60)
            except (TypeError, ValueError):
                timeout = _TIMEOUT_SECONDS

        passed = 0
        details: list[dict[str, object]] = []
        for idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
            ok, detail = _run_single_test(str(code), str(test_input), str(expected_output), timeout)
            passed += 1 if ok else 0
            details.append({"test_index": idx, "passed": ok, "detail": detail})

        total = len(test_inputs)
        if total == 0:
            return None, {"reason": "no_test_cases_after_filtering"}

        return passed == total, {
            "match_type": "test_execution",
            "tests_passed": passed,
            "tests_total": total,
            "pass_rate": passed / total,
            "test_details": details[:5],
            "test_details_truncated": len(details) > 5,
        }

    def _build_records(self) -> List[DatasetRecord]:
        rows = self._load_raw_rows()
        records: list[DatasetRecord] = []
        for idx, raw in enumerate(rows):
            record = self._convert_row(raw, idx)
            if record is not None:
                records.append(record)
        return records

    def _load_raw_rows(self) -> Sequence[MutableMapping[str, object]]:
        paths = [_HF_DATASET_FULL] if self._full else [_HF_DATASET, _HF_DATASET_FULL]
        last_exc: Exception | None = None
        for hf_path in paths:
            try:
                dataset = load_dataset(hf_path, split=self._split)
                if self._max_samples is not None:
                    dataset = dataset.select(range(min(self._max_samples, len(dataset))))
                rows = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)
                return [row if isinstance(row, MutableMapping) else dict(row) for row in rows]
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Failed to load LiveCodeBench: {last_exc}") from last_exc

    def _convert_row(
        self,
        raw: MutableMapping[str, object],
        idx: int,
    ) -> Optional[DatasetRecord]:
        problem_text = str(
            raw.get("question_content")
            or raw.get("problem_description")
            or raw.get("question")
            or raw.get("problem")
            or ""
        ).strip()
        if not problem_text:
            return None

        problem_id = str(
            raw.get("question_id")
            or raw.get("task_id")
            or raw.get("problem_id")
            or f"lcb-{idx}"
        )
        title = str(raw.get("question_title") or raw.get("title") or raw.get("name") or "").strip()
        difficulty = str(raw.get("difficulty") or raw.get("question_difficulty") or "").strip()
        platform = str(raw.get("platform") or raw.get("source") or raw.get("contest_source") or "").strip()

        all_test_inputs: list[str] = []
        all_test_outputs: list[str] = []
        test_inputs = _parse_json_field(raw.get("input", raw.get("test_inputs", [])))
        test_outputs = _parse_json_field(raw.get("output", raw.get("test_outputs", [])))
        public_cases = _parse_json_field(raw.get("public_test_cases", raw.get("sample_io", [])))
        hidden_cases = _parse_json_field(raw.get("hidden_test_cases", raw.get("test_cases", [])))

        for cases in (public_cases, hidden_cases):
            if not isinstance(cases, list):
                continue
            for case in cases:
                parsed = _parse_json_field(case)
                if isinstance(parsed, dict):
                    inp = str(parsed.get("input", "")).strip()
                    out = str(parsed.get("output", "")).strip()
                    if inp or out:
                        all_test_inputs.append(inp)
                        all_test_outputs.append(out)

        if not all_test_inputs and isinstance(test_inputs, list):
            all_test_inputs = [str(value).strip() for value in test_inputs]
        if not all_test_outputs and isinstance(test_outputs, list):
            all_test_outputs = [str(value).strip() for value in test_outputs]

        problem = _PROMPT_TEMPLATE.format(
            problem_statement=problem_text,
            input_format=str(raw.get("input_format", "") or "(see problem statement)").strip(),
            output_format=str(raw.get("output_format", "") or "(see problem statement)").strip(),
            constraints=str(raw.get("constraints", "") or "(see problem statement)").strip(),
            examples=_format_examples(all_test_inputs[:2], all_test_outputs[:2]),
        )

        metadata: MutableMapping[str, object] = {
            "dataset_name": self.dataset_name,
            "problem_id": problem_id,
            "title": title,
            "difficulty": difficulty,
            "platform": platform,
            "test_inputs": all_test_inputs,
            "test_outputs": all_test_outputs,
            "time_limit": raw.get("time_limit"),
            "memory_limit": raw.get("memory_limit"),
            "workload_type": "coding",
        }
        starter_code = str(raw.get("starter_code", raw.get("code_stub", "")) or "").strip()
        if starter_code:
            metadata["starter_code"] = starter_code

        subject = (
            f"{platform}/{difficulty}"
            if platform and difficulty
            else (platform or difficulty or "competitive-programming")
        )
        return DatasetRecord(
            problem=problem,
            answer="__livecodebench_test_execution__",
            subject=subject,
            dataset_metadata=metadata,
        )


__all__ = ["LiveCodeBenchDataset"]

"""Single-submission GDPval judging over produced deliverables.

This is intentionally not pairwise Elo. It judges one agent submission against
the task prompt, GDPval rubric, reference files, and all files the agent
submitted through the finish tool or left in its output directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler
from .deliverables import collect_file_evidence, extract_json_object, render_evidence_sections
from .gdpval import _parse_rubric

LOGGER = logging.getLogger(__name__)

_PASS_THRESHOLD = 0.70
_MAX_PROMPT_CHARS = 120_000

_SYSTEM_PROMPT = """You are an exacting evaluator for real-world professional work products.

You will judge a single candidate submission for a GDPval task. Evaluate the
actual submitted files, not just the candidate's prose summary. Use the task,
reference-file evidence, deliverable-file evidence, and rubric criteria.

Return only valid JSON. Do not wrap it in markdown.
"""

_USER_PROMPT = """## Task
{problem}

## Candidate finish summary / response
{model_answer}

{reference_section}

{deliverable_section}

## Rubric
{rubric}

## Required JSON response
Return exactly this shape:
{{
  "verdict": "pass|partial|fail",
  "score": 0.0,
  "missing_or_broken_deliverables": ["..."],
  "criteria": [
    {{
      "rubric_item_id": "...",
      "criterion": "...",
      "satisfied": true,
      "score_awarded": 0.0,
      "score_possible": 0.0,
      "evidence": "specific evidence from submitted files or reason it is missing"
    }}
  ],
  "notes": "short overall assessment"
}}

Scoring instructions:
- Base score on the rubric point weights where available.
- If a deliverable cannot be opened or is missing, mark affected criteria unsatisfied.
- A submission can pass only if it substantially satisfies the requested deliverables.
- Be strict about file type, file name, sheet names, visible content, formulas,
  citations, transcripts, and media requirements when the task/rubric mentions them.
"""

_REPAIR_PROMPT = """The previous evaluator response was not valid JSON.

Convert it into exactly one valid JSON object matching this schema:
{{
  "verdict": "pass|partial|fail",
  "score": 0.0,
  "missing_or_broken_deliverables": ["..."],
  "criteria": [
    {{
      "rubric_item_id": "...",
      "criterion": "...",
      "satisfied": true,
      "score_awarded": 0.0,
      "score_possible": 0.0,
      "evidence": "specific evidence or reason"
    }}
  ],
  "notes": "short overall assessment"
}}

Infer the score and verdict from the evaluator response. Return JSON only.

Evaluator response:
{raw}
"""


def _coerce_paths(raw: Any) -> list[Path]:
    if not raw:
        return []
    if isinstance(raw, (str, Path)):
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                parsed = json.loads(raw)
                return _coerce_paths(parsed)
            except Exception:
                pass
        return [Path(raw)]
    if isinstance(raw, list):
        return [Path(str(p)) for p in raw if p]
    return []


def _paths_from_metadata(metadata: Dict[str, object]) -> tuple[list[Path], list[Path]]:
    deliverable_paths: list[Path] = []
    reference_paths: list[Path] = []

    deliverable_paths.extend(_coerce_paths(metadata.get("gdpval_submitted_files")))
    outputs_dir = metadata.get("gdpval_outputs_dir")
    if outputs_dir:
        deliverable_paths.append(Path(str(outputs_dir)))

    inputs_dir = metadata.get("gdpval_inputs_dir")
    if inputs_dir:
        reference_paths.append(Path(str(inputs_dir)))

    # Deduplicate while preserving order.
    def dedupe(paths: list[Path]) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
        return out

    return dedupe(deliverable_paths), dedupe(reference_paths)


def _fallback_score_from_criteria(criteria: list[dict[str, Any]]) -> tuple[float, float, float]:
    achieved = 0.0
    possible = 0.0
    for item in criteria:
        try:
            possible_points = float(item.get("score_possible", item.get("points", 1.0)) or 0.0)
        except (TypeError, ValueError):
            possible_points = 1.0
        try:
            awarded = float(item.get("score_awarded", 0.0) or 0.0)
        except (TypeError, ValueError):
            awarded = possible_points if item.get("satisfied") is True else 0.0
        possible += possible_points
        achieved += max(0.0, min(awarded, possible_points))
    score = achieved / possible if possible > 0 else 0.0
    return score, achieved, possible


@EvaluationRegistry.register("gdpval-deliverable")
@EvaluationRegistry.register("gdpval-aa-single")
class GdpvalDeliverableHandler(EvaluationHandler):
    """Judge one GDPval submission using rubric and extracted file evidence."""

    evaluation_method = "gdpval-deliverable"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not hasattr(self._client, "chat"):
            return None, {"reason": "no_llm_client_for_judging"}

        rubric = _parse_rubric(metadata.get("rubric_json"))
        if not rubric:
            # Fall back to pretty rubric text if the JSON is absent.
            pretty = metadata.get("rubric_pretty") or reference
            if not pretty:
                return None, {"reason": "no_rubric"}
            rubric_text = str(pretty)
        else:
            rubric_text = json.dumps(rubric, indent=2, ensure_ascii=False)

        deliverable_paths, reference_paths = _paths_from_metadata(metadata)
        deliverable_evidence = collect_file_evidence(deliverable_paths)
        reference_evidence = collect_file_evidence(reference_paths, max_files=40, max_chars_per_file=10_000)

        if not deliverable_evidence:
            return False, {
                "reason": "no_deliverables_found",
                "searched_paths": [str(p) for p in deliverable_paths],
            }

        reference_section = render_evidence_sections("Reference File Evidence", reference_evidence)
        deliverable_section = render_evidence_sections("Submitted Deliverable Evidence", deliverable_evidence)

        user_prompt = _USER_PROMPT.format(
            problem=problem,
            model_answer=model_answer or "",
            reference_section=reference_section,
            deliverable_section=deliverable_section,
            rubric=rubric_text,
        )
        if len(user_prompt) > _MAX_PROMPT_CHARS:
            user_prompt = user_prompt[:_MAX_PROMPT_CHARS] + "\n... [prompt truncated]"

        try:
            raw = self._client.chat(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_output_tokens=4096,
            )
        except Exception as exc:
            LOGGER.warning("GDPval deliverable judge failed: %s", exc)
            return None, {"reason": "judge_error", "error": str(exc)}

        parsed = extract_json_object(raw)
        repair_output = None
        if parsed is None:
            try:
                repair_output = self._client.chat(
                    system_prompt="You repair evaluator output into strict JSON.",
                    user_prompt=_REPAIR_PROMPT.format(raw=raw[:12_000]),
                    temperature=0.0,
                    max_output_tokens=4096,
                )
                parsed = extract_json_object(repair_output)
            except Exception as exc:
                LOGGER.warning("GDPval deliverable judge JSON repair failed: %s", exc)
            if parsed is None:
                return False, {
                    "reason": "judge_returned_non_json",
                    "judge_output": raw[:4000],
                    "repair_output": (repair_output or "")[:4000],
                    "deliverables": [item.to_dict() for item in deliverable_evidence],
                }

        criteria = parsed.get("criteria")
        criteria_list: list[dict[str, Any]] = criteria if isinstance(criteria, list) else []
        score = parsed.get("score")
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            score_float, _, _ = _fallback_score_from_criteria(criteria_list)
        if score_float > 1.0:
            score_float = score_float / 100.0
        score_float = max(0.0, min(score_float, 1.0))

        verdict = str(parsed.get("verdict") or "").lower()
        missing = parsed.get("missing_or_broken_deliverables") or []
        is_correct = (
            score_float >= _PASS_THRESHOLD
            and verdict == "pass"
            and not missing
        )

        return is_correct, {
            "match_type": "gdpval_deliverable_judge",
            "score": score_float,
            "pass_threshold": _PASS_THRESHOLD,
            "verdict": verdict,
            "missing_or_broken_deliverables": missing,
            "criteria": criteria_list,
            "notes": parsed.get("notes"),
            "judge_output": raw[:8000],
            "repair_output": (repair_output or "")[:8000],
            "deliverables": [item.to_dict() for item in deliverable_evidence],
            "references": [item.to_dict() for item in reference_evidence],
        }


__all__ = ["GdpvalDeliverableHandler"]

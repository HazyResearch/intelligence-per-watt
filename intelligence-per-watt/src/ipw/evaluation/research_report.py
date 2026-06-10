"""LLM-as-judge scoring for deep research reports."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_DIMENSIONS = ("comprehensiveness", "depth", "instruction_following", "readability")

_GENERIC_RUBRIC = """Evaluate the research report on a 0-10 scale for:
1. comprehensiveness: breadth and depth of coverage
2. depth: analysis quality, insight, and evidence use
3. instruction_following: adherence to the original task
4. readability: structure, clarity, and presentation"""

_PROMPT_TEMPLATE = """You are an expert evaluator assessing an AI-generated research report.

## Original Research Task
{task}

## Reference Context
{reference}

## Report To Evaluate
{report}

## Rubric
{rubric}

Return only JSON with this schema:
{{
  "scores": {{
    "comprehensiveness": <0-10>,
    "depth": <0-10>,
    "instruction_following": <0-10>,
    "readability": <0-10>
  }},
  "weighted_total": <0-10>,
  "notes": "<brief justification>"
}}"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = [block.group(1)] if block else []

    depth = 0
    start = None
    for idx, char in enumerate(raw):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start : idx + 1])

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _normalize_scores(parsed: dict[str, Any]) -> tuple[float, dict[str, float]]:
    raw_scores = parsed.get("scores", {})
    scores: dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for dimension in _DIMENSIONS:
            value = raw_scores.get(dimension)
            if value is None and dimension == "instruction_following":
                value = raw_scores.get("instruction following")
            if isinstance(value, (int, float)):
                scores[dimension] = max(0.0, min(float(value), 10.0))

    total = parsed.get("weighted_total", parsed.get("overall_score", parsed.get("score")))
    if isinstance(total, (int, float)):
        weighted_total = max(0.0, min(float(total), 10.0))
    elif scores:
        weighted_total = sum(scores.values()) / len(scores)
    else:
        weighted_total = 0.0
    return weighted_total, scores


@EvaluationRegistry.register("deepresearchbench")
@EvaluationRegistry.register("liveresearchbench")
@EvaluationRegistry.register("research-report")
class ResearchReportHandler(EvaluationHandler):
    """Rubric-based LLM judge for long-form deep research outputs."""

    evaluation_method = "research-report"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response", "score": 0.0}
        if not hasattr(self._client, "chat"):
            return None, {"reason": "no_llm_client"}

        rubric = str(metadata.get("rubric") or _GENERIC_RUBRIC)
        prompt = _PROMPT_TEMPLATE.format(
            task=problem,
            reference=reference or "(no reference report provided)",
            report=model_answer,
            rubric=rubric,
        )
        try:
            raw = self._client.chat(
                system_prompt="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=4096,
            )
        except Exception as exc:
            LOGGER.error("Research report scoring failed: %s", exc)
            return None, {"reason": "judge_error", "error": str(exc)}

        parsed = _parse_json_object(raw)
        weighted_total, scores = _normalize_scores(parsed)
        normalized_score = weighted_total / 10.0
        return normalized_score >= 0.5, {
            "score": normalized_score,
            "weighted_total_0_10": weighted_total,
            "dimension_scores": scores,
            "notes": str(parsed.get("notes", "")) if parsed else "",
            "raw_judge_output": raw,
        }


__all__ = ["ResearchReportHandler"]

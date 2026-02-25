from __future__ import annotations

import logging
import re
import string
from typing import Any, Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)


def _normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def _is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def _exact_match(model_answer: str, ground_truth: str) -> bool:
    """GAIA exact-match scorer with normalization for numbers, lists, and strings."""
    if model_answer is None:
        model_answer = "None"

    if _is_float(ground_truth):
        normalized = _normalize_number_str(model_answer)
        return normalized == float(ground_truth)

    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                comparisons.append(
                    _normalize_number_str(ma_elem) == float(gt_elem)
                )
            else:
                comparisons.append(
                    _normalize_str(ma_elem, remove_punct=False)
                    == _normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    return _normalize_str(model_answer) == _normalize_str(ground_truth)


_LLM_FALLBACK_PROMPT = """Your job is to determine if the predicted answer is semantically equivalent to the gold target.

Question: {question}
Gold target: {ground_truth}
Predicted answer: {response}

Consider the following:
- Numerical answers should match exactly (accounting for different formats like $1,000 vs 1000)
- List answers should contain all elements (order may vary)
- String answers should have the same meaning (case and punctuation don't matter)

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer, or 'None' if no answer is present>
reasoning: <brief explanation of why the extracted answer matches or does not match the gold target>
correct: <yes or no>"""


@EvaluationRegistry.register("gaia")
class GAIAHandler(EvaluationHandler):
    """GAIA evaluation: exact match with normalization + LLM fallback."""

    evaluation_method = "gaia"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        if not reference or not reference.strip():
            return None, {"reason": "no_ground_truth"}

        # Try exact match first (fast, no API call)
        if _exact_match(model_answer, reference):
            return True, {"match_type": "exact"}

        # LLM fallback for semantic comparison
        if not hasattr(self._client, "chat"):
            return False, {
                "match_type": "exact_failed",
                "reason": "no_llm_client_for_fallback",
            }

        try:
            prompt = _LLM_FALLBACK_PROMPT.format(
                question=problem or "(No question provided)",
                response=model_answer,
                ground_truth=reference,
            )
            raw = self._client.chat(
                system_prompt="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=1024,
            )

            structured_match = re.search(
                r"^correct:\s*(yes|no)", raw, re.MULTILINE | re.IGNORECASE
            )
            if structured_match:
                is_correct = structured_match.group(1).lower() == "yes"
            else:
                is_correct = (
                    "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()
                )

            meta: Dict[str, object] = {
                "match_type": "llm_fallback",
                "raw_judge_output": raw,
            }
            extracted_match = re.search(
                r"^extracted_final_answer:\s*(.+)", raw, re.MULTILINE
            )
            if extracted_match:
                meta["extracted_answer"] = extracted_match.group(1).strip()

            return is_correct, meta

        except Exception as exc:
            LOGGER.error("GAIA LLM fallback failed: %s", exc)
            return False, {
                "match_type": "llm_fallback_error",
                "error": str(exc),
            }

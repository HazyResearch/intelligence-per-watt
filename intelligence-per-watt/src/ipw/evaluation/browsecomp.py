"""BrowseComp short-answer judge."""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_GRADER_TEMPLATE = """Judge whether the following response to a question is correct based on the precise correct answer.

[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}

Your judgement must use exactly this format:
extracted_final_answer: <the final exact answer extracted from the response, or 'None'>
reasoning: <brief explanation focused only on whether the extracted answer matches>
correct: <yes or no>
confidence: <confidence score between 0% and 100%>"""


@EvaluationRegistry.register("browsecomp")
class BrowseCompHandler(EvaluationHandler):
    """LLM judge for BrowseComp exact-answer tasks."""

    evaluation_method = "browsecomp"

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
        if not hasattr(self._client, "chat"):
            return None, {"reason": "no_llm_client"}

        prompt = _GRADER_TEMPLATE.format(
            question=problem,
            response=model_answer,
            correct_answer=reference,
        )
        try:
            raw = self._client.chat(
                system_prompt="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=1024,
            )
        except Exception as exc:
            LOGGER.error("BrowseComp scoring failed: %s", exc)
            return None, {"reason": "judge_error", "error": str(exc)}

        match = re.search(r"^correct:\s*(yes|no)", raw, re.MULTILINE | re.IGNORECASE)
        if not match:
            return None, {"reason": "missing_verdict", "raw_judge_output": raw}

        metadata_out: Dict[str, object] = {"raw_judge_output": raw}
        extracted = re.search(r"^extracted_final_answer:\s*(.+)", raw, re.MULTILINE)
        if extracted:
            metadata_out["extracted_answer"] = extracted.group(1).strip()
        confidence = re.search(r"^confidence:\s*(.+)", raw, re.MULTILINE)
        if confidence:
            metadata_out["confidence"] = confidence.group(1).strip()
        return match.group(1).lower() == "yes", metadata_out


__all__ = ["BrowseCompHandler"]

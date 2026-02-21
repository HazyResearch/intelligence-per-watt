from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_GRADER_TEMPLATE = """You are evaluating an AI system's answer to a multi-hop factual question.

Compare the predicted answer against the Ground Truth Answer and determine if the prediction is correct.

## Evaluation Guidelines

1. **Focus on semantic meaning**: Look for equivalent information - exact wording is not required.
2. **Assess factual accuracy**: Determine whether the essential facts from the Ground Truth are present in the answer.
3. **Ignore minor differences**: Capitalization, punctuation, formatting, and word order don't matter.
4. **Partial credit**: If the Ground Truth has multiple parts, all essential parts must be present for a correct rating.
5. **Additional information**: Extra correct information in the prediction is acceptable, but extra incorrect information is not.

## Question
{question}

## Ground Truth Answer
{ground_truth}

## Predicted Answer
{predicted_answer}

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer, or 'None' if no answer is present>
reasoning: <brief explanation of why the extracted answer is or is not correct>
correct: <yes or no>"""


@EvaluationRegistry.register("frames")
class FRAMESHandler(EvaluationHandler):
    """LLM-as-judge evaluation for FRAMES multi-hop factual retrieval."""

    evaluation_method = "frames"

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
            raise RuntimeError(
                "FRAMESHandler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        prompt = _GRADER_TEMPLATE.format(
            question=problem,
            ground_truth=reference,
            predicted_answer=model_answer,
        )

        try:
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
                response_upper = raw.upper().strip()
                if "TRUE" in response_upper:
                    is_correct = True
                elif "FALSE" in response_upper:
                    is_correct = False
                else:
                    LOGGER.warning("Could not parse grade from response: %s", raw[:50])
                    is_correct = False

            meta: Dict[str, object] = {
                "raw_judge_output": raw,
            }
            extracted = re.search(
                r"^extracted_final_answer:\s*(.+)", raw, re.MULTILINE
            )
            if extracted:
                meta["extracted_answer"] = extracted.group(1).strip()

            return is_correct, meta

        except Exception as exc:
            LOGGER.error("FRAMES scoring failed: %s", exc)
            return None, {"error": str(exc)}

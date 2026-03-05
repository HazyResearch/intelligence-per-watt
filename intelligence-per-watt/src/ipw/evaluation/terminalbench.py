from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_GRADER_TEMPLATE = """You are evaluating an AI system's answer to a terminal/CLI task.

Compare the predicted answer against the expected output and determine if the prediction is correct.

## Evaluation Guidelines

1. **Focus on functional equivalence**: The answer should produce the same result, even if the exact syntax differs.
2. **Ignore minor formatting**: Trailing whitespace, trailing newlines, and minor spacing differences are acceptable.
3. **Command equivalence**: Different commands that produce identical output are considered correct.
4. **Partial output**: If the expected output has multiple lines, all essential lines must be present.

## Task
{question}

## Expected Output
{expected_output}

## Predicted Answer
{predicted_answer}

Your response MUST use exactly this format:
reasoning: <brief explanation of why the answer is or is not correct>
correct: <yes or no>"""


class TerminalBenchHandler(EvaluationHandler):
    """LLM-as-judge evaluation for TerminalBench tasks."""

    evaluation_method = "terminalbench"

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
            return None, {"reason": "no_expected_output"}

        # Try exact match first (strip whitespace)
        if model_answer.strip() == reference.strip():
            return True, {"match_type": "exact"}

        if not hasattr(self._client, "chat"):
            raise RuntimeError(
                "TerminalBenchHandler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        prompt = _GRADER_TEMPLATE.format(
            question=problem,
            expected_output=reference,
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
                is_correct = (
                    "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()
                )

            meta: Dict[str, object] = {
                "match_type": "llm_judge",
                "raw_judge_output": raw,
            }
            return is_correct, meta

        except Exception as exc:
            LOGGER.error("TerminalBench scoring failed: %s", exc)
            return None, {"error": str(exc)}

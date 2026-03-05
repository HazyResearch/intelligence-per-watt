from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler
from .helpers import normalize_math_answer

LOGGER = logging.getLogger(__name__)


@EvaluationRegistry.register("math-500")
class Math500Handler(EvaluationHandler):
    """
    Evaluation for MATH-500-style problems.

    Two-stage approach:
    1. First, uses LLM to verify mathematical equivalence (with retry logic)
    2. If LLM verification fails, falls back to extraction + normalization
    """
    evaluation_method = "math-500"

    EXTRACT_PROMPT = r"""Extract the mathematical answer from the solution. The answer will typically be inside a \boxed{} command.
If there are multiple boxed expressions, extract the final one. Return only the mathematical expression without any surrounding text.

Example 1:
Input: Therefore, $x = \boxed{5}$ is the solution.
Output: 5

Example 2:
Input: The final answer is $\boxed{\frac{\sqrt{3}}{2}}$.
Output: \frac{\sqrt{3}}{2}

Example 3:
Input: We get $\boxed{x = 2}$ and $\boxed{y = 3}$, so $\boxed{x + y = 5}$.
Output: 5

Rules:
- Return ONLY the final numeric/algebraic expression
- Do NOT include headings or words (e.g., 'Step', 'Solution')
- Do NOT include LaTeX delimiters like $ ... $, \( ... \), or \[ ... \]
- If no clear final answer is present, return NONE"""

    VERIFY_PROMPT = r"""Compare these two mathematical solutions and determine if they are equivalent. Focus on:
1. The final numerical or mathematical answer (typically in a \boxed{} command)
2. Mathematical equivalence (e.g., 1/2 = 0.5 = \frac{1}{2})
3. Different but valid solution methods that arrive at the same result

Output format (strict):
Return ONLY a compact JSON object with these exact keys and types, no extra text:
{"explanation": string, "is_correct": boolean}
- explanation: short description on why they match or not.
- is_correct: true if equivalent, false otherwise."""

    def _extract_answer_with_llm(self, text: str) -> Optional[str]:
        """Extract answer using LLM."""
        # Assumes self._client is an OpenAIClient or compatible with a .chat() helper
        if not hasattr(self._client, "chat"):
            raise RuntimeError("Math500Handler requires a client with a .chat() helper (e.g. OpenAIClient)")

        try:
            raw_response = self._client.chat(
                system_prompt=self.EXTRACT_PROMPT,
                user_prompt=text,
                temperature=0.0,
                max_output_tokens=100,
            )

            extracted = raw_response.strip()
            normalized = normalize_math_answer(extracted)

            if normalized and normalized.upper() != "NONE":
                return normalized
            return None

        except Exception as exc:
            LOGGER.error(f"Error extracting math answer: {exc}")
            return None

    def _verify_with_llm(self, model_answer: str, reference: str) -> Tuple[Optional[bool], Optional[str]]:
        """Verify mathematical equivalence using LLM judge."""
        if not hasattr(self._client, "chat"):
            raise RuntimeError("Math500Handler requires a client with a .chat() helper (e.g. OpenAIClient)")

        try:
            user_prompt = f"Solution 1:\n{model_answer}\n\nSolution 2:\n{reference}\n\nRespond with JSON only."

            raw_response = self._client.chat(
                system_prompt=self.VERIFY_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_output_tokens=500,
            )

            content = raw_response.strip()

            # Extract first JSON object and parse
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                return None, None

            try:
                data = json.loads(m.group(0))
            except Exception:
                return None, None

            is_correct = data.get("is_correct")
            explanation = data.get("explanation", "")

            if isinstance(is_correct, bool):
                return is_correct, str(explanation).replace('\n', ' ')

            return None, None

        except Exception as exc:
            LOGGER.error(f"Error verifying math answer: {exc}")
            return None, None

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """
        Two-stage evaluation:
        1. Try LLM-based verification (with retries)
        2. Fall back to extraction + normalization if LLM fails
        """

        # Stage 1: Try LLM verification with retries
        NUM_RETRIES = 3
        judge_result = None
        judge_explanation = None

        for attempt in range(NUM_RETRIES):
            judge_result, judge_explanation = self._verify_with_llm(model_answer, reference)
            if judge_result is not None:
                break

        if judge_result is not None:
            return judge_result, {
                "method": "llm_verify",
                "explanation": judge_explanation,
            }

        # Stage 2: Fallback to extraction + normalization
        LOGGER.warning("MATH | LLM Judge failed, falling back to extraction")

        extracted_model = self._extract_answer_with_llm(model_answer)
        extracted_ref = self._extract_answer_with_llm(reference)

        # Also try direct normalization as additional fallback
        if not extracted_model:
            extracted_model = normalize_math_answer(model_answer)
        if not extracted_ref:
            extracted_ref = normalize_math_answer(reference)

        if not extracted_model or not extracted_ref:
            return None, {
                "reason": "unscorable_math_answer",
                "extracted_model": extracted_model,
                "extracted_ref": extracted_ref,
            }

        is_correct = extracted_model == extracted_ref
        return is_correct, {
            "method": "extraction_fallback",
            "extracted_model": extracted_model,
            "extracted_ref": extracted_ref,
        }

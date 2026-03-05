from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)


class BaseMCQHandler(EvaluationHandler):
    """Shared logic for multiple-choice (letter-based) evaluations."""

    def _valid_letters_from_options(self, metadata: Dict[str, object]) -> Optional[str]:
        options = metadata.get("options")
        if isinstance(options, list) and options:
            n = len(options)
            return "".join(chr(ord("A") + i) for i in range(n))
        return None

    def _extract_answer_with_llm(
        self,
        problem: str,
        model_answer: str,
        valid_letters: str,
    ) -> Optional[str]:
        """
        Use the evaluation LLM to extract the answer letter from the model response.
        """
        if not hasattr(self._client, "chat"):
            raise RuntimeError(
                "MCQ handler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        last_letter = chr(ord('A') + len(valid_letters) - 1) if valid_letters else 'D'

        system_prompt = (
            f"You are an answer extraction assistant. Extract the final multiple choice answer "
            f"from the response. Return ONLY a single letter (A-{last_letter}). "
            f"If no valid answer letter is found, return 'NONE'."
        )

        user_prompt = f"Problem: {problem}\nResponse: {model_answer}\n\nExtract the final answer letter:"

        try:
            raw_response = self._client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_output_tokens=5,
            )

            extracted = raw_response.strip().upper()

            # Try to extract letter from response (handles "The answer is: A" etc.)
            answer_match = re.search(r'(?:THE ANSWER IS:?\s*)?([A-Z])', extracted, re.IGNORECASE)
            if answer_match:
                extracted = answer_match.group(1).upper()

            # Validate it's in the valid set
            if extracted in valid_letters:
                return extracted

            return None

        except Exception as exc:
            LOGGER.error(f"Error in LLM-based answer extraction: {exc}")
            return None

    def _evaluate_mcq(
        self,
        *,
        problem: str,
        reference_letter: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        ref = (reference_letter or "").strip().upper()
        if not ref:
            return None, {"reason": "missing_reference_letter"}

        valid_letters = self._valid_letters_from_options(metadata)
        if valid_letters is None:
            # Fallback if options not available
            valid_letters = "ABCD"

        candidate = self._extract_answer_with_llm(problem, model_answer, valid_letters)
        if not candidate:
            return None, {"reason": "no_choice_letter_extracted"}

        is_correct = candidate == ref
        meta = {
            "reference_letter": ref,
            "candidate_letter": candidate,
            "valid_letters": valid_letters,
        }
        return is_correct, meta

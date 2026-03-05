from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)


@EvaluationRegistry.register("natural_reasoning")
@EvaluationRegistry.register("reasoning")  # Alias used in IPW dataset
class NaturalReasoningHandler(EvaluationHandler):
    """LLM-based evaluation comparing a response to a ground-truth answer."""

    evaluation_method = "natural_reasoning"

    GROUND_TRUTH_PROMPT = """
        You are evaluating the technical accuracy of a model's response by comparing it against a reference answer.

        TASK: Determine if the model's response is CORRECT or INCORRECT based on whether it matches the reference answer.

        EVALUATION CRITERIA:
        1. The model's answer must arrive at the same conclusion as the reference answer
        2. Mathematical results must match those in the reference (allowing for equivalent forms)
        3. For factual questions, the core facts must align with the reference
        4. For reasoning problems, the final answer must match even if the approach differs

        IMPORTANT RULES:
        - The reference answer is your ground truth - treat it as the correct answer
        - Different wording or approach is acceptable IF the conclusion matches the reference
        - This is a binary decision: TRUE if correct, FALSE if incorrect
        - No partial credit - the answer either matches the reference or it doesn't
        - Ignore differences in formatting, verbosity, or explanation style
        - Focus ONLY on whether the core answer aligns with the reference

        Question: {question}

        Reference Answer: {reference_answer}

        Model Response: {model_response}

        EVALUATION STEPS:
        1. Identify the core answer in the reference response
        2. Identify the core answer in the model's response
        3. Compare: Do they reach the same conclusion?
        4. If numerical: Are the values equivalent?
        5. If factual: Do the key facts match?
        6. If reasoning-based: Is the final answer the same?

        VERDICT: Return TRUE if the model's answer matches the reference answer, FALSE otherwise.

        Final Verdict (respond with only TRUE or FALSE):
        """

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not reference.strip():
            raise RuntimeError("NaturalReasoningHandler requires a non-empty reference answer.")

        if not hasattr(self._client, "chat"):
            raise RuntimeError(
                "NaturalReasoningHandler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        prompt = self.GROUND_TRUTH_PROMPT.format(
            question=problem,
            model_response=model_answer,
            reference_answer=reference,
        )

        raw = self._client.chat(
            system_prompt="",
            user_prompt=prompt,
            temperature=0.0,
            max_output_tokens=32,
        )

        # Match 'true' / 'false' as the entire response, case-insensitive, allowing for whitespace.
        m = re.search(r"^\s*(true|false)\s*$", raw, flags=re.IGNORECASE)
        meta = {
            "evaluation_mode": "ground_truth",
            "raw_judge_output": raw,
        }
        if not m:
            meta["unscorable"] = True
            return None, meta

        norm = m.group(1).lower()
        meta["parsed_label"] = norm
        result = norm == "true"
        return result, meta

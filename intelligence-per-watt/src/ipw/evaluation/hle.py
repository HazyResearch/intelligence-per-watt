from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

# HLE uses the same SimpleQA-style grading template (CORRECT / INCORRECT / NOT_ATTEMPTED).
_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the full answer is included AND the uncertainty expressed is googable.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not fully included in the answer.
    - No statements in the predicted answer contradict the gold target.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: Malia.
Predicted answer 2: Malia, Move, and Sasha.
Predicted answer 3: Barack Obama's children are named Malia and Sasha. I believe he also has a son named Lent.
Predicted answer 4: Barack Obama has two children, Malia and Sasha. It is possible that he has more children that I am not aware of.
Predicted answer 5: Barack Obama's children are named Malia and Sasha. She also has a daughter named Winkle, but she was born too late for me to know about her.
Predicted answer 6: I think Barack Obama's children are named Malia and Sasha. I'm less sure about this, but I think he also has another son.
Predicted answer 7: Malia Obama and Sasha Obama. It is possible he has more but I am not sure.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the predicted answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that") are also considered incorrect.


Grade the following example.

Question: {question}
Gold target: {target}
Predicted answer: {predicted}

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer>
reasoning: <brief explanation of why the extracted answer is or is not correct>
correct: <yes, no, or not_attempted>"""


def _parse_grade(response: str) -> str:
    """Parse grade from LLM response. Returns CORRECT, INCORRECT, or NOT_ATTEMPTED."""
    structured_match = re.search(
        r"^correct:\s*(yes|not_attempted|no)", response, re.MULTILINE | re.IGNORECASE
    )
    if structured_match:
        value = structured_match.group(1).lower()
        if value == "yes":
            return "CORRECT"
        if value == "no":
            return "INCORRECT"
        return "NOT_ATTEMPTED"

    response_upper = response.upper().strip()
    if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
        return "CORRECT"
    if "INCORRECT" in response_upper:
        return "INCORRECT"
    if "NOT_ATTEMPTED" in response_upper or "NOT ATTEMPTED" in response_upper:
        return "NOT_ATTEMPTED"

    LOGGER.warning("Could not parse HLE grade from response: %s", response[:100])
    return "NOT_ATTEMPTED"


@EvaluationRegistry.register("hle")
class HLEHandler(EvaluationHandler):
    """LLM-as-judge evaluation for Humanity's Last Exam (CORRECT / INCORRECT / NOT_ATTEMPTED)."""

    evaluation_method = "hle"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not model_answer or not model_answer.strip():
            return False, {"grade": "NOT_ATTEMPTED", "reason": "empty_response"}

        if not reference or not reference.strip():
            return None, {"reason": "no_ground_truth"}

        if not hasattr(self._client, "chat"):
            raise RuntimeError(
                "HLEHandler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        prompt = _GRADER_TEMPLATE.format(
            question=problem,
            target=reference,
            predicted=model_answer,
        )

        try:
            raw = self._client.chat(
                system_prompt="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=1024,
            )

            grade = _parse_grade(raw)
            is_correct = grade == "CORRECT"

            meta: Dict[str, object] = {
                "grade": grade,
                "raw_judge_output": raw,
            }
            extracted = re.search(
                r"^extracted_final_answer:\s*(.+)", raw, re.MULTILINE
            )
            if extracted:
                meta["extracted_answer"] = extracted.group(1).strip()

            return is_correct, meta

        except Exception as exc:
            LOGGER.error("HLE scoring failed: %s", exc)
            return None, {"error": str(exc)}

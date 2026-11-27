from __future__ import annotations

import re
from typing import Optional


def normalize_math_answer(text: Optional[str]) -> Optional[str]:
    """Best-effort normalization for simple mathematical expressions."""

    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    if not s:
        return None

    # Remove LaTeX math delimiters
    s = s.replace("$", "")

    # Extract content inside the last \boxed{...} if present
    boxed_matches = list(re.finditer(r"\\boxed\{([^}]*)\}", s))
    if boxed_matches:
        s = boxed_matches[-1].group(1)

    # Remove thin spaces / !, etc.
    s = re.sub(r"\\[!,]", "", s)

    # Replace \frac{a}{b} with a/b for simple integer fractions
    s = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", s)

    # Replace \sqrt{n} with √n for small integers
    s = re.sub(r"\\sqrt\{(\d+)\}", r"√\1", s)

    # Remove thousands separators
    s = s.replace(",", "")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


# The judge templates ask for a final "correct: <yes/no/...>" line. Judges also
# fill it with the grade names the templates define (CORRECT / INCORRECT /
# NOT_ATTEMPTED) or wrap it in Markdown, so accept those forms too.
_VERDICT_LINE = re.compile(
    r"^[ \t>*_#\u2022+-]*(?:\d+[.)][ \t]*)?correct[ \t*_]*:[ \t*_`\"'(\[]*"
    r"(yes|true|not[ _]attempted|incorrect|correct|false|no)",
    re.MULTILINE | re.IGNORECASE,
)
_VERDICT_GRADES = {
    "yes": "CORRECT",
    "true": "CORRECT",
    "correct": "CORRECT",
    "incorrect": "INCORRECT",
    "false": "INCORRECT",
    "no": "INCORRECT",
}
# The "correct:" label itself, so the keyword fallback does not read it as a verdict.
_VERDICT_LABEL = re.compile(r"\bcorrect[\s*_]*:", re.IGNORECASE)
_NEGATED_CORRECT = re.compile(r"\b(?:NOT|ISN['\u2019]T)\s+(?:AN?\s+)?CORRECT\b")


def parse_judge_grade(raw: str) -> Optional[str]:
    """Return CORRECT, INCORRECT or NOT_ATTEMPTED from an LLM judge reply, or None."""
    match = _VERDICT_LINE.search(raw)
    if match:
        value = match.group(1).lower()
        return _VERDICT_GRADES.get(value, "NOT_ATTEMPTED")

    # No verdict line: fall back to grade keywords in the free text.
    text = _VERDICT_LABEL.sub(" ", raw).upper()
    if "INCORRECT" in text or _NEGATED_CORRECT.search(text):
        return "INCORRECT"
    if "CORRECT" in text:
        return "CORRECT"
    if "NOT_ATTEMPTED" in text or "NOT ATTEMPTED" in text:
        return "NOT_ATTEMPTED"
    return None

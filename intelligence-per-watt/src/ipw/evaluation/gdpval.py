"""GDPval evaluation: rubric-based LLM-as-judge.

GDPval tasks have no reference answers — each task ships a ``rubric_json``
listing criteria with point values. We ask the judge model to score the
candidate answer against each criterion, then aggregate:

- A criterion is satisfied iff the judge returns "yes".
- Score = (achieved points) / (max points).
- ``is_correct`` is True iff every ``required: true`` criterion is satisfied
  AND the achieved score is at least the configured pass threshold (default
  0.5).
- If the rubric is malformed or no criteria are present, returns ``None``
  (unscorable).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_PASS_THRESHOLD = 0.5
_OUTPUTS_BUDGET_CHARS = 12_000  # per-file budget when summarising deliverables

_JUDGE_PROMPT = """You are grading a candidate response against a single rubric criterion.

## Task given to the candidate
{problem}

## Candidate response (plain-text answer)
{response}
{deliverables_section}
## Rubric criterion
{criterion}

Does the candidate response satisfy this criterion?

Reply with exactly one line in this format:
verdict: <yes or no>
reason: <one short sentence>
"""


def _extract_deliverable_text(outputs_dir: Path) -> str:
    """Return a concatenated text view of any deliverable files.

    Used so the judge can verify rubric criteria that reference produced
    files (e.g. ``Workbook contains tab 'Sample'``).
    """
    if not outputs_dir or not Path(outputs_dir).is_dir():
        return ""
    sections: list[str] = []
    for path in sorted(Path(outputs_dir).iterdir()):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        try:
            if suf in (".txt", ".md", ".csv", ".tsv", ".json"):
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suf in (".xlsx", ".xls", ".xlsm"):
                try:
                    from openpyxl import load_workbook
                except ImportError:
                    text = "<openpyxl not installed>"
                else:
                    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
                    parts: list[str] = []
                    for sheet in wb.worksheets:
                        parts.append(f"# Sheet: {sheet.title}")
                        for row in sheet.iter_rows(values_only=True):
                            cells = [("" if v is None else str(v)) for v in row]
                            parts.append("\t".join(cells))
                    text = "\n".join(parts)
            elif suf == ".pdf":
                try:
                    import pdfplumber
                except ImportError:
                    text = "<pdfplumber not installed>"
                else:
                    pages = []
                    with pdfplumber.open(str(path)) as pdf:
                        for page in pdf.pages:
                            pages.append(page.extract_text() or "")
                    text = "\n\n".join(pages)
            elif suf == ".docx":
                try:
                    import docx
                except ImportError:
                    text = "<python-docx not installed>"
                else:
                    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            text = f"<error reading {path.name}: {exc}>"

        if len(text) > _OUTPUTS_BUDGET_CHARS:
            text = text[:_OUTPUTS_BUDGET_CHARS] + "\n... [truncated]"
        sections.append(f"### Deliverable file: {path.name}\n{text}")
    return "\n\n".join(sections)


def _parse_rubric(raw: Any) -> List[Dict[str, Any]]:
    """Return the list of rubric items from a rubric_json blob."""
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            # Some rubrics wrap the list under a key
            items = (
                parsed.get("rubric")
                or parsed.get("criteria")
                or parsed.get("items")
                or []
            )
        else:
            return []
    else:
        return []

    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("criterion"):
            cleaned.append(item)
    return cleaned


def _judge_verdict(raw: str) -> bool:
    """Return True iff the judge said 'verdict: yes'."""
    if not raw:
        return False
    m = re.search(r"verdict\s*:\s*(yes|no)", raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    # Fallback: any standalone yes/no token in the first line
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    return bool(re.search(r"\byes\b", first_line, re.IGNORECASE))


@EvaluationRegistry.register("gdpval")
class GdpvalHandler(EvaluationHandler):
    """Rubric-based LLM-as-judge grading for GDPval tasks."""

    evaluation_method = "gdpval"

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

        rubric_raw = metadata.get("rubric_json")
        rubric = _parse_rubric(rubric_raw)
        if not rubric:
            return None, {"reason": "no_rubric"}

        if not hasattr(self._client, "chat"):
            return None, {"reason": "no_llm_client_for_judging"}

        # Optional deliverable-file content: the file-io agent writes outputs
        # under <workspace>/outputs/ and surfaces the path via metadata.
        outputs_dir = metadata.get("gdpval_outputs_dir")
        if not outputs_dir:
            # Conventional path: workspace sibling of metadata's instance_id
            # — fall back to scanning common locations.
            pass
        deliverable_text = _extract_deliverable_text(Path(outputs_dir)) if outputs_dir else ""
        deliverables_section = (
            f"\n## Deliverable files produced\n{deliverable_text}\n"
            if deliverable_text
            else ""
        )

        per_criterion: list[dict[str, Any]] = []
        achieved_points = 0.0
        max_points = 0.0
        required_failures = 0

        for idx, item in enumerate(rubric):
            criterion_text = str(item.get("criterion") or "").strip()
            if not criterion_text:
                continue

            try:
                points = float(item.get("score", 1.0))
            except (TypeError, ValueError):
                points = 1.0
            required = bool(item.get("required", False))
            rubric_item_id = item.get("rubric_item_id", f"item_{idx}")

            max_points += points

            prompt = _JUDGE_PROMPT.format(
                problem=problem[:4000],  # keep judge prompt bounded
                response=model_answer[:8000],
                deliverables_section=deliverables_section,
                criterion=criterion_text,
            )
            try:
                raw = self._client.chat(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=0.0,
                    max_output_tokens=512,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Rubric judge failed for item %s: %s", rubric_item_id, exc
                )
                per_criterion.append(
                    {
                        "rubric_item_id": rubric_item_id,
                        "criterion": criterion_text,
                        "points": points,
                        "required": required,
                        "satisfied": False,
                        "error": str(exc),
                    }
                )
                if required:
                    required_failures += 1
                continue

            satisfied = _judge_verdict(raw)
            if satisfied:
                achieved_points += points
            elif required:
                required_failures += 1

            per_criterion.append(
                {
                    "rubric_item_id": rubric_item_id,
                    "criterion": criterion_text,
                    "points": points,
                    "required": required,
                    "satisfied": satisfied,
                    "judge_output": raw[:500],
                }
            )

        score = achieved_points / max_points if max_points > 0 else 0.0
        is_correct = required_failures == 0 and score >= _PASS_THRESHOLD

        return is_correct, {
            "match_type": "rubric_llm_judge",
            "score": score,
            "achieved_points": achieved_points,
            "max_points": max_points,
            "required_failures": required_failures,
            "num_criteria": len(per_criterion),
            "per_criterion": per_criterion,
            "pass_threshold": _PASS_THRESHOLD,
        }


__all__ = ["GdpvalHandler"]

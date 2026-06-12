"""Parsers that turn raw LM output into a structured ParsedTurn.

Two modes:
- parse_function_calling: consumes OpenAI/Anthropic-style tool_calls dict
- parse_structured_text: regex extraction of THOUGHT/ACTION/INPUT blocks
  (compatible with native ReAct prompts and basic CodeAct outputs)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ipw.execution.errors import MalformedOutputError


@dataclass
class ParsedTurn:
    """Parsed result of one LM turn."""

    final_answer: Optional[str]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


_FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+?)\s*$", re.MULTILINE)
_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(\S+)\s*$", re.MULTILINE)
_INPUT_RE = re.compile(
    r"^\s*INPUT\s*:\s*(\{.*?\})\s*(?=$|\n\s*(?:THOUGHT|ACTION|FINAL_ANSWER|$))",
    re.DOTALL | re.MULTILINE,
)


def parse_structured_text(text: str) -> ParsedTurn:
    """Parse THOUGHT/ACTION/INPUT/FINAL_ANSWER blocks."""
    final_match = _FINAL_RE.search(text)
    if final_match:
        return ParsedTurn(final_answer=final_match.group(1).strip(), tool_calls=[])

    action_matches = list(_ACTION_RE.finditer(text))
    input_matches = list(_INPUT_RE.finditer(text))

    if not action_matches:
        raise MalformedOutputError(
            "no ACTION or FINAL_ANSWER block found in LM output"
        )
    if len(action_matches) != len(input_matches):
        raise MalformedOutputError(
            f"ACTION/INPUT block count mismatch ({len(action_matches)} vs {len(input_matches)})"
        )

    tool_calls: List[Dict[str, Any]] = []
    for a, i in zip(action_matches, input_matches):
        name = a.group(1).strip()
        raw_json = i.group(1).strip()
        try:
            parsed_input = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise MalformedOutputError(
                f"invalid JSON in INPUT block for {name!r}: {exc}"
            ) from exc
        tool_calls.append({"name": name, "input": parsed_input})

    return ParsedTurn(final_answer=None, tool_calls=tool_calls)


def parse_function_calling(raw: Dict[str, Any]) -> ParsedTurn:
    """Parse OpenAI/Anthropic-style chat completion message.

    Input shape: {"content": str | None, "tool_calls": [{"function": {"name": ..., "arguments": str}}]}
    """
    content = raw.get("content")
    tool_calls_raw = raw.get("tool_calls") or []

    if not tool_calls_raw:
        return ParsedTurn(final_answer=content if content is not None else "", tool_calls=[])

    tool_calls: List[Dict[str, Any]] = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            raise MalformedOutputError(
                f"tool_call entry missing function.name: {tc!r}"
            )
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw)
            except json.JSONDecodeError as exc:
                raise MalformedOutputError(
                    f"invalid JSON in tool_call arguments for {name!r}: {exc}"
                ) from exc
        else:
            parsed = args_raw
        tool_calls.append({"name": name, "input": parsed})

    return ParsedTurn(final_answer=None, tool_calls=tool_calls)


__all__ = ["ParsedTurn", "parse_structured_text", "parse_function_calling"]

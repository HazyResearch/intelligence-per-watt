"""Action space: the six-action JSON contract plus parse_action (tolerant first-object extraction)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ACTION_TYPES = frozenset({"think", "bash", "call_tool", "call_worker", "delegate_task", "finish"})


class ActionParseError(ValueError):
    """The model output did not contain a valid action JSON object."""


@dataclass(slots=True, frozen=True)
class Action:
    type: str
    text: str = ""
    target: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def _extract_json(text: str) -> tuple[dict[str, Any], str]:
    """Return (first valid JSON object, its exact source substring).

    Scans for the first ``{`` and uses ``json.JSONDecoder.raw_decode`` to parse one complete object
    (so nested braces — e.g. call_tool ``args`` — parse correctly). This handles ```json fences,
    <tool_call> tags, and bare JSON uniformly (the surrounding markers sit outside the object), and
    when a model emits MULTIPLE action objects in one turn it returns the FIRST (one action / turn).
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj, text[start:end]
        start = text.find("{", start + 1)
    raise ActionParseError(f"no JSON action object found in: {text[:200]!r}")


def _require(obj: dict[str, Any], key: str, atype: str) -> str:
    """Return ``obj[key]`` as a non-empty str, or raise ActionParseError.

    Guards the routing keys (``tool``/``worker``) so a well-typed-but-incomplete emission surfaces as
    an ActionParseError the brain's fallback can catch — never a raw KeyError that crashes the loop.
    """
    val = obj.get(key)
    if not val:
        raise ActionParseError(f"{atype!r} action missing required {key!r}: {obj}")
    return str(val)


def parse_action(text: str) -> Action:
    obj, raw = _extract_json(text)  # raw = the single executed action (not the whole emission)
    atype = obj.get("action") or obj.get("type")
    if atype not in ACTION_TYPES:
        raise ActionParseError(f"unknown action {atype!r}")
    if atype in ("think",):
        return Action(type=atype, text=str(obj.get("text", "")), raw=raw)
    if atype == "bash":
        return Action(type=atype, text=str(obj.get("command", "")), raw=raw)
    if atype == "finish":
        return Action(type=atype, text=str(obj.get("answer", "")), raw=raw)
    if atype == "call_tool":
        return Action(type=atype, target=_require(obj, "tool", atype), args=dict(obj.get("args", {})), raw=raw)
    if atype == "call_worker":
        return Action(
            type=atype,
            target=_require(obj, "worker", atype),
            args={"sub_prompt": str(obj.get("sub_prompt", "")), "context": obj.get("context")},
            raw=raw,
        )
    # delegate_task
    return Action(type=atype, target=_require(obj, "worker", atype), raw=raw)

"""SkillOrchestra-style pre-task router: one cheap cloud-classifier call scores a TB
task against a skill catalog and decides local-vs-cloud. stdlib + requests only.

Reimplemented from OpenJarvis agents/hybrid/skillorchestra.py (reference-only).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from ...cost import dollars_tokens

# (local, cloud) competence priors in [0,1] for Qwen3.5-27B vs a frontier cloud model.
# ROUGH ESTIMATES — recalibrate from per-skill TB success data.
COMPETENCE: dict[str, tuple[float, float]] = {
    "bash_scripting": (0.82, 0.95),
    "file_navigation": (0.85, 0.95),
    "test_execution": (0.78, 0.93),
    "debugging": (0.50, 0.93),
    "package_management": (0.80, 0.93),
    "system_administration": (0.62, 0.92),
    "networking": (0.58, 0.90),
    "data_processing": (0.70, 0.92),
    "multi_step_reasoning": (0.48, 0.94),
    "code_editing": (0.72, 0.94),
}
SKILL_CATALOG: tuple[str, ...] = tuple(COMPETENCE.keys())


@dataclass(slots=True, frozen=True)
class RouteDecision:
    tier: str                       # "local" | "cloud"
    weights: dict[str, float]
    local_score: float
    cloud_score: float
    classifier_tokens: int | None
    classifier_dollars: float | None


def _post(url: str, headers: dict, body: dict) -> dict:
    r = requests.post(url, json=body, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from possibly-fenced model output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _router_system_prompt() -> str:
    skills = ", ".join(SKILL_CATALOG)
    return (
        "You are a task-routing classifier for terminal tasks. Rate how much the task "
        f"relies on each of these skills: {skills}. Return STRICT JSON mapping each skill "
        "to a weight in [0,1] (0=irrelevant, 1=central). Output only the JSON object."
    )


def classify_task(task_text, *, base_url, api_key, model, call_fn=_post, max_tokens=400):
    """One classifier call -> (weights restricted+clamped to the catalog, tokens_in, tokens_out)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _router_system_prompt()},
            {"role": "user", "content": task_text},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = call_fn(base_url.rstrip("/") + "/chat/completions", headers, body)
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
    raw = _extract_json(content)
    weights = {s: max(0.0, min(1.0, float(raw.get(s, 0.0) or 0.0))) for s in SKILL_CATALOG}
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return weights, None, None
    return weights, usage.get("prompt_tokens"), usage.get("completion_tokens")


def route_task(task_text, *, base_url, api_key, model, margin=0.8,
               price_in: float | None = None, price_out: float | None = None,
               call_fn=_post) -> RouteDecision:
    """Classify the task, score local vs cloud expected competence, decide the tier."""
    weights, tin, tout = classify_task(task_text, base_url=base_url, api_key=api_key,
                                       model=model, call_fn=call_fn)
    local_score = sum(weights[s] * COMPETENCE[s][0] for s in SKILL_CATALOG)
    cloud_score = sum(weights[s] * COMPETENCE[s][1] for s in SKILL_CATALOG)
    tier = "local" if local_score >= margin * cloud_score else "cloud"
    classifier_tokens = None if tin is None or tout is None else tin + tout
    classifier_dollars = None
    if tin is not None and tout is not None and price_in is not None and price_out is not None:
        classifier_dollars = dollars_tokens(tin, tout, price_in, price_out)
    return RouteDecision(
        tier=tier, weights=weights, local_score=local_score, cloud_score=cloud_score,
        classifier_tokens=classifier_tokens, classifier_dollars=classifier_dollars,
    )

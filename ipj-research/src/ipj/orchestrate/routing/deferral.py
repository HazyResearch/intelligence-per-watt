"""Self-assessment deferral signal for the v2 router.

Instead of a fixed turn budget, periodically ask the LOCAL model itself whether it is
making progress (CONTINUE) or should hand off to a stronger model (DEFER). This attacks
the dominant cascade waste: the local model burns ~3/4 of its compute (J and latency) on
tasks it will escalate anyway, because failures are timeouts — the longest tasks (see the
inefficiency decomposition, EXP-20260626-02). A DEFER latches the task to the cloud tier,
reclaiming the rest of the local run.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFER_PROMPT = (
    "Pause and honestly assess your progress on the task above. Reply with EXACTLY one word:\n"
    "- CONTINUE if you are making clear progress and are likely to finish soon.\n"
    "- DEFER if you are stuck, looping, or unlikely to solve it, so a more capable model "
    "should take over.\n"
    "Answer:"
)


@dataclass
class DeferralDecision:
    defer: bool
    tokens_in: int = 0
    tokens_out: int = 0
    raw: str = ""


def parse_decision(text: str) -> bool:
    """DEFER only on an explicit DEFER signal; default to CONTINUE.

    A malformed/empty self-assessment must NOT escalate — otherwise a flaky probe silently
    inflates the cloud bill. So we require an explicit DEFER that is not paired with CONTINUE.
    """
    t = (text or "").strip().upper()
    if not t:
        return False
    if "DEFER" in t and "CONTINUE" not in t:
        return True
    return t.startswith("DEFER")


def assess_deferral(
    messages,
    *,
    base_url: str,
    api_key: str,
    model: str,
    forward_fn,
    prompt: str = DEFER_PROMPT,
    max_tokens: int = 8,
    disable_thinking: bool = True,
) -> DeferralDecision:
    """Ask ``model`` (the local serve) to gauge CONTINUE vs DEFER given the conversation.

    ``forward_fn(url, headers, body) -> dict`` is the same forwarder the proxy uses, so the
    probe shares the proxy's HTTP path in production and is trivially mockable in tests.
    On any probe error we return CONTINUE (fail-open to the cheaper tier, never escalate
    on a broken probe).

    ``disable_thinking`` sends ``chat_template_kwargs={"enable_thinking": false}`` so a reasoning
    local model emits the one-word verdict directly instead of burning the tiny ``max_tokens``
    budget on a thinking trace (validated on Qwen3.6: without it the probe only ever returns
    thinking tokens, so it never DEFERs).
    """
    probe = list(messages) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": probe, "max_tokens": max_tokens, "temperature": 0.0}
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = forward_fn(base_url.rstrip("/") + "/chat/completions", headers, body)
    except Exception:
        return DeferralDecision(defer=False, raw="<probe-error>")
    text = (((resp.get("choices") or [{}])[0]).get("message") or {}).get("content", "") or ""
    usage = resp.get("usage") or {}
    return DeferralDecision(
        defer=parse_decision(text),
        tokens_in=usage.get("prompt_tokens", 0) or 0,
        tokens_out=usage.get("completion_tokens", 0) or 0,
        raw=text.strip()[:60],
    )

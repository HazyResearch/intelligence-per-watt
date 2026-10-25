"""Chat-client adapters (OpenAI, Anthropic) normalizing provider SDKs to a common ChatResponse."""

from __future__ import annotations

from typing import Any

from .brain import ChatResponse

try:  # cost lookup is reused from IPW; degrade gracefully if unavailable
    from ipw.cost.pricing import calculate_cost_or_none
except Exception:  # pragma: no cover - exercised only without the IPW venv

    def calculate_cost_or_none(
        provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        return None


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


class AnthropicChatClient:
    """Adapter wrapping the Anthropic SDK; normalizes responses to ChatResponse (with prompt caching)."""

    def __init__(self, model: str, *, sdk: Any | None = None, api_key: str | None = None) -> None:
        self.model = model
        if sdk is None:  # pragma: no cover - real SDK construction
            import anthropic

            sdk = anthropic.Anthropic(api_key=api_key)
        self._sdk = sdk

    def complete(self, messages: list[dict[str, Any]], **params: Any) -> ChatResponse:
        system, rest = _split_system(messages)
        # Anthropic prompt caching: mark the (large, stable) system block as cacheable.
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        # Map every non-assistant role (e.g. our replayed "tool"/"worker" turns) to "user".
        mapped = [
            {"role": ("assistant" if m["role"] == "assistant" else "user"), "content": m["content"]} for m in rest
        ]
        # The Anthropic Messages API requires strictly alternating user/assistant turns and a
        # leading user turn; coalesce consecutive same-role turns and prepend a user turn if needed.
        anth_msgs: list[dict[str, Any]] = []
        for m in mapped:
            if anth_msgs and anth_msgs[-1]["role"] == m["role"]:
                anth_msgs[-1]["content"] = f"{anth_msgs[-1]['content']}\n\n{m['content']}"
            else:
                anth_msgs.append({"role": m["role"], "content": m["content"]})
        if anth_msgs and anth_msgs[0]["role"] != "user":
            anth_msgs.insert(0, {"role": "user", "content": "(continue)"})
        resp = self._sdk.messages.create(
            model=self.model,
            system=system_blocks,
            messages=anth_msgs,
            max_tokens=params.get("max_tokens", 4096),
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        tin = int(getattr(resp.usage, "input_tokens", 0))
        tout = int(getattr(resp.usage, "output_tokens", 0))
        return ChatResponse(
            content=text,
            tokens_in=tin,
            tokens_out=tout,
            dollars=calculate_cost_or_none("anthropic", self.model, tin, tout),
        )


class OpenAIChatClient:
    """Adapter wrapping the OpenAI SDK; normalizes responses to ChatResponse."""

    def __init__(
        self, model: str, *, sdk: Any | None = None, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        self.model = model
        self.base_url = base_url
        if sdk is None:  # pragma: no cover - real SDK construction
            import openai

            sdk = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._sdk = sdk

    def complete(self, messages: list[dict[str, Any]], **params: Any) -> ChatResponse:
        # OpenAI chat.completions rejects role:"tool" without a tool_call_id; our replayed
        # tool/worker results carry role:"tool", so map any non-standard role to "user".
        _allowed = {"system", "user", "assistant"}
        sdk_messages = [
            {"role": (m["role"] if m["role"] in _allowed else "user"), "content": m["content"]} for m in messages
        ]
        resp = self._sdk.chat.completions.create(
            model=self.model,
            messages=sdk_messages,
            max_completion_tokens=params.get("max_tokens", 4096),
        )
        text = resp.choices[0].message.content or ""
        tin = int(getattr(resp.usage, "prompt_tokens", 0))
        tout = int(getattr(resp.usage, "completion_tokens", 0))
        provider = "openai"
        return ChatResponse(
            content=text,
            tokens_in=tin,
            tokens_out=tout,
            dollars=calculate_cost_or_none(provider, self.model, tin, tout),
        )

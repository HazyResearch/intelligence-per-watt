"""Small OpenAI-compatible chat harness utilities."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from ipw.agents.base import BaseAgent
from ipw.cost.pricing import calculate_cost


@dataclass
class ChatCallResult:
    content: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    token_source: str = "missing"


def _approx_context_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _usage_int(usage: dict[str, Any], key: str) -> Optional[int]:
    value = usage.get(key)
    if value is None:
        return None
    return int(value)


def _cost_or_none(
    provider: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None
    return calculate_cost(provider, model, input_tokens, output_tokens)


def _strip_provider(model: str) -> tuple[str, str]:
    if model.startswith("anthropic/"):
        return "anthropic", model.split("/", 1)[1]
    if model.startswith(("gemini/", "google/")):
        return "gemini", model.split("/", 1)[1]
    if model.startswith("openai/"):
        return "openai", model.split("/", 1)[1]
    if model.startswith("claude-"):
        return "anthropic", model
    if model.startswith("gemini-"):
        return "gemini", model
    return "openai", model


def _extract_openai_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) for part in content)
    return str(content or "")


class OpenAICompatibleHarness(BaseAgent):
    """Base class for lightweight text/tool harnesses."""

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MAX_TURNS = 20
    DEFAULT_MAX_OUTPUT_TOKENS = 32_768

    def __init__(
        self,
        model: Any,
        *,
        mcp_tools: Optional[dict[str, Any]] = None,
        event_recorder=None,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        temperature: float = 0.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mcp_tools=mcp_tools, event_recorder=event_recorder)
        self.model_config = self._normalize_model_config(model, base_url or api_base, api_key)
        self.max_turns = max(1, int(max_turns))
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.instructions = instructions or "You are a helpful assistant."

    def _normalize_model_config(
        self,
        model: Any,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> dict[str, Any]:
        if api_key == "EMPTY":
            api_key = None
        if isinstance(model, dict):
            config = dict(model)
            config.setdefault("base_url", base_url)
            config.setdefault("api_key", api_key)
            if config.get("api_key") == "EMPTY":
                config["api_key"] = None
            return config
        if isinstance(model, str):
            return {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
            }
        return {"model_object": model, "model": str(model), "base_url": base_url, "api_key": api_key}

    @property
    def model_name(self) -> str:
        return str(self.model_config.get("model") or self.model_config.get("model_object") or "unknown")

    def _local_openai_context_window(self) -> int:
        return int(os.getenv("IPW_OPENAI_COMPAT_CONTEXT_WINDOW", "32768"))

    def _request_timeout_seconds(self) -> float:
        return float(
            self.model_config.get("timeout_seconds")
            or os.getenv("IPW_OPENAI_COMPAT_LLM_TIMEOUT", "120")
        )

    def _bounded_openai_max_tokens(self, messages: list[dict[str, str]]) -> int:
        if not self.model_config.get("base_url") or self.model_config.get("cloud"):
            return self.max_output_tokens
        prompt_text = json.dumps(messages)
        prompt_tokens = max(_approx_context_tokens(prompt_text), len(prompt_text) // 3)
        buffer_tokens = int(os.getenv("IPW_OPENAI_COMPAT_CONTEXT_BUFFER_TOKENS", "512"))
        available = max(1, self._local_openai_context_window() - prompt_tokens - buffer_tokens)
        return min(self.max_output_tokens, available)

    def _trim_messages_to_context(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Keep local OpenAI-compatible requests inside the server context."""
        if not self.model_config.get("base_url") or self.model_config.get("cloud"):
            return messages

        context_window = self._local_openai_context_window()
        buffer_tokens = int(os.getenv("IPW_OPENAI_COMPAT_CONTEXT_BUFFER_TOKENS", "512"))
        output_reserve = min(self.max_output_tokens, 4096)
        max_prompt_tokens = max(1024, context_window - buffer_tokens - output_reserve)

        def token_estimate(items: list[dict[str, str]]) -> int:
            text = json.dumps(items)
            return max(_approx_context_tokens(text), len(text) // 3)

        if token_estimate(messages) <= max_prompt_tokens:
            return messages

        anchors: list[dict[str, str]] = []
        cursor = 0
        if messages and messages[0].get("role") == "system":
            anchors.append(messages[0])
            cursor = 1
        if cursor < len(messages) and messages[cursor].get("role") == "user":
            anchors.append(messages[cursor])
            cursor += 1

        notice = {
            "role": "user",
            "content": "[Earlier tool observations omitted to fit the local model context window.]",
        }
        kept_reversed: list[dict[str, str]] = []
        base = anchors + [notice]
        for message in reversed(messages[cursor:]):
            candidate = base + list(reversed(kept_reversed + [message]))
            if token_estimate(candidate) <= max_prompt_tokens:
                kept_reversed.append(message)
            elif not kept_reversed:
                truncated = dict(message)
                content = str(truncated.get("content", ""))
                truncated["content"] = content[-max(2048, context_window * 2):]
                kept_reversed.append(truncated)
                break

        return anchors + [notice] + list(reversed(kept_reversed))

    def _post_openai_chat(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> requests.Response:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self._request_timeout_seconds(),
        )
        if response.status_code != 400:
            return response
        text = response.text.lower()
        if (
            not self.model_config.get("base_url")
            or self.model_config.get("cloud")
            or "maximum context length" not in text
        ):
            return response
        max_tokens = int(payload.get("max_tokens") or 0)
        if max_tokens <= 1024:
            return response
        retry_payload = dict(payload)
        retry_payload["max_tokens"] = max(1024, max_tokens // 2)
        return requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            data=json.dumps(retry_payload),
            timeout=self._request_timeout_seconds(),
        )

    def _chat(self, messages: list[dict[str, str]]) -> ChatCallResult:
        model_object = self.model_config.get("model_object")
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if model_object is not None:
            return self._chat_model_object(model_object, prompt_text)

        model = str(self.model_config.get("model") or "")
        local_openai_endpoint = bool(self.model_config.get("base_url")) and not self.model_config.get("cloud")
        if local_openai_endpoint:
            provider, provider_model = "openai", model.removeprefix("openai/")
        else:
            provider, provider_model = _strip_provider(model)
        self._record_event("lm_inference_start", model=model)
        try:
            if provider == "anthropic":
                result = self._chat_anthropic(provider_model, messages)
            elif provider == "gemini":
                result = self._chat_gemini(provider_model, messages)
            else:
                result = self._chat_openai(provider_model, messages)
            self._record_event(
                "lm_inference_end",
                model=model,
                prompt_tokens=result.input_tokens,
                completion_tokens=result.output_tokens,
                total_tokens=(
                    result.input_tokens + result.output_tokens
                    if result.input_tokens is not None
                    and result.output_tokens is not None
                    else None
                ),
                cost_usd=result.cost_usd,
            )
            return result
        except Exception as exc:
            self._record_event("lm_inference_end", model=model, error=str(exc))
            raise

    def _chat_model_object(self, model_object: Any, prompt: str) -> ChatCallResult:
        self._record_event("lm_inference_start", model=str(model_object))
        try:
            if hasattr(model_object, "response"):
                response = model_object.response(prompt)
            elif hasattr(model_object, "run"):
                response = model_object.run(prompt)
            else:
                raise RuntimeError(f"Unsupported model object: {type(model_object)!r}")
            content = str(getattr(response, "content", response) or "")
            metrics = getattr(response, "metrics", None)
            has_usage = (
                metrics is not None
                and getattr(metrics, "input_tokens", None) is not None
                and getattr(metrics, "output_tokens", None) is not None
            )
            input_tokens = int(getattr(metrics, "input_tokens")) if has_usage else None
            output_tokens = int(getattr(metrics, "output_tokens")) if has_usage else None
            self._record_event(
                "lm_inference_end",
                model=str(model_object),
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            token_source = "model_object_metrics" if has_usage else "missing_usage"
            return ChatCallResult(content, input_tokens, output_tokens, 0.0 if has_usage else None, token_source)
        except Exception as exc:
            self._record_event("lm_inference_end", model=str(model_object), error=str(exc))
            raise

    def _chat_openai(self, model: str, messages: list[dict[str, str]]) -> ChatCallResult:
        messages = self._trim_messages_to_context(messages)
        default_base = "https://api.openai.com/v1" if self.model_config.get("cloud") else self.DEFAULT_BASE_URL
        base_url = str(self.model_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or default_base).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        api_key = (
            self.model_config.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("IPW_API_KEY")
            or "EMPTY"
        )
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
        }
        max_tokens = self._bounded_openai_max_tokens(messages)
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = self.temperature
            payload["max_tokens"] = max_tokens
        response = self._post_openai_chat(
            base_url=base_url,
            headers=headers,
            payload=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = _extract_openai_content(data)
        usage = data.get("usage") or {}
        has_usage = usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None
        input_tokens = _usage_int(usage, "prompt_tokens") if has_usage else None
        output_tokens = _usage_int(usage, "completion_tokens") if has_usage else None
        return ChatCallResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_or_none("openai", model, input_tokens, output_tokens),
            token_source="openai_response_usage" if has_usage else "missing_usage",
        )

    def _chat_anthropic(self, model: str, messages: list[dict[str, str]]) -> ChatCallResult:
        api_key = self.model_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "EMPTY":
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic models")
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_messages = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": model,
            "system": system,
            "messages": user_messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": str(api_key),
                "anthropic-version": "2023-06-01",
            },
            data=json.dumps(payload),
            timeout=self._request_timeout_seconds(),
        )
        response.raise_for_status()
        data = response.json()
        content_parts = data.get("content") or []
        content = "\n".join(str(part.get("text", "")) for part in content_parts if isinstance(part, dict))
        usage = data.get("usage") or {}
        has_usage = usage.get("input_tokens") is not None and usage.get("output_tokens") is not None
        input_tokens = _usage_int(usage, "input_tokens") if has_usage else None
        output_tokens = _usage_int(usage, "output_tokens") if has_usage else None
        return ChatCallResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_or_none("anthropic", model, input_tokens, output_tokens),
            token_source="anthropic_response_usage" if has_usage else "missing_usage",
        )

    def _chat_gemini(self, model: str, messages: list[dict[str, str]]) -> ChatCallResult:
        api_key = (
            self.model_config.get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key or api_key == "EMPTY":
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini models")
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_text = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self._request_timeout_seconds(),
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        content = ""
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            content = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        usage = data.get("usageMetadata") or {}
        has_usage = (
            usage.get("promptTokenCount") is not None
            and usage.get("candidatesTokenCount") is not None
        )
        input_tokens = _usage_int(usage, "promptTokenCount") if has_usage else None
        output_tokens = _usage_int(usage, "candidatesTokenCount") if has_usage else None
        return ChatCallResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_or_none("gemini", model, input_tokens, output_tokens),
            token_source="gemini_response_usage" if has_usage else "missing_usage",
        )

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                action = parsed.get("action") or parsed.get("tool")
                action_input = parsed.get("action_input") or parsed.get("input") or parsed.get("query")
                if action and action_input is not None:
                    return str(action), str(action_input)
            except json.JSONDecodeError:
                pass

        action_match = re.search(
            r"^Action:\s*([A-Za-z0-9_\-]+)\s*$",
            text,
            re.MULTILINE,
        )
        input_marker = re.search(r"^Action Input:\s*", text, re.MULTILINE)
        action = None
        if action_match:
            action = action_match.group(1).strip()
        if action and input_marker:
            tail = text[input_marker.end():]
            stop_match = re.search(
                r"(?=^(?:Action:|Final(?: Answer)?:|Observation:)\s*)",
                tail,
                re.MULTILINE | re.DOTALL,
            )
            action_input = tail[: stop_match.start()] if stop_match else tail
            return action, action_input.strip()
        return None

    def _extract_final(self, text: str) -> str | None:
        final_match = re.search(r"^Final(?: Answer)?:\s*(.*)", text, re.MULTILINE | re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        return None

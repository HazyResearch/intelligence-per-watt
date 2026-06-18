from __future__ import annotations

import httpx

from ipw.agents.mcp.vllm_server import VLLMMCPServer


def test_vllm_mcp_retry_uses_reported_context_budget(monkeypatch) -> None:
    payloads: list[dict[str, object]] = []

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]):
            payloads.append(dict(json))
            request = httpx.Request("POST", url)
            if len(payloads) == 1:
                return httpx.Response(
                    400,
                    text=(
                        "This model's maximum context length is 4096 tokens. "
                        "However, you requested 4097 tokens (3900 in the messages, "
                        "197 in the completion). max_tokens is too large."
                    ),
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 3900,
                        "completion_tokens": 10,
                        "total_tokens": 3910,
                    },
                },
                request=request,
            )

    monkeypatch.setattr("ipw.agents.mcp.vllm_server.httpx.Client", _Client)

    server = VLLMMCPServer(
        model_name="test-model",
        vllm_url="http://127.0.0.1:1",
        max_tokens=197,
    )

    result = server.execute("hello")

    assert result.content == "ok"
    assert payloads[0]["max_tokens"] == 197
    assert payloads[1]["max_tokens"] == 195
    assert result.metadata["max_tokens_capped"] is True

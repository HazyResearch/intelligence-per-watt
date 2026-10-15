"""Tests for agents/mcp/tool_server.py — CalculatorServer, ThinkServer and WebSearchServer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.agents.mcp.tool_server import CalculatorServer, ThinkServer, WebSearchServer
from ipw.cost.pricing import FIRECRAWL_COST_PER_CREDIT


class TestCalculatorServer:
    """Test CalculatorServer safe evaluation."""

    @pytest.fixture()
    def calc(self) -> CalculatorServer:
        return CalculatorServer()

    def test_basic_arithmetic(self, calc: CalculatorServer) -> None:
        result = calc.execute("2 + 3")
        assert result.content == "5"
        assert result.cost_usd == 0.0

    def test_multiplication(self, calc: CalculatorServer) -> None:
        result = calc.execute("6 * 7")
        assert result.content == "42"

    def test_division(self, calc: CalculatorServer) -> None:
        result = calc.execute("10 / 4")
        assert result.content == "2.5"

    def test_exponentiation(self, calc: CalculatorServer) -> None:
        result = calc.execute("2 ** 10")
        assert result.content == "1024"

    def test_caret_exponentiation(self, calc: CalculatorServer) -> None:
        result = calc.execute("2 ^ 10")
        assert result.content == "1024"

    def test_nested_expression(self, calc: CalculatorServer) -> None:
        result = calc.execute("(2 + 3) * 4")
        assert result.content == "20"

    def test_sqrt_function(self, calc: CalculatorServer) -> None:
        result = calc.execute("sqrt(16)")
        assert result.content == "4.0"

    def test_negative_numbers(self, calc: CalculatorServer) -> None:
        result = calc.execute("-5 + 3")
        assert result.content == "-2"

    def test_extract_expression_from_prompt(self, calc: CalculatorServer) -> None:
        result = calc.execute("calculate 2 + 3")
        assert result.content == "5"

    def test_extract_what_is(self, calc: CalculatorServer) -> None:
        result = calc.execute("what is 10 * 5?")
        assert result.content == "50"

    def test_invalid_expression(self, calc: CalculatorServer) -> None:
        result = calc.execute("not a math expression @#$")
        assert "Error" in result.content

    def test_metadata_contains_tool_name(self, calc: CalculatorServer) -> None:
        result = calc.execute("1 + 1")
        assert result.metadata["tool"] == "calculator"

    def test_zero_token_usage(self, calc: CalculatorServer) -> None:
        result = calc.execute("1 + 1")
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0


class TestThinkServer:
    """Test ThinkServer pass-through behavior."""

    @pytest.fixture()
    def think(self) -> ThinkServer:
        return ThinkServer()

    def test_passthrough(self, think: ThinkServer) -> None:
        result = think.execute("Let me think step by step...")
        assert "[Thinking]" in result.content
        assert "Let me think step by step..." in result.content

    def test_cost_is_zero(self, think: ThinkServer) -> None:
        result = think.execute("thinking")
        assert result.cost_usd == 0.0

    def test_metadata(self, think: ThinkServer) -> None:
        result = think.execute("test")
        assert result.metadata["tool"] == "think"

    def test_zero_token_usage(self, think: ThinkServer) -> None:
        result = think.execute("test")
        assert result.usage["total_tokens"] == 0


class TestWebSearchServerFirecrawl:
    """Test the Firecrawl provider of WebSearchServer against mocked HTTP."""

    PAYLOAD = {
        "success": True,
        "creditsUsed": 2,
        "data": {
            "web": [
                {"title": "Result A", "url": "https://example.com/a", "description": "Desc A"},
                {"title": "Result B", "url": "https://example.com/b", "description": "Desc B"},
            ]
        },
    }

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("TAVILY_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
            monkeypatch.delenv(key, raising=False)

    def _mock_post(self, monkeypatch: pytest.MonkeyPatch, payload=None, status: int = 200) -> dict:
        import httpx

        calls: dict = {}

        def _post(url, **kwargs):
            calls["url"] = url
            calls["json"] = kwargs.get("json")
            calls["headers"] = kwargs.get("headers")
            request = httpx.Request("POST", url)
            return httpx.Response(status, json=payload if payload is not None else self.PAYLOAD, request=request)

        monkeypatch.setattr(httpx, "post", _post)
        return calls

    def test_auto_uses_firecrawl_when_only_its_key_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        assert WebSearchServer()._resolve_provider() == "firecrawl"

    def test_auto_keeps_tavily_when_its_key_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Adding a Firecrawl key must not change an existing Tavily setup."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        assert WebSearchServer()._resolve_provider() == "tavily"

    def test_no_keys_keeps_the_tavily_message(self) -> None:
        result = WebSearchServer().execute("q")
        assert "TAVILY_API_KEY" in result.content
        assert result.metadata["error"] == "no_api_key"

    def test_explicit_provider_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        assert WebSearchServer(provider="firecrawl")._resolve_provider() == "firecrawl"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown web_search provider"):
            WebSearchServer(provider="bing")

    def test_request_and_formatting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        calls = self._mock_post(monkeypatch)

        result = WebSearchServer(max_results=3).execute("latest AI news")

        assert calls["url"] == "https://api.firecrawl.dev/v2/search"
        assert calls["headers"]["Authorization"] == "Bearer fc-test"
        assert calls["json"] == {"query": "latest AI news", "limit": 3, "origin": "intelligence-per-watt"}
        assert result.content.startswith("Web search results for: latest AI news")
        assert "1. Result A" in result.content
        assert "   URL: https://example.com/a" in result.content
        assert "   Desc A" in result.content
        assert result.metadata["provider"] == "firecrawl"
        assert result.metadata["num_results"] == 2
        assert result.metadata["credits_used"] == 2
        assert result.cost_usd == pytest.approx(2 * FIRECRAWL_COST_PER_CREDIT)

    def test_self_hosted_api_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002/")
        calls = self._mock_post(monkeypatch)

        result = WebSearchServer().execute("q")

        assert calls["url"] == "http://localhost:3002/v2/search"
        assert result.cost_usd == 0.0

    def test_keyless_self_hosted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Self-hosted instances usually run without auth."""
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        calls = self._mock_post(monkeypatch)

        server = WebSearchServer()
        result = server.execute("q")

        assert server._resolve_provider() == "firecrawl"
        assert "Authorization" not in calls["headers"]
        assert result.metadata["num_results"] == 2

    def test_cost_follows_reported_credits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        self._mock_post(monkeypatch, payload={**self.PAYLOAD, "creditsUsed": 4})

        result = WebSearchServer(max_results=20).execute("q")

        assert result.cost_usd == pytest.approx(4 * FIRECRAWL_COST_PER_CREDIT)

    def test_backend_is_recorded_on_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runs must show which search provider produced their results."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        self._mock_post(monkeypatch)
        recorder = MagicMock()
        server = WebSearchServer()
        server.event_recorder = recorder

        server.execute("q")

        backends = {c.kwargs.get("backend") for c in recorder.record.call_args_list}
        assert backends == {"firecrawl"}
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        assert WebSearchServer()._get_backend() == "tavily"

    def test_tavily_results_carry_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        server = WebSearchServer()
        client = MagicMock()
        client.search.return_value = {
            "answer": "An answer",
            "results": [{"title": "T", "url": "https://example.com", "content": "C"}],
        }
        server._client = client

        result = server.execute("q")

        assert result.metadata["provider"] == "tavily"
        assert "Summary: An answer" in result.content
        assert "1. T" in result.content
        assert result.cost_usd == WebSearchServer.COST_PER_SEARCH

    def test_explicit_provider_without_key(self) -> None:
        result = WebSearchServer(provider="firecrawl").execute("q")
        assert result.metadata["error"] == "no_api_key"
        assert "FIRECRAWL_API_KEY" in result.content

    def test_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setattr(httpx, "post", MagicMock(side_effect=httpx.ConnectError("down")))

        result = WebSearchServer().execute("q")

        assert result.content.startswith("Search error: ConnectError")
        assert result.cost_usd == 0.0

    def test_truncation_limits_apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        payload = {"data": {"web": [{"title": "Long", "url": "https://example.com", "description": "x" * 500}]}}
        self._mock_post(monkeypatch, payload=payload)

        result = WebSearchServer(max_content_chars=50, max_total_chars=200).execute("q")

        assert "... (result truncated)" in result.content
        assert "x" * 51 not in result.content

    def test_rejected_key_names_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-bad")
        self._mock_post(monkeypatch, payload={"success": False}, status=401)

        result = WebSearchServer().execute("q")

        assert "FIRECRAWL_API_KEY" in result.content
        assert result.cost_usd == 0.0

    def test_exhausted_credits_point_to_the_key_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        self._mock_post(monkeypatch, payload={"success": False}, status=402)

        result = WebSearchServer().execute("q")

        assert "credits" in result.content
        assert "firecrawl.dev/app/api-keys" in result.content
        assert result.cost_usd == 0.0

    def test_rate_limit_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        self._mock_post(monkeypatch, payload={"success": False}, status=429)

        result = WebSearchServer().execute("q")

        assert "rate limit" in result.content

    def test_other_http_error_reports_the_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        self._mock_post(monkeypatch, payload={"success": False}, status=500)

        result = WebSearchServer().execute("q")

        assert "HTTP 500" in result.content

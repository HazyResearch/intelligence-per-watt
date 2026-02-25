"""Tests for agents/mcp/openai_server.py -- OpenAIMCPServer pricing."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from ipw.cost.pricing import OPENAI_PRICING, calculate_cost


@pytest.fixture(autouse=True)
def _mock_openai_module():
    """Temporarily mock the openai module for the duration of each test.

    Uses patch.dict so the original sys.modules state is always restored,
    preventing pollution of the openai module for downstream tests.
    """
    mock_openai = MagicMock()
    mock_openai.__spec__ = importlib.machinery.ModuleSpec("openai", None)

    # Save and remove any cached ipw.agents.mcp.openai_server so it
    # re-imports with the mocked openai.
    saved = sys.modules.pop("ipw.agents.mcp.openai_server", None)
    try:
        with patch.dict("sys.modules", {"openai": mock_openai}):
            yield mock_openai
    finally:
        # Remove the re-imported module so it doesn't leak either
        sys.modules.pop("ipw.agents.mcp.openai_server", None)
        if saved is not None:
            sys.modules["ipw.agents.mcp.openai_server"] = saved


def _make_server(mock_openai: MagicMock, model_name: str = "gpt-4o"):
    """Create an OpenAIMCPServer with mocked openai client."""
    from ipw.agents.mcp.openai_server import OpenAIMCPServer

    mock_openai.OpenAI.reset_mock()
    mock_openai.OpenAI.return_value = MagicMock()
    return OpenAIMCPServer(model_name=model_name, api_key="test-key")


class TestOpenAIMCPServerPricing:
    """Test OpenAI server pricing calculations with mocked openai."""

    def test_pricing_calculation_gpt4o(self, _mock_openai_module: MagicMock) -> None:
        server = _make_server(_mock_openai_module, "gpt-4o")

        assert server.pricing is not None
        assert server.pricing["input"] == 2.50
        assert server.pricing["output"] == 10.00

        cost = calculate_cost("openai", "gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_unknown_model_falls_back_to_gpt4o_pricing(
        self, _mock_openai_module: MagicMock
    ) -> None:
        server = _make_server(_mock_openai_module, "totally-unknown-model")
        assert server.pricing["input"] == 2.50
        assert server.pricing["output"] == 10.00

    def test_name_includes_model(self, _mock_openai_module: MagicMock) -> None:
        server = _make_server(_mock_openai_module, "gpt-4o-mini")
        assert server.name == "openai:gpt-4o-mini"

    def test_execute_impl_with_mocked_stream(
        self, _mock_openai_module: MagicMock
    ) -> None:
        from ipw.agents.mcp.openai_server import OpenAIMCPServer

        _mock_openai_module.OpenAI.reset_mock()
        _mock_openai_module.OpenAIError = type("OpenAIError", (Exception,), {})

        mock_client = MagicMock()
        _mock_openai_module.OpenAI.return_value = mock_client

        # Build mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.usage = None

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = []
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_chunk3.usage = mock_usage

        mock_client.chat.completions.create.return_value = iter(
            [mock_chunk1, mock_chunk2, mock_chunk3]
        )

        server = OpenAIMCPServer(model_name="gpt-4o", api_key="test-key")
        result = server._execute_impl("test prompt")

        assert result.content == "Hello world"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.cost_usd > 0

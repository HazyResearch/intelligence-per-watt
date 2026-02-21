"""Tests for agents/mcp/openai_server.py -- OpenAIMCPServer pricing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ipw.cost.pricing import OPENAI_PRICING, calculate_cost

# Install a persistent mock for the openai module so that the lazy
# ``from openai import OpenAI`` inside OpenAIMCPServer.__init__ succeeds.
# This is done at module-level so it survives across tests without
# repeatedly patching sys.modules (which can destabilize pyarrow).
# We set __spec__ so importlib.util.find_spec("openai") doesn't crash.
import importlib
_MOCK_OPENAI = MagicMock()
_MOCK_OPENAI.__spec__ = importlib.machinery.ModuleSpec("openai", None)
if "openai" not in sys.modules:
    sys.modules["openai"] = _MOCK_OPENAI

from ipw.agents.mcp.openai_server import OpenAIMCPServer  # noqa: E402


def _make_server(model_name: str = "gpt-4o") -> OpenAIMCPServer:
    """Create an OpenAIMCPServer with mocked openai client."""
    _MOCK_OPENAI.OpenAI.reset_mock()
    _MOCK_OPENAI.OpenAI.return_value = MagicMock()
    return OpenAIMCPServer(model_name=model_name, api_key="test-key")


class TestOpenAIMCPServerPricing:
    """Test OpenAI server pricing calculations with mocked openai."""

    def test_pricing_calculation_gpt4o(self) -> None:
        server = _make_server("gpt-4o")

        assert server.pricing is not None
        assert server.pricing["input"] == 2.50
        assert server.pricing["output"] == 10.00

        cost = calculate_cost("openai", "gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_unknown_model_falls_back_to_gpt4o_pricing(self) -> None:
        server = _make_server("totally-unknown-model")
        assert server.pricing["input"] == 2.50
        assert server.pricing["output"] == 10.00

    def test_name_includes_model(self) -> None:
        server = _make_server("gpt-4o-mini")
        assert server.name == "openai:gpt-4o-mini"

    def test_execute_impl_with_mocked_stream(self) -> None:
        _MOCK_OPENAI.OpenAI.reset_mock()
        _MOCK_OPENAI.OpenAIError = type("OpenAIError", (Exception,), {})

        mock_client = MagicMock()
        _MOCK_OPENAI.OpenAI.return_value = mock_client

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

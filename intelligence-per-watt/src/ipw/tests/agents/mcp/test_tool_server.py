"""Tests for agents/mcp/tool_server.py — CalculatorServer and ThinkServer."""

from __future__ import annotations

import pytest

from ipw.agents.mcp.tool_server import CalculatorServer, ThinkServer


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

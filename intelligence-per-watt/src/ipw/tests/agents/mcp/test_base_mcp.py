"""Tests for agents/mcp/base.py — MCPToolResult and BaseMCPServer."""

from __future__ import annotations

import pytest

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult
from ipw.telemetry.events import EventRecorder


class TestMCPToolResult:
    """Test MCPToolResult dataclass fields."""

    def test_default_values(self) -> None:
        result = MCPToolResult(content="hello")
        assert result.content == "hello"
        assert result.usage == {}
        assert result.cost_usd is None
        assert result.telemetry_samples == []
        assert result.latency_seconds == 0.0
        assert result.ttft_seconds is None
        assert result.metadata == {}

    def test_all_fields(self) -> None:
        result = MCPToolResult(
            content="answer",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost_usd=0.001,
            telemetry_samples=[{"power": 100}],
            latency_seconds=0.5,
            ttft_seconds=0.1,
            metadata={"model": "test"},
        )
        assert result.cost_usd == 0.001
        assert result.latency_seconds == 0.5
        assert result.ttft_seconds == 0.1
        assert result.usage["total_tokens"] == 15

    def test_usage_dict_is_independent(self) -> None:
        r1 = MCPToolResult(content="a")
        r2 = MCPToolResult(content="b")
        r1.usage["key"] = 1
        assert "key" not in r2.usage


class TestBaseMCPServer:
    """Test BaseMCPServer contract."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseMCPServer(name="test")

    def test_concrete_subclass(self) -> None:
        class DummyServer(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                return MCPToolResult(content=f"echo: {prompt}")

        server = DummyServer(name="dummy")
        assert server.name == "dummy"
        assert repr(server) == "DummyServer(name='dummy')"

    def test_execute_wraps_impl(self) -> None:
        class DummyServer(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                return MCPToolResult(content=f"echo: {prompt}")

        server = DummyServer(name="dummy")
        result = server.execute("hello")
        assert result.content == "echo: hello"
        assert result.latency_seconds > 0

    def test_execute_records_events(self) -> None:
        recorder = EventRecorder()

        class DummyServer(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                return MCPToolResult(
                    content="ok",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    cost_usd=0.001,
                )

        server = DummyServer(name="dummy", event_recorder=recorder)
        server.execute("test")

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "submodel_call_start"
        assert events[1].event_type == "submodel_call_end"
        assert events[1].metadata["total_tokens"] == 15
        assert events[1].metadata["cost_usd"] == 0.001

    def test_get_backend_from_name(self) -> None:
        class DummyServer(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                return MCPToolResult(content="ok")

        server = DummyServer(name="vllm:model")
        assert server._get_backend() == "vllm"

    def test_get_backend_unknown(self) -> None:
        class DummyServer(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                return MCPToolResult(content="ok")

        server = DummyServer(name="calculator")
        assert server._get_backend() == "unknown"

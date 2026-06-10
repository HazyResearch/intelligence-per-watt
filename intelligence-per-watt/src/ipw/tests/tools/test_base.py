"""Tests for tools/base.py — BaseTool, ToolSpec, ToolResult, ToolCallMode."""

from __future__ import annotations

import asyncio

import pytest

from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType
from ipw.tools.base import BaseTool, ToolCallMode, ToolResult, ToolSpec


class TestToolSpec:
    def test_minimal_spec(self) -> None:
        spec = ToolSpec(name="calc", description="basic calculator", parameters={})
        assert spec.name == "calc"
        assert spec.description == "basic calculator"
        assert spec.parameters == {}
        assert spec.side_effect_conflict is False
        assert spec.requires_docker is False
        assert spec.requires_network is False

    def test_side_effect_tags(self) -> None:
        spec = ToolSpec(
            name="shell",
            description="shell",
            parameters={"cmd": {"type": "string"}},
            side_effect_conflict=True,
            requires_docker=True,
            requires_network=False,
        )
        assert spec.side_effect_conflict is True
        assert spec.requires_docker is True


class TestToolResult:
    def test_success_result(self) -> None:
        r = ToolResult(content="42", success=True)
        assert r.success is True
        assert r.content == "42"
        assert r.error is None

    def test_error_result(self) -> None:
        r = ToolResult(content="", success=False, error="div by zero")
        assert r.success is False
        assert r.error == "div by zero"


class TestToolCallMode:
    def test_modes_exist(self) -> None:
        assert ToolCallMode.FUNCTION_CALLING.value == "function_calling"
        assert ToolCallMode.STRUCTURED_TEXT.value == "structured_text"


class _EchoTool(BaseTool):
    spec = ToolSpec(
        name="echo",
        description="echoes back its input",
        parameters={"text": {"type": "string"}},
    )

    async def run(self, **kwargs: object) -> ToolResult:
        text = kwargs.get("text", "")
        return ToolResult(content=str(text), success=True)


class TestBaseTool:
    def test_call_returns_result(self) -> None:
        tool = _EchoTool()
        result = asyncio.run(tool(text="hello"))
        assert isinstance(result, ToolResult)
        assert result.content == "hello"
        assert result.success is True

    def test_call_publishes_start_and_end_events(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.TOOL_CALL_START, received.append)
        bus.subscribe(EventType.TOOL_CALL_END, received.append)

        tool = _EchoTool(bus=bus)
        asyncio.run(tool(text="hi"))

        types = [e.event_type for e in received]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types

    def test_call_propagates_correlation_id(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(None, received.append)
        tool = _EchoTool(bus=bus)
        asyncio.run(tool(correlation_id="cid-7", text="x"))
        cids = {e.correlation_id for e in received if e.correlation_id is not None}
        assert "cid-7" in cids

    def test_call_records_tool_name_in_payload(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.TOOL_CALL_START, received.append)
        tool = _EchoTool(bus=bus)
        asyncio.run(tool(text="x"))
        assert received[0].payload["tool"] == "echo"

    def test_end_event_records_status(self) -> None:
        bus = EventBus()
        ended: list[Event] = []
        bus.subscribe(EventType.TOOL_CALL_END, ended.append)
        tool = _EchoTool(bus=bus)
        asyncio.run(tool(text="x"))
        assert ended[0].payload["status"] == "ok"

    def test_exception_in_run_emits_error_end_event(self) -> None:
        class _Boom(BaseTool):
            spec = ToolSpec(name="boom", description="raises", parameters={})

            async def run(self, **kwargs: object) -> ToolResult:
                raise RuntimeError("kaboom")

        bus = EventBus()
        ended: list[Event] = []
        bus.subscribe(EventType.TOOL_CALL_END, ended.append)
        tool = _Boom(bus=bus)

        with pytest.raises(RuntimeError, match="kaboom"):
            asyncio.run(tool())

        assert ended and ended[0].payload["status"] == "error"
        assert "kaboom" in ended[0].payload["error"]

    def test_auto_generates_correlation_id_when_none_passed(self) -> None:
        # When the caller omits correlation_id, __call__ generates a UUID4.
        # Without this test, an Executor regression that drops correlation_id
        # is silently masked by the auto-generation path.
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(None, received.append)
        tool = _EchoTool(bus=bus)
        asyncio.run(tool(text="x"))
        cids = [e.correlation_id for e in received if e.correlation_id]
        assert len(cids) == 2          # START + END
        assert all(len(c) == 36 for c in cids)  # UUID4 string format
        assert cids[0] == cids[1]      # same cid across both events

    def test_set_workspace_stores_path(self) -> None:
        tool = _EchoTool()
        assert tool._default_cwd is None
        tool.set_workspace("/tmp/ws")
        assert tool._default_cwd == "/tmp/ws"

    def test_set_workspace_clears_with_none(self) -> None:
        tool = _EchoTool()
        tool.set_workspace("/tmp/ws")
        tool.set_workspace(None)
        assert tool._default_cwd is None

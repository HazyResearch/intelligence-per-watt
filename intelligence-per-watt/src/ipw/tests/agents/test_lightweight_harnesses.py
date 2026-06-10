from __future__ import annotations

from dataclasses import dataclass

from ipw.agents.dspy_rlm import DSPyRLM
from ipw.agents.forgecode import ForgeCode
from ipw.agents.mcp.base import MCPToolResult
from ipw.core.types import AgentRunResult
from ipw.telemetry.events import EventRecorder, EventType


@dataclass
class _Response:
    content: str
    metrics: object | None = None


class _FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def response(self, prompt: str) -> _Response:
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return _Response(output)


class _Tool:
    def execute(self, prompt: str) -> MCPToolResult:
        return MCPToolResult(content=f"tool saw {prompt}", usage={}, cost_usd=0.0, metadata={})


class _RecordingTool(_Tool):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def execute(self, prompt: str) -> MCPToolResult:
        self.prompts.append(prompt)
        return super().execute(prompt)


def test_dspy_rlm_emits_lm_and_tool_events() -> None:
    recorder = EventRecorder()
    model = _FakeModel(["Action: calculator\nAction Input: 2+2", "Final: 4"])
    agent = DSPyRLM(model=model, mcp_tools={"calculator": _Tool()}, event_recorder=recorder)

    result = agent.run("What is 2+2?")

    assert isinstance(result, AgentRunResult)
    assert result.content == "4"
    assert result.tool_calls_attempted == 1
    assert result.tool_calls_succeeded == 1

    event_types = [event.event_type for event in recorder.get_events()]
    assert event_types.count(EventType.LM_INFERENCE_START) == 2
    assert event_types.count(EventType.LM_INFERENCE_END) == 2
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_END in event_types


def test_dspy_rlm_prioritizes_action_before_final_text() -> None:
    recorder = EventRecorder()
    model = _FakeModel(
        [
            "<answer></answer>\nAction: bash\nAction Input: echo hi\nFinal: not yet",
            "Final: done",
        ]
    )
    agent = DSPyRLM(model=model, mcp_tools={"bash": _Tool()}, event_recorder=recorder)

    result = agent.run("Run a command")

    assert result.content == "done"
    assert result.tool_calls_attempted == 1
    assert result.tool_names_used == ["bash"]


def test_dspy_rlm_ignores_non_contract_inline_tool_call() -> None:
    recorder = EventRecorder()
    model = _FakeModel(
        [
            "<|tool_call>call:bash Action: bash Action Input: find . -name '*.py' Final: <answer></answer>",
        ]
    )
    tool = _RecordingTool()
    agent = DSPyRLM(model=model, mcp_tools={"bash": tool}, event_recorder=recorder)

    result = agent.run("Run a command")

    assert result.tool_calls_attempted == 0
    assert result.tool_names_used == []
    assert tool.prompts == []


def test_forgecode_adds_workspace_context(tmp_path) -> None:
    model = _FakeModel(["Final: diff --git a/a.py b/a.py"])
    agent = ForgeCode(model=model)
    agent.set_workspace(str(tmp_path))
    result = agent.run("Fix the repo")
    assert "diff --git" in result.content
    assert model.calls == 1


def test_local_openai_context_trimming_preserves_task(monkeypatch) -> None:
    monkeypatch.setenv("IPW_OPENAI_COMPAT_CONTEXT_WINDOW", "2048")
    agent = DSPyRLM(
        model={"model": "m", "base_url": "http://127.0.0.1:1/v1", "api_key": "EMPTY"},
        max_output_tokens=1024,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
    ]
    messages.extend(
        {"role": "user" if idx % 2 else "assistant", "content": "x" * 2000}
        for idx in range(20)
    )

    trimmed = agent._trim_messages_to_context(messages)

    assert trimmed[0]["content"] == "system"
    assert trimmed[1]["content"] == "original task"
    assert len(trimmed) < len(messages)
    assert any("omitted" in message["content"] for message in trimmed)

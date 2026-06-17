from __future__ import annotations

import json
import subprocess
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
        self.prompts: list[str] = []

    def response(self, prompt: str) -> _Response:
        self.prompts.append(prompt)
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


def test_forgecode_prompt_includes_concrete_tool_examples() -> None:
    model = _FakeModel(["Final: done"])
    agent = ForgeCode(
        model=model,
        mcp_tools={"bash": _Tool(), "file_read": _Tool(), "file_write": _Tool()},
    )

    agent.run("Fix the repo")

    prompt = model.prompts[0]
    assert "Action: bash\nAction Input:" in prompt
    assert "Action: file_read\nAction Input:" in prompt
    assert "Action: file_write\nAction Input:" in prompt
    assert "Do not include chat-template markers" in prompt
    assert "Do not spend the whole run reading files" in prompt
    assert "make the smallest plausible edit" in prompt
    assert "Final:\n```diff" in prompt


def test_forgecode_rejects_excessive_read_only_inspection(tmp_path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    model = _FakeModel(["Action: bash\nAction Input: grep -R foo ."] * 5)
    tool = _RecordingTool()
    agent = ForgeCode(model=model, mcp_tools={"bash": tool}, max_turns=5)
    agent.set_workspace(str(tmp_path))

    result = agent.run("Fix the repo")

    assert len(tool.prompts) == 3
    assert result.tool_calls_attempted == 5
    assert result.tool_calls_succeeded == 3
    assert any("Read-only inspection budget exhausted" in prompt for prompt in model.prompts)


def test_forgecode_turn_limit_nudges_final_diff(tmp_path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    model = _FakeModel(
        [
            "Action: bash\nAction Input: grep -R foo .",
            "Action: bash\nAction Input: grep -R bar .",
            "Action: bash\nAction Input: grep -R baz .",
            "Action: bash\nAction Input: grep -R qux .",
            "Final:\n```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n```",
        ]
    )
    tool = _RecordingTool()
    agent = ForgeCode(model=model, mcp_tools={"bash": tool}, max_turns=4)
    agent.set_workspace(str(tmp_path))

    result = agent.run("Fix the repo")

    assert len(tool.prompts) == 3
    assert model.calls == 5
    assert "Tool budget exhausted" in model.prompts[-1]
    assert "diff --git a/a.py b/a.py" in result.content


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


def test_local_openai_context_retry_uses_reported_vllm_budget(monkeypatch) -> None:
    agent = DSPyRLM(
        model={"model": "m", "base_url": "http://127.0.0.1:1/v1", "api_key": "EMPTY"},
        max_output_tokens=1000,
    )
    payloads: list[dict[str, object]] = []

    class _Response:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    def _post(*_args, **kwargs):
        payloads.append(json.loads(kwargs["data"]))
        if len(payloads) == 1:
            return _Response(
                400,
                "This model's maximum context length is 4096 tokens. "
                "However, you requested 4097 tokens (3900 in the messages, "
                "197 in the completion).",
            )
        return _Response(200)

    monkeypatch.setattr("ipw.agents.openai_compat.requests.post", _post)

    response = agent._post_openai_chat(
        base_url="http://127.0.0.1:1/v1",
        headers={},
        payload={"model": "m", "messages": [], "max_tokens": 197},
    )

    assert response.status_code == 200
    assert payloads[0]["max_tokens"] == 197
    assert payloads[1]["max_tokens"] == 195

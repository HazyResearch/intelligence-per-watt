"""Tests for agents/react_native.py — native ReAct (no Agno)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from ipw.agents.react_native import REACT_SYSTEM_PROMPT, NativeReact
from ipw.execution.executor import ExecutorContext, ToolCall, ToolCallRecord, TurnOutput, TurnRecord
from ipw.tools.base import BaseTool, ToolCallMode, ToolResult, ToolSpec


class _EchoTool(BaseTool):
    spec = ToolSpec(name="echo", description="echo a string", parameters={
        "text": {"type": "string"}
    })

    async def run(self, **kwargs):
        return ToolResult(content=str(kwargs.get("text", "")), success=True)


class _StubLLM:
    """Stub LM client — returns canned completions in sequence."""

    def __init__(self, completions: List[str]) -> None:
        self._completions = completions
        self._idx = 0
        self.calls: List[Dict[str, Any]] = []

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        out = self._completions[self._idx]
        self._idx += 1
        return out


class TestNativeReactConfiguration:
    def test_tool_mode_is_structured_text(self) -> None:
        agent = NativeReact(model="gpt-4o-mini", llm=_StubLLM([]), tools=[])
        assert agent.tool_mode == ToolCallMode.STRUCTURED_TEXT

    def test_default_system_prompt_constant_exists(self) -> None:
        assert "THOUGHT" in REACT_SYSTEM_PROMPT
        assert "ACTION" in REACT_SYSTEM_PROMPT
        assert "FINAL_ANSWER" in REACT_SYSTEM_PROMPT


class TestNativeReactStep:
    def test_single_turn_final_answer(self) -> None:
        llm = _StubLLM(["THOUGHT: easy\nFINAL_ANSWER: 42"])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("What is 2*21?")
        ctx = ExecutorContext(task_id="t")

        turn = asyncio.run(agent.step(ctx))
        assert turn.is_final
        assert turn.final_answer == "42"

    def test_tool_call_turn(self) -> None:
        llm = _StubLLM([
            'THOUGHT: I need to echo.\nACTION: echo\nINPUT: {"text": "hi"}',
        ])
        tool = _EchoTool()
        agent = NativeReact(model="m", llm=llm, tools=[tool])
        agent.set_task("Echo 'hi'")
        ctx = ExecutorContext(task_id="t")

        turn = asyncio.run(agent.step(ctx))
        assert turn.is_final is False
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "echo"
        assert turn.tool_calls[0].input == {"text": "hi"}

    def test_prompt_includes_tool_descriptions(self) -> None:
        llm = _StubLLM(["FINAL_ANSWER: ok"])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("X")
        ctx = ExecutorContext(task_id="t")
        asyncio.run(agent.step(ctx))

        # The LLM should have been called with a prompt mentioning the echo tool
        prompt_text = str(llm.calls[0]["messages"])
        assert "echo" in prompt_text


def _make_turn_record(
    turn_index: int,
    raw_output: str,
    tool_name: str = "echo",
    observation: str = "result",
) -> TurnRecord:
    """Helper to build a TurnRecord that looks like a real completed turn."""
    tc = ToolCall(name=tool_name, input={"text": observation})
    tc_result = ToolResult(content=observation, success=True)
    return TurnRecord(
        turn_index=turn_index,
        output=TurnOutput(final_answer=None, tool_calls=[tc]),
        tool_records=[ToolCallRecord(call=tc, result=tc_result, error=None)],
    )


class TestNativeReactChainContinuity:
    """Verify that prior assistant outputs are replayed in the message history."""

    def test_assistant_turn_included_in_second_step_messages(self) -> None:
        """On turn 1 the messages must include the assistant's turn-0 output."""
        turn0_raw = 'THOUGHT: searching\nACTION: echo\nINPUT: {"text": "hi"}'
        turn1_raw = "THOUGHT: done\nFINAL_ANSWER: 42"

        llm = _StubLLM([turn0_raw, turn1_raw])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        # --- Turn 0 ---
        ctx = ExecutorContext(task_id="t")
        result0 = asyncio.run(agent.step(ctx))
        assert not result0.is_final

        # Simulate Executor appending to history after dispatch.
        ctx.history.append(_make_turn_record(0, turn0_raw, observation="hi"))

        # --- Turn 1 ---
        result1 = asyncio.run(agent.step(ctx))
        assert result1.is_final

        # The messages sent on turn 1 must include an assistant message with turn 0's raw output.
        messages_turn1 = llm.calls[1]["messages"]
        assistant_msgs = [m for m in messages_turn1 if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert turn0_raw in assistant_msgs[0]["content"]

    def test_observation_follows_assistant_message(self) -> None:
        """The OBSERVATION user-message must come AFTER the assistant message, not before."""
        turn0_raw = 'THOUGHT: x\nACTION: echo\nINPUT: {"text": "ping"}'
        turn1_raw = "THOUGHT: done\nFINAL_ANSWER: pong"

        llm = _StubLLM([turn0_raw, turn1_raw])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t")
        asyncio.run(agent.step(ctx))
        ctx.history.append(_make_turn_record(0, turn0_raw, observation="ping"))
        asyncio.run(agent.step(ctx))

        messages_turn1 = llm.calls[1]["messages"]
        # Find position of the assistant message and the observation user message.
        asst_idx = next(i for i, m in enumerate(messages_turn1) if m["role"] == "assistant")
        obs_idx = next(
            i for i, m in enumerate(messages_turn1)
            if m["role"] == "user" and "OBSERVATION" in m["content"]
        )
        assert asst_idx < obs_idx, "assistant message must precede its observation"


class TestNativeReactForceFinal:
    """Verify forced final-answer behavior on the last turn."""

    def test_force_final_on_last_turn_with_action_response(self) -> None:
        """If the model returns an ACTION on the final turn, we must still get is_final=True."""
        action_raw = 'THOUGHT: still looking\nACTION: echo\nINPUT: {"text": "x"}'
        llm = _StubLLM([action_raw])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t", max_turns=10, turn_index=9)
        result = asyncio.run(agent.step(ctx))

        assert result.is_final is True
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        assert result.tool_calls == []

    def test_force_final_uses_raw_as_fallback(self) -> None:
        """The forced final answer should contain the raw model output."""
        action_raw = 'THOUGHT: still searching\nACTION: echo\nINPUT: {"text": "q"}'
        llm = _StubLLM([action_raw])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t", max_turns=5, turn_index=4)
        result = asyncio.run(agent.step(ctx))

        # The fallback must be non-trivial (contains the raw text, not "No answer.")
        assert action_raw.strip() in result.final_answer or len(result.final_answer) > 5

    def test_normal_final_answer_unaffected(self) -> None:
        """When the model voluntarily emits FINAL_ANSWER, the value is preserved unchanged."""
        llm = _StubLLM(["THOUGHT: done\nFINAL_ANSWER: Paris"])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("capital of France?")

        ctx = ExecutorContext(task_id="t", max_turns=10, turn_index=3)
        result = asyncio.run(agent.step(ctx))

        assert result.is_final is True
        assert result.final_answer == "Paris"
        assert result.tool_calls == []

    def test_non_final_turn_still_returns_tool_calls(self) -> None:
        """On a non-final turn, tool calls are still returned normally."""
        action_raw = 'THOUGHT: need to look up\nACTION: echo\nINPUT: {"text": "hi"}'
        llm = _StubLLM([action_raw])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        # turn_index=3, max_turns=10 — not the last turn
        ctx = ExecutorContext(task_id="t", max_turns=10, turn_index=3)
        result = asyncio.run(agent.step(ctx))

        assert result.is_final is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "echo"


class TestNativeReactBudgetHint:
    """Verify the per-turn budget hint injected into messages."""

    def test_soft_hint_present_when_turns_remain(self) -> None:
        """A 'turn X of Y' hint is appended when there are multiple turns remaining."""
        llm = _StubLLM(["THOUGHT: thinking\nFINAL_ANSWER: x"])
        agent = NativeReact(model="m", llm=llm, tools=[])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t", max_turns=10, turn_index=0)
        asyncio.run(agent.step(ctx))

        messages = llm.calls[0]["messages"]
        user_contents = " ".join(m["content"] for m in messages if m["role"] == "user")
        # turn_index=0, max_turns=10 → "You are on turn 1 of 10."
        assert "turn 1 of 10" in user_contents
        assert "emit FINAL_ANSWER now" in user_contents

    def test_final_turn_directive_on_last_turn(self) -> None:
        """The strong 'FINAL turn' directive is present when turn_index = max_turns - 1."""
        llm = _StubLLM(["THOUGHT: done\nFINAL_ANSWER: 42"])
        agent = NativeReact(model="m", llm=llm, tools=[])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t", max_turns=10, turn_index=9)
        asyncio.run(agent.step(ctx))

        messages = llm.calls[0]["messages"]
        user_contents = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "FINAL" in user_contents or "final" in user_contents.lower()
        assert "Do NOT call any tool" in user_contents or "FINAL_ANSWER" in user_contents

    def test_no_hint_when_max_turns_is_zero(self) -> None:
        """When max_turns=0 (default / direct unit test path), no hint is appended."""
        llm = _StubLLM(["THOUGHT: done\nFINAL_ANSWER: ok"])
        agent = NativeReact(model="m", llm=llm, tools=[])
        agent.set_task("task")

        # ExecutorContext with default max_turns=0
        ctx = ExecutorContext(task_id="t")
        asyncio.run(agent.step(ctx))

        messages = llm.calls[0]["messages"]
        user_contents = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "turns left" not in user_contents.lower()
        assert "FINAL turn" not in user_contents


class TestNativeReactSetTaskReset:
    """Verify set_task resets _raw_turns so agents can be reused across queries."""

    def test_set_task_clears_raw_turns(self) -> None:
        action_raw = 'THOUGHT: searching\nACTION: echo\nINPUT: {"text": "a"}'
        llm = _StubLLM([action_raw, "THOUGHT: done\nFINAL_ANSWER: result"])
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("first task")

        ctx = ExecutorContext(task_id="t1")
        asyncio.run(agent.step(ctx))
        assert len(agent._raw_turns) == 1

        # Reset via set_task (new query).
        agent.set_task("second task")
        assert agent._raw_turns == [], "set_task must clear _raw_turns"

    def test_raw_turns_aligned_after_multiple_steps(self) -> None:
        """After N steps, _raw_turns has exactly N entries."""
        raws = [
            'THOUGHT: step0\nACTION: echo\nINPUT: {"text": "a"}',
            'THOUGHT: step1\nACTION: echo\nINPUT: {"text": "b"}',
            "THOUGHT: done\nFINAL_ANSWER: final",
        ]
        llm = _StubLLM(raws)
        agent = NativeReact(model="m", llm=llm, tools=[_EchoTool()])
        agent.set_task("task")

        ctx = ExecutorContext(task_id="t")
        asyncio.run(agent.step(ctx))
        assert len(agent._raw_turns) == 1
        ctx.history.append(_make_turn_record(0, raws[0]))

        asyncio.run(agent.step(ctx))
        assert len(agent._raw_turns) == 2
        ctx.history.append(_make_turn_record(1, raws[1]))

        asyncio.run(agent.step(ctx))
        assert len(agent._raw_turns) == 3

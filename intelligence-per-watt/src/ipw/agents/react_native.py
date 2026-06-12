"""Native ReAct agent driven by the Executor.

No Agno dependency. The LM contract is a simple async `complete(messages)`
function returning a string. Prompts use the THOUGHT/ACTION/INPUT/FINAL_ANSWER
format parsed by `ipw.execution.parsers.parse_structured_text`.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Protocol

from ipw.agents.base import ToolUsingAgent
from ipw.core.registry import AgentRegistry
from ipw.execution.executor import ExecutorContext, ToolCall, TurnOutput
from ipw.execution.parsers import parse_structured_text
from ipw.tools.base import BaseTool, ToolCallMode

REACT_SYSTEM_PROMPT = """\
You are a careful step-by-step problem solver. You have a LIMITED number of
turns. Use the following format for every response:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
INPUT: <JSON object of arguments>

When you have the final answer, emit:

THOUGHT: <your reasoning>
FINAL_ANSWER: <the answer>

IMPORTANT RULES:
- Do NOT repeat an action you have already taken if you got a useful result.
- As soon as you have enough information to answer, STOP using tools and emit
  FINAL_ANSWER.
- If you are uncertain near the end of your turn budget, give your single best
  answer rather than continuing to search.

Available tools:
{tool_block}
"""


class _LLMProtocol(Protocol):
    async def complete(self, messages: List[dict], **kwargs) -> str: ...


@AgentRegistry.register("react-native")
class NativeReact(ToolUsingAgent):
    """ReAct agent using structured-text parsing instead of function calling."""

    tool_mode = ToolCallMode.STRUCTURED_TEXT

    def __init__(
        self,
        model: str,
        llm: _LLMProtocol,
        tools: List[BaseTool],
        event_recorder: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(event_recorder=event_recorder)
        self.model = model
        self._llm = llm
        self.tools = tools
        self._task: Optional[str] = None
        self._system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_block=self._format_tools()
        )
        # Per-query transcript of raw assistant outputs, aligned with context.history.
        # Index i corresponds to the assistant's raw text for history turn i.
        self._raw_turns: List[str] = []

    def set_task(self, task: str) -> None:
        self._task = task
        # Reset per-query state so reused agents don't leak prior transcripts.
        self._raw_turns = []

    def _format_tools(self) -> str:
        lines: List[str] = []
        for tool in self.tools:
            s = tool.spec
            params_str = json.dumps(s.parameters) if s.parameters else "{}"
            lines.append(f"- {s.name}: {s.description}\n  parameters: {params_str}")
        return "\n".join(lines)

    def _build_messages(self, context: ExecutorContext) -> List[dict]:
        messages: List[dict] = [{"role": "system", "content": self._system_prompt}]
        if self._task is not None:
            messages.append({"role": "user", "content": self._task})

        for i, turn in enumerate(context.history):
            # Replay the assistant's prior reasoning so the model sees its own chain.
            if i < len(self._raw_turns):
                messages.append({"role": "assistant", "content": self._raw_turns[i]})
            # Then replay tool observations for that turn.
            for tc in turn.tool_records:
                tool_msg = (
                    f"OBSERVATION ({tc.call.name}): "
                    f"{tc.result.content if tc.result else f'ERROR: {tc.error}'}"
                )
                messages.append({"role": "user", "content": tool_msg})

        # Append a per-turn turn-budget hint so the model knows when to conclude.
        if context.max_turns > 0:
            remaining = max(context.max_turns - context.turn_index, 0)
            if remaining <= 1:
                hint = (
                    "This is your FINAL turn. You MUST emit `FINAL_ANSWER:` now with "
                    "your best answer. Do NOT call any tool."
                )
            else:
                hint = (
                    f"You are on turn {context.turn_index + 1} of {context.max_turns}. "
                    "If you already have enough information, emit FINAL_ANSWER now."
                )
            messages.append({"role": "user", "content": hint})

        return messages

    async def step(self, context: ExecutorContext) -> TurnOutput:
        messages = self._build_messages(context)
        raw = await self._llm.complete(messages, model=self.model)
        parsed = parse_structured_text(raw)

        if parsed.is_final:
            self._raw_turns.append(raw)
            return TurnOutput(final_answer=parsed.final_answer, tool_calls=[])

        # Detect if this is the final turn and force a non-empty answer.
        is_last_turn = (
            context.max_turns > 0
            and context.turn_index >= context.max_turns - 1
        )
        if is_last_turn:
            # Best-effort answer from whatever the model produced.
            best_effort = (
                parsed.final_answer
                or raw.strip()
                or "No answer."
            )
            self._raw_turns.append(raw)
            return TurnOutput(final_answer=best_effort, tool_calls=[])

        self._raw_turns.append(raw)
        tool_calls = [
            ToolCall(name=tc["name"], input=tc["input"])
            for tc in parsed.tool_calls
        ]
        return TurnOutput(final_answer=None, tool_calls=tool_calls)


__all__ = ["NativeReact", "REACT_SYSTEM_PROMPT"]

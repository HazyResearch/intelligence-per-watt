"""Executor — runs an agent's turn loop with retry, parallel tool dispatch, and bus telemetry.

Stateless per query (no SQLite, no message queue); persistence can be added later
if needed. The Executor is the canonical run path for native agents such as
``react-native``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ipw.execution.errors import FatalError, RetryableError, RetryExhaustedError, classify_error
from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType
from ipw.tools.base import BaseTool, ToolResult

_log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    input: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TurnOutput:
    final_answer: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


@dataclass
class ToolCallRecord:
    call: ToolCall
    result: Optional[ToolResult]
    error: Optional[str]


@dataclass
class TurnRecord:
    turn_index: int
    output: TurnOutput
    tool_records: List[ToolCallRecord] = field(default_factory=list)
    n_retries: int = 0


@dataclass
class ExecutorContext:
    task_id: str
    history: List[TurnRecord] = field(default_factory=list)
    tool_results: List[ToolCallRecord] = field(default_factory=list)
    # Updated by Executor before each step() call so agents can read turn budget.
    turn_index: int = 0
    max_turns: int = 0


@dataclass
class ExecutorResult:
    status: str  # "success" | "failed" | "max_turns_exhausted"
    final_answer: Optional[str]
    n_turns: int
    n_tool_calls: int
    n_retries: int
    error: Optional[str] = None
    history: List[TurnRecord] = field(default_factory=list)


class _AgentProtocol(Protocol):
    tools: List[BaseTool]

    async def step(self, context: ExecutorContext) -> TurnOutput: ...


class Executor:
    """Stateless per-query agent runner with retry + parallel tool dispatch."""

    def __init__(
        self,
        bus: EventBus,
        *,
        max_attempts_per_turn: int = 3,
        base_backoff_s: float = 1.0,
        step_timeout_s: float = 300.0,
        tool_timeout_s: float = 120.0,
    ) -> None:
        self._bus = bus
        self._max_attempts = max_attempts_per_turn
        self._base_backoff_s = base_backoff_s
        self._step_timeout_s = step_timeout_s
        self._tool_timeout_s = tool_timeout_s

    async def execute(
        self,
        agent: _AgentProtocol,
        *,
        task_id: str,
        max_turns: int = 10,
        agent_name: Optional[str] = None,
    ) -> ExecutorResult:
        context = ExecutorContext(task_id=task_id)
        a_name = agent_name or getattr(agent, "name", agent.__class__.__name__)

        self._publish(EventType.AGENT_START, payload={
            "agent_name": a_name, "task_id": task_id,
        })

        total_retries = 0
        total_tool_calls = 0

        for turn_idx in range(max_turns):
            self._publish(EventType.TURN_START, payload={"turn_idx": turn_idx, "task_id": task_id})
            context.turn_index = turn_idx
            context.max_turns = max_turns
            try:
                turn_output, attempts_used = await self._step_with_retry(agent, context, turn_idx)
            except FatalError as e:
                self._publish(EventType.ERROR_CLASSIFIED, payload={"fatal": True, "error": str(e)})
                self._publish(EventType.AGENT_END, payload={
                    "agent_name": a_name, "status": "failed", "n_turns": turn_idx + 1,
                })
                return ExecutorResult(
                    status="failed", final_answer=None, n_turns=turn_idx + 1,
                    n_tool_calls=total_tool_calls, n_retries=total_retries,
                    error=str(e), history=context.history,
                )

            total_retries += attempts_used - 1
            tool_records: List[ToolCallRecord] = []
            if turn_output.tool_calls:
                tool_records = await self._dispatch_tools(agent.tools, turn_output.tool_calls)
                total_tool_calls += len(tool_records)

            turn_record = TurnRecord(
                turn_index=turn_idx, output=turn_output,
                tool_records=tool_records, n_retries=attempts_used - 1,
            )
            context.history.append(turn_record)
            context.tool_results.extend(tool_records)

            self._publish(EventType.TURN_END, payload={
                "turn_idx": turn_idx, "is_final": turn_output.is_final,
                "n_tool_calls": len(tool_records),
            })

            if turn_output.is_final:
                self._publish(EventType.AGENT_END, payload={
                    "agent_name": a_name, "status": "success", "n_turns": turn_idx + 1,
                })
                return ExecutorResult(
                    status="success", final_answer=turn_output.final_answer,
                    n_turns=turn_idx + 1, n_tool_calls=total_tool_calls,
                    n_retries=total_retries, history=context.history,
                )

        self._publish(EventType.AGENT_END, payload={
            "agent_name": a_name, "status": "max_turns_exhausted", "n_turns": max_turns,
        })
        return ExecutorResult(
            status="max_turns_exhausted", final_answer=None, n_turns=max_turns,
            n_tool_calls=total_tool_calls, n_retries=total_retries,
            history=context.history,
        )

    async def _step_with_retry(
        self, agent: _AgentProtocol, context: ExecutorContext, turn_idx: int,
    ) -> tuple[TurnOutput, int]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                step_coro = agent.step(context)
                if self._step_timeout_s is not None:
                    turn_output = await asyncio.wait_for(step_coro, timeout=self._step_timeout_s)
                else:
                    turn_output = await step_coro
                return turn_output, attempt
            except Exception as exc:
                classified = classify_error(exc)
                self._publish(EventType.RETRY_ATTEMPT, payload={
                    "turn_idx": turn_idx, "attempt": attempt,
                    "error_class": "retryable" if isinstance(classified, RetryableError) else "fatal",
                    "error_type": exc.__class__.__name__,
                })
                if isinstance(classified, RetryableError) and attempt < self._max_attempts:
                    await asyncio.sleep(self._base_backoff_s * (2 ** (attempt - 1)))
                    continue
                if isinstance(classified, RetryableError):
                    # Out of retries — distinct from "fatal from attempt 1"
                    raise RetryExhaustedError(str(exc)) from exc
                raise classified from exc
        # Unreachable: loop body always exits via return or raise.
        raise FatalError(f"step exhausted retries on turn {turn_idx}")  # pragma: no cover

    async def _dispatch_tools(
        self, tools: List[BaseTool], calls: List[ToolCall],
    ) -> List[ToolCallRecord]:
        tool_by_name = {t.spec.name: t for t in tools}
        has_conflict = any(
            tool_by_name.get(c.name) and tool_by_name[c.name].spec.side_effect_conflict
            for c in calls
        )

        records: List[ToolCallRecord] = []
        if has_conflict or len(calls) == 1:
            for call in calls:
                records.append(await self._dispatch_one(tool_by_name, call))
        else:
            tasks = [self._dispatch_one(tool_by_name, c) for c in calls]
            records = list(await asyncio.gather(*tasks))
        return records

    async def _dispatch_one(
        self, tool_by_name: Dict[str, BaseTool], call: ToolCall,
    ) -> ToolCallRecord:
        tool = tool_by_name.get(call.name)
        if tool is None:
            return ToolCallRecord(call=call, result=None, error=f"tool {call.name!r} not found")
        try:
            tool_coro = tool(correlation_id=call.correlation_id, **call.input)
            if self._tool_timeout_s is not None:
                result = await asyncio.wait_for(tool_coro, timeout=self._tool_timeout_s)
            else:
                result = await tool_coro
            return ToolCallRecord(call=call, result=result, error=None)
        except asyncio.TimeoutError:
            _log.warning("tool %s timed out after %ss", call.name, self._tool_timeout_s)
            return ToolCallRecord(
                call=call, result=None,
                error=f"tool timeout after {self._tool_timeout_s}s",
            )
        except Exception as exc:
            _log.exception("tool %s raised", call.name)
            return ToolCallRecord(call=call, result=None, error=str(exc))

    def _publish(self, event_type: EventType, *, payload: Dict[str, Any]) -> None:
        self._bus.publish(Event(
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            payload=payload,
        ))


__all__ = [
    "Executor", "ExecutorContext", "ExecutorResult",
    "ToolCall", "TurnOutput", "ToolCallRecord", "TurnRecord",
]

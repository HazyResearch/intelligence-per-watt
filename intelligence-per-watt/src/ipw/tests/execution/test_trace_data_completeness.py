"""Tests for per-step / per-trace data completeness.

Validates that TurnTrace and QueryTrace objects contain all required fields,
and that export round-trips preserve the data.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ipw.core.types import AgentRunResult, DatasetRecord
from ipw.execution.agentic_runner import AgenticRunner
from ipw.execution.exporters import export_hf_dataset, export_jsonl
from ipw.execution.trace import QueryTrace, TurnTrace
from ipw.telemetry.events import EventRecorder, EventType


# All fields that must be present on TurnTrace
REQUIRED_TURN_FIELDS = [
    "turn_index",
    "input_tokens",
    "output_tokens",
    "tools_called",
    "tool_latencies_s",
    "wall_clock_s",
    "gpu_energy_joules",
    "cpu_energy_joules",
    "gpu_power_avg_watts",
    "cpu_power_avg_watts",
    "cost_usd",
]


def _make_complete_turn(index: int = 0) -> TurnTrace:
    """Create a TurnTrace with ALL required fields populated."""
    return TurnTrace(
        turn_index=index,
        input_tokens=120,
        output_tokens=60,
        tools_called=["web_search", "calculator"],
        tool_latencies_s={"web_search": 0.5, "calculator": 0.1},
        wall_clock_s=2.5,
        gpu_energy_joules=12.5,
        cpu_energy_joules=3.2,
        gpu_power_avg_watts=155.0,
        cpu_power_avg_watts=48.0,
        cost_usd=0.0015,
    )


def _make_complete_trace() -> QueryTrace:
    """Create a QueryTrace with multi-turn data and all fields populated."""
    return QueryTrace(
        query_id="q0000",
        workload_type="agentic",
        query_text="What is the capital of France?",
        response_text="Paris",
        turns=[_make_complete_turn(0), _make_complete_turn(1)],
        total_wall_clock_s=5.0,
        completed=True,
    )


class TestTurnTraceCompleteness:
    """Verify TurnTrace objects have all required fields."""

    def test_all_required_fields_present(self) -> None:
        turn = _make_complete_turn()
        turn_dict = turn.to_dict()
        for field in REQUIRED_TURN_FIELDS:
            assert field in turn_dict, f"Missing field: {field}"
            assert turn_dict[field] is not None, f"Field is None: {field}"

    def test_round_trip_preserves_all_fields(self) -> None:
        original = _make_complete_turn()
        restored = TurnTrace.from_dict(original.to_dict())

        assert restored.turn_index == original.turn_index
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.tools_called == original.tools_called
        assert restored.tool_latencies_s == original.tool_latencies_s
        assert restored.wall_clock_s == original.wall_clock_s
        assert restored.gpu_energy_joules == original.gpu_energy_joules
        assert restored.cpu_energy_joules == original.cpu_energy_joules
        assert restored.gpu_power_avg_watts == original.gpu_power_avg_watts
        assert restored.cpu_power_avg_watts == original.cpu_power_avg_watts
        assert restored.cost_usd == original.cost_usd


class TestQueryTraceAggregation:
    """Verify QueryTrace correctly aggregates per-turn data."""

    def test_total_input_tokens(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_input_tokens == 240  # 120 * 2

    def test_total_output_tokens(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_output_tokens == 120  # 60 * 2

    def test_total_tool_calls(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_tool_calls == 4  # 2 tools * 2 turns

    def test_total_gpu_energy(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_gpu_energy_joules == 25.0  # 12.5 * 2

    def test_total_cost(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_cost_usd == 0.003  # 0.0015 * 2

    def test_num_turns(self) -> None:
        trace = _make_complete_trace()
        assert trace.num_turns == 2

    def test_total_wall_clock(self) -> None:
        trace = _make_complete_trace()
        assert trace.total_wall_clock_s == 5.0


class TestJSONLPreservesPerTurnData:
    """Verify JSONL export preserves all per-turn data."""

    def test_jsonl_round_trip(self, tmp_path: Path) -> None:
        traces = [_make_complete_trace()]
        path = export_jsonl(traces, tmp_path / "traces.jsonl")

        loaded = QueryTrace.load_jsonl(path)
        assert len(loaded) == 1

        turn = loaded[0].turns[0]
        assert turn.input_tokens == 120
        assert turn.output_tokens == 60
        assert turn.tools_called == ["web_search", "calculator"]
        assert turn.tool_latencies_s["web_search"] == 0.5
        assert turn.wall_clock_s == 2.5
        assert turn.gpu_energy_joules == 12.5
        assert turn.cpu_energy_joules == 3.2
        assert turn.gpu_power_avg_watts == 155.0
        assert turn.cpu_power_avg_watts == 48.0
        assert turn.cost_usd == 0.0015


class TestHFDatasetPreservesTraceData:
    """Verify HF dataset export includes trace_json with full turn details."""

    def test_hf_export_round_trip(self, tmp_path: Path) -> None:
        from datasets import load_from_disk

        traces = [_make_complete_trace()]
        path = export_hf_dataset(traces, tmp_path / "hf_dataset")

        ds = load_from_disk(str(path))
        assert len(ds) == 1

        row = ds[0]
        assert row["total_input_tokens"] == 240
        assert row["total_output_tokens"] == 120
        assert row["total_tool_calls"] == 4
        assert row["total_gpu_energy_joules"] == 25.0
        assert row["total_cost_usd"] == 0.003

        # Verify trace_json contains full turn details
        parsed = json.loads(row["trace_json"])
        assert len(parsed["turns"]) == 2
        turn0 = parsed["turns"][0]
        assert turn0["input_tokens"] == 120
        assert turn0["gpu_energy_joules"] == 12.5
        assert turn0["cost_usd"] == 0.0015


class TestMultiTurnRunProducesCompleteTraces:
    """Verify a multi-turn agentic run produces TurnTrace with all fields."""

    def test_event_recorder_produces_complete_turns(self) -> None:
        """Simulate an agent run with event recorder and verify turn traces."""
        recorder = EventRecorder()
        agent = MagicMock()

        def agent_run(prompt: str, **kwargs) -> AgentRunResult:
            # Turn 1: inference + tool call
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(EventType.TOOL_CALL_START, tool="search")
            recorder.record(EventType.TOOL_CALL_END, tool="search")
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=100,
                completion_tokens=50,
            )
            # Turn 2: inference only
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=80,
                completion_tokens=30,
            )
            return AgentRunResult(content="Final answer")

        agent.run.side_effect = agent_run

        dataset = MagicMock()
        records = [DatasetRecord(problem="Q1", answer="A1", subject="test")]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 1

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test"},
            event_recorder=recorder,
        )
        runner._event_recorder = recorder

        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        assert trace.num_turns == 2

        # Turn 1: should have tool call
        t0 = trace.turns[0]
        assert t0.input_tokens == 100
        assert t0.output_tokens == 50
        assert t0.tools_called == ["search"]
        assert "search" in t0.tool_latencies_s
        assert t0.wall_clock_s > 0

        # Turn 2: inference only, no tools
        t1 = trace.turns[1]
        assert t1.input_tokens == 80
        assert t1.output_tokens == 30
        assert t1.tools_called == []
        assert t1.wall_clock_s > 0

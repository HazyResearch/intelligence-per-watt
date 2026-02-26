"""Tests for execution/trace.py — TurnTrace and QueryTrace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipw.execution.trace import QueryTrace, TurnTrace


class TestTurnTrace:
    """Test TurnTrace construction and serialization."""

    def test_creation_with_all_fields(self) -> None:
        turn = TurnTrace(
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            tool_result_tokens=20,
            tools_called=["calc", "search"],
            tool_latencies_s={"calc": 0.1, "search": 0.5},
            wall_clock_s=1.5,
            error=None,
            gpu_energy_joules=10.0,
            cpu_energy_joules=3.0,
            gpu_power_avg_watts=150.0,
            cpu_power_avg_watts=45.0,
            cost_usd=0.005,
        )
        assert turn.turn_index == 0
        assert turn.input_tokens == 100
        assert turn.tools_called == ["calc", "search"]
        assert turn.gpu_energy_joules == 10.0

    def test_to_dict_roundtrip(self) -> None:
        turn = TurnTrace(
            turn_index=2,
            input_tokens=50,
            output_tokens=25,
            tools_called=["think"],
            wall_clock_s=0.8,
            gpu_energy_joules=5.0,
            cost_usd=0.001,
        )
        d = turn.to_dict()
        restored = TurnTrace.from_dict(d)
        assert restored.turn_index == turn.turn_index
        assert restored.input_tokens == turn.input_tokens
        assert restored.output_tokens == turn.output_tokens
        assert restored.tools_called == turn.tools_called
        assert restored.wall_clock_s == turn.wall_clock_s
        assert restored.gpu_energy_joules == turn.gpu_energy_joules
        assert restored.cost_usd == turn.cost_usd

    def test_from_dict_defaults(self) -> None:
        d = {"turn_index": 0}
        turn = TurnTrace.from_dict(d)
        assert turn.input_tokens == 0
        assert turn.output_tokens == 0
        assert turn.tools_called == []
        assert turn.error is None


class TestQueryTrace:
    """Test QueryTrace construction, properties, and serialization."""

    def _make_trace(self) -> QueryTrace:
        turns = [
            TurnTrace(
                turn_index=0,
                input_tokens=100,
                output_tokens=50,
                tools_called=["calc"],
                wall_clock_s=1.0,
                gpu_energy_joules=5.0,
                cost_usd=0.002,
            ),
            TurnTrace(
                turn_index=1,
                input_tokens=80,
                output_tokens=30,
                tools_called=["search", "think"],
                wall_clock_s=2.0,
                gpu_energy_joules=8.0,
                cost_usd=0.003,
            ),
        ]
        return QueryTrace(
            query_id="q0001",
            workload_type="agentic",
            query_text="What is 2+2?",
            response_text="4",
            turns=turns,
            total_wall_clock_s=3.0,
            completed=True,
        )

    def test_num_turns(self) -> None:
        trace = self._make_trace()
        assert trace.num_turns == 2

    def test_total_input_tokens(self) -> None:
        trace = self._make_trace()
        assert trace.total_input_tokens == 180

    def test_total_output_tokens(self) -> None:
        trace = self._make_trace()
        assert trace.total_output_tokens == 80

    def test_total_tool_calls(self) -> None:
        trace = self._make_trace()
        assert trace.total_tool_calls == 3

    def test_total_gpu_energy_joules(self) -> None:
        trace = self._make_trace()
        assert trace.total_gpu_energy_joules == 13.0

    def test_total_gpu_energy_joules_none_when_no_values(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0)],
        )
        assert trace.total_gpu_energy_joules is None

    def test_total_cost_usd(self) -> None:
        trace = self._make_trace()
        assert trace.total_cost_usd == pytest.approx(0.005)

    def test_total_cost_usd_none_when_no_values(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0)],
        )
        assert trace.total_cost_usd is None

    def test_query_level_gpu_energy_fallback(self) -> None:
        """total_gpu_energy_joules falls back to query_gpu_energy_joules when no turns have energy."""
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0)],
            query_gpu_energy_joules=42.0,
        )
        assert trace.total_gpu_energy_joules == 42.0

    def test_query_level_gpu_energy_not_used_when_turns_have_energy(self) -> None:
        """Per-turn energy takes precedence over query-level energy."""
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=10.0)],
            query_gpu_energy_joules=42.0,
        )
        assert trace.total_gpu_energy_joules == 10.0

    def test_total_cpu_energy_joules_from_turns(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[
                TurnTrace(turn_index=0, cpu_energy_joules=3.0),
                TurnTrace(turn_index=1, cpu_energy_joules=5.0),
            ],
        )
        assert trace.total_cpu_energy_joules == 8.0

    def test_total_cpu_energy_joules_fallback(self) -> None:
        """total_cpu_energy_joules falls back to query_cpu_energy_joules."""
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0)],
            query_cpu_energy_joules=15.0,
        )
        assert trace.total_cpu_energy_joules == 15.0

    def test_total_cpu_energy_joules_none_when_empty(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0)],
        )
        assert trace.total_cpu_energy_joules is None

    def test_query_energy_fields_in_to_dict(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            query_gpu_energy_joules=10.0,
            query_cpu_energy_joules=5.0,
            query_gpu_power_avg_watts=200.0,
            query_cpu_power_avg_watts=80.0,
        )
        d = trace.to_dict()
        assert d["query_gpu_energy_joules"] == 10.0
        assert d["query_cpu_energy_joules"] == 5.0
        assert d["query_gpu_power_avg_watts"] == 200.0
        assert d["query_cpu_power_avg_watts"] == 80.0

    def test_query_energy_fields_roundtrip(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            query_gpu_energy_joules=10.0,
            query_cpu_energy_joules=5.0,
            query_gpu_power_avg_watts=200.0,
            query_cpu_power_avg_watts=80.0,
        )
        restored = QueryTrace.from_dict(trace.to_dict())
        assert restored.query_gpu_energy_joules == 10.0
        assert restored.query_cpu_energy_joules == 5.0
        assert restored.query_gpu_power_avg_watts == 200.0
        assert restored.query_cpu_power_avg_watts == 80.0

    def test_to_dict_from_dict_roundtrip(self) -> None:
        trace = self._make_trace()
        d = trace.to_dict()
        restored = QueryTrace.from_dict(d)
        assert restored.query_id == trace.query_id
        assert restored.workload_type == trace.workload_type
        assert restored.num_turns == trace.num_turns
        assert restored.total_input_tokens == trace.total_input_tokens
        assert restored.completed == trace.completed

    def test_jsonl_save_load_roundtrip(self, tmp_path: Path) -> None:
        trace = self._make_trace()
        jsonl_path = tmp_path / "traces.jsonl"

        trace.save_jsonl(jsonl_path)
        loaded = QueryTrace.load_jsonl(jsonl_path)

        assert len(loaded) == 1
        assert loaded[0].query_id == "q0001"
        assert loaded[0].num_turns == 2
        assert loaded[0].total_input_tokens == 180

    def test_jsonl_multiple_traces(self, tmp_path: Path) -> None:
        trace1 = self._make_trace()
        trace2 = QueryTrace(
            query_id="q0002",
            workload_type="agentic",
            query_text="Another question",
            response_text="Another answer",
            turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5)],
            total_wall_clock_s=0.5,
            completed=True,
        )

        jsonl_path = tmp_path / "traces.jsonl"
        trace1.save_jsonl(jsonl_path)
        trace2.save_jsonl(jsonl_path)

        loaded = QueryTrace.load_jsonl(jsonl_path)
        assert len(loaded) == 2
        assert loaded[0].query_id == "q0001"
        assert loaded[1].query_id == "q0002"

    def test_is_resolved_in_to_dict(self) -> None:
        trace = QueryTrace(
            query_id="q0",
            workload_type="test",
            is_resolved=True,
        )
        d = trace.to_dict()
        assert d["is_resolved"] is True

    def test_is_resolved_roundtrip(self) -> None:
        for val in (True, False, None):
            trace = QueryTrace(
                query_id="q0",
                workload_type="test",
                is_resolved=val,
            )
            restored = QueryTrace.from_dict(trace.to_dict())
            assert restored.is_resolved is val

    def test_is_resolved_none_by_default(self) -> None:
        trace = QueryTrace(query_id="q0", workload_type="test")
        assert trace.is_resolved is None

    def test_to_hf_dataset(self) -> None:
        try:
            from datasets import Dataset
        except ImportError:
            pytest.skip("datasets package not available")

        trace = self._make_trace()
        ds = QueryTrace.to_hf_dataset([trace])

        assert len(ds) == 1
        row = ds[0]
        assert row["query_id"] == "q0001"
        assert row["num_turns"] == 2
        assert row["total_input_tokens"] == 180
        assert row["total_output_tokens"] == 80
        assert row["total_tool_calls"] == 3
        assert row["completed"] is True
        # trace_json should be valid JSON
        parsed = json.loads(row["trace_json"])
        assert parsed["query_id"] == "q0001"

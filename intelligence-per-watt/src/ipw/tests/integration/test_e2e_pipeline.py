"""Tier 3 end-to-end integration tests for the full profiling pipeline.

These tests validate the full pipeline from CLI invocation through export,
using mocked agent and dataset to avoid needing a live inference server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ipw.core.types import AgentRunResult, DatasetRecord
from ipw.execution.agentic_runner import AgenticRunner
from ipw.execution.exporters import export_hf_dataset, export_jsonl, export_summary_json
from ipw.execution.trace import QueryTrace, TurnTrace
from ipw.telemetry.events import EventRecorder, EventType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
]


def _make_traces_with_turn_data() -> list[QueryTrace]:
    """Create realistic traces with per-turn data for testing exports."""
    traces = []
    for i in range(3):
        turns = [
            TurnTrace(
                turn_index=0,
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                tools_called=["web_search"],
                tool_latencies_s={"web_search": 0.5},
                wall_clock_s=2.0,
                gpu_energy_joules=10.0 + i,
                cpu_energy_joules=2.0 + i * 0.5,
                gpu_power_avg_watts=150.0,
                cpu_power_avg_watts=45.0,
                cost_usd=0.001,
            ),
            TurnTrace(
                turn_index=1,
                input_tokens=80 + i * 5,
                output_tokens=40 + i * 3,
                tools_called=["calculator"],
                tool_latencies_s={"calculator": 0.1},
                wall_clock_s=1.5,
                gpu_energy_joules=8.0,
                cpu_energy_joules=1.5,
                gpu_power_avg_watts=140.0,
                cpu_power_avg_watts=42.0,
                cost_usd=0.0008,
            ),
        ]
        traces.append(
            QueryTrace(
                query_id=f"q{i:04d}",
                workload_type="agentic",
                query_text=f"Question {i}",
                response_text=f"Answer {i}",
                turns=turns,
                total_wall_clock_s=3.5,
                completed=True,
            )
        )
    return traces


class TestE2EPipeline:
    """End-to-end pipeline tests with mocked agent/dataset."""

    def test_agentic_run_produces_traces(self, tmp_path: Path) -> None:
        """Test AgenticRunner produces traces from mocked agent."""
        from unittest.mock import MagicMock

        recorder = EventRecorder()

        agent = MagicMock()
        call_count = 0

        def agent_run(prompt: str, **kwargs) -> AgentRunResult:
            nonlocal call_count
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(EventType.TOOL_CALL_START, tool="search")
            recorder.record(EventType.TOOL_CALL_END, tool="search")
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=100,
                completion_tokens=50,
            )
            call_count += 1
            return AgentRunResult(content=f"Answer {call_count}")

        agent.run.side_effect = agent_run

        dataset = MagicMock()
        records = [
            DatasetRecord(problem=f"Q{i}", answer=f"A{i}", subject="test")
            for i in range(3)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 3

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            config={"model": "test-model"},
            event_recorder=recorder,
        )

        traces = asyncio.run(runner.run(max_queries=3))

        assert len(traces) == 3
        assert all(t.completed for t in traces)
        assert all(t.num_turns >= 1 for t in traces)

    def test_jsonl_export_produces_valid_traces(self, tmp_path: Path) -> None:
        """Test JSONL export produces valid, reloadable traces."""
        traces = _make_traces_with_turn_data()
        jsonl_path = export_jsonl(traces, tmp_path / "traces.jsonl")

        assert jsonl_path.exists()

        # Reload and verify
        loaded = QueryTrace.load_jsonl(jsonl_path)
        assert len(loaded) == 3
        for original, reloaded in zip(traces, loaded):
            assert reloaded.query_id == original.query_id
            assert reloaded.num_turns == original.num_turns
            assert reloaded.total_input_tokens == original.total_input_tokens

    def test_jsonl_preserves_per_turn_data(self, tmp_path: Path) -> None:
        """Test JSONL round-trip preserves all per-turn fields."""
        traces = _make_traces_with_turn_data()
        jsonl_path = export_jsonl(traces, tmp_path / "traces.jsonl")

        loaded = QueryTrace.load_jsonl(jsonl_path)
        turn = loaded[0].turns[0]
        assert turn.input_tokens == traces[0].turns[0].input_tokens
        assert turn.output_tokens == traces[0].turns[0].output_tokens
        assert turn.tools_called == ["web_search"]
        assert "web_search" in turn.tool_latencies_s
        assert turn.gpu_energy_joules is not None
        assert turn.cpu_energy_joules is not None
        assert turn.gpu_power_avg_watts is not None
        assert turn.cpu_power_avg_watts is not None
        assert turn.cost_usd is not None

    def test_hf_dataset_export_produces_loadable_dataset(self, tmp_path: Path) -> None:
        """Test HF dataset export produces loadable Arrow dataset."""
        from datasets import load_from_disk

        traces = _make_traces_with_turn_data()
        hf_path = export_hf_dataset(traces, tmp_path / "hf_dataset")

        assert hf_path.exists()

        ds = load_from_disk(str(hf_path))
        assert len(ds) == 3
        assert "query_id" in ds.column_names
        assert "total_input_tokens" in ds.column_names
        assert "trace_json" in ds.column_names

    def test_hf_dataset_includes_trace_json(self, tmp_path: Path) -> None:
        """Test HF dataset export includes trace_json with full turn details."""
        from datasets import load_from_disk

        traces = _make_traces_with_turn_data()
        hf_path = export_hf_dataset(traces, tmp_path / "hf_dataset")

        ds = load_from_disk(str(hf_path))
        trace_json_str = ds[0]["trace_json"]
        parsed = json.loads(trace_json_str)
        assert "turns" in parsed
        assert len(parsed["turns"]) == 2
        assert parsed["turns"][0]["input_tokens"] > 0

    def test_summary_json_contains_required_fields(self, tmp_path: Path) -> None:
        """Test summary.json contains all required aggregate fields."""
        traces = _make_traces_with_turn_data()
        config = {"agent": "react", "model": "test", "dataset": "gaia"}

        summary_path = export_summary_json(traces, config, tmp_path / "summary.json")
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert "config" in summary
        assert "totals" in summary
        assert "averages" in summary

        totals = summary["totals"]
        assert totals["queries"] == 3
        assert totals["completed"] == 3
        assert totals["turns"] == 6
        assert totals["input_tokens"] > 0
        assert totals["output_tokens"] > 0
        assert totals["wall_clock_s"] > 0
        assert totals["gpu_energy_joules"] is not None
        assert totals["cost_usd"] is not None

        averages = summary["averages"]
        assert averages["turns_per_query"] == 2.0
        assert averages["wall_clock_per_query_s"] > 0

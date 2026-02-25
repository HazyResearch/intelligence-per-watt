"""Tests for execution/exporters.py — export functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipw.execution.exporters import export_jsonl, export_summary_json
from ipw.execution.trace import QueryTrace, TurnTrace


def _make_traces() -> list[QueryTrace]:
    return [
        QueryTrace(
            query_id="q0001",
            workload_type="agentic",
            query_text="What is 2+2?",
            response_text="4",
            turns=[
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
                    tools_called=["search"],
                    wall_clock_s=2.0,
                    gpu_energy_joules=8.0,
                    cost_usd=0.003,
                ),
            ],
            total_wall_clock_s=3.0,
            completed=True,
        ),
        QueryTrace(
            query_id="q0002",
            workload_type="agentic",
            query_text="Capital of France?",
            response_text="Paris",
            turns=[
                TurnTrace(
                    turn_index=0,
                    input_tokens=20,
                    output_tokens=5,
                    wall_clock_s=0.5,
                ),
            ],
            total_wall_clock_s=0.5,
            completed=True,
        ),
    ]


class TestExportJsonl:
    """Test JSONL export."""

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        traces = _make_traces()
        path = tmp_path / "traces.jsonl"
        result_path = export_jsonl(traces, path)
        assert result_path == path
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert "query_id" in parsed
            assert "turns" in parsed

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        traces = _make_traces()
        path = tmp_path / "traces.jsonl"
        export_jsonl(traces, path)

        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_roundtrip_export_load(self, tmp_path: Path) -> None:
        traces = _make_traces()
        path = tmp_path / "traces.jsonl"
        export_jsonl(traces, path)

        loaded = QueryTrace.load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].query_id == "q0001"
        assert loaded[0].num_turns == 2
        assert loaded[1].query_id == "q0002"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        traces = _make_traces()
        path = tmp_path / "nested" / "dir" / "traces.jsonl"
        export_jsonl(traces, path)
        assert path.exists()

    def test_empty_traces(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        export_jsonl([], path)
        assert path.exists()
        assert path.read_text() == ""


class TestExportSummaryJson:
    """Test summary JSON export."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {"agent": "react", "model": "test"}
        path = tmp_path / "summary.json"
        result_path = export_summary_json(traces, config, path)
        assert result_path == path

        summary = json.loads(path.read_text())
        assert "config" in summary
        assert "totals" in summary
        assert "averages" in summary

    def test_schema_fields(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {"agent": "react"}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        totals = summary["totals"]
        assert totals["queries"] == 2
        assert totals["completed"] == 2
        assert totals["turns"] == 3
        assert totals["tool_calls"] == 2
        assert totals["input_tokens"] == 200
        assert totals["output_tokens"] == 85
        assert totals["wall_clock_s"] == pytest.approx(3.5)

    def test_config_is_preserved(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {"agent": "react", "model": "test-model", "dataset": "gaia"}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        assert summary["config"]["agent"] == "react"
        assert summary["config"]["model"] == "test-model"

    def test_averages_computed(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        averages = summary["averages"]
        assert averages["turns_per_query"] == 1.5
        assert averages["wall_clock_per_query_s"] == pytest.approx(1.75)

    def test_empty_traces(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        export_summary_json([], {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["queries"] == 0
        assert summary["averages"]["turns_per_query"] == 0

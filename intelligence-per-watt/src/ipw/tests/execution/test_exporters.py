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

    def test_statistics_section_exists(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {"agent": "react"}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        assert "statistics" in summary
        stats = summary["statistics"]
        # Check expected metric keys
        for key in [
            "wall_clock_s", "gpu_energy_joules", "cpu_energy_joules",
            "gpu_power_watts", "cpu_power_watts",
            "input_tokens", "output_tokens", "total_tokens",
            "throughput_tokens_per_sec", "energy_per_token_joules",
            "cost_usd", "turns", "tool_calls",
        ]:
            assert key in stats, f"Missing statistics key: {key}"
            assert "avg" in stats[key]
            assert "median" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]
            assert "std" in stats[key]

    def test_statistics_wall_clock_values(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        wc = summary["statistics"]["wall_clock_s"]
        assert wc["avg"] == pytest.approx(1.75)
        assert wc["min"] == pytest.approx(0.5)
        assert wc["max"] == pytest.approx(3.0)

    def test_total_tokens_in_totals(self, tmp_path: Path) -> None:
        traces = _make_traces()
        config = {}
        path = tmp_path / "summary.json"
        export_summary_json(traces, config, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["total_tokens"] == 285  # 200 input + 85 output

    def test_empty_traces_has_statistics(self, tmp_path: Path) -> None:
        """Statistics section is present even with empty traces, with all-None values."""
        path = tmp_path / "summary.json"
        export_summary_json([], {}, path)

        summary = json.loads(path.read_text())
        assert "statistics" in summary
        for key in summary["statistics"]:
            assert summary["statistics"][key]["avg"] is None

    def test_empty_traces(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        export_summary_json([], {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["queries"] == 0
        assert summary["totals"]["total_tokens"] == 0
        assert summary["averages"]["turns_per_query"] == 0


# ---------------------------------------------------------------------------
# Helper for accuracy / efficiency tests
# ---------------------------------------------------------------------------

def _make_traces_with_accuracy() -> list[QueryTrace]:
    """Return 2 traces: one resolved (is_resolved=True), one not (False).

    Both have GPU energy and power data via turn-level fields and
    query-level MBU so the efficiency metrics can be computed.
    """
    return [
        QueryTrace(
            query_id="a001",
            workload_type="agentic",
            query_text="Solve X",
            response_text="42",
            turns=[
                TurnTrace(
                    turn_index=0,
                    input_tokens=100,
                    output_tokens=50,
                    wall_clock_s=2.0,
                    gpu_energy_joules=10.0,
                    gpu_power_avg_watts=200.0,
                    cpu_energy_joules=3.0,
                    cpu_power_avg_watts=50.0,
                ),
            ],
            total_wall_clock_s=2.0,
            completed=True,
            is_resolved=True,
            query_mbu_avg_pct=45.0,
            query_mbu_max_pct=60.0,
        ),
        QueryTrace(
            query_id="a002",
            workload_type="agentic",
            query_text="Solve Y",
            response_text="wrong",
            turns=[
                TurnTrace(
                    turn_index=0,
                    input_tokens=80,
                    output_tokens=40,
                    wall_clock_s=3.0,
                    gpu_energy_joules=15.0,
                    gpu_power_avg_watts=250.0,
                    cpu_energy_joules=5.0,
                    cpu_power_avg_watts=60.0,
                ),
            ],
            total_wall_clock_s=3.0,
            completed=True,
            is_resolved=False,
            query_mbu_avg_pct=55.0,
            query_mbu_max_pct=70.0,
        ),
    ]


class TestEfficiencySection:
    """Tests for the efficiency, accuracy, MBU, and normalized sections."""

    def test_efficiency_section_present(self, tmp_path: Path) -> None:
        """Efficiency section exists with accuracy, ipj, ipw."""
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert "efficiency" in summary
        eff = summary["efficiency"]
        assert "accuracy" in eff
        assert "ipj" in eff
        assert "ipw" in eff
        assert "total_gpu_energy_joules" in eff
        assert "total_cpu_energy_joules" in eff
        assert "avg_gpu_power_watts" in eff
        assert "avg_cpu_power_watts" in eff

        # accuracy = 1 resolved / 2 scored = 0.5
        assert eff["accuracy"] == pytest.approx(0.5)
        # ipj = accuracy / total_gpu_energy = 0.5 / 25.0
        assert eff["ipj"] == pytest.approx(0.5 / 25.0)
        # ipw = accuracy / avg_gpu_power (avg of 200 and 250 = 225)
        assert eff["ipw"] == pytest.approx(0.5 / 225.0)

    def test_totals_has_accuracy(self, tmp_path: Path) -> None:
        """totals.accuracy is set correctly."""
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert "accuracy" in summary["totals"]
        assert summary["totals"]["accuracy"] == pytest.approx(0.5)

    def test_mbu_in_statistics(self, tmp_path: Path) -> None:
        """mbu_avg_pct appears in the statistics section."""
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert "mbu_avg_pct" in summary["statistics"]
        mbu = summary["statistics"]["mbu_avg_pct"]
        assert "avg" in mbu
        assert "median" in mbu
        # avg of 45.0 and 55.0 = 50.0
        assert mbu["avg"] == pytest.approx(50.0)

    def test_normalized_statistics_present(self, tmp_path: Path) -> None:
        """With 40 traces, normalized sections exist with correct outlier metadata."""
        # Create 40 traces with varying wall_clock_s so trimming is meaningful.
        traces = []
        for i in range(40):
            wc = 1.0 + i * 0.1  # 1.0..4.9
            traces.append(
                QueryTrace(
                    query_id=f"n{i:03d}",
                    workload_type="agentic",
                    query_text=f"Q{i}",
                    response_text=f"A{i}",
                    turns=[
                        TurnTrace(
                            turn_index=0,
                            input_tokens=50,
                            output_tokens=20,
                            wall_clock_s=wc,
                            gpu_energy_joules=5.0,
                            gpu_power_avg_watts=200.0,
                            cpu_energy_joules=2.0,
                            cpu_power_avg_watts=40.0,
                        ),
                    ],
                    total_wall_clock_s=wc,
                    completed=True,
                    is_resolved=(i % 2 == 0),  # 50% accuracy
                    query_mbu_avg_pct=30.0,
                ),
            )

        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert "normalized_statistics" in summary
        ns = summary["normalized_statistics"]
        assert ns is not None
        assert "_description" in ns
        assert "_outliers_removed" in ns
        meta = ns["_outliers_removed"]
        assert meta["queries_before"] == 40
        # 5% of 40 = 2 removed from each end -> 36 remain
        assert meta["queries_after"] == 36
        assert meta["top_pct"] == 5
        assert meta["bottom_pct"] == 5

        # Trimmed set excludes the 2 lowest and 2 highest wall_clock_s values.
        # Lowest 2: 1.0, 1.1  Highest 2: 4.8, 4.9
        # Remaining: indices 2..37 -> wall_clock 1.2..4.7
        assert "wall_clock_s" in ns
        assert ns["wall_clock_s"]["min"] == pytest.approx(1.2)
        assert ns["wall_clock_s"]["max"] == pytest.approx(4.7)

    def test_normalized_efficiency_present(self, tmp_path: Path) -> None:
        """normalized_efficiency has accuracy / ipj / ipw fields."""
        traces = []
        for i in range(20):
            wc = 1.0 + i * 0.5
            traces.append(
                QueryTrace(
                    query_id=f"e{i:03d}",
                    workload_type="agentic",
                    query_text=f"Q{i}",
                    response_text=f"A{i}",
                    turns=[
                        TurnTrace(
                            turn_index=0,
                            input_tokens=50,
                            output_tokens=20,
                            wall_clock_s=wc,
                            gpu_energy_joules=10.0,
                            gpu_power_avg_watts=200.0,
                            cpu_energy_joules=3.0,
                            cpu_power_avg_watts=50.0,
                        ),
                    ],
                    total_wall_clock_s=wc,
                    completed=True,
                    is_resolved=True,
                    query_mbu_avg_pct=40.0,
                ),
            )

        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert "normalized_efficiency" in summary
        ne = summary["normalized_efficiency"]
        assert ne is not None
        assert "accuracy" in ne
        assert "ipj" in ne
        assert "ipw" in ne
        assert "_description" in ne
        assert "_outliers_removed" in ne

    def test_efficiency_with_no_resolved(self, tmp_path: Path) -> None:
        """When no is_resolved is set, accuracy/ipj/ipw are None."""
        traces = [
            QueryTrace(
                query_id="u001",
                workload_type="agentic",
                query_text="Q1",
                response_text="A1",
                turns=[
                    TurnTrace(
                        turn_index=0,
                        input_tokens=50,
                        output_tokens=20,
                        wall_clock_s=1.0,
                        gpu_energy_joules=5.0,
                        gpu_power_avg_watts=200.0,
                    ),
                ],
                total_wall_clock_s=1.0,
                completed=True,
                # is_resolved not set -> defaults to None
            ),
        ]

        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        eff = summary["efficiency"]
        assert eff["accuracy"] is None
        assert eff["ipj"] is None
        assert eff["ipw"] is None

    def test_normalized_none_when_fewer_than_4_traces(self, tmp_path: Path) -> None:
        """With fewer than 4 traces, normalized sections are None."""
        traces = _make_traces_with_accuracy()  # only 2 traces
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["normalized_statistics"] is None
        assert summary["normalized_efficiency"] is None

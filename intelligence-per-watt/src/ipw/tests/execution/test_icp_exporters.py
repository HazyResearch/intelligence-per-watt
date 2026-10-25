"""Tests for execution/exporters.py — export functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipw.execution.exporters import export_jsonl, export_summary_json
from ipw.execution.trace import QueryTrace, TurnTrace

_RIGOROUS_NATIVE_UTILITY = {
    "utility_complete": True,
    "utility_provenance": "task_native_verifier",
    "utility_verifier_id": "terminalbench-native-verifier",
    "utility_verifier_version": "2.1.0",
    "utility_task_manifest_id": "terminalbench-v2.1-fixture",
    "utility_task_membership_id": "fixture-task-0",
    "utility_task_manifest_size": 1,
}


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
            "wall_clock_s",
            "gpu_energy_joules",
            "cpu_energy_joules",
            "gpu_power_watts",
            "cpu_power_watts",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "throughput_tokens_per_sec",
            "energy_per_token_joules",
            "cost_usd",
            "turns",
            "tool_calls",
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

    def test_all_unknown_costs_remain_unknown(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="unknown",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, cost_usd=None)],
            )
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["cost_usd"] is None
        assert summary["totals"]["cost_usd_compatibility_status"] == "legacy_lm_only_partial"
        assert summary["totals"]["complete_cost_usd"] is None
        assert summary["totals"]["legacy_lm_only_cost_usd"] is None
        assert summary["totals"]["cost_known_queries"] == 0
        assert summary["totals"]["cost_unknown_queries"] == 1
        assert summary["totals"]["cost_coverage"] == 0.0
        assert summary["statistics"]["cost_usd"]["avg"] is None
        assert summary["statistics"]["cost_usd"]["compatibility_status"] == "legacy_lm_only_partial"
        assert summary["statistics"]["complete_cost_usd"]["avg"] is None
        assert summary["statistics"]["legacy_lm_only_cost_usd"]["avg"] is None

    def test_mixed_costs_preserve_legacy_partial_but_complete_fails_closed(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="known",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, cost_usd=0.25)],
            ),
            QueryTrace(
                query_id="unknown",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, cost_usd=None)],
            ),
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        # Established keys reproduce the pre-Phase-0 partial LM-only view for
        # two releases. Only the additive complete fields are IPD-safe.
        assert summary["totals"]["cost_usd"] == pytest.approx(0.25)
        assert summary["totals"]["cost_usd_compatibility_status"] == "legacy_lm_only_partial"
        assert summary["totals"]["complete_cost_usd"] is None
        assert summary["totals"]["legacy_lm_only_cost_usd"] == pytest.approx(0.25)
        assert summary["totals"]["cost_known_queries"] == 1
        assert summary["totals"]["cost_unknown_queries"] == 1
        assert summary["totals"]["cost_coverage"] == pytest.approx(0.5)
        assert summary["statistics"]["cost_usd"]["avg"] == pytest.approx(0.25)
        assert summary["statistics"]["cost_usd"]["compatibility_status"] == "legacy_lm_only_partial"
        assert summary["statistics"]["complete_cost_usd"]["avg"] is None
        assert summary["statistics"]["legacy_lm_only_cost_usd"]["avg"] == pytest.approx(0.25)

    def test_complete_zero_costs_remain_zero(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="free",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, cost_usd=0.0)],
            )
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["cost_usd"] == 0.0
        assert summary["totals"]["cost_usd_compatibility_status"] == "legacy_lm_only_partial"
        assert summary["totals"]["complete_cost_usd"] == 0.0
        assert summary["totals"]["legacy_lm_only_cost_usd"] == 0.0
        assert summary["totals"]["cost_coverage"] == 1.0
        assert summary["statistics"]["cost_usd"]["avg"] == 0.0
        assert summary["statistics"]["cost_usd"]["compatibility_status"] == "legacy_lm_only_partial"

    @pytest.mark.parametrize("invalid", [-1.0, float("inf"), True, "0.25"])
    def test_invalid_cost_is_unknown_for_summary_coverage(self, tmp_path: Path, invalid: object) -> None:
        traces = [
            QueryTrace(
                query_id="invalid",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, cost_usd=invalid)],
            )
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["cost_usd"] is None
        assert summary["totals"]["complete_cost_usd"] is None
        assert summary["totals"]["legacy_lm_only_cost_usd"] is None
        assert summary["totals"]["cost_known_queries"] == 0
        assert summary["totals"]["cost_unknown_queries"] == 1
        assert summary["totals"]["cost_coverage"] == 0.0


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
        # Established IPJ remains accuracy / partially observed joules.
        assert eff["ipj"] == pytest.approx(0.5 / 25.0)
        assert eff["legacy_ipj"] == pytest.approx(0.5 / 25.0)
        assert eff["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert eff["utility_total"] == 1
        assert eff["gpu_energy_coverage"] == 1.0
        assert eff["ipj_eligible"] is True
        assert eff["rigorous_ipj"] is None
        assert eff["rigorous_ipj_eligible"] is False
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

    def test_zero_utility_preserves_legacy_none_and_rigorous_zero(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="wrong",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=False,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **_RIGOROUS_NATIVE_UTILITY,
            )
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["utility_total"] == 0
        # HEAD historically suppressed IPJ for zero accuracy. Preserve that
        # established key while the rigorous ratio correctly represents zero.
        assert efficiency["ipj"] is None
        assert efficiency["legacy_ipj"] is None
        assert efficiency["ipj_eligible"] is False
        assert efficiency["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert efficiency["rigorous_ipj"] == 0.0
        assert efficiency["rigorous_ipj_eligible"] is True

    def test_rigorous_ipj_requires_explicit_complete_measured_energy(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="resolved",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=True,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_membership_id": "fixture-task-0",
                    "utility_task_manifest_size": 2,
                },
            ),
            QueryTrace(
                query_id="unresolved",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=15.0)],
                completed=True,
                is_resolved=False,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_membership_id": "fixture-task-1",
                    "utility_task_manifest_size": 2,
                },
            ),
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["utility_total"] == 1
        assert efficiency["rigorous_total_gpu_energy_joules"] == 20.0
        assert efficiency["rigorous_gpu_energy_coverage"] == 1.0
        assert efficiency["rigorous_ipj"] == pytest.approx(0.05)
        assert efficiency["rigorous_ipj_eligible"] is True

    @pytest.mark.parametrize(
        ("complete", "provenance"),
        [
            (None, None),
            (False, "measured_dedicated"),
            (True, "estimated"),
            (True, "measured_shared_allocated"),
        ],
    )
    def test_rigorous_ipj_rejects_incomplete_or_nonmeasured_energy(
        self,
        tmp_path: Path,
        complete: bool | None,
        provenance: str | None,
    ) -> None:
        trace = QueryTrace(
            query_id="not-rigorous",
            workload_type="agentic",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
            completed=True,
            is_resolved=True,
            gpu_energy_complete=complete,
            gpu_energy_provenance=provenance,
        )
        path = tmp_path / "summary.json"

        export_summary_json([trace], {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_total_gpu_energy_joules"] is None
        assert efficiency["rigorous_gpu_energy_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_rigorous_ipj_rejects_unproven_utility(self, tmp_path: Path) -> None:
        trace = QueryTrace(
            query_id="unproven-utility",
            workload_type="agentic",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
            completed=True,
            is_resolved=True,
            gpu_energy_complete=True,
            gpu_energy_provenance="measured_dedicated",
            accelerator_device_coverage_complete=True,
        )
        path = tmp_path / "summary.json"

        export_summary_json([trace], {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_total_gpu_energy_joules"] == 5.0
        assert efficiency["rigorous_utility_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    @pytest.mark.parametrize(
        "missing_field",
        [
            "utility_verifier_id",
            "utility_verifier_version",
            "utility_task_manifest_id",
            "utility_task_membership_id",
            "utility_task_manifest_size",
        ],
    )
    def test_rigorous_ipj_rejects_unbound_utility_evidence(self, tmp_path: Path, missing_field: str) -> None:
        evidence = dict(_RIGOROUS_NATIVE_UTILITY)
        evidence[missing_field] = ""
        trace = QueryTrace(
            query_id="unbound-utility",
            workload_type="agentic",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
            completed=True,
            is_resolved=True,
            gpu_energy_complete=True,
            gpu_energy_provenance="measured_dedicated",
            accelerator_device_coverage_complete=True,
            **evidence,
        )
        path = tmp_path / f"{missing_field}.json"

        export_summary_json([trace], {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_total_gpu_energy_joules"] == 5.0
        assert efficiency["rigorous_utility_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_rigorous_ipj_rejects_mixed_utility_manifests(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id=f"q{index}",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=True,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_manifest_id": f"manifest-{index}",
                    "utility_task_membership_id": f"fixture-task-{index}",
                    "utility_task_manifest_size": 2,
                },
            )
            for index in range(2)
        ]
        path = tmp_path / "mixed-manifests.json"

        export_summary_json(traces, {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_utility_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    @pytest.mark.parametrize("membership_ids", [["task-0"], ["task-0", "task-0"]])
    def test_rigorous_ipj_rejects_manifest_subsets_and_duplicates(
        self, tmp_path: Path, membership_ids: list[str]
    ) -> None:
        traces = [
            QueryTrace(
                query_id=f"q{index}",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=True,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_membership_id": membership_id,
                    "utility_task_manifest_size": 2,
                },
            )
            for index, membership_id in enumerate(membership_ids)
        ]
        path = tmp_path / f"population-{len(membership_ids)}.json"

        export_summary_json(traces, {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_utility_coverage"] == 0.5
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_benchmark_energy_cannot_bypass_mixed_utility_manifests(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id=f"q{index}",
                workload_type="agentic",
                completed=True,
                is_resolved=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_manifest_id": f"manifest-{index}",
                    "utility_task_membership_id": f"fixture-task-{index}",
                    "utility_task_manifest_size": 2,
                },
            )
            for index in range(2)
        ]
        path = tmp_path / "mixed-manifest-benchmark.json"

        export_summary_json(
            traces,
            {},
            path,
            bench_energy={
                "gpu_energy_joules": 10.0,
                "gpu_energy_complete": True,
                "gpu_energy_provenance": "measured_dedicated",
                "accelerator_device_coverage_complete": True,
            },
        )

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["rigorous_utility_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_rigorous_ipj_rejects_partial_turn_energy_despite_legacy_sum(self, tmp_path: Path) -> None:
        trace = QueryTrace(
            query_id="partial-turn-energy",
            workload_type="agentic",
            turns=[
                TurnTrace(turn_index=0, gpu_energy_joules=5.0),
                TurnTrace(turn_index=1, gpu_energy_joules=None),
            ],
            completed=True,
            is_resolved=True,
            **_RIGOROUS_NATIVE_UTILITY,
            gpu_energy_complete=True,
            gpu_energy_provenance="measured_dedicated",
            accelerator_device_coverage_complete=True,
        )
        path = tmp_path / "summary.json"

        export_summary_json([trace], {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert trace.total_gpu_energy_joules == 5.0
        assert trace.complete_total_gpu_energy_joules is None
        assert efficiency["legacy_ipj"] == pytest.approx(0.2)
        assert efficiency["rigorous_total_gpu_energy_joules"] is None
        assert efficiency["rigorous_gpu_energy_coverage"] == 0.0
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_missing_query_energy_only_invalidates_rigorous_ipj(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="measured",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=True,
            ),
            QueryTrace(
                query_id="unknown",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=None)],
                completed=True,
                is_resolved=False,
            ),
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["gpu_energy_joules"] == 5.0
        efficiency = summary["efficiency"]
        assert efficiency["ipj"] == pytest.approx(0.5 / 5.0)
        assert efficiency["legacy_ipj"] == pytest.approx(0.5 / 5.0)
        assert efficiency["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert efficiency["gpu_energy_coverage"] == pytest.approx(0.5)
        assert efficiency["ipj_eligible"] is True
        assert efficiency["rigorous_total_gpu_energy_joules"] is None
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    @pytest.mark.parametrize(
        ("lm_energy_measurable", "shared_device_warning"),
        [(False, False), (True, True)],
    )
    def test_unattributable_query_energy_is_only_legacy_ipj_eligible(
        self,
        tmp_path: Path,
        lm_energy_measurable: bool,
        shared_device_warning: bool,
    ) -> None:
        trace = QueryTrace(
            query_id="unattributable",
            workload_type="agentic",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
            completed=True,
            is_resolved=True,
            lm_energy_measurable=lm_energy_measurable,
            shared_device_warning=shared_device_warning,
        )
        path = tmp_path / "summary.json"

        export_summary_json([trace], {}, path)

        summary = json.loads(path.read_text())
        assert summary["totals"]["gpu_energy_joules"] == 5.0
        efficiency = summary["efficiency"]
        assert efficiency["gpu_energy_coverage"] == 1.0
        assert efficiency["ipj"] == pytest.approx(0.2)
        assert efficiency["legacy_ipj"] == pytest.approx(0.2)
        assert efficiency["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert efficiency["ipj_eligible"] is True
        assert efficiency["rigorous_total_gpu_energy_joules"] is None
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_benchmark_energy_requires_complete_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        trace = QueryTrace(
            query_id="benchmark-window",
            workload_type="agentic",
            completed=True,
            is_resolved=True,
            lm_energy_measurable=False,
            **_RIGOROUS_NATIVE_UTILITY,
        )

        cases = [
            ({"gpu_energy_joules": 5.0}, 5.0),
            (
                {
                    "gpu_energy_joules": float("inf"),
                    "gpu_energy_complete": True,
                    "gpu_energy_provenance": "measured_dedicated",
                },
                None,
            ),
            (
                {
                    "gpu_energy_joules": 5.0,
                    "gpu_energy_complete": True,
                    "gpu_energy_provenance": "estimated",
                },
                5.0,
            ),
        ]
        for index, (bench_energy, expected_legacy_energy) in enumerate(cases):
            path = tmp_path / f"invalid-{index}.json"
            export_summary_json([trace], {}, path, bench_energy=bench_energy)
            summary = json.loads(path.read_text())
            assert summary["totals"]["gpu_energy_joules"] == expected_legacy_energy
            assert summary["efficiency"]["rigorous_total_gpu_energy_joules"] is None
            assert summary["efficiency"]["rigorous_ipj_eligible"] is False
            assert summary["efficiency"]["rigorous_ipj"] is None

    def test_complete_benchmark_energy_can_back_workload_ipj(
        self,
        tmp_path: Path,
    ) -> None:
        trace = QueryTrace(
            query_id="benchmark-window",
            workload_type="agentic",
            completed=True,
            is_resolved=True,
            **_RIGOROUS_NATIVE_UTILITY,
        )
        path = tmp_path / "summary.json"

        export_summary_json(
            [trace],
            {},
            path,
            bench_energy={
                "gpu_energy_joules": 5.0,
                "gpu_energy_complete": True,
                "gpu_energy_provenance": "measured_dedicated",
                "accelerator_device_coverage_complete": True,
                "avg_gpu_power_watts": 0.0,
            },
        )

        summary = json.loads(path.read_text())
        efficiency = summary["efficiency"]
        assert summary["totals"]["gpu_energy_joules"] == 5.0
        assert efficiency["gpu_energy_coverage"] == 1.0
        assert efficiency["avg_gpu_power_watts"] == 0.0
        assert efficiency["ipj"] == pytest.approx(0.2)
        assert efficiency["ipj_eligible"] is True
        assert efficiency["ipj_energy_basis"] == "legacy_benchmark_aggregate"
        assert efficiency["rigorous_total_gpu_energy_joules"] == 5.0
        assert efficiency["rigorous_ipj"] == pytest.approx(0.2)
        assert efficiency["rigorous_ipj_eligible"] is True

    @pytest.mark.parametrize(
        ("lm_energy_measurable", "shared_device_warning"),
        [(False, False), (True, True)],
    )
    def test_benchmark_energy_rejects_unattributable_workload(
        self,
        tmp_path: Path,
        lm_energy_measurable: bool,
        shared_device_warning: bool,
    ) -> None:
        trace = QueryTrace(
            query_id="benchmark-window",
            workload_type="agentic",
            completed=True,
            is_resolved=True,
            lm_energy_measurable=lm_energy_measurable,
            shared_device_warning=shared_device_warning,
            **_RIGOROUS_NATIVE_UTILITY,
        )
        path = tmp_path / "unattributable-benchmark.json"

        export_summary_json(
            [trace],
            {},
            path,
            bench_energy={
                "gpu_energy_joules": 5.0,
                "gpu_energy_complete": True,
                "gpu_energy_provenance": "measured_dedicated",
                "accelerator_device_coverage_complete": True,
            },
        )

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["legacy_ipj"] == pytest.approx(0.2)
        assert efficiency["rigorous_total_gpu_energy_joules"] is None
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_posthoc_normalized_efficiency_is_never_rigorous(
        self,
        tmp_path: Path,
    ) -> None:
        traces = [
            QueryTrace(
                query_id=f"q{index}",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=10.0)],
                total_wall_clock_s=float(index + 1),
                completed=True,
                is_resolved=index % 2 == 0,
                gpu_energy_complete=True,
                gpu_energy_provenance="measured_dedicated",
                accelerator_device_coverage_complete=True,
                **{
                    **_RIGOROUS_NATIVE_UTILITY,
                    "utility_task_membership_id": f"fixture-task-{index}",
                    "utility_task_manifest_size": 4,
                },
            )
            for index in range(4)
        ]
        path = tmp_path / "normalized.json"

        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["efficiency"]["rigorous_ipj_eligible"] is True
        normalized = summary["normalized_efficiency"]
        assert normalized["legacy_ipj"] is not None
        assert normalized["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert normalized["rigorous_ipj"] is None
        assert normalized["rigorous_ipj_eligible"] is False

    def test_unscored_query_only_invalidates_rigorous_ipj(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="scored",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=True,
                is_resolved=True,
            ),
            QueryTrace(
                query_id="unscored",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
                completed=False,
                is_resolved=None,
            ),
        ]
        path = tmp_path / "summary.json"

        export_summary_json(traces, {}, path)

        efficiency = json.loads(path.read_text())["efficiency"]
        assert efficiency["unscored_queries"] == 1
        assert efficiency["ipj"] == pytest.approx(0.1)
        assert efficiency["legacy_ipj"] == pytest.approx(0.1)
        assert efficiency["ipj_compatibility_status"] == "legacy_nonrigorous"
        assert efficiency["ipj_eligible"] is True
        assert efficiency["rigorous_ipj"] is None
        assert efficiency["rigorous_ipj_eligible"] is False

    def test_normalized_none_when_fewer_than_4_traces(self, tmp_path: Path) -> None:
        """With fewer than 4 traces, normalized sections are None."""
        traces = _make_traces_with_accuracy()  # only 2 traces
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)

        summary = json.loads(path.read_text())
        assert summary["normalized_statistics"] is None
        assert summary["normalized_efficiency"] is None


class TestResolvedUnresolvedCounts:
    """Tests for resolved/unresolved aggregation in totals."""

    def _make_trace(self, qid: str, is_resolved=None) -> QueryTrace:
        return QueryTrace(
            query_id=qid,
            workload_type="test",
            turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5, wall_clock_s=1.0)],
            total_wall_clock_s=1.0,
            completed=True,
            is_resolved=is_resolved,
        )

    def test_all_resolved(self, tmp_path: Path) -> None:
        traces = [self._make_trace(f"q{i}", is_resolved=True) for i in range(3)]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        totals = json.loads(path.read_text())["totals"]
        assert totals["resolved"] == 3
        assert totals["unresolved"] == 0

    def test_all_unresolved(self, tmp_path: Path) -> None:
        traces = [self._make_trace(f"q{i}", is_resolved=False) for i in range(3)]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        totals = json.loads(path.read_text())["totals"]
        assert totals["resolved"] == 0
        assert totals["unresolved"] == 3

    def test_all_none_is_resolved(self, tmp_path: Path) -> None:
        traces = [self._make_trace(f"q{i}", is_resolved=None) for i in range(3)]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        totals = json.loads(path.read_text())["totals"]
        assert totals["resolved"] == 0
        assert totals["unresolved"] == 0

    def test_mixed_resolved_states(self, tmp_path: Path) -> None:
        traces = [
            self._make_trace("q0", is_resolved=True),
            self._make_trace("q1", is_resolved=False),
            self._make_trace("q2", is_resolved=None),
            self._make_trace("q3", is_resolved=True),
            self._make_trace("q4", is_resolved=False),
        ]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        totals = json.loads(path.read_text())["totals"]
        assert totals["resolved"] == 2
        assert totals["unresolved"] == 2

    def test_empty_traces_resolved_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        export_summary_json([], {}, path)
        totals = json.loads(path.read_text())["totals"]
        assert totals["resolved"] == 0
        assert totals["unresolved"] == 0

    def test_accuracy_computed_from_resolved(self, tmp_path: Path) -> None:
        """accuracy = resolved / (resolved + unresolved), ignoring None."""
        traces = [
            self._make_trace("q0", is_resolved=True),
            self._make_trace("q1", is_resolved=False),
            self._make_trace("q2", is_resolved=False),
            self._make_trace("q3", is_resolved=None),  # not scored
        ]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        # accuracy = 1 / 3 (only 3 scored, 1 resolved)
        assert summary["totals"]["accuracy"] == pytest.approx(1.0 / 3.0)
        assert summary["efficiency"]["accuracy"] == pytest.approx(1.0 / 3.0)

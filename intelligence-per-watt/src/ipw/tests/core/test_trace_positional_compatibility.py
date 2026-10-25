"""Two-release constructor and cost-view compatibility for trace models."""

from __future__ import annotations

from dataclasses import fields

import pytest

from ipw.execution.trace import QueryTrace, TurnTrace

_LEGACY_TURN_FIELDS = [
    "turn_index",
    "input_tokens",
    "output_tokens",
    "tool_result_tokens",
    "tools_called",
    "tool_latencies_s",
    "wall_clock_s",
    "error",
    "gpu_energy_joules",
    "cpu_energy_joules",
    "gpu_power_avg_watts",
    "cpu_power_avg_watts",
    "cost_usd",
    "dram_energy_joules",
    "peak_watts",
    "reasoning_tokens",
    "cached_tokens",
    "has_parallel_tools",
    "shared_device_warning",
]

_LEGACY_QUERY_FIELDS = [
    "query_id",
    "workload_type",
    "query_text",
    "response_text",
    "turns",
    "total_wall_clock_s",
    "completed",
    "timed_out",
    "query_gpu_energy_joules",
    "query_cpu_energy_joules",
    "query_gpu_power_avg_watts",
    "query_cpu_power_avg_watts",
    "judge_gpu_energy_joules",
    "judge_cpu_energy_joules",
    "judge_gpu_power_avg_watts",
    "judge_cpu_power_avg_watts",
    "judge_wall_clock_s",
    "query_mbu_avg_pct",
    "query_mbu_max_pct",
    "is_resolved",
    "unscorable_reason",
    "score_metadata",
    "lm_energy_measurable",
    "shared_device_warning",
    "accuracy_score",
    "accuracy_metadata",
    "n_retries",
]


def test_turn_trace_preserves_exact_legacy_positional_signature() -> None:
    legacy_values = [
        7,
        11,
        12,
        13,
        ["shell"],
        {"shell": 0.5},
        1.25,
        "legacy error",
        2.1,
        3.2,
        4.3,
        5.4,
        0.13,
        6.5,
        7.6,
        8,
        9,
        True,
        True,
    ]

    trace = TurnTrace(*legacy_values)

    assert [item.name for item in fields(TurnTrace)][: len(_LEGACY_TURN_FIELDS)] == _LEGACY_TURN_FIELDS
    assert [getattr(trace, name) for name in _LEGACY_TURN_FIELDS] == legacy_values
    assert trace.tool_cost_usd is None
    assert trace.tool_cost_complete is False
    assert trace.legacy_lm_only_cost_usd == pytest.approx(0.13)
    assert trace.total_cost_usd is None


def test_query_trace_preserves_exact_legacy_positional_signature() -> None:
    turn = TurnTrace(turn_index=0, cost_usd=0.25)
    legacy_values = [
        "legacy-query",
        "agentic",
        "question",
        "answer",
        [turn],
        12.5,
        True,
        False,
        101.0,
        51.0,
        201.0,
        71.0,
        3.0,
        2.0,
        30.0,
        20.0,
        0.4,
        55.0,
        70.0,
        True,
        None,
        {"judge": "native"},
        False,
        True,
        0.75,
        {"metric": "pass"},
        2,
    ]

    trace = QueryTrace(*legacy_values)

    assert [item.name for item in fields(QueryTrace)][: len(_LEGACY_QUERY_FIELDS)] == _LEGACY_QUERY_FIELDS
    assert [getattr(trace, name) for name in _LEGACY_QUERY_FIELDS] == legacy_values
    assert trace.utility_complete is False
    assert trace.utility_provenance == "unknown"
    assert trace.utility_calibration_id is None
    assert trace.gpu_energy_complete is None
    assert trace.gpu_energy_provenance is None
    assert trace.accelerator_device_coverage_complete is None
    assert trace.utility_verifier_id is None
    assert trace.utility_verifier_version is None
    assert trace.utility_task_manifest_id is None
    assert trace.utility_task_membership_id is None
    assert trace.utility_task_manifest_size is None


def test_legacy_total_and_alias_do_not_weaken_complete_total() -> None:
    priced_lm = TurnTrace(turn_index=0, cost_usd=0.25)
    unknown_tool = TurnTrace(
        turn_index=1,
        tools_called=["paid_search"],
        cost_usd=0.5,
    )
    trace = QueryTrace(
        query_id="cost-compat",
        workload_type="agentic",
        turns=[priced_lm, unknown_tool],
    )

    assert priced_lm.total_cost_usd == pytest.approx(0.25)
    assert unknown_tool.total_cost_usd is None
    assert trace.total_cost_usd == pytest.approx(0.75)
    assert trace.legacy_lm_only_cost_usd == pytest.approx(0.75)
    assert trace.complete_total_cost_usd is None
    assert trace.cost_known_turns == 1

    turn_record = unknown_tool.to_dict()
    query_record = trace.to_dict()
    assert turn_record["cost_usd"] == pytest.approx(0.5)
    assert turn_record["legacy_lm_only_cost_usd"] == pytest.approx(0.5)
    assert turn_record["total_cost_usd"] is None
    assert query_record["legacy_lm_only_cost_usd"] == pytest.approx(0.75)
    assert query_record["total_cost_usd"] == pytest.approx(0.75)
    assert query_record["complete_total_cost_usd"] is None

    unknown_tool.tool_cost_usd = 0.1
    unknown_tool.tool_cost_complete = True
    assert trace.total_cost_usd == pytest.approx(0.75)
    assert trace.legacy_lm_only_cost_usd == pytest.approx(0.75)
    assert trace.complete_total_cost_usd == pytest.approx(0.85)
    assert trace.cost_known_turns == 2


def test_legacy_lm_only_alias_is_none_without_any_known_lm_cost() -> None:
    trace = QueryTrace(
        query_id="unknown",
        workload_type="agentic",
        turns=[TurnTrace(turn_index=0, cost_usd=None)],
    )

    assert trace.legacy_lm_only_cost_usd is None
    assert trace.to_dict()["legacy_lm_only_cost_usd"] is None

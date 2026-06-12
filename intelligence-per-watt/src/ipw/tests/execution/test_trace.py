"""Tests for execution/trace.py schema extensions."""

from __future__ import annotations

from ipw.execution.trace import QueryTrace, TurnTrace


class TestTurnTraceAdditiveFields:
    def test_new_fields_default_none_or_false(self) -> None:
        t = TurnTrace(turn_index=0)
        assert t.dram_energy_joules is None
        assert t.peak_watts is None
        assert t.reasoning_tokens is None
        assert t.cached_tokens is None
        assert t.has_parallel_tools is False
        assert t.shared_device_warning is False

    def test_new_fields_serialize_when_set(self) -> None:
        t = TurnTrace(
            turn_index=0,
            dram_energy_joules=12.5,
            peak_watts=420.0,
            reasoning_tokens=100,
            cached_tokens=50,
            has_parallel_tools=True,
            shared_device_warning=True,
        )
        d = t.to_dict()
        assert d["dram_energy_joules"] == 12.5
        assert d["peak_watts"] == 420.0
        assert d["reasoning_tokens"] == 100
        assert d["cached_tokens"] == 50
        assert d["has_parallel_tools"] is True
        assert d["shared_device_warning"] is True

    def test_new_fields_present_in_dict_when_default(self) -> None:
        t = TurnTrace(turn_index=0)
        d = t.to_dict()
        assert d["turn_index"] == 0
        assert d["input_tokens"] == 0
        assert d["gpu_energy_joules"] is None
        assert "dram_energy_joules" in d
        assert "peak_watts" in d

    def test_roundtrip_from_dict_preserves_new_fields(self) -> None:
        original = TurnTrace(
            turn_index=1,
            input_tokens=100,
            dram_energy_joules=5.0,
            has_parallel_tools=True,
        )
        restored = TurnTrace.from_dict(original.to_dict())
        assert restored.dram_energy_joules == 5.0
        assert restored.has_parallel_tools is True

    def test_from_dict_legacy_payload_uses_safe_defaults(self) -> None:
        # Simulating an old trace dict without the newer additive fields
        legacy = {
            "turn_index": 0,
            "input_tokens": 10,
            "output_tokens": 20,
            "tool_result_tokens": 0,
            "tools_called": [],
            "tool_latencies_s": {},
            "wall_clock_s": 1.0,
            "error": None,
            "gpu_energy_joules": None,
            "cpu_energy_joules": None,
            "gpu_power_avg_watts": None,
            "cpu_power_avg_watts": None,
            "cost_usd": None,
        }
        restored = TurnTrace.from_dict(legacy)
        assert restored.dram_energy_joules is None
        assert restored.peak_watts is None
        assert restored.reasoning_tokens is None
        assert restored.cached_tokens is None
        assert restored.has_parallel_tools is False
        assert restored.shared_device_warning is False


class TestQueryTraceAdditiveFields:
    def test_new_fields_default(self) -> None:
        q = QueryTrace(query_id="q1", workload_type="gaia")
        assert q.lm_energy_measurable is True
        assert q.shared_device_warning is False
        assert q.accuracy_score is None
        assert q.accuracy_metadata == {}
        assert q.n_retries == 0

    def test_new_fields_serialize(self) -> None:
        q = QueryTrace(
            query_id="q1",
            workload_type="gaia",
            lm_energy_measurable=False,
            shared_device_warning=True,
            accuracy_score=1.0,
            accuracy_metadata={"match_type": "exact"},
            n_retries=2,
        )
        d = q.to_dict()
        assert d["lm_energy_measurable"] is False
        assert d["shared_device_warning"] is True
        assert d["accuracy_score"] == 1.0
        assert d["accuracy_metadata"] == {"match_type": "exact"}
        assert d["n_retries"] == 2

    def test_from_dict_legacy_payload_uses_safe_defaults(self) -> None:
        legacy = {
            "query_id": "q1",
            "workload_type": "gaia",
            "query_text": "",
            "response_text": "",
            "turns": [],
            "total_wall_clock_s": 0.0,
            "completed": False,
            "timed_out": False,
            "query_gpu_energy_joules": None,
            "query_cpu_energy_joules": None,
            "query_gpu_power_avg_watts": None,
            "query_cpu_power_avg_watts": None,
            "query_mbu_avg_pct": None,
            "query_mbu_max_pct": None,
            "is_resolved": None,
        }
        restored = QueryTrace.from_dict(legacy)
        assert restored.lm_energy_measurable is True
        assert restored.shared_device_warning is False
        assert restored.accuracy_score is None
        assert restored.accuracy_metadata == {}
        assert restored.n_retries == 0

    def test_accuracy_metadata_to_dict_is_isolated(self) -> None:
        q = QueryTrace(query_id="q1", workload_type="gaia", accuracy_metadata={"k": "v"})
        d = q.to_dict()
        d["accuracy_metadata"]["k"] = "mutated"
        assert q.accuracy_metadata["k"] == "v"

    def test_roundtrip_from_dict_preserves_new_fields(self) -> None:
        original = QueryTrace(
            query_id="q1",
            workload_type="gaia",
            lm_energy_measurable=False,
            accuracy_score=0.75,
            accuracy_metadata={"match_type": "fuzzy"},
            n_retries=3,
            turns=[TurnTrace(turn_index=0, dram_energy_joules=2.0, has_parallel_tools=True)],
        )
        restored = QueryTrace.from_dict(original.to_dict())
        assert restored.lm_energy_measurable is False
        assert restored.accuracy_score == 0.75
        assert restored.accuracy_metadata == {"match_type": "fuzzy"}
        assert restored.n_retries == 3
        assert len(restored.turns) == 1
        assert restored.turns[0].dram_energy_joules == 2.0
        assert restored.turns[0].has_parallel_tools is True

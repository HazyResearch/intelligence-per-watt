from __future__ import annotations

from ipj.efficiency.record import PerQueryRecord, build_record
from ipj.efficiency.tb_metadata import TBMeta

SAMPLE = {
    "task_name": "terminal-bench/kv-store-grpc",
    "started_at": "2026-06-23T05:59:42.000000",
    "finished_at": "2026-06-23T06:02:42.000000",  # +180s
    "verifier_result": {"rewards": {"reward": 1.0}},
    "agent_result": {"n_input_tokens": 4000, "n_output_tokens": 1000, "cost_usd": None},
    "exception_info": None,
}
META = TBMeta(difficulty="medium", category="software-engineering", tags=["grpc"],
              expert_time_min=30.0, junior_time_min=120.0)

def test_build_record_solved_local():
    r = build_record(SAMPLE, approach="Qwen/Qwen3.6-27B-FP8", category="open",
                     hardware="H100", precision="FP8", tb_meta=META, model_params_b=27.0, joules=4000.0)
    assert isinstance(r, PerQueryRecord)
    assert r.task_id == "terminal-bench/kv-store-grpc"
    assert r.solved is True and r.failure_mode == "solved"
    assert r.difficulty == "medium" and r.tb_category == "software-engineering"
    assert r.input_tokens == 4000 and r.output_tokens == 1000
    assert r.dollars == 0.0  # local = free
    assert r.latency_s == 180.0
    assert r.flops == 2.0 * 27.0 * 1e9 * 5000
    assert r.joules == 4000.0

def test_build_record_timeout_no_flops_when_params_unknown():
    sample = {**SAMPLE, "verifier_result": {"rewards": {"reward": 0.0}},
              "exception_info": {"exception_type": "AgentTimeoutError"},
              "agent_result": {"n_input_tokens": 100, "n_output_tokens": 0, "cost_usd": 0.5}}
    r = build_record(sample, approach="gpt-5.5", category="closed",
                     hardware="cloud", precision="n/a", tb_meta=META, model_params_b=None)
    assert r.solved is False and r.failure_mode == "timeout"
    assert r.flops is None          # unknown params
    assert r.dollars == 0.5         # cloud uses cost_usd
    assert r.joules is None         # cloud unmeasurable

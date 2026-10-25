from __future__ import annotations

import time

from ipj.metrics import MEASURED_DEDICATED_JOULES_PROVENANCE
from ipj.orchestrate.harness.dispatch import (
    AcceleratorEnergyMeasurement,
    WorkerDispatch,
    WorkerResult,
)
from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC, WorkerPool


class FakeServer:
    def __init__(self, binding):
        self.binding = binding
        self.calls = []

    def execute(self, prompt, **params):
        self.calls.append(prompt)
        return WorkerResult(content=f"[{self.binding.model_id}] {prompt}", tokens_in=10, tokens_out=5, dollars=0.01, latency_s=0.1)


def _dispatch(**kw):
    pool = WorkerPool.from_spec(DEFAULT_POOL_SPEC)
    return WorkerDispatch(pool, server_factory=FakeServer, **kw), pool


def test_call_worker_records_cost_and_concats_context():
    d, _ = _dispatch()
    r = d.call_worker("cloud-mini", sub_prompt="solve", context="ctx")
    assert "ctx" in r.content and "solve" in r.content
    assert r.dollars == 0.01
    assert r.gpu_energy_j is None  # cloud: no measured Joules


def test_local_worker_gets_energy_from_probe():
    d, _ = _dispatch(energy_probe=lambda t0, t1: 12.3)
    r = d.call_worker("local-fast", sub_prompt="ls", context=None)
    assert r.gpu_energy_j == 12.3  # legacy numeric probe remains visible
    assert r.gpu_energy_complete is False  # but cannot invent measurement provenance


def test_local_worker_accepts_explicit_complete_measurement():
    d, _ = _dispatch(
        energy_probe=lambda t0, t1: AcceleratorEnergyMeasurement(
            12.3,
            provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
            complete=True,
            accelerator_device_coverage_complete=True,
        )
    )
    result = d.call_worker("local-fast", sub_prompt="ls", context=None)
    assert result.gpu_energy_j == 12.3
    assert result.gpu_energy_complete is True
    assert result.gpu_energy_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE


def test_local_worker_preserves_server_supplied_complete_measurement():
    class MeasuredServer(FakeServer):
        def execute(self, prompt, **params):
            return WorkerResult(
                content="ok",
                gpu_energy_j=6.0,
                gpu_energy_provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    pool = WorkerPool.from_spec(DEFAULT_POOL_SPEC)
    dispatch = WorkerDispatch(pool, server_factory=MeasuredServer)
    result = dispatch.call_worker("local-fast", sub_prompt="ls", context=None)
    assert result.gpu_energy_j == 6.0
    assert result.gpu_energy_complete is True


def test_cloud_worker_ignores_energy_probe():
    # Even with a probe supplied, a CLOUD worker (is_local=False) must report no local Joules.
    d, _ = _dispatch(energy_probe=lambda t0, t1: 99.0)
    r = d.call_worker("cloud-mini", sub_prompt="solve", context=None)
    assert r.gpu_energy_j is None


def test_delegate_uses_runner_clean_restart():
    seen = {}

    def runner(binding, task_prompt):
        seen["binding"] = binding.alias
        seen["prompt"] = task_prompt
        return WorkerResult(content="DONE", dollars=0.5)

    d, _ = _dispatch(delegate_runner=runner)
    r = d.delegate_task("cloud-frontier", task_prompt="the whole task")
    assert r.content == "DONE"
    assert seen == {"binding": "cloud-frontier", "prompt": "the whole task"}


def test_delegate_fallback_executes_fresh_prompt():
    d, _ = _dispatch()
    r = d.delegate_task("cloud-frontier", task_prompt="whole task")
    assert "whole task" in r.content


def test_default_clock_matches_epoch_timestamp_domain():
    dispatch, _ = _dispatch()
    assert dispatch.clock is time.time

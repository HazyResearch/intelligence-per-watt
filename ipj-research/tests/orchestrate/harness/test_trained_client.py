from __future__ import annotations

import time

from ipj.metrics import MEASURED_DEDICATED_JOULES_PROVENANCE
from ipj.orchestrate.harness.dispatch import AcceleratorEnergyMeasurement
from ipj.orchestrate.harness.prompt import render_message
from ipj.orchestrate.harness.trained_client import LocalAdapterChatClient


def test_build_prompt_uses_shared_render_and_assistant_cue():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    c = LocalAdapterChatClient("base", _generate=lambda p: ("x", 0, 0))
    prompt = c._build_prompt(msgs)
    assert prompt == render_message(msgs[0]) + render_message(msgs[1]) + "<|assistant|>\n"


def test_complete_wraps_generation_without_inventing_local_cost_or_energy_provenance():
    calls = {}
    def fake_gen(prompt):
        calls["prompt"] = prompt
        return ('{"action":"finish","answer":"42"}', 11, 4)  # content, n_in, n_out
    ticks = iter([100.0, 100.5])
    c = LocalAdapterChatClient(
        "base", energy_probe=lambda a, b: 3.0, clock=lambda: next(ticks), _generate=fake_gen
    )
    resp = c.complete([{"role": "user", "content": "hi"}])
    assert resp.content == '{"action":"finish","answer":"42"}'
    assert resp.tokens_in == 11 and resp.tokens_out == 4
    assert resp.dollars is None
    assert resp.gpu_energy_j == 3.0           # from the injected probe over the [t0,t1] window
    assert resp.gpu_energy_complete is False  # a bare float has no verified provenance
    assert resp.latency_s == 0.5              # from the injected clock
    assert "<|assistant|>" in calls["prompt"]


def test_complete_no_probe_leaves_energy_none():
    c = LocalAdapterChatClient("base", _generate=lambda p: ("ok", 1, 1))
    assert c.complete([{"role": "user", "content": "x"}]).gpu_energy_j is None


def test_complete_uses_explicit_cost_estimator_and_energy_measurement():
    ticks = iter([10.0, 10.5])
    client = LocalAdapterChatClient(
        "base",
        energy_probe=lambda start, end: AcceleratorEnergyMeasurement(
            4.0,
            provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
            complete=True,
            accelerator_device_coverage_complete=True,
        ),
        cost_estimator=lambda tokens_in, tokens_out, latency_s: 0.25,
        clock=lambda: next(ticks),
        _generate=lambda prompt: ("ok", 2, 1),
    )
    response = client.complete([{"role": "user", "content": "x"}])
    assert response.dollars == 0.25
    assert response.gpu_energy_complete is True
    assert response.gpu_energy_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE


def test_default_clock_matches_epoch_timestamp_domain():
    client = LocalAdapterChatClient("base", _generate=lambda prompt: ("ok", 1, 1))
    assert client.clock is time.time

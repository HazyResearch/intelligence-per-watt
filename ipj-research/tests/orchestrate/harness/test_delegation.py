from __future__ import annotations

import pytest

from ipj.orchestrate.harness.actions import parse_action
from ipj.orchestrate.harness.brain import BrainStep
from ipj.orchestrate.harness.delegation import (
    WORKER_SYSTEM,
    chat_client_for_binding,
    make_container_delegate_runner,
)
from ipj.orchestrate.harness.dispatch import WorkerResult
from ipj.orchestrate.harness.workers import WorkerBinding
from tests.orchestrate.conftest import FakeSession, ScriptBrain


def _binding(*, provider="anthropic", model_id="x", base_url=None, is_local=False):
    return WorkerBinding(
        alias="cloud-frontier", provider=provider, model_id=model_id, base_url=base_url, is_local=is_local
    )


def test_container_delegate_runner_acts_in_container_and_returns_result():
    # A delegated worker must SOLVE the whole task by acting in the SAME container (editing /testbed
    # via bash) and then finishing — not by emitting a text-only MCP answer.
    fake_session = FakeSession("patched file\n__RC=0\n")
    emissions = [
        '{"action":"bash","command":"sed -i s/bug/fix/ /testbed/app.py"}',
        '{"action":"finish","answer":"fixed the bug"}',
    ]
    runner = make_container_delegate_runner(fake_session, brain_factory=lambda binding: ScriptBrain(emissions))
    result = runner(_binding(), "fix the bug")

    assert isinstance(result, WorkerResult)
    # the worker owned the task end to end and finished with the final answer
    assert result.content == "fixed the bug"
    # Token sums reflect both worker turns. The container tool has no verified
    # dollar measurement, so its unknown component must poison the cost rollup.
    assert result.tokens_in == 14
    assert result.tokens_out == 6
    assert result.dollars is None
    # the worker ACTED in the container: its bash command reached the session
    assert any("sed -i s/bug/fix/ /testbed/app.py" in str(call["keys"]) for call in fake_session.sent)


def test_container_delegate_runner_handles_no_finish():
    # If the worker never finishes (max_turns exhausted), content is "" and sums still reflect steps.
    fake_session = FakeSession("ok\n__RC=0\n")
    emissions = ['{"action":"bash","command":"ls /testbed"}'] * 5
    runner = make_container_delegate_runner(
        fake_session, max_turns=2, brain_factory=lambda binding: ScriptBrain(emissions)
    )
    result = runner(_binding(), "explore")
    assert result.content == ""  # no finish -> empty answer, never None
    assert result.tokens_in == 14  # capped at 2 turns


def test_container_delegate_runner_preserves_unknown_rollups():
    class UnknownAccountingBrain:
        system_prompt = "x"

        def propose(self, ctx):
            return BrainStep(
                action=parse_action('{"action":"finish","answer":"done"}'),
                tokens_in=None,
                tokens_out=1,
                dollars=None,
                latency_s=None,
                gpu_energy_j=2.0,
            )

    runner = make_container_delegate_runner(
        FakeSession(""),
        brain_factory=lambda binding: UnknownAccountingBrain(),
    )
    result = runner(_binding(), "task")
    assert result.dollars is None
    assert result.tokens_in is None
    assert result.latency_s is None
    assert result.gpu_energy_j is None
    assert result.gpu_energy_complete is False


def test_worker_system_prompt_does_not_subdelegate():
    # A delegated worker owns the task: WORKER_SYSTEM must NOT advertise the delegation actions.
    assert "call_worker" not in WORKER_SYSTEM
    assert "delegate_task" not in WORKER_SYSTEM
    # it should still allow the self-acting actions
    assert "bash" in WORKER_SYSTEM and "finish" in WORKER_SYSTEM


def test_chat_client_for_binding_routes_providers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # openrouter routes through the OpenAI-compatible client with the OpenRouter base_url
    c = chat_client_for_binding(_binding(provider="openrouter", model_id="qwen/x"))
    assert c.base_url == "https://openrouter.ai/api/v1"
    assert c.model == "qwen/x"
    # unknown provider is a hard error (no silent default)
    with pytest.raises(ValueError):
        chat_client_for_binding(_binding(provider="mystery"))


def test_chat_client_for_binding_vllm_uses_binding_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # SDK needs a key to construct
    c = chat_client_for_binding(_binding(provider="vllm", model_id="m", base_url="http://host:9000"))
    assert c.base_url == "http://host:9000"

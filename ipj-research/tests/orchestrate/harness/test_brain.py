from __future__ import annotations

from dataclasses import dataclass, field

from ipj.orchestrate.harness.brain import BrainStep, ChatResponse, TeacherBrain, build_messages
from ipj.orchestrate.harness.trace import TaskMeta
from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC, WorkerPool


@dataclass
class FakeCtx:
    task: TaskMeta
    card_catalog: list
    tool_catalog: list
    budget_card: str
    history: list = field(default_factory=list)


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.seen = None

    def complete(self, messages, **params):
        self.seen = messages
        return ChatResponse(content=self.content, tokens_in=20, tokens_out=8, dollars=0.004)


def _ctx():
    pool = WorkerPool.from_spec(DEFAULT_POOL_SPEC)
    meta = TaskMeta(adp_source="s", task_id="t", domain="d", prompt="do X", verifiable=True)
    return FakeCtx(task=meta, card_catalog=pool.card_catalog(), tool_catalog=[{"name": "web_search"}], budget_card="low")


def test_teacher_proposes_parsed_action_with_cost():
    client = FakeClient('{"action":"call_worker","worker":"local-fast","sub_prompt":"go"}')
    step = TeacherBrain(client).propose(_ctx())
    assert isinstance(step, BrainStep)
    assert step.action.type == "call_worker"
    assert step.action.target == "local-fast"
    assert step.dollars == 0.004


def test_messages_include_anonymized_cards_not_model_ids():
    client = FakeClient('{"action":"finish","answer":"ok"}')
    TeacherBrain(client).propose(_ctx())
    blob = str(client.seen)
    assert "local-fast" in blob  # alias visible
    assert "Qwen/Qwen3.5-9B" not in blob  # real model id NEVER shown to the brain
    assert "do X" in blob  # task prompt present


def test_build_messages_has_system_and_budget():
    msgs = build_messages(_ctx(), system_prompt="SYS")
    assert msgs[0]["role"] == "system"
    assert "low" in str(msgs)  # budget card surfaced


def test_teacher_direct_answer_becomes_finish():
    # A strong teacher that ignores the action grammar and answers directly must still yield a
    # usable trace: the raw answer is wrapped as a `finish` action (tier-1 self-answer).
    step = TeacherBrain(FakeClient("55")).propose(_ctx())
    assert step.action.type == "finish"
    assert step.action.text == "55"

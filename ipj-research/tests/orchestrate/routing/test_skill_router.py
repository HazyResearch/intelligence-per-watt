from __future__ import annotations

from ipj.orchestrate.routing.skill_router import (
    COMPETENCE,
    SKILL_CATALOG,
    _extract_json,
    classify_task,
    route_task,
)


def _fake_call(content: str, *, tin: int = 50, tout: int = 20):
    def call_fn(url, headers, body):
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": tin, "completion_tokens": tout}}
    return call_fn


def test_competence_keys_match_catalog():
    assert tuple(COMPETENCE.keys()) == SKILL_CATALOG
    assert all(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 for lo, hi in COMPETENCE.values())


def test_classify_task_parses_and_clamps_weights():
    cf = _fake_call('{"bash_scripting": 0.9, "debugging": 1.7, "nonsense": 1.0}')
    weights, tin, tout = classify_task("do x", base_url="http://x", api_key="k", model="m", call_fn=cf)
    assert weights["bash_scripting"] == 0.9
    assert weights["debugging"] == 1.0          # clamped to [0,1]
    assert weights["multi_step_reasoning"] == 0.0  # missing -> 0
    assert "nonsense" not in weights             # restricted to the catalog
    assert (tin, tout) == (50, 20)


def test_classify_task_missing_usage_is_unknown():
    def call_fn(url, headers, body):
        return {"choices": [{"message": {"content": '{"bash_scripting": 1.0}'}}]}

    _, tin, tout = classify_task("do x", base_url="http://x", api_key="k", model="m", call_fn=call_fn)
    assert tin is None and tout is None


def test_route_task_routes_local_for_routine_task():
    cf = _fake_call('{"bash_scripting": 1.0, "file_navigation": 1.0}')
    d = route_task("ls and grep", base_url="x", api_key="k", model="m", margin=0.8, call_fn=cf)
    assert d.tier == "local"
    assert d.local_score > 0 and d.cloud_score > 0


def test_route_task_routes_cloud_for_hard_task():
    cf = _fake_call('{"debugging": 1.0, "multi_step_reasoning": 1.0}')
    d = route_task("find subtle race", base_url="x", api_key="k", model="m", margin=0.8, call_fn=cf)
    assert d.tier == "cloud"


def test_route_task_margin_is_the_knob():
    cf = _fake_call('{"system_administration": 1.0, "networking": 1.0}')
    assert route_task("net cfg", base_url="x", api_key="k", model="m", margin=0.5, call_fn=cf).tier == "local"
    assert route_task("net cfg", base_url="x", api_key="k", model="m", margin=1.0, call_fn=cf).tier == "cloud"


def test_route_task_prices_classifier_tokens():
    cf = _fake_call('{"bash_scripting": 1.0}', tin=1000, tout=500)
    d = route_task("x", base_url="x", api_key="k", model="m", price_in=1.0, price_out=2.0, call_fn=cf)
    assert d.classifier_tokens == 1500
    assert d.classifier_dollars == (1000 / 1e6) * 1.0 + (500 / 1e6) * 2.0


def test_route_task_unknown_price_does_not_become_free():
    d = route_task(
        "x",
        base_url="x",
        api_key="k",
        model="m",
        call_fn=_fake_call('{"bash_scripting": 1.0}'),
    )
    assert d.classifier_tokens == 70
    assert d.classifier_dollars is None


def test_route_task_explicit_zero_prices_remain_known_zero():
    d = route_task(
        "x",
        base_url="x",
        api_key="k",
        model="m",
        price_in=0.0,
        price_out=0.0,
        call_fn=_fake_call('{"bash_scripting": 1.0}'),
    )
    assert d.classifier_dollars == 0.0


def test_extract_json_handles_fenced_output():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json("garbage") == {}

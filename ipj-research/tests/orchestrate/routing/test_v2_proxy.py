from __future__ import annotations

import pytest
import requests

from ipj.orchestrate.routing.v2_proxy import (
    ProxyServer,
    RoutingState,
    _normalize_cloud_req,
    handle_chat,
    task_key,
)


class TestNormalizeCloudReq:
    def test_gpt5_renames_max_tokens(self):
        out = _normalize_cloud_req({"max_tokens": 500, "model": "gpt-5.5-2026-04-23"}, "gpt-5.5-2026-04-23")
        assert out["max_completion_tokens"] == 500 and "max_tokens" not in out

    def test_gpt5_drops_nondefault_temperature(self):
        out = _normalize_cloud_req({"temperature": 0.0, "model": "gpt-5"}, "gpt-5")
        assert "temperature" not in out

    def test_gpt5_keeps_default_temperature(self):
        out = _normalize_cloud_req({"temperature": 1.0, "model": "gpt-5"}, "gpt-5")
        assert out["temperature"] == 1.0

    def test_non_openai_model_untouched(self):
        body = {"max_tokens": 500, "temperature": 0.0, "model": "gemini-3.1-pro-preview"}
        assert _normalize_cloud_req(dict(body), "gemini-3.1-pro-preview") == body

    def test_handle_chat_normalizes_cloud_forward(self):
        st = RoutingState(local_turn_budget=0)  # everything escalates immediately
        seen = {}

        def fwd(url, headers, body):
            seen["body"] = body
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        ups = {"local": ("http://l/v1", "E"), "cloud": ("http://c/v1", "K", "gpt-5.5-2026-04-23")}
        handle_chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 99},
                    st, ups, fwd)
        assert seen["body"]["max_completion_tokens"] == 99 and "max_tokens" not in seen["body"]


def test_route_local_until_budget_then_sticky_cloud():
    st = RoutingState(local_turn_budget=2)
    assert [st.route("t1") for _ in range(4)] == ["local", "local", "cloud", "cloud"]
    assert st.route("t2") == "local"          # a different task starts fresh
    assert st.n_escalated == 1                # only t1 escalated


def test_task_key_stable_across_growth_and_distinct_across_tasks():
    a1 = [{"role": "user", "content": "task A"}]
    a2 = [{"role": "user", "content": "task A"}, {"role": "assistant", "content": "x"}]
    b = [{"role": "user", "content": "task B"}]
    assert task_key(a1) == task_key(a2)       # same messages[0] -> same key
    assert task_key(a1) != task_key(b)
    assert task_key([]) == task_key([])       # empty is stable, not a crash


def test_record_tallies_per_tier():
    st = RoutingState()
    st.record("local", 100, 50)
    st.record("cloud", 10, 5)
    assert st.tokens["local"] == {"in": 100, "out": 50}
    assert st.tokens["cloud"] == {"in": 10, "out": 5}
    assert st.requests == {"local": 1, "cloud": 1}


def test_record_missing_usage_keeps_request_count_and_makes_tokens_unknown():
    st = RoutingState()
    st.record("cloud", None, None)
    st.record("cloud", 10, 5)

    assert st.requests["cloud"] == 2
    assert st.tokens["cloud"] == {"in": None, "out": None}


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "3"])
def test_record_invalid_usage_is_unknown(invalid):
    st = RoutingState()
    st.record("cloud", invalid, 0)

    assert st.requests["cloud"] == 1
    assert st.tokens["cloud"] == {"in": None, "out": 0}


def test_handle_chat_preserves_known_zero_usage():
    st = RoutingState(local_turn_budget=0)
    ups = {"local": ("http://l/v1", "E"), "cloud": ("http://c/v1", "K", "m")}

    handle_chat(
        {"messages": [{"role": "user", "content": "x"}]},
        st,
        ups,
        lambda *args: {"usage": {"prompt_tokens": 0, "completion_tokens": 0}},
    )

    assert st.requests["cloud"] == 1
    assert st.tokens["cloud"] == {"in": 0, "out": 0}


def test_handle_chat_missing_usage_is_not_recorded_as_zero():
    st = RoutingState(local_turn_budget=0)
    ups = {"local": ("http://l/v1", "E"), "cloud": ("http://c/v1", "K", "m")}

    handle_chat(
        {"messages": [{"role": "user", "content": "x"}]},
        st,
        ups,
        lambda *args: {"choices": [{"message": {"content": "ok"}}]},
    )

    assert st.requests["cloud"] == 1
    assert st.tokens["cloud"] == {"in": None, "out": None}


def test_handle_chat_routes_local_then_cloud_and_rewrites_model():
    st = RoutingState(local_turn_budget=1)
    calls = []

    def fake_forward(url, headers, body):
        calls.append((url, body["model"], headers["Authorization"]))
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    ups = {"local": ("http://local/v1", "EMPTY"), "cloud": ("http://cloud/v1", "sk-x", "gpt-5.5")}
    base = {"model": "Qwen", "messages": [{"role": "user", "content": "task A"}]}

    handle_chat(base, st, ups, fake_forward)  # 1st turn of task A -> local
    grown = {"model": "Qwen", "messages": [{"role": "user", "content": "task A"},
                                           {"role": "assistant", "content": "a"}]}
    handle_chat(grown, st, ups, fake_forward)  # 2nd turn of same task -> cloud (budget=1)

    assert calls[0] == ("http://local/v1/chat/completions", "Qwen", "Bearer EMPTY")
    assert calls[1] == ("http://cloud/v1/chat/completions", "gpt-5.5", "Bearer sk-x")
    assert st.tokens["local"] == {"in": 10, "out": 5}
    assert st.tokens["cloud"] == {"in": 10, "out": 5}


@pytest.mark.integration
def test_proxy_server_loopback_routes_local_then_cloud():
    st = RoutingState(local_turn_budget=1)
    seen = []

    def fake_forward(url, headers, body):
        seen.append(url)
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    ups = {"local": ("http://local/v1", "EMPTY"), "cloud": ("http://cloud/v1", "k", "gpt-5.5")}
    srv = ProxyServer(st, ups, forward_fn=fake_forward)
    port = srv.start()
    try:
        body = {"model": "Qwen", "messages": [{"role": "user", "content": "A"}]}
        r1 = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=10)
        requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=10)  # triggers cloud route
        assert r1.status_code == 200
        assert r1.json()["choices"][0]["message"]["content"] == "ok"
        assert seen[0].startswith("http://local") and seen[1].startswith("http://cloud")
    finally:
        srv.stop()


def test_force_tier_overrides_turn_budget():
    from ipj.orchestrate.routing.v2_proxy import RoutingState
    st = RoutingState(local_turn_budget=1)
    st.force_tier("k", "local", classifier_tokens=70, classifier_dollars=0.001)
    assert st.is_routed("k")
    # Past the budget, a forced-local task stays local.
    assert [st.route("k") for _ in range(3)] == ["local", "local", "local"]
    assert st.classifier_tokens == 70
    assert st.classifier_dollars == 0.001
    assert (st.n_local, st.n_cloud) == (1, 0)


def test_unknown_classifier_cost_is_sticky():
    st = RoutingState()
    st.force_tier("a", "local", classifier_tokens=None, classifier_dollars=None)
    st.force_tier("b", "cloud", classifier_tokens=10, classifier_dollars=0.0)

    assert st.classifier_tokens is None
    assert st.classifier_dollars is None


def test_unforced_task_still_uses_budget():
    from ipj.orchestrate.routing.v2_proxy import RoutingState
    st = RoutingState(local_turn_budget=1)
    assert st.route("k") == "local"   # turn 1
    assert st.route("k") == "cloud"   # turn 2 > budget
    assert not st.is_routed("k")
    assert (st.n_local, st.n_cloud) == (0, 0)


def test_handle_chat_classifies_once_per_task(monkeypatch):
    from ipj.orchestrate.routing import skill_router
    from ipj.orchestrate.routing.skill_router import RouteDecision
    from ipj.orchestrate.routing.v2_proxy import RoutingState, handle_chat

    calls = {"n": 0}

    def fake_route_task(task_text, **kw):
        calls["n"] += 1
        return RouteDecision(tier="cloud", weights={}, local_score=0.0, cloud_score=1.0,
                             classifier_tokens=30, classifier_dollars=0.002)

    monkeypatch.setattr(skill_router, "route_task", fake_route_task)
    st = RoutingState()
    upstreams = {"local": ("http://l", "k"), "cloud": ("http://c", "k", "cloud-model")}

    def fwd(url, headers, body):
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    router = {"base_url": "http://r", "api_key": "k", "model": "m",
              "margin": 0.8, "price_in": 0.0, "price_out": 0.0}
    body = {"messages": [{"role": "system", "content": "task X"}]}
    for _ in range(3):
        handle_chat(body, st, upstreams, fwd, router=router)
    assert calls["n"] == 1                     # classified exactly once
    assert st.n_cloud == 1
    assert st.classifier_tokens == 30


def test_handle_chat_router_failure_falls_back_to_budget(monkeypatch):
    from ipj.orchestrate.routing import skill_router
    from ipj.orchestrate.routing.v2_proxy import RoutingState, handle_chat, task_key

    def boom(task_text, **kw):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(skill_router, "route_task", boom)
    st = RoutingState(local_turn_budget=15)
    upstreams = {"local": ("http://l", "k"), "cloud": ("http://c", "k", "cloud-model")}

    def fwd(url, headers, body):
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    router = {"base_url": "http://r", "api_key": "k", "model": "m",
              "margin": 0.8, "price_in": 0.0, "price_out": 0.0}
    messages = [{"role": "system", "content": "t"}]
    handle_chat({"messages": messages}, st, upstreams, fwd, router=router)
    assert not st.is_routed(task_key(messages))  # no force on failure -> budget path
    assert st.classifier_tokens is None
    assert st.classifier_dollars is None

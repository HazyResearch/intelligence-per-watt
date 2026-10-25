from __future__ import annotations

from ipj.orchestrate.routing.deferral import DeferralDecision, assess_deferral, parse_decision
from ipj.orchestrate.routing.v2_proxy import RoutingState, handle_chat


def _resp(content, pin=5, pout=2):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": pin, "completion_tokens": pout}}


class TestParseDecision:
    def test_defer(self):
        assert parse_decision("DEFER") is True
        assert parse_decision("defer") is True
        assert parse_decision("DEFER - I am stuck in a loop") is True

    def test_continue(self):
        assert parse_decision("CONTINUE") is False
        assert parse_decision("continue working on it") is False

    def test_empty_or_ambiguous_is_continue(self):
        # fail-safe: never escalate on a broken/empty/ambiguous probe (cost protection)
        assert parse_decision("") is False
        assert parse_decision("   ") is False
        assert parse_decision("maybe CONTINUE or DEFER, unsure") is False


class TestAssessDeferral:
    def test_defer_response_and_probe_shape(self):
        calls = []

        def fwd(url, headers, body):
            calls.append(body)
            return _resp("DEFER")

        d = assess_deferral([{"role": "user", "content": "task"}],
                            base_url="http://x/v1", api_key="k", model="m", forward_fn=fwd)
        assert d.defer is True and d.tokens_in == 5 and d.tokens_out == 2
        last = calls[0]["messages"][-1]
        assert last["role"] == "user" and "DEFER" in last["content"]
        assert calls[0]["max_tokens"] == 8 and calls[0]["temperature"] == 0.0
        # validated on Qwen3.6: without this the reasoning model never emits a verdict in 8 tokens
        assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_can_be_left_on(self):
        calls = []

        def fwd(url, headers, body):
            calls.append(body)
            return _resp("CONTINUE")

        assess_deferral([{"role": "user", "content": "t"}], base_url="http://x/v1",
                        api_key="k", model="m", forward_fn=fwd, disable_thinking=False)
        assert "chat_template_kwargs" not in calls[0]

    def test_continue_response(self):
        d = assess_deferral([{"role": "user", "content": "t"}], base_url="http://x/v1",
                            api_key="k", model="m", forward_fn=lambda *a: _resp("CONTINUE"))
        assert d.defer is False

    def test_probe_error_fails_open(self):
        def boom(*a):
            raise RuntimeError("serve down")
        d = assess_deferral([{"role": "user", "content": "t"}], base_url="http://x/v1",
                            api_key="k", model="m", forward_fn=boom)
        assert d.defer is False  # never escalate on a broken probe


class TestProxyDeferralIntegration:
    def _setup(self, decisions):
        state = RoutingState(local_turn_budget=999, defer_every_k=2)  # high budget -> only deferral routes
        upstreams = {"local": ("http://local/v1", "EMPTY"), "cloud": ("http://cloud/v1", "K", "gpt-5.5")}
        forwarded = []

        def fwd(url, headers, body):
            forwarded.append(url)
            return _resp("ok")

        it = iter(decisions)

        def deferral(messages):
            return DeferralDecision(defer=next(it), tokens_in=3, tokens_out=1, raw="x")

        return state, upstreams, fwd, deferral, forwarded

    def test_defer_at_turn2_latches_to_cloud(self):
        state, up, fwd, deferral, fwded = self._setup([True])
        body = {"messages": [{"role": "user", "content": "task A"}]}
        for _ in range(3):
            handle_chat(dict(body), state, up, fwd, deferral=deferral)
        # turn1 local (no check), turn2 DEFER->cloud, turn3 latched cloud
        assert fwded == ["http://local/v1/chat/completions",
                         "http://cloud/v1/chat/completions",
                         "http://cloud/v1/chat/completions"]
        assert state.n_deferred == 1 and len(state.deferrals) == 1

    def test_continue_stays_local(self):
        state, up, fwd, deferral, fwded = self._setup([False, False])
        body = {"messages": [{"role": "user", "content": "task B"}]}
        for _ in range(4):  # self-checks at turn 2 and 4, both CONTINUE
            handle_chat(dict(body), state, up, fwd, deferral=deferral)
        assert all("local" in u for u in fwded)
        assert state.n_deferred == 0 and len(state.deferrals) == 2  # 2 probes, neither deferred

    def test_no_deferral_when_disabled(self):
        state = RoutingState(local_turn_budget=999, defer_every_k=0)  # disabled
        up = {"local": ("http://local/v1", "EMPTY"), "cloud": ("http://cloud/v1", "K", "m")}
        fwded = []

        def fwd(url, headers, body):
            fwded.append(url)
            return _resp("ok")

        def deferral(m):
            raise AssertionError("deferral must not be called when defer_every_k=0")

        body = {"messages": [{"role": "user", "content": "t"}]}
        for _ in range(4):
            handle_chat(dict(body), state, up, fwd, deferral=deferral)
        assert all("local" in u for u in fwded)

    def test_probe_tokens_count_as_classifier_overhead(self):
        state, up, fwd, deferral, _ = self._setup([False])
        body = {"messages": [{"role": "user", "content": "t"}]}
        for _ in range(2):  # one check at turn 2
            handle_chat(dict(body), state, up, fwd, deferral=deferral)
        assert state.classifier_tokens == 4  # 3 in + 1 out from the probe

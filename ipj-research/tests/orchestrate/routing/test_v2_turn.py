from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipj.cost import H100_HOURLY_USD, dollars_local, dollars_tokens
from ipj.harbor.run import RunWindow
from ipj.orchestrate.routing.v2_turn import run_v2


def _write_result(dirpath: Path, resolved, n_tasks, accuracy):
    # Write JSON matching parse_harbor_summary's schema (Harbor 0.13.2 format):
    #   n_total_trials at the top level, accuracy under stats.evals[key].metrics[0].mean
    job = dirpath / "job"
    job.mkdir(parents=True, exist_ok=True)
    evals = {
        "e": {
            "metrics": [{"mean": accuracy}],
            "reward_stats": {"reward": {"1.0": [f"{t}__0" for t in resolved]}},
        }
    }
    (job / "result.json").write_text(
        json.dumps({"n_total_trials": n_tasks, "stats": {"evals": evals, "n_input_tokens": 0, "n_output_tokens": 0}})
    )


class _FakeCap:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def joules(self, start, end): return 4000.0
    def accelerator_device_coverage_complete(self, start, end): return True


class _FakeProxy:
    """Simulates a run: pre-populates the routing state with per-tier tokens + one escalation."""
    def __init__(self, state, upstreams, **kw):
        state.record("local", 1000, 500)
        state.record("cloud", 200, 100)
        state._task["t1"] = {"count": 99, "escalated": True}
        self.state = state

    def start(self, host="127.0.0.1", port=0): return 9999
    def stop(self): pass


def test_run_v2_builds_cell_from_proxy_tallies(tmp_path: Path):
    def fake_run(cmd, *, cwd=None, env=None):
        assert "127.0.0.1:9999" in env["OPENAI_BASE_URL"]   # harbor routed through the proxy
        _write_result(tmp_path, resolved=["t1", "t2"], n_tasks=3, accuracy=2 / 3)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v2(
        local_model="Qwen/Qwen3.6-27B-FP8", cloud_model="gpt-5.5", dataset="ds",
        output_dir=tmp_path, local_base_url="http://l/v1", cloud_base_url="http://c/v1",
        cloud_api_key="k", n_tasks=3, cloud_price_in=1.25, cloud_price_out=10.0,
        energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=_FakeProxy,
    )

    assert res.cell.accuracy == pytest.approx(2 / 3)
    assert res.cell.tokens_in == 1200 and res.cell.tokens_out == 600   # local + cloud
    assert res.n_escalated == 1
    assert res.cell.joules == pytest.approx(4000.0)  # measured local only
    assert res.cell.cloud_joules_estimate is not None
    assert res.cell.cloud_joules_estimate.joules == pytest.approx(800.0)
    assert res.cell.cloud_joules_estimate.lo == pytest.approx(560.0)
    assert res.cell.cloud_joules_estimate.hi == pytest.approx(1040.0)
    assert res.cell.joules_provenance == "measured_dedicated"
    assert res.cell.joules_scope == "local-accelerator"
    assert res.cell.joules_complete is False
    # Fix vs brief: dollars_local requires gpu_hourly_rate as a positional arg (no default).
    # run_v2 defaults to H100_HOURLY_USD, so the assertion must pass that explicitly.
    expected_dollars = dollars_local(10.0, H100_HOURLY_USD) + dollars_tokens(200, 100, 1.25, 10.0)
    assert res.cell.dollars == pytest.approx(expected_dollars)
    assert res.cell.dollars_basis == "gpu-hours+api"


def test_run_v2_threads_n_concurrent_into_harbor_cmd(tmp_path):
    captured = {}

    def fake_run(cmd, *, cwd=None, env=None):
        captured["cmd"] = cmd
        _write_result(tmp_path, resolved=["t1"], n_tasks=2, accuracy=0.5)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    run_v2(
        local_model="Qwen/Qwen3.6-27B-FP8", cloud_model="gpt-5.5", dataset="ds",
        output_dir=tmp_path, local_base_url="http://l/v1", cloud_base_url="http://c/v1",
        cloud_api_key="k", cloud_price_in=0.0, cloud_price_out=0.0,
        n_tasks=2, n_concurrent=8,
        run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=_FakeProxy,
    )
    flat = " ".join(captured["cmd"])
    assert "8" in captured["cmd"] and "--n-concurrent" in flat


def test_run_v2_does_not_turn_missing_local_telemetry_into_zero(tmp_path):
    class UnavailableCap(_FakeCap):
        def joules(self, start, end):
            return None

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v2(
        local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
        local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
        cloud_price_in=0.0, cloud_price_out=0.0,
        energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=UnavailableCap, proxy_factory=_FakeProxy,
    )

    assert res.cell.joules is None
    assert res.cell.cloud_joules_estimate is None
    assert res.cell.joules_provenance == "unknown"
    assert res.cell.joules_complete is False


def test_run_v2_threads_router_and_folds_classifier_cost(tmp_path):
    """Router config is passed to proxy_factory; classifier $ accumulates in RoutingState
    and is folded into cell.dollars + surfaced on V2Result."""
    from ipj.orchestrate.routing import v2_turn
    from ipj.orchestrate.routing.v2_proxy import RoutingState  # noqa: F401

    # Use the same fixture shape as the existing test — result.json lives in a subdir.
    _write_result(tmp_path, resolved=["t1", "t2"], n_tasks=2, accuracy=0.5)

    captured = {}

    class FakeProxy:
        def __init__(self, state, upstreams, router=None, deferral=None):
            captured["router"] = router
            captured["deferral"] = deferral
            self.state = state
            # Simulate pre-task classifier forcing one local + one cloud task, each costing $0.01.
            state.force_tier("a", "local", classifier_tokens=40, classifier_dollars=0.01)
            state.force_tier("b", "cloud", classifier_tokens=40, classifier_dollars=0.01)

        def start(self, host="127.0.0.1", port=0):
            return 12345

        def stop(self):
            pass

    class _Win:
        start, end, returncode = 0.0, 1.0, 0

    class FakeCap:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def joules(self, s, e):
            return 100.0

    def fake_run(cmd, *, cwd=None, env=None):
        return _Win()

    res = v2_turn.run_v2(
        local_model="Qwen/Q", cloud_model="gpt-5.5", dataset="d", output_dir=tmp_path,
        local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
        cloud_price_in=0.0, cloud_price_out=0.0,
        pre_route=True, router_model="gpt-5-mini", router_margin=0.8,
        run_fn=fake_run, capture_factory=FakeCap, proxy_factory=FakeProxy,
    )

    # Router config was threaded through to the proxy factory.
    assert captured["router"]["model"] == "gpt-5-mini"
    assert captured["router"]["margin"] == pytest.approx(0.8)

    # Routing counts come from RoutingState.force_tier calls above.
    assert res.n_local == 1
    assert res.n_cloud == 1

    # Classifier costs accumulated correctly (2 × $0.01).
    assert res.classifier_dollars == pytest.approx(0.02)
    assert res.cell.classifier_dollars == pytest.approx(0.02)

    # Classifier $ is folded into cell.dollars (total >= classifier cost alone).
    assert res.cell.dollars >= 0.02
    assert res.cell.joules_provenance == "unknown"
    assert res.cell.joules_complete is False


def test_run_v2_rejects_used_cloud_with_unknown_default_prices(tmp_path):
    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    with pytest.raises(ValueError, match="unknown cost.*cloud API"):
        run_v2(
            local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
            local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
            run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=_FakeProxy,
        )


def test_run_v2_unused_cloud_is_known_zero_with_unknown_prices(tmp_path):
    class LocalOnlyProxy:
        def __init__(self, state, upstreams, **kw):
            state.record("local", 100, 50)

        def start(self, host="127.0.0.1", port=0):
            return 9999

        def stop(self):
            pass

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v2(
        local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
        local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
        energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=LocalOnlyProxy,
    )

    assert res.cell.dollars == pytest.approx(dollars_local(10.0, H100_HOURLY_USD))
    assert res.cell.joules_complete is True


def test_cloud_pre_router_invalidates_local_worker_energy_completeness(tmp_path):
    class PreRoutedLocalProxy:
        def __init__(self, state, upstreams, **kw):
            state.force_tier(
                "local-task",
                "local",
                classifier_tokens=10,
                classifier_dollars=0.0,
            )
            state.record("local", 100, 50)

        def start(self, host="127.0.0.1", port=0):
            return 9999

        def stop(self):
            pass

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v2(
        local_model="local",
        cloud_model="cloud",
        dataset="ds",
        output_dir=tmp_path,
        local_base_url="http://l/v1",
        cloud_base_url="http://c/v1",
        cloud_api_key="k",
        pre_route=True,
        cloud_price_in=0.0,
        cloud_price_out=0.0,
        energy_provenance="measured_dedicated",
        run_fn=fake_run,
        capture_factory=_FakeCap,
        proxy_factory=PreRoutedLocalProxy,
    )

    assert res.n_cloud == 0
    assert res.cell.joules == pytest.approx(4000.0)
    assert res.cell.joules_complete is False


def test_run_v2_rejects_missing_cloud_usage_even_with_known_prices(tmp_path):
    class MissingUsageProxy:
        def __init__(self, state, upstreams, **kw):
            state.record("local", 100, 50)
            state.record("cloud", None, None)

        def start(self, host="127.0.0.1", port=0):
            return 9999

        def stop(self):
            pass

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    with pytest.raises(ValueError, match="unknown cost.*cloud API"):
        run_v2(
            local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
            local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
            cloud_price_in=1.0, cloud_price_out=2.0,
            run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=MissingUsageProxy,
        )


@pytest.mark.parametrize("cloud_signal", ["request", "forced-tier"])
def test_run_v2_zero_token_cloud_use_still_makes_energy_incomplete(tmp_path, cloud_signal):
    class ZeroTokenCloudProxy:
        def __init__(self, state, upstreams, **kw):
            if cloud_signal == "request":
                state.record("cloud", 0, 0)
            else:
                state.force_tier("cloud-task", "cloud")

        def start(self, host="127.0.0.1", port=0):
            return 9999

        def stop(self):
            pass

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v2(
        local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
        local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
        cloud_price_in=0.0, cloud_price_out=0.0, energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=ZeroTokenCloudProxy,
    )

    assert res.cell.dollars == pytest.approx(dollars_local(10.0, H100_HOURLY_USD))
    assert res.cell.dollars_basis == "gpu-hours+api"
    assert res.cell.joules_complete is False


def test_run_v2_rejects_unknown_classifier_cost(tmp_path):
    class UnknownClassifierProxy:
        def __init__(self, state, upstreams, **kw):
            state.record("local", 100, 50)
            state.force_tier("local-task", "local", classifier_tokens=10, classifier_dollars=None)

        def start(self, host="127.0.0.1", port=0):
            return 9999

        def stop(self):
            pass

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(tmp_path, resolved=["t1"], n_tasks=1, accuracy=1.0)
        return RunWindow(start=0.0, end=10.0, returncode=0)

    with pytest.raises(ValueError, match="unknown cost.*router classifier"):
        run_v2(
            local_model="local", cloud_model="cloud", dataset="ds", output_dir=tmp_path,
            local_base_url="http://l/v1", cloud_base_url="http://c/v1", cloud_api_key="k",
            run_fn=fake_run, capture_factory=_FakeCap, proxy_factory=UnknownClassifierProxy,
        )

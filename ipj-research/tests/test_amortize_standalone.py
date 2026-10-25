from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

# load the standalone amortizer by path (it lives under ipj/experiments, not the package)
_SA = Path(__file__).resolve().parents[1] / "experiments" / "fast-rerun" / "amortize_standalone.py"
_spec = importlib.util.spec_from_file_location("amortize_standalone", _SA)
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_run(tmp_path, n=100, dur=1.0, power_w=700.0, resolve_every=3):
    jobs = tmp_path / "jobs" / "2026-06-20__00-00-00"
    t0 = 1781990000.0
    for i in range(1, n + 1):
        d = jobs / f"task{i:03d}__h{i}"
        d.mkdir(parents=True, exist_ok=True)
        end = t0 + i
        json.dump({
            "started_at": _iso(end - dur), "finished_at": _iso(end),
            "agent_result": {"n_input_tokens": 100, "n_output_tokens": 50},
            "verifier_result": {"rewards": {"reward": 1.0 if i % resolve_every == 0 else 0.0}},
        }, open(d / "result.json", "w"))
    power = tmp_path / "power.txt"
    with open(power, "w") as f:
        for t in range(0, n + 2):
            f.write(f"{t0 + t} {power_w}\n")
    return str(tmp_path / "jobs"), str(power)


def test_load_events_parses_harbor_schema(tmp_path):
    jobs, _ = _make_run(tmp_path, n=10)
    evs = sa.load_events(jobs)
    assert len(evs) == 10
    assert sum(e["resolved"] for e in evs) == 3            # i=3,6,9
    assert all(e["tin"] == 100 and e["tout"] == 50 for e in evs)


def test_p50_central_window_constant_power(tmp_path):
    jobs, power = _make_run(tmp_path, n=100, power_w=700.0)
    m = sa.window(sa.load_events(jobs), sa.load_power(power), 50)
    assert m["n"] == 50                                     # central 50 of 100
    assert m["window_s"] == pytest.approx(50.0)             # ends t=26..75 -> [25,75]
    assert m["avg_power_w"] == pytest.approx(700.0, rel=1e-3)
    assert m["energy_per_query_j"] == pytest.approx(700.0 * 50.0 / 50.0, rel=1e-3)
    assert m["tokens_out_per_query"] == pytest.approx(50.0)


def test_window_widths(tmp_path):
    jobs, power = _make_run(tmp_path, n=100)
    evs, pw = sa.load_events(jobs), sa.load_power(power)
    assert [sa.window(evs, pw, p)["n"] for p in (100, 90, 80, 50)] == [100, 90, 80, 50]


def test_dollars_amortized_over_window(tmp_path):
    jobs, power = _make_run(tmp_path, n=100, power_w=700.0)
    m = sa.window(sa.load_events(jobs), sa.load_power(power), 50, gpu_hourly_rate=1.10, n_gpus=1)
    assert m["dollars"] == pytest.approx((50.0 / 3600.0) * 1.10, rel=1e-3)
    assert m["dollars_per_query"] == pytest.approx(m["dollars"] / 50, rel=1e-3)


def test_matches_canonical_percentile_amortize(tmp_path):
    """Guard against drift from ipj/src/ipj/percentile_amortize.py on the same events."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ipj.percentile_amortize import QueryEvent, window_amortize
    except Exception:
        pytest.skip("canonical percentile_amortize not importable")
    jobs, power = _make_run(tmp_path, n=60, power_w=512.0)
    evs = sa.load_events(jobs)
    pw = sa.load_power(power)
    canon_events = [QueryEvent(start=e["start"], end=e["end"], tokens_in=e["tin"],
                               tokens_out=e["tout"], resolved=e["resolved"]) for e in evs]
    for p in (100, 90, 80, 50):
        a = sa.window(evs, pw, p)
        b = window_amortize(canon_events, pw, p)
        assert a["n"] == b.n_queries
        assert a["energy_per_query_j"] == pytest.approx(b.energy_per_query_j, rel=1e-6)
        assert a["avg_power_w"] == pytest.approx(b.avg_power_w, rel=1e-6)
        assert a["tokens_out_per_query"] == pytest.approx(b.tokens_out_per_query, rel=1e-6)


def test_cli_writes_metrics(tmp_path, capsys):
    jobs, power = _make_run(tmp_path, n=40)
    out = str(tmp_path / "metrics.json")
    import sys
    argv = sys.argv
    sys.argv = ["amortize_standalone.py", "--jobs-dir", jobs, "--power", power,
                "--gpu-hourly-rate", "1.1", "--label", "test", "--out", out]
    try:
        sa.main()
    finally:
        sys.argv = argv
    d = json.load(open(out))
    assert d["n"] == 40 and set(str(p) for p in d["windows"]) >= {"100", "50"} or set(d["windows"]) >= {100, 50}
    assert "P50" in capsys.readouterr().out

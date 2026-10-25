from __future__ import annotations

import json

from ipj.orchestrate.filter.pipeline import Stats, build_sft_dataset


def _write(tmp_path, trajs):
    d = tmp_path / "traces"
    d.mkdir()
    for i, t in enumerate(trajs):
        (d / f"{i}.json").write_text(json.dumps(t.to_dict()))
    return d


def test_pipeline_end_to_end(tmp_path, make_traj):
    # task A: two successful (cheap 0.2 kept, pricey 0.9 -> not_cost_optimal)
    # task B: one failed (-> failed). plus a duplicate of A-cheap (-> duplicate).
    a_cheap = make_traj("A", dollars=0.2, uniq=1)
    a_dup = make_traj("A", dollars=0.2, uniq=1)  # identical signature to a_cheap
    a_pricey = make_traj("A", dollars=0.9, uniq=2)
    b_fail = make_traj("B", success=False, uniq=3)
    traces = _write(tmp_path, [a_cheap, a_dup, a_pricey, b_fail])

    out = tmp_path / "out"
    stats = build_sft_dataset(traces, out)

    assert isinstance(stats, Stats)
    assert stats.n_loaded == 4
    assert stats.dropped["failed"] == 1
    assert stats.dropped["duplicate"] == 1
    assert stats.dropped["not_cost_optimal"] == 1
    assert stats.n_kept == 1  # only A-cheap survives
    lines = (out / "sft.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["metadata"]["task_id"] == "A"
    saved = json.loads((out / "stats.json").read_text())
    assert saved["n_kept"] == 1 and saved["strategy_distribution"] == {"self": 1}
    assert saved["dollars_known_n"] == 1
    assert saved["gpu_energy_complete_n"] == 0


def test_pipeline_surfaces_unknown_cost_drop(tmp_path, make_traj):
    trajectory = make_traj("A")
    trajectory.outcome.total_dollars = None
    traces = _write(tmp_path, [trajectory])
    stats = build_sft_dataset(traces, tmp_path / "out")
    assert stats.n_kept == 0
    assert stats.dropped["unknown_cost"] == 1
    assert stats.dollars_known_n == 0

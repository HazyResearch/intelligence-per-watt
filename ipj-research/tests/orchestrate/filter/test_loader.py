from __future__ import annotations

import json

from ipj.orchestrate.filter.loader import load_trajectories


def test_loads_all_valid_traces_skips_garbage(tmp_path, make_traj):
    (tmp_path / "a.json").write_text(json.dumps(make_traj("a", with_outcome=False).to_dict()))
    (tmp_path / "b.json").write_text(json.dumps(make_traj("b", with_outcome=False).to_dict()))
    (tmp_path / "broken.json").write_text("{not valid json")
    trajs = load_trajectories(tmp_path)
    assert sorted(t.task.task_id for t in trajs) == ["a", "b"]


def test_empty_dir_returns_empty(tmp_path):
    assert load_trajectories(tmp_path) == []

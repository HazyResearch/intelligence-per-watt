from __future__ import annotations

import json
from pathlib import Path

from ipj.harbor.per_task import per_task_outcomes, solve_matrix

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_summary.json"


def test_per_task_from_real_fixture():
    # Recorded oracle run: 1 resolved (write-compressor) + 4 errored (RuntimeError).
    by = {o.task: o for o in per_task_outcomes(FIXTURE)}
    assert by["write-compressor"].status == "resolved"
    assert by["write-compressor"].reward == 1.0
    assert by["kv-store-grpc"].status == "errored"
    assert by["kv-store-grpc"].reward is None
    assert sum(1 for o in per_task_outcomes(FIXTURE) if o.status == "errored") == 4


def test_resolved_unresolved_errored(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"n_total_trials": 3, "stats": {"evals": {"terminus-2__d": {
        "reward_stats": {"reward": {"1.0": ["a__x"], "0.0": ["b__y"]}},
        "exception_stats": {"TimeoutError": ["c__z"]},
    }}}}))
    outs = {o.task: o.status for o in per_task_outcomes(p)}
    assert outs == {"a": "resolved", "b": "unresolved", "c": "errored"}


def test_solve_matrix_across_models(tmp_path):
    def _run(name, resolved, unresolved):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"stats": {"evals": {"m__d": {
            "reward_stats": {"reward": {"1.0": [f"{t}__i" for t in resolved],
                                        "0.0": [f"{t}__i" for t in unresolved]}},
        }}}}))
        return str(p)

    m = solve_matrix({
        "qwen": _run("qwen", ["t1"], ["t2"]),
        "gpt": _run("gpt", ["t1", "t2"], []),
    })
    assert m["t1"] == {"qwen": "resolved", "gpt": "resolved"}
    assert m["t2"] == {"qwen": "unresolved", "gpt": "resolved"}

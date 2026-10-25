from __future__ import annotations

import json
from pathlib import Path

from ipj.harbor.results import HarborSummary, parse_harbor_summary

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_summary.json"


def test_parses_real_fixture():
    """Parse the recorded real fixture and assert the real values."""
    s = parse_harbor_summary(FIXTURE)
    assert isinstance(s, HarborSummary)
    assert s.n_tasks == 5
    assert s.accuracy == 0.2  # 1 of 5 trials rewarded in the recorded run
    assert s.tokens_in == 0 and s.tokens_out == 0  # oracle makes no model calls (null → 0)
    assert s.cost_usd is None  # oracle has no billed cost


def test_parses_model_run_with_tokens(tmp_path):
    """Synthetic MODEL run with tokens populated and non-oracle eval key."""
    p = tmp_path / "result.json"
    p.write_text(json.dumps({
        "n_total_trials": 5,
        "stats": {
            "evals": {"terminus-2__terminal-bench/terminal-bench-2-1": {"metrics": [{"mean": 0.6}]}},
            "n_input_tokens": 120000,
            "n_output_tokens": 18000,
            "cost_usd": 1.23,
        },
    }))
    s = parse_harbor_summary(p)
    assert s.n_tasks == 5
    assert s.accuracy == 0.6
    assert s.tokens_in == 120000
    assert s.tokens_out == 18000
    assert s.cost_usd == 1.23  # Harbor's billed $ is surfaced


def test_missing_evals_defaults_accuracy_zero(tmp_path):
    """Empty/missing evals → accuracy 0.0, n_tasks still read."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"n_total_trials": 3, "stats": {"evals": {}}}))
    s = parse_harbor_summary(p)
    assert s.n_tasks == 3
    assert s.accuracy == 0.0

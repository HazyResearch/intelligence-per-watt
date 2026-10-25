"""Unit tests for the router training-data writer (§7.2 / §7.5)."""

from __future__ import annotations

import json

import pytest

from ipj.router.dataset import (
    RouterRow,
    decision_for,
    read_classifier_table,
    to_sft_example,
    to_training_matrix,
    validate_row,
    write_classifier_table,
    write_sft_jsonl,
)

SOLVED_ROW = RouterRow(
    task_id="repo__abc123",
    task_text="Fix the failing test in module foo.",
    embedding=[0.1, 0.2, 0.3],
    local_solved=True,
    cloud_solved=True,
    cloud_cost=0.10,
)
FAILED_ROW = RouterRow(
    task_id="repo__def456",
    task_text="Implement the new CLI flag.",
    embedding=[0.4, 0.5, 0.6],
    local_solved=False,
    cloud_solved=True,
    cloud_cost=0.90,  # above the $0.225 cheap threshold
)


def test_classifier_table_round_trip(tmp_path):
    path = tmp_path / "clf.jsonl"
    rows = [SOLVED_ROW, FAILED_ROW]
    write_classifier_table(rows, path)
    got = read_classifier_table(path)
    assert got == rows


def test_to_training_matrix_unpacks_rows():
    X, y, tasks = to_training_matrix([SOLVED_ROW, FAILED_ROW])
    assert X == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert y == [True, False]
    assert tasks == ["repo__abc123", "repo__def456"]


def test_sft_loss_mask_invariant(tmp_path):
    path = tmp_path / "sft.jsonl"
    write_sft_jsonl([SOLVED_ROW, FAILED_ROW], path)
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    for ex in lines:
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]
        # only the assistant turn carries loss=True
        assert [m["loss"] for m in ex["messages"]] == [False, False, True]
        loss_true = [m for m in ex["messages"] if m["loss"]]
        assert len(loss_true) == 1 and loss_true[0]["role"] == "assistant"


def test_sft_assistant_decision_encodes_escalate_and_cost():
    ex = to_sft_example(SOLVED_ROW)
    decision = json.loads(ex["messages"][-1]["content"])
    assert set(decision) == {"escalate", "p_local_solve", "cheap_cloud_solvable"}
    # solved locally → do not escalate, p_local_solve = 1.0, cloud cheap (0.10 <= 0.225)
    assert decision == {"escalate": False, "p_local_solve": 1.0, "cheap_cloud_solvable": True}


def test_decision_for_failed_row_is_cost_aware():
    d = decision_for(FAILED_ROW)
    # local failed → escalate; cloud solved but $0.90 > $0.225 → not cheaply solvable
    assert d == {"escalate": True, "p_local_solve": 0.0, "cheap_cloud_solvable": False}


def test_decision_cheap_threshold_configurable():
    d = decision_for(FAILED_ROW, cheap_cloud_cost=1.0)
    assert d["cheap_cloud_solvable"] is True


def test_row_from_dict_round_trip():
    d = SOLVED_ROW.to_dict()
    assert RouterRow.from_dict(d) == SOLVED_ROW


def test_validate_row_missing_field_raises():
    d = SOLVED_ROW.to_dict()
    del d["cloud_cost"]
    with pytest.raises(ValueError, match="cloud_cost"):
        validate_row(d)


def test_validate_row_wrong_type_raises():
    d = SOLVED_ROW.to_dict()
    d["embedding"] = "not-a-list"
    with pytest.raises(ValueError, match="embedding"):
        validate_row(d)

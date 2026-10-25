from __future__ import annotations

from ipj.efficiency.failure_modes import EXCLUDED_MODES, MODEL_FAILURE_MODES, classify


def test_solved():
    assert classify(1.0, None) == "solved"

def test_capability_completed_but_wrong():
    assert classify(0.0, None) == "capability"

def test_timeout():
    assert classify(0.0, "AgentTimeoutError") == "timeout"

def test_infra_is_excluded():
    assert classify(0.0, "InternalServerError") == "infra"
    assert "infra" in EXCLUDED_MODES

def test_harness_is_excluded():
    assert classify(0.0, "VerifierTimeoutError") == "harness"
    assert classify(None, "CancelledError") == "harness"

def test_context_and_parsing():
    assert classify(0.0, "ContextLengthExceededError") == "context"
    assert classify(0.0, "ToolCallParseError") == "parsing"

def test_mode_sets_partition():
    assert MODEL_FAILURE_MODES == frozenset({"timeout", "capability", "parsing", "context"})
    assert EXCLUDED_MODES == frozenset({"infra", "harness"})

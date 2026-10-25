from __future__ import annotations

MODEL_FAILURE_MODES = frozenset({"timeout", "capability", "parsing", "context"})
EXCLUDED_MODES = frozenset({"infra", "harness"})


def classify(reward: float | None, exception_type: str | None) -> str:
    """Map a per-task outcome to a failure mode.

    solved        reward >= 0.999
    capability    completed, reward 0, no exception (the honest wrong answer)
    timeout       AgentTimeoutError (model too slow / stuck looping)
    context       request exceeded the serve's context window
    parsing       agent couldn't act on the output (malformed tool-calls)
    harness       verifier/cancellation (VerifierTimeoutError, CancelledError) -> excluded
    infra         any other exception (InternalServerError, connection, OOM) -> excluded / re-run
    """
    if reward is not None and reward >= 0.999:
        return "solved"
    et = (exception_type or "").lower()
    if not et:
        return "capability"
    if "agenttimeout" in et:
        return "timeout"
    if "verifiertimeout" in et or "cancel" in et:
        return "harness"
    if "context" in et:
        return "context"
    if "parse" in et or "parsing" in et or "toolcall" in et:
        return "parsing"
    return "infra"

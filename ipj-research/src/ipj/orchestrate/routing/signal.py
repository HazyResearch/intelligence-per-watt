"""Escalation-signal predicates for the v1/v2 cascade.

Both are `(status: str) -> bool`, suitable for `cascade_ladder(escalate_when=...)`.
The REALIZED router may only see the proxy signal (did the local run finish cleanly?);
the ORACLE may peek at the eval (resolved) to bound the ceiling.
"""

from __future__ import annotations


def escalate_when_realized(status: str) -> bool:
    """Realized v1 proxy: escalate only when the local run did not finish cleanly
    (errored / hit its turn budget). Never inspects resolved/unresolved (the hidden eval)."""
    return status == "errored"


def escalate_when_oracle(status: str) -> bool:
    """Oracle ceiling: escalate any task the tier did not resolve."""
    return status != "resolved"

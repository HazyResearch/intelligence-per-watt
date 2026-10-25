"""Build the two eval arms: 'trained' (adapter brain + full worker pool) vs 'base_solo' (base model, tools-only)."""

from __future__ import annotations

from typing import Any, Callable

from ipj.orchestrate.harness.brain import TeacherBrain, TrainedBrain
from ipj.orchestrate.harness.dispatch import WorkerDispatch
from ipj.orchestrate.harness.ipw_glue import mcp_server_factory
from ipj.orchestrate.harness.trained_client import LocalAdapterChatClient
from ipj.orchestrate.harness.workers import WorkerPool

ARMS = ("trained", "base_solo")


def build_arm(
    arm: str,
    base_model: str,
    adapter_dir: str | None,
    pool_spec: dict[str, Any],
    *,
    energy_probe: Callable[[float, float], "float | None"] | None = None,
    client_factory: Callable[..., Any] = LocalAdapterChatClient,
    server_factory: Callable[..., Any] = mcp_server_factory,
) -> tuple[Any, WorkerDispatch, WorkerPool]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choices: {ARMS}")
    if arm == "trained":
        client = client_factory(base_model, adapter_dir, energy_probe=energy_probe)
        brain: Any = TrainedBrain(client)
        pool = WorkerPool.from_spec(pool_spec)
    else:  # base_solo: base model, no adapter, NO workers (tools-only self-solve)
        client = client_factory(base_model, None, energy_probe=energy_probe)
        brain = TeacherBrain(client)
        pool = WorkerPool.from_spec({"workers": {}})
    return brain, WorkerDispatch(pool, server_factory=server_factory), pool

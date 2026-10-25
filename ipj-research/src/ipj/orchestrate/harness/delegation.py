"""Container-acting delegation: run a delegated worker as a sub-agent that ACTS in the task container.

When the orchestrator delegates the WHOLE task to a worker, that worker should not just return a
text answer over MCP — it should edit /testbed and run commands in the SAME container, exactly like
the local->cloud escalation in the cascade. ``make_container_delegate_runner`` builds a
``DelegateRunner`` that drives a worker brain through the standard action loop with a container-backed
tool surface, then rolls the worker's per-step cost/energy back up into a single ``WorkerResult``.
"""

from __future__ import annotations

import os

from .brain import ChatClient, TeacherBrain
from .clients import AnthropicChatClient, OpenAIChatClient
from .dispatch import DelegateRunner, WorkerDispatch, WorkerResult
from .loop import OrchestratorHarness
from .trace import TaskMeta, summarize_outcome
from .workers import WorkerBinding, WorkerPool

# A delegated worker owns the WHOLE task end to end inside the container. It uses only the
# self-acting actions (think/bash/finish) and must NOT sub-delegate — it never sees other workers,
# so call_worker/delegate_task are deliberately absent from its grammar.
WORKER_SYSTEM = (
    "You are a capable software engineer working directly inside the task's container. Solve the "
    "WHOLE task yourself by acting in the container: read and edit files under /testbed and run "
    "commands with bash, iterating until the task is done. Each turn, emit EXACTLY ONE JSON action "
    "object and nothing else, one of: "
    '{"action":"think","text":"..."} to reason privately, '
    '{"action":"bash","command":"..."} to run a shell command in the container, or '
    '{"action":"finish","answer":"..."} to report the final result once the task is complete. '
    "Do not ask for help or hand the task off — you own it from start to finish."
)


def chat_client_for_binding(binding: WorkerBinding) -> ChatClient:
    """Map a worker binding's provider to a concrete chat client (api keys come from env via the SDK)."""
    provider = binding.provider
    if provider == "openai":
        return OpenAIChatClient(binding.model_id)
    if provider == "anthropic":
        return AnthropicChatClient(binding.model_id)
    if provider == "openrouter":
        return OpenAIChatClient(
            binding.model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    if provider == "vllm":
        return OpenAIChatClient(binding.model_id, base_url=binding.base_url or "http://localhost:8000")
    raise ValueError(f"no chat client for provider {provider!r} (alias {binding.alias!r})")


def make_container_delegate_runner(session, *, max_turns: int = 20, brain_factory=None) -> DelegateRunner:
    """Build a ``DelegateRunner`` that runs a delegated worker as a sub-agent acting in ``session``.

    The returned ``runner(binding, task_prompt)`` drives a worker brain through the action loop with a
    container-backed tool surface (same TmuxSession), so the worker edits /testbed and runs commands
    in place. Per-step cost/energy/latency are summed into one ``WorkerResult``. Pass ``brain_factory``
    (binding -> brain) to inject a brain in tests; otherwise a ``TeacherBrain`` is built from the
    binding's provider with the worker system prompt.
    """

    def runner(binding: WorkerBinding, task_prompt: str) -> WorkerResult:
        # Lazy import to avoid an import cycle: generate.session_tools lives under the generate
        # package, which imports the harness — importing it at module scope would couple the two.
        from ..generate.session_tools import build_session_tool_surface

        brain = (
            brain_factory(binding)
            if brain_factory is not None
            else TeacherBrain(chat_client_for_binding(binding), system_prompt=WORKER_SYSTEM)
        )
        tools = build_session_tool_surface(session)
        # The worker never sub-delegates, so it gets an empty pool and a dispatch that is never used.
        empty_pool = WorkerPool.from_spec({"workers": {}})
        dispatch = WorkerDispatch(empty_pool, server_factory=lambda b: None)
        traj = OrchestratorHarness(
            brain,
            dispatch,
            tools,
            brain_name=f"worker:{binding.alias}",
            max_turns=max_turns,
        ).run(
            TaskMeta("delegate", "delegated", "swe", task_prompt, True),
            empty_pool,
        )
        steps = traj.steps
        rolled_up = summarize_outcome(steps, success=False, source="delegation_rollup")
        return WorkerResult(
            content=traj.final_answer or "",
            tokens_in=rolled_up.total_tokens_in,
            tokens_out=rolled_up.total_tokens_out,
            dollars=rolled_up.total_dollars,
            latency_s=rolled_up.total_latency_s,
            gpu_energy_j=rolled_up.total_gpu_energy_j,
            gpu_energy_provenance=rolled_up.gpu_energy_provenance,
            gpu_energy_complete=rolled_up.gpu_energy_complete,
            accelerator_device_coverage_complete=(
                rolled_up.accelerator_device_coverage_complete
            ),
        )

    return runner

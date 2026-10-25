"""Generation driver: run each converted task through the harness and capture the verified Trajectory.

Bounded-parallel + budget-capped + resume-safe scheduler. Every task that is skipped (resume) or
dropped (budget) is logged -- nothing is silently truncated.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ipj.orchestrate.harness.trace import Trajectory, summarize_outcome

from .swe_tasks import SweTask

RunOne = Callable[[SweTask, Path], "tuple[Trajectory, bool]"]
ConvertFn = Callable[[SweTask, Path], Path]
LogFn = Callable[[str], None]

_log = logging.getLogger(__name__)


def _default_log(msg: str) -> None:
    _log.info(msg)


def _default_convert(task: SweTask, tasks_root: Path) -> Path:
    # Lazy: convert pulls in terminal_bench, which need not be importable to schedule (or unit-test).
    from .convert import swe_task_to_harbor

    return swe_task_to_harbor(task, tasks_root)


@dataclass(slots=True)
class GenResult:
    instance_id: str
    resolved: bool
    dollars: float | None
    trajectory: Trajectory


def finalize_trace(traj: Trajectory, is_resolved: bool) -> Trajectory:
    traj.outcome = summarize_outcome(traj.steps, success=is_resolved, source="verifier")
    return traj


def generate(
    tasks: list[SweTask],
    out_root: Path,
    *,
    run_one: RunOne,
    budget: float | None = None,
    n_concurrent: int = 1,
    convert: ConvertFn | None = None,
    log: LogFn | None = None,
) -> list[GenResult]:
    """Run each task through ``run_one``, capturing its verified Trajectory under ``out_root``.

    - ``n_concurrent``: at most this many tasks run at once (bounded concurrency; default serial).
    - ``budget``: once cumulative spend >= budget, no further tasks are *scheduled* (in-flight ones
      finish). Remaining tasks are dropped and each is logged.
    - resume-safe: a task whose trace json already exists under ``out_root/traces`` is skipped (logged).
    - ``convert`` / ``log`` are injectable for testing; defaults are the real Harbor converter + logger.

    Results are returned in scheduling order (deterministic), not completion order.
    """
    out_root = Path(out_root)
    traces_dir = out_root / "traces"
    tasks_dir = out_root / "tasks"
    traces_dir.mkdir(parents=True, exist_ok=True)
    convert = convert if convert is not None else _default_convert
    log = log if log is not None else _default_log
    n_concurrent = max(1, int(n_concurrent))

    # Resume: skip tasks already verified in a prior run (log every skip -- no silent truncation).
    pending: list[SweTask] = []
    for task in tasks:
        if (traces_dir / f"{task.instance_id}.json").exists():
            log(f"generate: skip {task.instance_id} (resume: trace already exists)")
            continue
        pending.append(task)

    def _run(task: SweTask) -> GenResult:
        task_dir = convert(task, tasks_dir)
        traj, resolved = run_one(task, task_dir)
        finalize_trace(traj, resolved)
        cost = traj.outcome.total_dollars if traj.outcome else None
        if budget is not None and cost is None:
            raise ValueError(
                f"cannot enforce budget for {task.instance_id}: trajectory cost is unknown"
            )
        (traces_dir / f"{task.instance_id}.json").write_text(json.dumps(traj.to_dict(), default=str, indent=2))
        return GenResult(task.instance_id, resolved, cost, traj)

    results: dict[str, GenResult] = {}
    order: list[str] = []
    spent = 0.0
    task_iter = iter(pending)
    in_flight: dict = {}

    def _budget_reached() -> bool:
        return budget is not None and spent >= budget

    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        while True:
            # Fill up to n_concurrent slots, unless the budget already stopped scheduling.
            while len(in_flight) < n_concurrent and not _budget_reached():
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                order.append(task.instance_id)
                in_flight[pool.submit(_run, task)] = task
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.pop(fut)
                res = fut.result()
                results[res.instance_id] = res
                if res.dollars is not None:
                    spent += res.dollars
        # Anything the budget prevented us from scheduling is dropped -- log each explicitly.
        for task in task_iter:
            log(f"generate: drop {task.instance_id} (budget ${budget} reached; spent ${spent:.4f})")

    return [results[i] for i in order]

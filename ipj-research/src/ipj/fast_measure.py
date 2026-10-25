"""Pure helpers for a fast-served measurement cell with a makespan cutoff.

The fast-serving rerun (compiled vLLM) is orchestrated by bash on the GPU box, but every
parsing/scoring/energy decision lives here so it is unit-tested and reused by the cutoff
watcher. Two conventions make the numbers honest:

- **Full-N denominator.** Accuracy = resolved / `n_total`; tasks that never reached a
  terminal state (because the 95% makespan cutoff killed the laggard tail) count as unsolved.
- **All terminal tokens count.** Errored/timed-out tasks still consumed tokens (they ran
  until the agent timeout), so their `agent_result` tokens are included.

Per-task `result.json` is Harbor 0.13.2's shape: `verifier_result.rewards.reward` (0/1),
`exception_info.exception_type` (errored), `agent_result.n_input_tokens/n_output_tokens`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .cost import H100_HOURLY_USD, dollars_local
from .metrics import Cell


class IncompleteRunError(RuntimeError):
    """Raised when a run is too truncated, or its power trace too empty, to describe honestly."""


class ImplausiblePowerError(RuntimeError):
    """Raised when a cell's mean power cannot have come from the boards it claims."""


class IdleBoardError(RuntimeError):
    """Raised when a board the cell claims was never observed doing work."""


class NoInferenceError(RuntimeError):
    """Raised when the tasks reached terminal states without the model ever generating."""


# Per-board sustained power ceiling (W), by the substring `detect_hardware` puts in the label.
# A board cannot exceed its firmware `power.limit`, so a mean above it is proof the trace covers
# devices the run did not own -- exactly how the first Gemma cell went wrong: 886.9 W peak over
# 2 allocated L40S = 443 W/board against a 350 W hard limit, because the sampler logged the whole
# 8-GPU node. A floor is checked too: a mean at idle means the GPUs were not working, which is how
# a stalled agent (400s, NCCL deadlock, dead endpoint) produces a cell that looks measured.
BOARD_POWER_LIMIT_W: dict[str, float] = {
    "h200": 700.0, "h100": 700.0, "l40s": 350.0, "a100": 400.0,
    "mi350x": 1000.0, "gb10": 200.0, "rtx5090": 575.0,
}
BOARD_IDLE_FLOOR_W: dict[str, float] = {
    "h200": 60.0, "h100": 55.0, "l40s": 30.0, "a100": 40.0,
    # MI350X idle measured at 246 W on matx-amd-1 (rocm-smi, 2026-08-14), far above the
    # NVIDIA parts. Floor set below it so only a powered-down board trips -- like L40S,
    # the floor cannot distinguish "idle" from "lightly loaded" on this silicon.
    "mi350x": 200.0, "gb10": 10.0, "rtx5090": 20.0,
}


def check_power_plausible(joules: float, wall_s: float, n_gpus: int, hardware: str) -> None:
    """Raise if mean power per board is impossible for the silicon the cell claims.

    Silent on unknown hardware rather than guessing a limit -- a wrong ceiling would either wave
    through a bad cell or reject a good one, and both are worse than declining to judge.
    """
    if not (wall_s > 0 and n_gpus > 0) or joules is None:
        return
    key = next((k for k in BOARD_POWER_LIMIT_W if k in (hardware or "").lower()), None)
    if key is None:
        return
    per_board = joules / wall_s / n_gpus
    limit, floor = BOARD_POWER_LIMIT_W[key], BOARD_IDLE_FLOOR_W[key]
    if per_board > limit:
        raise ImplausiblePowerError(
            f"mean {per_board:.1f} W/board over {n_gpus}x {key} exceeds the {limit:.0f} W "
            f"firmware limit. The power trace covers devices this run did not own -- sample only "
            f"the allocated boards (see the Gemma whole-node defect)."
        )
    if per_board < floor:
        raise ImplausiblePowerError(
            f"mean {per_board:.1f} W/board over {n_gpus}x {key} is at or below the ~{floor:.0f} W "
            f"idle floor. The GPUs were not working: the run stalled (rejected requests, a dead "
            f"endpoint, a startup deadlock) and these Joules describe idling, not inference."
        )


# Fraction of in-window samples a board must spend above `BOARD_ACTIVE_PCT` utilisation before
# its Joules describe inference. A floor against DEAD, not a target -- the failure it exists to
# catch reads 0.00.
#
# CALIBRATED, not assumed (2026-08-14, live H200 runs on sc):
#   q122, 2xH200, 5,458 samples -- gpu0 and gpu1 both 100.0% above 5%, median 100, mean 99.9
#   q38,  1xH200,   576 samples -- gpu0 94.4% above 5%, median 100, mean 94.2
#   MI350X container-escape defect                    -- 0.0%
# vLLM's continuous batching keeps the board pegged even though the agent spends much of the
# makespan on container shell work, so real runs land at 94-100%, not the "idles between tasks"
# figure this threshold was first written against. 0.10 therefore carries ~9x margin below the
# worst real observation while still rejecting the 0.00 case outright. Do not raise it toward the
# observed band: the gap between 0% and 94% is the whole signal, and a threshold close to real
# duty cycle would start refusing honest runs the moment one is genuinely slower.
BOARD_ACTIVE_PCT = 5.0
BOARD_MIN_ACTIVE_FRAC = 0.10


def summarize_board_activity(
    samples: list[tuple[float, list[float]]], start: float, end: float,
    active_pct: float = BOARD_ACTIVE_PCT,
) -> list[float]:
    """Per-board fraction of in-window samples showing utilisation above `active_pct`.

    `samples` is `(epoch, [util_pct per board])` in the board order the sampler queried, which
    must be the order the run was allocated. Returns one fraction per board, or [] if the window
    caught no samples.
    """
    rows = [u for ts, u in samples if start <= ts <= end and u]
    if not rows:
        return []
    n_boards = min(len(u) for u in rows)
    return [
        sum(1 for u in rows if u[i] > active_pct) / len(rows)
        for i in range(n_boards)
    ]


def check_boards_active(
    active_fracs: list[float], n_gpus: int, min_active_frac: float = BOARD_MIN_ACTIVE_FRAC,
) -> None:
    """Raise unless every board the cell claims was observed doing work.

    THE FOURTH GATE, and the only one that catches a mis-attributed sampler. The other three ask
    whether the number is well-formed; this asks whether it came from the right silicon. On
    matx-amd-1 (2026-08-14) the container escaped its Slurm allocation: vLLM ran on GPU 3 at 436 W
    while the sampler read the allocated GPU 0, idle at 239 W. That cell passed completeness,
    joules>0, and power plausibility -- 239 W sits mid-band for MI350X -- and reported the energy
    of a board that did nothing. Only utilisation separates the two.

    Note the asymmetry: non-zero utilisation is NECESSARY, not sufficient. A vLLM startup deadlock
    pins NVML at 100% with only the CUDA context resident (see fast_cell.sh's health wait), so this
    gate cannot certify that work happened -- only refuse when it demonstrably did not.

    Silent when `active_fracs` is empty: absent telemetry is a missing check, not a passed one, and
    the caller (harvest_cell.py) decides whether to demand it.
    """
    if not active_fracs:
        return
    if len(active_fracs) != n_gpus:
        raise IdleBoardError(
            f"utilisation trace covers {len(active_fracs)} board(s) but the cell claims {n_gpus}. "
            f"The sampler and the allocation disagree; one of them is wrong."
        )
    dead = [(i, f) for i, f in enumerate(active_fracs) if f < min_active_frac]
    if dead:
        detail = ", ".join(f"board {i} active {f:.1%} of samples" for i, f in dead)
        raise IdleBoardError(
            f"{detail} (need >= {min_active_frac:.0%} above {BOARD_ACTIVE_PCT:.0f}% util). "
            f"These Joules describe a board that was not doing the work -- either the sampler read "
            f"a device the run did not use (the MI350X container-escape defect) or the run stalled."
        )


# Fraction of TERMINAL tasks that may have produced no output tokens before the run stops
# describing inference at all. Calibrated on real runs (2026-08-15): q397 0.0%, q122 0.0%,
# q38 2.3% -- against 98.8% for a run whose every task died in Docker setup. A ~10x margin.
MAX_NO_OUTPUT_FRAC = 0.25


def check_work_happened(n_terminal: int, n_no_output: int,
                        max_frac: float = MAX_NO_OUTPUT_FRAC) -> None:
    """Raise if the tasks reached terminal states without the model ever generating.

    THE FIFTH GATE, and the one the other four cannot cover. They ask whether the number is
    complete, non-empty, plausible for the silicon, and taken from the right board. All four
    passed this cell:

        acc=0.011  joules=208,585  wall=346s  85/89 terminal  602 W/board  gpu0 96.6% active

    Every one of its 85 tasks had died instantly in `docker compose up` -- "all predefined
    address pools have been fully subnetted", because two harbor runs shared a node and
    exhausted Docker's ~31-network pool. The tasks DID reach terminal states (they errored),
    the GPU WAS busy (vLLM sat loaded and warming), and the Joules were real. The cell was
    well-formed, verified by harvest_cell.py, and claimed Qwen3.8-27B scores 1.1% on TB v2.1.

    Zero output tokens is the signal, because it is the one thing an infrastructure failure and
    a real run can never share: a task that reached the agent always generates something, and a
    task that died in container setup never can. Wall-clock would also have caught this instance
    (346 s for 85 tasks is impossible) but not a slow partial failure; token production is the
    direct evidence that inference occurred.
    """
    if n_terminal <= 0:
        return
    frac = n_no_output / n_terminal
    if frac > max_frac:
        raise NoInferenceError(
            f"{n_no_output}/{n_terminal} terminal tasks produced ZERO output tokens "
            f"({frac:.1%} > {max_frac:.0%}). These tasks never reached the model -- the usual "
            f"cause is the harness failing before the agent runs (docker compose, image pull, "
            f"network pool exhaustion). The Joules are real but they measure an idle server, "
            f"not inference."
        )


def latest_jobs_run(jobs_dir: str | Path) -> Path | None:
    """Newest `<jobs>/<timestamp>/` run subdir (timestamp names sort chronologically)."""
    p = Path(jobs_dir)
    if not p.is_dir():
        return None
    runs = sorted((d for d in p.iterdir() if d.is_dir()), key=lambda d: d.name)
    return runs[-1] if runs else None


def _task_results(run_dir: str | Path) -> list[Path]:
    """Per-task `result.json` paths under a run dir (the `<task>__<id>/` subdirs)."""
    return sorted(Path(run_dir).glob("*__*/result.json"))


def count_completed(run_dir: str | Path) -> int:
    """How many tasks have reached a terminal state (have a per-task result.json)."""
    return len(_task_results(run_dir))


def cutoff_reached(run_dir: str | Path, n_total: int, frac: float) -> bool:
    """True once at least ⌈frac·n_total⌉ tasks have completed."""
    return count_completed(run_dir) >= math.ceil(frac * n_total)


def trapezoid_joules(samples: list[tuple[float, float]], start: float, end: float) -> float:
    """∫ power dt (Joules) over [start, end] from (epoch_s, watts) samples, trapezoidal.

    Samples are sorted and clamped to the window; <2 in-window samples → 0.0.
    """
    pts = sorted((t, w) for t, w in samples if start <= t <= end)
    return sum((pts[i][1] + pts[i - 1][1]) / 2.0 * (pts[i][0] - pts[i - 1][0])
               for i in range(1, len(pts)))


@dataclass(frozen=True)
class RunSummary:
    accuracy: float
    n_resolved: int
    n_terminal: int     # tasks that produced a result (resolved + unresolved + errored)
    tokens_in: int
    tokens_out: int
    n_no_output: int = 0    # terminal tasks whose agent emitted ZERO output tokens


def _parse_task(result_json: Path) -> tuple[str, str, bool, int, int]:
    """(task_name, status, verifier_solved, n_input_tokens, n_output_tokens) for one result.json.

    The task name is normalized to its bare form — Harbor stores it dataset-prefixed
    (``terminal-bench/<task>``), but the cascade must key on the bare ``<task>`` to align
    with the cloud outcomes from ``harbor.per_task.per_task_outcomes``.

    TWO ANSWERS, DELIBERATELY, because two different questions are being asked of the same file:

    * ``status`` is three-valued and lets an exception WIN over the reward. This is what the
      cascade escalates on (``escalate_when=lambda s: s == "errored"``), and it is right there:
      at deploy time you observe that the agent timed out and must decide to escalate *without*
      being able to see the verifier's reward.
    * ``verifier_solved`` is the reward alone, independent of any exception. This is what the
      BOARD means by solved — `per_task_vectors_v2.json` carries `solved` and `errored` as
      independent booleans, and every published tier has tasks that are both (qwen36_27b 3,
      GLM_local 8, opus5 5, kimi_k3 4).

    Conflating them made `summarize_run` understate local accuracy against the board's own
    convention, and ONLY for local tiers, since cloud tiers reach the spine through
    harvest_detail.py which counts by reward. Measured on the 2026-08-14 runs: q38 42 vs 46
    (47.2% vs 51.7%), q122 35 vs 36, q397 40 vs 40. Every one of those gaps was an
    AgentTimeoutError on a task the verifier had already passed — the agent overran its budget
    after solving the task, which does not un-solve it.
    """
    d = json.loads(result_json.read_text())
    raw = d.get("task_name") or result_json.parent.name.rsplit("__", 1)[0]
    task = raw.split("/")[-1]
    exc = d.get("exception_info") or {}
    reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    solved = reward is not None and reward >= 1.0
    if exc.get("exception_type"):
        status = "errored"
    else:
        status = "resolved" if solved else "unresolved"
    ar = d.get("agent_result") or {}
    return task, status, solved, int(ar.get("n_input_tokens") or 0), int(ar.get("n_output_tokens") or 0)


def task_status_map(run_dir: str | Path) -> dict[str, str]:
    """{task_name: status} from the per-task result.jsons — the cascade local-tier outcomes.

    Deliberately the escalation view (exception wins), NOT the board's solved view; see
    `_parse_task`. Do not swap one for the other to "fix" an accuracy mismatch.
    """
    return {t: s for t, s, _, _, _ in map(_parse_task, _task_results(run_dir))}


def summarize_run(run_dir: str | Path, n_total: int) -> RunSummary:
    """Accuracy (over the full n_total) + token totals from the completed per-task results."""
    n_resolved = tin = tout = n_no_out = 0
    results = _task_results(run_dir)
    for rj in results:
        # Count by the VERIFIER, not by the escalation status -- see `_parse_task`. Using the
        # status here made a task that the verifier passed but whose agent then overran its
        # timeout count as unsolved, understating local tiers against the board's own convention.
        _, _status, solved, ti, to = _parse_task(rj)
        n_resolved += solved
        tin += ti
        tout += to
        n_no_out += (to == 0)
    acc = n_resolved / n_total if n_total else 0.0
    return RunSummary(accuracy=acc, n_resolved=n_resolved, n_terminal=len(results),
                      tokens_in=tin, tokens_out=tout, n_no_output=n_no_out)


def detect_hardware(default: str = "unknown-local-fast") -> str:
    """Best-effort silicon label from NVML, e.g. 'l40s-local-fast', 'h100-local-fast'.

    Falls back to `default` rather than guessing H100 -- a wrong-but-plausible label is worse
    than an obviously-unknown one, because it reads as a measurement on hardware we never used.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
        if not names:
            return default
        n = names[0].lower()
        for tag in ("h200", "h100", "l40s", "a100", "mi350x", "rtx 5090", "gb10"):
            if tag in n:
                return f"{tag.replace(' ', '')}-local-fast"
        return f"{n.split()[-1]}-local-fast"
    except Exception:
        return default


def build_fast_cell(
    model: str,
    run_dir: str | Path,
    n_total: int,
    *,
    joules: float,
    wall_s: float,
    cutoff_frac: float,
    gpu_rate: float = H100_HOURLY_USD,
    n_gpus: int = 1,
    dataset: str = "terminal-bench/terminal-bench-2-1",
    harness: str = "terminus-2",
    hardware: str | None = None,
    allow_partial: bool = False,
    board_active_fracs: list[float] | None = None,
) -> Cell:
    """Assemble a fast-served Cell: $ from the (cutoff) makespan, energy measured, full-N accuracy.

    `hardware` must describe the silicon the run actually used -- it labels the energy number on
    the figures. It was previously hardcoded to "h100-local-fast", which silently mislabels any
    run on other silicon (L40S, MI350X, H200); pass it explicitly, or leave it None to detect
    the device from NVML.
    """
    s = summarize_run(run_dir, n_total)

    # REFUSE to emit a cell from a run that did not reach its cutoff. Until 2026-08-13 this was
    # never checked: `summarize_run` computed `n_terminal` and `build_fast_cell` threw it away, so
    # a truncated run produced a fully-formed cell claiming the REQUESTED n_tasks. The only thing
    # preventing that in practice was that Slurm's preemption SIGTERM kills the driver shell
    # before it reaches the emit step -- an accident of signal delivery, not a guarantee. Adding
    # an ordinary `trap ... EXIT` cleanup (the idiom already used in serve_fast.sh and
    # run_strategies.sh) is enough to flip it on. Replayed against a real preempted tree, the old
    # path emitted accuracy=0.034 / joules=344,390 / wall=377.5 s from 4 of 89 tasks, stamped
    # "95% makespan cutoff" -- a cutoff it never reached.
    required = math.ceil(cutoff_frac * n_total)
    if s.n_terminal < required and not allow_partial:
        raise IncompleteRunError(
            f"{s.n_terminal}/{n_total} tasks reached a terminal state, need >= {required} "
            f"({int(round(cutoff_frac * 100))}% cutoff). Refusing to emit a cell: its accuracy "
            f"would be computed over {n_total} and its Joules over a truncated makespan. "
            f"Pass allow_partial=True only to inspect, never to publish."
        )

    # A measured-energy label must describe actual samples. Zero or negative Joules means the
    # power log was empty, unreadable, or shorter than two in-window samples -- emitting that as
    # "measured nvml gpu" would launder a non-measurement into the field ADR-0003 reserves for
    # real ones, and ADR-0004 would then price dollars from it.
    if joules is not None and joules <= 0:
        raise IncompleteRunError(
            f"joules={joules} — the power trace yielded no usable samples over [start, end]. "
            "Refusing to stamp that 'measured'."
        )

    check_work_happened(s.n_terminal, s.n_no_output)

    hw = hardware or detect_hardware()
    check_power_plausible(joules, wall_s, n_gpus, hw)
    check_boards_active(board_active_fracs or [], n_gpus)

    return Cell(
        model=model, harness=harness, hardware=hw, dataset=dataset,
        n_tasks=n_total, accuracy=s.accuracy, tokens_in=s.tokens_in, tokens_out=s.tokens_out,
        dollars=dollars_local(wall_s, gpu_rate, n_gpus=n_gpus), dollars_basis="gpu-hours",
        joules=round(joules, 1), wall_clock_s=wall_s,
        energy_source=f"measured nvml gpu (compiled, {int(round(cutoff_frac * 100))}% makespan cutoff)",
        run_dir=str(run_dir), n_terminal=s.n_terminal, n_gpus=n_gpus,
    )

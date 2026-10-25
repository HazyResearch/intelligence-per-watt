"""Guards that stop a truncated or unmeasured run from becoming a publishable cell.

Both defects below were live until 2026-08-13 and were found by replaying the emit step against a
real preempted run tree (job 16756473, ev_q122, killed by Slurm at 4/89 tasks). The emit produced
a fully-formed cell -- accuracy 0.034, joules 344,390, wall 377.5 s -- stamped
"measured nvml gpu (compiled, 95% makespan cutoff)" for a cutoff it never reached, in 0.115 s,
well inside Slurm's 30 s KillWait window.

Nothing rejected it. It only failed to reach disk because bash died on its own SIGTERM first --
an accident of signal delivery. An ordinary `trap ... EXIT` cleanup (the idiom already used in
serve_fast.sh:44 and run_strategies.sh:39) would have been enough to let it through.
"""
from __future__ import annotations

import json
import math

import pytest

from ipj.fast_measure import (
    IdleBoardError,
    ImplausiblePowerError,
    IncompleteRunError,
    NoInferenceError,
    build_fast_cell,
    check_boards_active,
    check_power_plausible,
    summarize_board_activity,
    task_status_map,
)


def _task(dirpath, name, *, reward, tok_in=100, tok_out=10):
    d = dirpath / f"{name}__abc123"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{name}",
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {"n_input_tokens": tok_in, "n_output_tokens": tok_out},
    }))


def _run_dir(tmp_path, n_done, n_solved=0):
    run = tmp_path / "jobs" / "2026-08-12__19-53-48"
    run.mkdir(parents=True)
    for i in range(n_done):
        _task(run, f"task{i:03d}", reward=1.0 if i < n_solved else 0.0)
    return run


def test_truncated_run_is_refused(tmp_path):
    """The preemption case: 4 of 89 tasks terminal, 95% cutoff needs 85."""
    run = _run_dir(tmp_path, n_done=4, n_solved=3)
    with pytest.raises(IncompleteRunError) as e:
        build_fast_cell("openai/x", run, 89, joules=344390.3, wall_s=377.5, cutoff_frac=0.95)
    msg = str(e.value)
    assert "4/89" in msg and "85" in msg, msg


def test_cutoff_boundary_is_inclusive(tmp_path):
    """85/89 is exactly ceil(0.95*89) and must be accepted; 84 must not."""
    ok = build_fast_cell("openai/x", _run_dir(tmp_path / "a", n_done=85, n_solved=40),
                         89, joules=1e6, wall_s=1000.0, cutoff_frac=0.95,
                         hardware="test-noboard")
    assert ok.n_terminal == 85
    with pytest.raises(IncompleteRunError):
        build_fast_cell("openai/x", _run_dir(tmp_path / "b", n_done=84),
                        89, joules=1e6, wall_s=1000.0, cutoff_frac=0.95,
                        hardware="test-noboard")


def test_empty_power_trace_is_not_stamped_measured(tmp_path):
    """trapezoid_joules returns 0.0 for <2 in-window samples. Emitting that as 'measured nvml
    gpu' would launder a non-measurement into the field ADR-0003 reserves for real ones, and
    ADR-0004 would then price local dollars from it."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    with pytest.raises(IncompleteRunError) as e:
        build_fast_cell("openai/x", run, 89, joules=0.0, wall_s=11134.0, cutoff_frac=0.95)
    assert "no usable samples" in str(e.value)


def test_allow_partial_is_opt_in_and_still_records_truncation(tmp_path):
    """Inspecting a partial run stays possible, but the cell carries n_terminal so the
    truncation is visible downstream -- previously n_tasks recorded what was REQUESTED and
    nothing recorded what actually finished."""
    run = _run_dir(tmp_path, n_done=4, n_solved=3)
    c = build_fast_cell("openai/x", run, 89, joules=344390.3, wall_s=377.5,
                        cutoff_frac=0.95, allow_partial=True, hardware="test-noboard")
    assert c.n_terminal == 4 and c.n_tasks == 89
    assert c.accuracy == pytest.approx(3 / 89)


def test_cell_records_n_gpus_for_power_checks(tmp_path):
    """n_gpus was a local in the emit script and vanished, so a published cell could not be
    checked against the boards' power limit, nor re-priced on the amortized basis."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=6.9e6, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=4, hardware="test-noboard")
    assert c.n_gpus == 4


def test_power_above_the_board_limit_is_refused(tmp_path):
    """The Gemma defect: 10,443,922 J over 13,241 s attributed to 2 allocated L40S is
    394 W/board against a 350 W firmware limit. A board cannot exceed its power.limit, so the
    trace must have covered devices the run did not own -- it logged the whole 8-GPU node."""
    run = _run_dir(tmp_path, n_done=89, n_solved=25)
    with pytest.raises(ImplausiblePowerError) as e:
        build_fast_cell("openai/x", run, 89, joules=10443922.0, wall_s=13241.0,
                        cutoff_frac=0.95, n_gpus=2, hardware="l40s-local-fast")
    assert "exceeds the 350 W" in str(e.value)


def test_power_at_the_idle_floor_is_refused(tmp_path):
    """A mean AT idle means the GPUs never worked at all -- e.g. an NCCL startup deadlock, where
    NVML still reports 100% utilisation while only the CUDA context is resident.

    Scope note, deliberately narrow: this catches *idle*, not merely *degraded*. The 2026-08-10
    L40S Qwen run averaged 96.7 W over 2 boards = 48 W/board, which is above the 33 W measured
    L40S idle, so no power threshold could have flagged it without also rejecting healthy runs.
    That run's real signals were 486 HTTP 400s and 40 agent timeouts -- caught by the serving
    checks in fast_cell.sh, not here. Raising this floor to catch it would trade a true positive
    for false rejections of legitimate low-utilisation runs."""
    run = _run_dir(tmp_path, n_done=89, n_solved=10)
    idle_joules = 28.0 * 2 * 20000.0        # both boards parked just under the 30 W floor
    with pytest.raises(ImplausiblePowerError) as e:
        build_fast_cell("openai/x", run, 89, joules=idle_joules, wall_s=20000.0,
                        cutoff_frac=0.95, n_gpus=2, hardware="l40s-local-fast")
    assert "idle floor" in str(e.value)


def test_plausible_power_passes(tmp_path):
    """The clean H200 Qwen cell: 6,945,279 J over 11,134 s on 1 board = 624 W, under 700 W."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast")
    assert c.n_gpus == 1 and c.joules > 0


def test_unknown_hardware_is_not_judged(tmp_path):
    """Silent on silicon we have no limit for -- a guessed ceiling would either wave through a
    bad cell or reject a good one, both worse than declining to judge."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=9e9, wall_s=100.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="some-new-accelerator")
    assert c.joules > 0


# --- Fourth gate: did the sampler watch the silicon that did the work? ------------------------
#
# The three gates above ask whether a number is well-formed. None of them asks whether it came
# from the right board. On matx-amd-1 (2026-08-14) `--device=/dev/dri` exposed every GPU on the
# node, the container escaped its Slurm allocation, and vLLM ran on GPU 3 at 436 W / 100% use
# while the sampler faithfully recorded the *allocated* GPU 0: idle, 239 W, 0% use. The resulting
# cell was complete, positive, and mid-band for MI350X's [200, 1000] W range. It passed
# everything and described a board that did nothing.


def test_mi350x_container_escape_is_refused(tmp_path):
    """Replay of the real defect. 239 W/board over 1 MI350X is a *plausible* mean -- that is the
    whole problem -- so only the 0%-utilisation trace can reject it."""
    run = _run_dir(tmp_path, n_done=89, n_solved=30)
    wall = 18000.0
    with pytest.raises(IdleBoardError) as e:
        build_fast_cell("openai/x", run, 89, joules=239.0 * wall, wall_s=wall,
                        cutoff_frac=0.95, n_gpus=1, hardware="mi350x-local-fast",
                        board_active_fracs=[0.0])
    assert "board 0 active 0.0%" in str(e.value)
    # and confirm the same cell sails through the power gate it was designed to defeat
    check_power_plausible(239.0 * wall, wall, 1, "mi350x-local-fast")


def test_board_count_mismatch_is_refused(tmp_path):
    """The sampler and the allocation must agree on how many boards exist. A trace covering 8
    boards on a cell claiming 2 is the Gemma whole-node defect seen from the other side --
    catchable here even when the arithmetic happens to land inside the power band."""
    run = _run_dir(tmp_path, n_done=89, n_solved=30)
    with pytest.raises(IdleBoardError) as e:
        build_fast_cell("openai/x", run, 89, joules=300.0 * 2 * 10000.0, wall_s=10000.0,
                        cutoff_frac=0.95, n_gpus=2, hardware="l40s-local-fast",
                        board_active_fracs=[0.9] * 8)
    assert "covers 8 board(s) but the cell claims 2" in str(e.value)


def test_one_dead_board_in_a_tp_group_is_refused(tmp_path):
    """A tensor-parallel run where one board never worked is not a 2-GPU measurement; billing it
    as gpu-hours x 2 would overstate dollars while its Joules understate the real per-board draw."""
    run = _run_dir(tmp_path, n_done=89, n_solved=30)
    with pytest.raises(IdleBoardError) as e:
        build_fast_cell("openai/x", run, 89, joules=300.0 * 2 * 10000.0, wall_s=10000.0,
                        cutoff_frac=0.95, n_gpus=2, hardware="h200-local-fast",
                        board_active_fracs=[0.94, 0.02])
    assert "board 1 active 2.0%" in str(e.value)


def test_busy_boards_pass(tmp_path):
    """The clean H200 Qwen cell's shape: one board, working nearly the whole makespan."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast",
                        board_active_fracs=[0.87])
    assert c.joules > 0


def test_idle_gaps_between_tasks_do_not_trip_the_gate(tmp_path):
    """A serving GPU idles between tasks; the floor is against DEAD, not against imperfect
    duty cycle. 18% active must still pass or the gate would reject honest runs."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast",
                        board_active_fracs=[0.18])
    assert c.joules > 0


def test_missing_utilisation_is_a_missing_check_not_a_passed_one(tmp_path):
    """build_fast_cell stays silent without a trace so pre-2026-08-14 cells remain harvestable;
    the refusal lives in harvest_cell.py, where a human is choosing to publish."""
    run = _run_dir(tmp_path, n_done=89, n_solved=44)
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast")
    assert c.joules > 0


def test_summarize_board_activity_windows_and_orders(tmp_path):
    """Per-board fractions come back in sampler order, counting only in-window samples."""
    s = [
        (100.0, [90.0, 0.0]),   # before window
        (200.0, [90.0, 0.0]),
        (300.0, [90.0, 0.0]),
        (400.0, [0.0, 0.0]),
        (900.0, [90.0, 90.0]),  # after window
    ]
    assert summarize_board_activity(s, 200.0, 400.0) == [2 / 3, 0.0]
    assert summarize_board_activity(s, 1000.0, 2000.0) == []


# --- Accuracy counts by the verifier, not by the escalation status ---------------------------
#
# A task whose verifier awarded reward=1.0 but whose agent then overran its timeout is SOLVED.
# The agent spending its budget after passing the tests does not un-pass them. The board has
# always counted this way -- per_task_vectors_v2.json carries `solved` and `errored` as
# independent booleans and every published tier has tasks that are both (qwen36_27b 3,
# GLM_local 8, opus5 5, kimi_k3 4). summarize_run did not, so local cells understated themselves
# against the very board they feed, and ONLY local cells, because cloud tiers reach the spine
# through harvest_detail.py which counts by reward.


def _task_errored_but_solved(dirpath, name):
    d = dirpath / f"{name}__abc123"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{name}",
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": {"exception_type": "AgentTimeoutError"},
        "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10},
    }))


def test_verifier_pass_with_agent_timeout_counts_as_solved(tmp_path):
    """The real q38 case: 4 of 46 solves were AgentTimeoutError on tasks the verifier passed.
    Counting by status gave 42/89 (47.2%); the board's own convention gives 46/89 (51.7%)."""
    run = _run_dir(tmp_path, n_done=85, n_solved=42)
    for i in range(4):
        _task_errored_but_solved(run, f"late{i:03d}")
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast")
    assert round(c.accuracy * 89) == 46


def test_escalation_status_still_treats_those_as_errored(tmp_path):
    """The cascade must keep seeing them as errored: at deploy time you observe the timeout and
    escalate WITHOUT being able to see the reward. Fixing accuracy must not silently change which
    tasks a deploy_honest cascade escalates."""
    run = _run_dir(tmp_path, n_done=2, n_solved=1)
    _task_errored_but_solved(run, "late000")
    assert task_status_map(run)["late000"] == "errored"


# --- Fifth gate: did inference actually happen? ----------------------------------------------
#
# The other four ask whether the number is complete, non-empty, plausible for the silicon, and
# taken from the right board. On 2026-08-15 a run passed ALL FOUR and was verified by
# harvest_cell.py while being entirely worthless:
#
#     acc=0.011  joules=208,585  wall=346s  85/89 terminal  602 W/board  gpu0 96.6% active
#
# Every one of its 85 tasks died instantly in `docker compose up` -- "all predefined address
# pools have been fully subnetted" -- because two harbor runs shared a node and exhausted
# Docker's ~31-network pool. The tasks reached terminal states, the GPU was genuinely busy
# (vLLM loaded and warming), and the Joules were real. The cell claimed Qwen3.8-27B scores 1.1%.


def _task_no_output(dirpath, name):
    d = dirpath / f"{name}__abc123"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{name}",
        "verifier_result": {"rewards": {"reward": 0.0}},
        "exception_info": {"exception_type": "RuntimeError",
                           "exception_message": "Docker compose command failed ... fully subnetted"},
        "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0},
    }))


def test_infrastructure_failure_is_refused(tmp_path):
    """Replay of the real defect: 84 of 85 terminal tasks never reached the model."""
    run = tmp_path / "jobs" / "2026-08-15__00-00-00"
    run.mkdir(parents=True)
    _task(run, "one-that-ran", reward=1.0, tok_in=100, tok_out=50)
    for i in range(84):
        _task_no_output(run, f"dead{i:03d}")
    with pytest.raises(NoInferenceError) as e:
        build_fast_cell("openai/x", run, 89, joules=208585.2, wall_s=346.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast",
                        board_active_fracs=[0.966])
    assert "ZERO output tokens" in str(e.value)


def test_all_four_earlier_gates_pass_that_same_cell(tmp_path):
    """The point of the fifth gate: nothing before it objects. If any of these start raising,
    this test is wrong, not the gate."""
    check_power_plausible(208585.2, 346.0, 1, "h200-local-fast")   # 602 W/board, in band
    check_boards_active([0.966], 1)                                # board genuinely busy
    assert 85 >= math.ceil(0.95 * 89)                              # completeness satisfied
    assert 208585.2 > 0                                            # joules non-empty


def test_a_few_zero_token_tasks_are_tolerated(tmp_path):
    """Real runs do carry some: q38 had 2/86 (2.3%), q397 and q122 zero. The threshold sits at
    25% so ordinary noise never trips it."""
    run = tmp_path / "jobs" / "2026-08-15__00-00-00"
    run.mkdir(parents=True)
    for i in range(84):
        _task(run, f"ok{i:03d}", reward=1.0 if i < 44 else 0.0, tok_in=1000, tok_out=100)
    for i in range(2):
        _task_no_output(run, f"dead{i:03d}")
    c = build_fast_cell("openai/x", run, 89, joules=6945278.9, wall_s=11134.0,
                        cutoff_frac=0.95, n_gpus=1, hardware="h200-local-fast",
                        board_active_fracs=[0.87])
    assert c.joules > 0

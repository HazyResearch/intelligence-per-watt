"""Convert an ADP SWE task into a terminal-bench task dir (task.yaml + Docker + tests + run-tests)."""

from __future__ import annotations

import shlex
from pathlib import Path

try:
    from terminal_bench.handlers.trial_handler import Task
except ImportError as e:  # pragma: no cover — surfaced only in bare venvs
    raise ImportError(
        "terminal_bench is required for ipj.orchestrate.generate. "
        "Install with:  uv pip install -e '../ipj[orchestrate]'  (or ipw[terminus])."
    ) from e

from .swe_tasks import SweTask

# --- run-tests.sh -------------------------------------------------------------------
# terminal-bench copies run-tests.sh + the tests/ dir into the container's /tests, then
# runs this script. It applies the task's test patch and runs the SWE-bench targets: the
# instance is "resolved" iff every FAIL_TO_PASS test passes AND every PASS_TO_PASS test
# still passes. The /testbed path + conda env follow the SWE-bench image convention and
# are pinned against a real image in the smoke (Task 6).
_RUN_TESTS = """#!/bin/bash
set -uo pipefail
cd /testbed || exit 1

# Apply the task's test patch (the tests that encode the fix). A failed apply must abort:
# a broken patch invalidates the verification, so don't swallow the error.
if ! git apply --whitespace=nowarn /tests/test_patch.diff 2>/dev/null; then
    patch -p1 < /tests/test_patch.diff || exit 1
fi

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate testbed 2>/dev/null || true

fail_to_pass=(__FAIL_TARGETS__)
pass_to_pass=(__PASS_TARGETS__)

# Nothing to verify without FAIL_TO_PASS targets -> fail rather than run the whole suite.
if [ ${#fail_to_pass[@]} -eq 0 ]; then
    echo "run-tests: no FAIL_TO_PASS targets; cannot verify resolution" >&2
    exit 1
fi

targets=("${fail_to_pass[@]}")
if [ ${#pass_to_pass[@]} -gt 0 ]; then
    targets+=("${pass_to_pass[@]}")
fi

# Neutralize the repo's own pytest addopts (moto etc. force -n auto / coverage); run just our
# targets serially. Exit 0 iff every FAIL_TO_PASS + PASS_TO_PASS target passes.
python -m pytest -o addopts="" -p no:cacheprovider -rA "${targets[@]}"
exit $?
"""

# --- docker-compose.yaml ------------------------------------------------------------
# Mirrors terminal-bench's stock template (cli/template-task/docker-compose.yaml) so the
# DockerComposeManager can find the container by name (containers.get) and mount session
# logs. The image is built from the task Dockerfile (below), which adds tmux + asciinema
# -- both required by TmuxSession. ipw's TerminalBenchTaskEnv sets T_BENCH_TASK_LOGS_PATH
# but not T_BENCH_TASK_AGENT_LOGS_PATH, so only the session-logs mount is declared (an
# unset ${VAR} source would make the compose unsubstitutable).
_COMPOSE = """services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    working_dir: /testbed
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
"""

# --- Dockerfile ---------------------------------------------------------------------
# FROM the SWE-bench eval image (the runnable env) + tmux & asciinema, which TmuxSession
# checks for via `tmux -V` / `asciinema --version` and refuses to start without. apt puts
# both on the default PATH so `docker exec` finds them (a pipx user install would not be).
_DOCKERFILE_BODY = (
    "RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema "
    "&& rm -rf /var/lib/apt/lists/*\n"
)


def swe_task_to_harbor(task: SweTask, out_root: Path) -> Path:
    task_dir = Path(out_root) / task.instance_id
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    spec = Task(
        instruction=task.problem_statement,
        category="software_engineering",
        parser_name="pytest",
        difficulty="medium",
        author_name="ipj-adp-harbor",
        author_email="ipj@local",
        tags=["adp", "swe-gym", task.repo],
        max_agent_timeout_sec=1200.0,
        max_test_timeout_sec=300.0,
    )
    spec.to_yaml(task_dir / "task.yaml")

    # The test patch must live in tests/: terminal-bench's run_tests() only copies
    # run-tests.sh + the tests/ dir into the container's /tests, so a root-level
    # test_patch.diff would never reach `git apply /tests/test_patch.diff`.
    (tests_dir / "test_patch.diff").write_text(task.test_patch)

    fail_targets = " ".join(shlex.quote(t) for t in task.fail_to_pass)
    pass_targets = " ".join(shlex.quote(t) for t in task.pass_to_pass)
    run_tests = _RUN_TESTS.replace("__FAIL_TARGETS__", fail_targets).replace("__PASS_TARGETS__", pass_targets)
    (task_dir / "run-tests.sh").write_text(run_tests)

    (task_dir / "docker-compose.yaml").write_text(_COMPOSE)
    (task_dir / "Dockerfile").write_text(f"FROM {task.image_ref}\n{_DOCKERFILE_BODY}")
    return task_dir

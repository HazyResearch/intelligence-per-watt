from __future__ import annotations

import pytest

# The `terminal_bench` import below is a hard dep of ipj[orchestrate]. Skip the whole file at
# collection time if it is missing (bare `ipj[dev]` venv), and tag the surviving tests
# `integration` so the default fast tier excludes them even when the extra IS installed.
pytest.importorskip("terminal_bench")
pytestmark = pytest.mark.integration

from terminal_bench.handlers.trial_handler import Task  # noqa: E402

from ipj.orchestrate.generate.convert import swe_task_to_harbor  # noqa: E402
from ipj.orchestrate.generate.swe_tasks import SweTask  # noqa: E402


def _task():
    return SweTask(
        instance_id="getmoto__moto-7365",
        repo="getmoto/moto",
        base_commit="7f6c9cb1",
        problem_statement="DynamoDB update_item floating-point bug",
        test_patch="diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n",
        fail_to_pass=["tests/t.py::test_a"],
        pass_to_pass=["tests/t.py::test_b"],
        image_ref="docker.io/swebench/sweb.eval.x86_64.getmoto_1776_moto-7365:latest",
    )


def test_emits_task_dir_with_required_files(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    assert task_dir == tmp_path / "getmoto__moto-7365"
    assert (task_dir / "task.yaml").exists()
    assert (task_dir / "run-tests.sh").exists()
    assert (task_dir / "docker-compose.yaml").exists()
    assert (task_dir / "Dockerfile").exists()


def test_task_yaml_validates_and_carries_instruction(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    loaded = Task.from_yaml(task_dir / "task.yaml")
    assert "DynamoDB" in loaded.instruction
    assert loaded.category == "software_engineering"
    assert loaded.parser_name.value == "pytest"


def test_run_tests_references_targets_and_patch(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    script = (task_dir / "run-tests.sh").read_text()
    assert "tests/t.py::test_a" in script  # FAIL_TO_PASS target
    assert "tests/t.py::test_b" in script  # PASS_TO_PASS target (resolved => both pass)
    assert "/tests/test_patch.diff" in script  # the patch path inside the container
    assert "git apply" in script or "patch" in script  # the test_patch is applied


def test_run_tests_does_not_swallow_a_failed_patch_or_run_whole_suite(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    script = (task_dir / "run-tests.sh").read_text()
    # A failed patch must abort (explicit `exit 1`), not fall through silently.
    assert "exit 1" in script
    # No FAIL_TO_PASS targets must fail instead of running the entire test suite.
    assert "fail_to_pass[@]} -eq 0" in script


def test_test_patch_written_into_tests_subdir(tmp_path):
    # run_tests() only copies run-tests.sh + the tests/ dir into /tests, so the patch
    # must live in tests/ (a root-level file would never reach the container).
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    assert (task_dir / "tests" / "test_patch.diff").read_text().startswith("diff --git")
    assert not (task_dir / "test_patch.diff").exists()


def test_compose_is_terminal_bench_runnable(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    compose = (task_dir / "docker-compose.yaml").read_text()
    # DockerComposeManager looks the container up by exact name; without this line
    # containers.get() raises NotFound.
    assert "container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}" in compose
    # Service name + build-from-Dockerfile (so tmux/asciinema get installed).
    assert "client:" in compose
    assert "dockerfile: Dockerfile" in compose
    # Session-logs volume mount, using the template's env vars verbatim.
    assert "${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}" in compose


def test_dockerfile_makes_image_terminal_bench_runnable(tmp_path):
    task_dir = swe_task_to_harbor(_task(), tmp_path)
    dockerfile = (task_dir / "Dockerfile").read_text()
    # FROM the instance's SWE-bench eval image...
    assert "FROM docker.io/swebench/sweb.eval.x86_64.getmoto_1776_moto-7365:latest" in dockerfile
    # ...plus the tools TmuxSession requires.
    assert "tmux" in dockerfile
    assert "asciinema" in dockerfile

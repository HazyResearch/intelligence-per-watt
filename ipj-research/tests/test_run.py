from __future__ import annotations

from ipj.harbor.run import harbor_command, run_command


def test_run_command_window_brackets_sleep():
    win = run_command(["sleep", "1"])
    assert win.returncode == 0
    assert (win.end - win.start) >= 1.0


def test_harbor_command_local_model():
    cmd = harbor_command(
        "terminal-bench/terminal-bench-2-1", "terminus-2", "openai/Qwen/Qwen3.5-27B-FP8",
        output_path="runs/p0/local", k=5, n_tasks=5,
    )
    assert cmd[:4] == ["harbor", "run", "-d", "terminal-bench/terminal-bench-2-1"]
    assert "-a" in cmd and "terminus-2" in cmd
    assert "-m" in cmd and "openai/Qwen/Qwen3.5-27B-FP8" in cmd
    assert "--n-concurrent" in cmd and "1" in cmd  # sequential for clean energy windows
    # Flags verified against Harbor 0.13.2: --jobs-dir (output), -l (task limit), -y (auto-confirm)
    assert cmd[cmd.index("--jobs-dir") + 1] == "runs/p0/local"
    assert cmd[cmd.index("-l") + 1] == "5"
    assert "-y" in cmd
    assert "--output-path" not in cmd


def test_harbor_command_omits_model_when_none():
    cmd = harbor_command("terminal-bench/terminal-bench-2-1", "oracle", None, output_path="runs/p0/o")
    assert "-m" not in cmd


def test_harbor_command_omits_task_limit_when_none():
    cmd = harbor_command("terminal-bench/terminal-bench-2-1", "oracle", None, output_path="runs/p0/o")
    assert "-l" not in cmd

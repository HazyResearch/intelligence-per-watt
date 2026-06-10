from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import MutableMapping


def prepare_git_workspace(
    metadata: MutableMapping[str, object],
    workspace: Path,
    *,
    timeout_s: int | None = None,
) -> None:
    """Clone the benchmark repository into a per-query workspace."""
    repo = str(metadata.get("repo") or "").strip()
    base_commit = str(metadata.get("base_commit") or "").strip()
    timeout = timeout_s or int(os.getenv("IPW_WORKSPACE_SETUP_TIMEOUT", "300"))

    workspace.mkdir(parents=True, exist_ok=True)
    metadata["workspace_path"] = str(workspace)

    if (workspace / ".git").exists():
        metadata["workspace_prepared"] = True
        return

    existing = [p for p in workspace.iterdir()]
    if existing:
        metadata["workspace_prepared"] = False
        metadata["workspace_error"] = "workspace_not_empty"
        return

    if not repo:
        metadata["workspace_prepared"] = False
        metadata["workspace_error"] = "missing_repo"
        return

    repo_url = repo if repo.startswith(("http://", "https://", "git@")) else f"https://github.com/{repo}.git"
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    log: list[str] = []

    def run(cmd: list[str]) -> None:
        log.append("$ " + " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.stdout:
            log.append(result.stdout[-4000:])
        if result.returncode != 0:
            raise RuntimeError(f"command failed with exit code {result.returncode}")

    try:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(workspace)])
        if base_commit:
            run(["git", "-C", str(workspace), "fetch", "--depth", "1", "origin", base_commit])
            run(["git", "-C", str(workspace), "checkout", "--force", base_commit])
        else:
            run(["git", "-C", str(workspace), "checkout", "--force"])
        metadata["workspace_prepared"] = True
        metadata["workspace_repo_url"] = repo_url
        metadata["workspace_base_commit"] = base_commit or None
    except Exception as exc:
        metadata["workspace_prepared"] = False
        metadata["workspace_error"] = str(exc)
        metadata["workspace_setup_log"] = "\n".join(log)[-12000:]
        (workspace / "WORKSPACE_SETUP_ERROR.txt").write_text(
            f"{exc}\n\n" + metadata["workspace_setup_log"],
            encoding="utf-8",
        )

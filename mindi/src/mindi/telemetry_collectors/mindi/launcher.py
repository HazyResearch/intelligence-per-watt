"""Process management helpers for the bundled Mindi energy monitor."""

from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from typing import Iterator, Mapping, MutableMapping, Sequence

import grpc

from . import _binaries
from .proto import get_stub_bundle

DEFAULT_TARGET = "127.0.0.1:50053"


def normalize_target(target: str) -> str:
    """Normalize a target string into ``host:port`` form."""

    if target.startswith("grpc://"):
        target = target[len("grpc://") :]
    if target.startswith("http://"):
        target = target[len("http://") :]
    if target.startswith("https://"):
        target = target[len("https://") :]
    if "/" in target:
        target = target.split("/", 1)[0]
    if ":" not in target:
        target = f"{target}:50052"
    return target


def wait_for_ready(target: str = DEFAULT_TARGET, *, timeout: float = 5.0) -> bool:
    """Return True when the energy monitor responds to a health check."""

    normalized = normalize_target(target)
    bundle = get_stub_bundle()
    channel = grpc.insecure_channel(normalized)
    stub = bundle.stub_factory(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        stub.Health(bundle.HealthRequestCls(), timeout=timeout)
        return True
    except Exception:
        return False
    finally:
        channel.close()


def launch_monitor(
    args: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 5.0,
    target: str = DEFAULT_TARGET,
) -> tuple[int, str]:
    """Launch the energy monitor and wait for it to become ready.

    Returns a tuple of ``(pid, target)``. Raises ``RuntimeError`` if readiness
    is not achieved within ``timeout`` seconds.
    """

    normalized = normalize_target(target)
    host, port = normalized.rsplit(":", 1)

    launch_args = list(args) if args is not None else []

    def _has_flag(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in launch_args)

    if not _has_flag("--port"):
        launch_args.extend(["--port", port])
    if not _has_flag("--bind-address"):
        launch_args.extend(["--bind-address", host])

    process = _binaries.launch("mindi-energy-monitor", launch_args, env=env)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if wait_for_ready(normalized, timeout=0.5):
            return process.pid, normalized
        time.sleep(0.25)

    try:
        process.terminate()
    except OSError:
        pass
    raise RuntimeError("Energy monitor failed to become ready")


@contextmanager
def ensure_monitor(
    target: str = DEFAULT_TARGET,
    *,
    timeout: float = 5.0,
    launch: bool = True,
    launch_args: Sequence[str] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Iterator[str]:
    """Ensure a monitor is reachable, optionally launching one in the background.

    Yields the normalized target string. If ``launch`` is True, a background
    process will be spawned when the target is unavailable. The caller is
    responsible for any teardown when the context exits.
    """

    normalized = normalize_target(target)
    pid: int | None = None
    if not wait_for_ready(normalized, timeout=timeout) and launch:
        env_map: Mapping[str, str] | None = env if env is not None else os.environ
        pid, normalized = launch_monitor(
            launch_args,
            env=env_map,
            timeout=timeout,
            target=normalized,
        )

    try:
        yield normalized
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


__all__ = [
    "DEFAULT_TARGET",
    "normalize_target",
    "wait_for_ready",
    "launch_monitor",
    "ensure_monitor",
]


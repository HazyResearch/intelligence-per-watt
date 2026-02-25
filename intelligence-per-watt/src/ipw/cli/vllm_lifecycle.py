"""vLLM server lifecycle management with lock files and process detection.

This module provides robust management of vLLM server processes:
- Lock files to track ownership and prevent conflicts
- Process detection to find vLLM instances on specific ports
- Model verification to ensure correct model is loaded
- Orphan cleanup to handle crashed/abandoned servers
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PortConflictError(Exception):
    """Raised when a port is in use by a different owner."""

    def __init__(
        self,
        port: int,
        existing_model: Optional[str] = None,
        requested_model: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        self.port = port
        self.existing_model = existing_model
        self.requested_model = requested_model
        self.owner = owner
        msg = f"Port {port} is in use"
        if existing_model:
            msg += f" by model '{existing_model}'"
        if owner:
            msg += f" (owner: {owner})"
        if requested_model:
            msg += f". Requested model: '{requested_model}'"
        super().__init__(msg)


class ModelMismatchError(Exception):
    """Raised when server is running but has wrong model loaded."""

    def __init__(
        self,
        port: int,
        expected_model: str,
        actual_model: str,
    ):
        self.port = port
        self.expected_model = expected_model
        self.actual_model = actual_model
        super().__init__(
            f"Model mismatch on port {port}: expected '{expected_model}', "
            f"got '{actual_model}'"
        )


@dataclass
class VLLMServerInfo:
    """Information about a running vLLM server for lock files."""

    pid: int
    model_id: str
    port: int
    gpu_ids: List[int] = field(default_factory=list)
    started_at: str = ""
    owner: str = "unknown"  # "ipw_cli", "external"
    owner_pid: Optional[int] = None

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        if self.owner_pid is None:
            self.owner_pid = os.getpid()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLLMServerInfo":
        """Create from dictionary."""
        return cls(**data)


class VLLMServerRegistry:
    """Lock file based registry for vLLM servers.

    Lock files are stored in ~/.cache/ipw/vllm/ with format:
        port_8000.lock - JSON file with VLLMServerInfo

    This allows multiple processes to coordinate access to vLLM servers
    and detect orphaned servers from crashed processes.
    """

    LOCK_DIR = Path.home() / ".cache" / "ipw" / "vllm"

    def __init__(self, lock_dir: Optional[Path] = None):
        self.lock_dir = lock_dir or self.LOCK_DIR
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def _lock_path(self, port: int) -> Path:
        """Get path to lock file for a port."""
        return self.lock_dir / f"port_{port}.lock"

    def acquire_lock(self, port: int, info: VLLMServerInfo) -> bool:
        """Acquire lock for a port.

        Returns:
            True if lock acquired, False if already locked by another.
        """
        lock_path = self._lock_path(port)

        existing = self.get_lock_info(port)
        if existing:
            if existing.owner_pid and _is_process_alive(existing.owner_pid):
                logger.warning(
                    f"Port {port} locked by PID {existing.owner_pid} "
                    f"(model: {existing.model_id})"
                )
                return False
            else:
                logger.info(f"Removing stale lock for port {port}")
                self.release_lock(port)

        try:
            lock_path.write_text(json.dumps(info.to_dict(), indent=2))
            logger.debug(f"Acquired lock for port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to acquire lock for port {port}: {e}")
            return False

    def release_lock(self, port: int) -> None:
        """Release lock for a port."""
        lock_path = self._lock_path(port)
        try:
            if lock_path.exists():
                lock_path.unlink()
                logger.debug(f"Released lock for port {port}")
        except Exception as e:
            logger.warning(f"Failed to release lock for port {port}: {e}")

    def get_lock_info(self, port: int) -> Optional[VLLMServerInfo]:
        """Get lock info for a port."""
        lock_path = self._lock_path(port)
        try:
            if lock_path.exists():
                data = json.loads(lock_path.read_text())
                return VLLMServerInfo.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to read lock for port {port}: {e}")
        return None

    def cleanup_stale_locks(self) -> List[int]:
        """Remove stale locks where owner process is dead.

        Returns:
            List of ports that had stale locks removed.
        """
        cleaned = []
        for lock_file in self.lock_dir.glob("port_*.lock"):
            try:
                port = int(lock_file.stem.replace("port_", ""))
                info = self.get_lock_info(port)
                if info and info.owner_pid:
                    if not _is_process_alive(info.owner_pid):
                        logger.info(
                            f"Cleaning stale lock for port {port} "
                            f"(owner PID {info.owner_pid} is dead)"
                        )
                        self.release_lock(port)
                        cleaned.append(port)
            except Exception as e:
                logger.warning(f"Error checking lock {lock_file}: {e}")
        return cleaned

    def list_locks(self) -> Dict[int, VLLMServerInfo]:
        """List all current locks."""
        locks = {}
        for lock_file in self.lock_dir.glob("port_*.lock"):
            try:
                port = int(lock_file.stem.replace("port_", ""))
                info = self.get_lock_info(port)
                if info:
                    locks[port] = info
            except Exception as e:
                logger.warning(f"Error reading lock {lock_file}: {e}")
        return locks


class VLLMProcessDetector:
    """Detect vLLM processes on the system.

    Uses a combination of:
    - lsof to find process holding a port
    - /proc to identify vLLM processes
    - HTTP queries to get model info
    """

    def find_vllm_on_port(self, port: int) -> Optional[Dict[str, Any]]:
        """Find vLLM process listening on a port."""
        if not self._is_port_in_use(port):
            return None

        pid = self._find_pid_on_port(port)
        if pid is None:
            return {"pid": None, "model": None, "port": port}

        model = self.query_loaded_model(port)
        return {
            "pid": pid,
            "model": model,
            "port": port,
            "is_vllm": self._is_vllm_process(pid),
        }

    def find_all_vllm_processes(self) -> List[Dict[str, Any]]:
        """Find all vLLM processes on the system."""
        vllm_procs = []
        try:
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        try:
                            pid = int(line.strip())
                            port = self._find_port_for_pid(pid)
                            model = self.query_loaded_model(port) if port else None
                            vllm_procs.append({
                                "pid": pid,
                                "port": port,
                                "model": model,
                            })
                        except ValueError:
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return vllm_procs

    def query_loaded_model(self, port: int, timeout: float = 5.0) -> Optional[str]:
        """Query /v1/models to get loaded model ID."""
        import json as _json
        import urllib.error
        import urllib.request

        url = f"http://localhost:{port}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = _json.loads(response.read())
                    models = data.get("data", [])
                    if models:
                        return models[0].get("id")
        except (urllib.error.URLError, TimeoutError, Exception):
            pass
        return None

    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill a process."""
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            logger.info(f"Killed process {pid} with signal {sig.name}")
            return True
        except ProcessLookupError:
            logger.debug(f"Process {pid} not found")
            return True
        except PermissionError:
            logger.warning(f"Permission denied to kill process {pid}")
            return False
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _find_pid_on_port(self, port: int) -> Optional[int]:
        """Find PID of process listening on port using lsof."""
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                first_line = result.stdout.strip().split("\n")[0]
                return int(first_line)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None

    def _find_port_for_pid(self, pid: int) -> Optional[int]:
        """Find which port a PID is listening on."""
        try:
            result = subprocess.run(
                ["lsof", "-Pan", "-p", str(pid), "-i", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "LISTEN" in line:
                        parts = line.split()
                        for part in parts:
                            if ":" in part:
                                try:
                                    port_str = part.split(":")[-1]
                                    return int(port_str)
                                except ValueError:
                                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _is_vllm_process(self, pid: int) -> bool:
        """Check if a PID is a vLLM process."""
        try:
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            if cmdline_path.exists():
                cmdline = cmdline_path.read_text()
                return "vllm" in cmdline.lower()
        except Exception:
            pass
        return False


def _is_process_alive(pid: int) -> bool:
    """Check if a process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_orphaned_servers(
    registry: Optional[VLLMServerRegistry] = None,
    detector: Optional[VLLMProcessDetector] = None,
    kill_orphans: bool = True,
) -> List[int]:
    """Clean up orphaned vLLM servers.

    Finds servers where:
    1. Lock file exists but owner process is dead
    2. vLLM process is running but has no lock file (external)

    Returns:
        List of ports that were cleaned up.
    """
    registry = registry or VLLMServerRegistry()
    detector = detector or VLLMProcessDetector()

    cleaned = []

    stale_ports = registry.cleanup_stale_locks()
    for port in stale_ports:
        proc_info = detector.find_vllm_on_port(port)
        if proc_info and proc_info.get("pid") and kill_orphans:
            logger.info(f"Killing orphaned vLLM on port {port} (PID {proc_info['pid']})")
            detector.kill_process(proc_info["pid"], force=True)
            cleaned.append(port)

    return cleaned


__all__ = [
    "VLLMServerInfo",
    "VLLMServerRegistry",
    "VLLMProcessDetector",
    "PortConflictError",
    "ModelMismatchError",
    "cleanup_orphaned_servers",
]

"""Tests for vLLM server lifecycle management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipw.cli.vllm_lifecycle import (
    ModelMismatchError,
    PortConflictError,
    VLLMProcessDetector,
    VLLMServerInfo,
    VLLMServerRegistry,
    _is_process_alive,
    cleanup_orphaned_servers,
)


class TestVLLMServerInfo:
    """Tests for VLLMServerInfo dataclass."""

    def test_create_with_defaults(self) -> None:
        info = VLLMServerInfo(pid=1234, model_id="test/model", port=8000)
        assert info.pid == 1234
        assert info.model_id == "test/model"
        assert info.port == 8000
        assert info.gpu_ids == []
        assert info.started_at != ""
        assert info.owner_pid is not None

    def test_to_dict_round_trip(self) -> None:
        info = VLLMServerInfo(
            pid=1234,
            model_id="test/model",
            port=8000,
            gpu_ids=[0, 1],
            owner="ipw_cli",
        )
        d = info.to_dict()
        restored = VLLMServerInfo.from_dict(d)
        assert restored.pid == info.pid
        assert restored.model_id == info.model_id
        assert restored.port == info.port
        assert restored.gpu_ids == info.gpu_ids
        assert restored.owner == info.owner

    def test_to_dict_is_json_serializable(self) -> None:
        info = VLLMServerInfo(pid=1234, model_id="test/model", port=8000)
        serialized = json.dumps(info.to_dict())
        assert "test/model" in serialized


class TestVLLMServerRegistry:
    """Tests for lock file based server registry."""

    def test_acquire_and_release_lock(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)
        info = VLLMServerInfo(pid=1234, model_id="test/model", port=8000)

        assert registry.acquire_lock(8000, info)
        assert (tmp_path / "port_8000.lock").exists()

        registry.release_lock(8000)
        assert not (tmp_path / "port_8000.lock").exists()

    def test_get_lock_info(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)
        info = VLLMServerInfo(pid=1234, model_id="test/model", port=8000)

        registry.acquire_lock(8000, info)
        retrieved = registry.get_lock_info(8000)

        assert retrieved is not None
        assert retrieved.pid == 1234
        assert retrieved.model_id == "test/model"

    def test_get_lock_info_nonexistent(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)
        assert registry.get_lock_info(9999) is None

    def test_list_locks(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)

        info1 = VLLMServerInfo(pid=1001, model_id="model-a", port=8000)
        info2 = VLLMServerInfo(pid=1002, model_id="model-b", port=8001)
        registry.acquire_lock(8000, info1)
        registry.acquire_lock(8001, info2)

        locks = registry.list_locks()
        assert len(locks) == 2
        assert 8000 in locks
        assert 8001 in locks

    def test_cleanup_stale_locks_removes_dead_process(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)

        # Use a PID that definitely doesn't exist
        info = VLLMServerInfo(
            pid=99999999,
            model_id="test/model",
            port=8000,
            owner_pid=99999999,
        )
        # Write the lock file directly to bypass alive check
        lock_path = tmp_path / "port_8000.lock"
        lock_path.write_text(json.dumps(info.to_dict()))

        cleaned = registry.cleanup_stale_locks()
        assert 8000 in cleaned
        assert not lock_path.exists()

    def test_acquire_fails_when_locked_by_alive_process(self, tmp_path: Path) -> None:
        registry = VLLMServerRegistry(lock_dir=tmp_path)

        # Use current process PID (which is alive)
        import os
        info = VLLMServerInfo(
            pid=os.getpid(),
            model_id="test/model",
            port=8000,
            owner_pid=os.getpid(),
        )
        assert registry.acquire_lock(8000, info)

        # Second acquire should fail
        info2 = VLLMServerInfo(
            pid=9999,
            model_id="other/model",
            port=8000,
            owner_pid=os.getpid(),
        )
        assert not registry.acquire_lock(8000, info2)


class TestPortConflictError:
    """Tests for PortConflictError."""

    def test_basic_error(self) -> None:
        err = PortConflictError(port=8000)
        assert "8000" in str(err)

    def test_with_model_info(self) -> None:
        err = PortConflictError(
            port=8000,
            existing_model="old/model",
            requested_model="new/model",
            owner="ipw_cli",
        )
        assert "old/model" in str(err)
        assert "new/model" in str(err)
        assert "ipw_cli" in str(err)


class TestModelMismatchError:
    """Tests for ModelMismatchError."""

    def test_error_message(self) -> None:
        err = ModelMismatchError(
            port=8000,
            expected_model="expected/model",
            actual_model="actual/model",
        )
        assert "expected/model" in str(err)
        assert "actual/model" in str(err)
        assert "8000" in str(err)


class TestIsProcessAlive:
    """Tests for _is_process_alive helper."""

    def test_current_process_is_alive(self) -> None:
        import os
        assert _is_process_alive(os.getpid())

    def test_nonexistent_pid_is_not_alive(self) -> None:
        assert not _is_process_alive(99999999)


class TestVLLMProcessDetector:
    """Tests for VLLMProcessDetector."""

    def test_port_not_in_use_returns_none(self) -> None:
        detector = VLLMProcessDetector()
        # Use a port that's very unlikely to be in use
        result = detector.find_vllm_on_port(59999)
        assert result is None

    def test_kill_nonexistent_process(self) -> None:
        detector = VLLMProcessDetector()
        # Should return True (process already dead)
        assert detector.kill_process(99999999)

"""Inference server lifecycle management for vLLM and Ollama.

This module provides centralized management of inference server processes,
ensuring that startup/warmup costs are excluded from energy profiling.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ipw.cli._console import error, info, warning
from ipw.cli.vllm_lifecycle import (
    ModelMismatchError,
    PortConflictError,
    VLLMProcessDetector,
    VLLMServerInfo,
    VLLMServerRegistry,
)


def find_available_gpus(
    min_free_memory_gb: float = 20.0,
    count: int = 1,
) -> List[int]:
    """Find GPU indices with sufficient free memory.

    Args:
        min_free_memory_gb: Minimum free memory required per GPU in GB.
        count: Number of GPUs needed.

    Returns:
        List of GPU indices with sufficient free memory.
        Returns empty list if not enough GPUs are available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        available = []
        min_free_mb = min_free_memory_gb * 1024

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    gpu_index = int(parts[0].strip())
                    free_mb = float(parts[1].strip())
                    if free_mb >= min_free_mb:
                        available.append(gpu_index)
                except (ValueError, IndexError):
                    continue

        return available[:count] if len(available) >= count else []

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return []


def _is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if not _is_port_in_use(port):
            return port
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}"
    )


@dataclass
class ServerConfig:
    """Configuration for an inference server instance."""

    model_id: str
    alias: str
    backend: str  # "vllm" or "ollama"
    port: int = 8000
    gpu_ids: List[int] = field(default_factory=list)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    extra_args: Dict[str, Any] = field(default_factory=dict)
    launch_module: str = "vllm.entrypoints.openai.api_server"
    env_vars: Dict[str, str] = field(default_factory=dict)


class InferenceServerManager:
    """Manages vLLM and Ollama server lifecycle with telemetry isolation.

    This class ensures that server startup, warmup, and shutdown costs are
    excluded from energy profiling by handling these operations outside
    the telemetry session.

    Example usage:
        configs = [
            ServerConfig(model_id="Qwen/Qwen3-4B", alias="main", backend="vllm", port=8000),
        ]
        manager = InferenceServerManager(configs)

        # Before telemetry session
        urls = manager.start_all()
        manager.warmup_all()

        # ... run benchmark with telemetry ...

        # After telemetry session
        manager.stop_all()
    """

    def __init__(
        self,
        configs: List[ServerConfig],
        auto_assign_gpus: bool = True,
        conflict_policy: Literal["fail", "kill", "skip_port"] = "fail",
        owner: str = "ipw_cli",
    ):
        self.configs = configs
        self._processes: Dict[str, subprocess.Popen] = {}
        self._urls: Dict[str, str] = {}
        self._conflict_policy = conflict_policy
        self._owner = owner

        self._registry = VLLMServerRegistry()
        self._detector = VLLMProcessDetector()

        if auto_assign_gpus:
            self._assign_gpus()

    def _assign_gpus(self) -> None:
        """Auto-assign GPUs to models without explicit GPU assignment."""
        total_gpus_needed = sum(
            config.tensor_parallel_size
            for config in self.configs
            if config.backend == "vllm" and not config.gpu_ids
        )

        if total_gpus_needed == 0:
            return

        available_gpus = find_available_gpus(
            min_free_memory_gb=20.0,
            count=total_gpus_needed,
        )

        if len(available_gpus) < total_gpus_needed:
            warning(
                f"Only found {len(available_gpus)} GPUs with sufficient memory, "
                f"need {total_gpus_needed}. Falling back to sequential assignment."
            )
            next_gpu = 0
            for config in self.configs:
                if config.backend == "vllm" and not config.gpu_ids:
                    config.gpu_ids = list(
                        range(next_gpu, next_gpu + config.tensor_parallel_size)
                    )
                    next_gpu += config.tensor_parallel_size
            return

        info(f"Found {len(available_gpus)} available GPU(s): {available_gpus}")
        gpu_idx = 0
        for config in self.configs:
            if config.backend == "vllm" and not config.gpu_ids:
                config.gpu_ids = available_gpus[gpu_idx : gpu_idx + config.tensor_parallel_size]
                gpu_idx += config.tensor_parallel_size

    def start_all(self, wait_timeout: float = 300.0) -> Dict[str, str]:
        """Start all configured servers.

        Returns:
            Dictionary mapping alias to base URL for each server.
        """
        for config in self.configs:
            if config.backend == "vllm":
                self._start_vllm(config, wait_timeout)
            elif config.backend == "ollama":
                self._start_ollama(config, wait_timeout)
            else:
                raise ValueError(f"Unknown backend: {config.backend}")

        return self._urls.copy()

    def stop_all(self) -> None:
        """Stop all managed servers."""
        for alias, process in list(self._processes.items()):
            info(f"Stopping server: {alias}")
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except Exception as e:
                warning(f"Error stopping {alias}: {e}")
            finally:
                del self._processes[alias]

        for config in self.configs:
            if config.backend == "vllm":
                self._registry.release_lock(config.port)

        self._urls.clear()

    def verify_model_match(self, url: str, expected_model: str) -> bool:
        """Query /v1/models and verify expected model is loaded."""
        import json
        import urllib.error
        import urllib.request

        models_url = url.rstrip("/")
        if not models_url.endswith("/v1"):
            models_url = f"{models_url}/v1"
        models_url = f"{models_url}/models"

        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
                loaded_models = [m["id"] for m in data.get("data", [])]
                for loaded in loaded_models:
                    if expected_model in loaded or loaded in expected_model:
                        return True
                return False
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            warning(f"Failed to verify model: {e}")
            return False

    def _handle_port_conflict(self, port: int, expected_model: str) -> int:
        """Handle port conflict according to configured policy."""
        proc_info = self._detector.find_vllm_on_port(port)
        existing_model = proc_info.get("model") if proc_info else None
        existing_pid = proc_info.get("pid") if proc_info else None

        lock_info = self._registry.get_lock_info(port)

        if self._conflict_policy == "fail":
            owner = lock_info.owner if lock_info else "external"
            raise PortConflictError(
                port=port,
                existing_model=existing_model,
                requested_model=expected_model,
                owner=owner,
            )

        elif self._conflict_policy == "kill":
            if existing_pid:
                info(f"Killing conflicting process {existing_pid} on port {port}")
                self._detector.kill_process(existing_pid, force=True)
                time.sleep(2)
            self._registry.release_lock(port)
            return port

        elif self._conflict_policy == "skip_port":
            new_port = _find_available_port(port + 1)
            info(f"Port {port} in use, using port {new_port} instead")
            return new_port

        return port

    def warmup(self, alias: str, warmup_prompt: str = "Hello") -> None:
        """Run a warmup query on a specific server."""
        if alias not in self._urls:
            warning(f"Cannot warmup unknown server: {alias}")
            return

        config = self._get_config_by_alias(alias)
        if config is None:
            return

        url = self._urls[alias]
        info(f"Warming up {alias} at {url}...")

        try:
            if config.backend == "vllm":
                self._warmup_vllm(url, config.model_id, warmup_prompt)
            elif config.backend == "ollama":
                self._warmup_ollama(url, config.model_id, warmup_prompt)
        except Exception as e:
            warning(f"Warmup failed for {alias}: {e}")

    def warmup_all(self, warmup_prompt: str = "Hello") -> None:
        """Run warmup queries on all servers."""
        for config in self.configs:
            self.warmup(config.alias, warmup_prompt)

    def _get_config_by_alias(self, alias: str) -> Optional[ServerConfig]:
        """Get server config by alias."""
        for config in self.configs:
            if config.alias == alias:
                return config
        return None

    def _start_vllm(self, config: ServerConfig, wait_timeout: float) -> None:
        """Start a vLLM server process."""
        actual_port = config.port
        if _is_port_in_use(config.port):
            lock_info = self._registry.get_lock_info(config.port)
            if lock_info and lock_info.model_id == config.model_id:
                url = f"http://localhost:{config.port}/v1"
                if self.verify_model_match(url, config.model_id):
                    info(f"Reusing existing vLLM server on port {config.port} ({config.model_id})")
                    self._urls[config.alias] = url
                    return

            actual_port = self._handle_port_conflict(config.port, config.model_id)
            config.port = actual_port

        info(f"Starting vLLM server: {config.alias} ({config.model_id})")
        info(f"  Port: {config.port}, GPUs: {config.gpu_ids}")

        env = os.environ.copy()
        env.pop("RUST_LOG", None)
        env.pop("LD_LIBRARY_PATH", None)

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            info(f"  Using user-specified GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        elif config.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_ids)

        if config.env_vars:
            env.update(config.env_vars)

        import sys

        tool_parser = config.extra_args.pop("tool_call_parser", None)

        cmd = [
            sys.executable,
            "-m",
            config.launch_module,
            "--model",
            config.model_id,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-model-len",
            str(config.max_model_len),
        ]

        if tool_parser:
            cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])

        for key, value in config.extra_args.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=None,
            stderr=None,
            preexec_fn=os.setsid,
        )

        self._processes[config.alias] = process
        base_url = f"http://localhost:{config.port}/v1"
        self._urls[config.alias] = base_url

        lock_info = VLLMServerInfo(
            pid=process.pid,
            model_id=config.model_id,
            port=config.port,
            gpu_ids=config.gpu_ids,
            owner=self._owner,
        )
        if not self._registry.acquire_lock(config.port, lock_info):
            warning(f"Failed to acquire lock for port {config.port}")

        if not self._wait_for_vllm_ready(config.port, wait_timeout, config.model_id, process):
            if process.poll() is not None:
                error(f"vLLM process exited with code {process.returncode}")
            self._registry.release_lock(config.port)
            raise RuntimeError(
                f"vLLM server {config.alias} failed to start within {wait_timeout}s"
            )

        if not self.verify_model_match(base_url, config.model_id):
            actual_model = self._detector.query_loaded_model(config.port)
            process.terminate()
            process.wait(timeout=10)
            self._registry.release_lock(config.port)
            raise ModelMismatchError(
                port=config.port,
                expected_model=config.model_id,
                actual_model=actual_model or "unknown",
            )

        info(f"vLLM server {config.alias} ready at {base_url}")

    def _start_ollama(self, config: ServerConfig, wait_timeout: float) -> None:
        """Ensure Ollama is running and pull/load the model."""
        info(f"Setting up Ollama model: {config.alias} ({config.model_id})")

        base_url = f"http://localhost:{config.port}"
        self._urls[config.alias] = base_url

        if not self._wait_for_ollama_ready(config.port, timeout=10.0):
            info("Ollama not running, attempting to start...")
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if not self._wait_for_ollama_ready(config.port, timeout=30.0):
                    raise RuntimeError("Failed to start Ollama server")
            except FileNotFoundError:
                raise RuntimeError(
                    "Ollama not installed. Install from https://ollama.ai"
                )

        info(f"Pulling Ollama model: {config.model_id}")
        result = subprocess.run(
            ["ollama", "pull", config.model_id],
            capture_output=True,
            timeout=wait_timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to pull Ollama model: {result.stderr.decode()}"
            )

        info(f"Ollama model {config.alias} ready at {base_url}")

    def _wait_for_vllm_ready(
        self,
        port: int,
        timeout: float,
        model_id: str | None = None,
        process: subprocess.Popen | None = None,
    ) -> bool:
        """Wait for vLLM server to be fully ready for inference."""
        import json
        import urllib.error
        import urllib.request

        models_url = f"http://localhost:{port}/v1/models"
        completions_url = f"http://localhost:{port}/v1/chat/completions"
        start_time = time.time()

        def _process_alive() -> bool:
            if process is None:
                return True
            return process.poll() is None

        # Phase 1: Wait for /v1/models to return 200
        while time.time() - start_time < timeout:
            if not _process_alive():
                error(f"vLLM process exited early (code {process.returncode}) during startup")
                return False
            try:
                req = urllib.request.Request(models_url, method="GET")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        if model_id is None:
                            data = json.loads(response.read())
                            if data.get("data"):
                                model_id = data["data"][0]["id"]
                        break
            except (urllib.error.URLError, TimeoutError, Exception):
                time.sleep(1.0)
        else:
            return False

        if model_id is None:
            return False

        # Phase 2: Wait for model to actually handle inference
        while time.time() - start_time < timeout:
            if not _process_alive():
                error(f"vLLM process exited early (code {process.returncode}) during model load")
                return False
            try:
                data = json.dumps(
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    }
                ).encode()
                req = urllib.request.Request(
                    completions_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status == 200:
                        return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    time.sleep(2.0)
                    continue
                time.sleep(1.0)
            except (urllib.error.URLError, TimeoutError, Exception):
                time.sleep(1.0)

        return False

    def _wait_for_ollama_ready(self, port: int, timeout: float) -> bool:
        """Wait for Ollama server to respond."""
        import urllib.error
        import urllib.request

        url = f"http://localhost:{port}/api/version"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, Exception):
                time.sleep(0.5)

        return False

    def _warmup_vllm(self, base_url: str, model_id: str, prompt: str) -> None:
        """Run warmup query on vLLM server."""
        import json
        import urllib.request

        url = f"{base_url}/chat/completions"
        data = json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            _ = response.read()

    def _warmup_ollama(self, base_url: str, model_id: str, prompt: str) -> None:
        """Run warmup query on Ollama server."""
        import json
        import urllib.request

        url = f"{base_url}/api/generate"
        data = json.dumps(
            {
                "model": model_id,
                "prompt": prompt,
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            _ = response.read()

    def get_url(self, alias: str) -> Optional[str]:
        """Get the base URL for a server by alias."""
        return self._urls.get(alias)

    def is_running(self, alias: str) -> bool:
        """Check if a server is running."""
        if alias in self._processes:
            return self._processes[alias].poll() is None
        return alias in self._urls


def parse_submodel_spec(spec: str) -> ServerConfig:
    """Parse a submodel specification string into ServerConfig.

    Format: alias:backend:model_id
    Examples:
        - "math:vllm:Qwen/Qwen2.5-Math-72B-Instruct"
        - "small:ollama:llama3.2:1b"
    """
    parts = spec.split(":", 2)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid submodel spec: {spec}. "
            f"Expected format: alias:backend:model_id"
        )

    alias, backend, model_id = parts[0], parts[1], parts[2]

    if backend not in ("vllm", "ollama"):
        raise ValueError(
            f"Invalid backend: {backend}. Supported: vllm, ollama"
        )

    return ServerConfig(
        model_id=model_id,
        alias=alias,
        backend=backend,
        port=11434 if backend == "ollama" else 8000,
    )


def build_server_configs(
    main_model: str,
    main_alias: str,
    submodel_specs: List[str],
    base_port: int = 8000,
    main_backend: str = "vllm",
) -> List[ServerConfig]:
    """Build server configurations for main model and submodels."""
    configs = []
    vllm_port = base_port

    main_config = ServerConfig(
        model_id=main_model,
        alias=main_alias,
        backend=main_backend,
        port=vllm_port if main_backend == "vllm" else 11434,
    )
    configs.append(main_config)
    if main_backend == "vllm":
        vllm_port += 1

    for spec in submodel_specs:
        config = parse_submodel_spec(spec)
        if config.backend == "vllm":
            config.port = vllm_port
            vllm_port += 1
        configs.append(config)

    return configs


__all__ = [
    "ServerConfig",
    "InferenceServerManager",
    "parse_submodel_spec",
    "build_server_configs",
    "find_available_gpus",
    "PortConflictError",
    "ModelMismatchError",
]

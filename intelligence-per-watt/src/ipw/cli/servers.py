"""Inference server management commands.

Provides commands to start, stop, and check status of inference servers
used for local model serving (Ollama, vLLM).

Usage:
    # Start servers
    ipw servers start --ollama
    ipw servers start --vllm --model Qwen/Qwen3-4B

    # Launch and wait for ready (recommended for benchmarking)
    ipw servers launch --vllm --model Qwen/Qwen3-4B --wait-timeout 120

    # Stop servers
    ipw servers stop --all

    # Check status
    ipw servers status
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

import click

from ipw.cli._console import error, info, success, warning

# Server health check timeouts
DEFAULT_WAIT_TIMEOUT = 60  # seconds
POLL_INTERVAL = 1.0  # seconds


@click.group()
def servers() -> None:
    """Manage inference servers (Ollama, vLLM)."""
    pass


@servers.command()
@click.option("--ollama", is_flag=True, help="Start Ollama server")
@click.option("--vllm", is_flag=True, help="Start vLLM server")
@click.option("--model", type=str, default=None, help="Model to load (required for vLLM)")
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to run server on (default: 11434 for Ollama, 8000 for vLLM)",
)
@click.option(
    "--num-parallel",
    type=int,
    default=None,
    help="Ollama: number of requests to process in parallel (OLLAMA_NUM_PARALLEL)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (default: 0.9)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Number of GPUs for tensor parallelism (default: 1)",
)
def start(
    ollama: bool,
    vllm: bool,
    model: Optional[str],
    port: Optional[int],
    num_parallel: Optional[int],
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> None:
    """Start inference server(s).

    Examples:
        ipw servers start --ollama
        ipw servers start --ollama --num-parallel 8
        ipw servers start --vllm --model Qwen/Qwen3-4B
        ipw servers start --vllm --model Qwen/Qwen3-4B --tensor-parallel-size 2
    """
    if not ollama and not vllm:
        error("Please specify --ollama or --vllm")
        raise click.Abort()

    if ollama and vllm:
        error("Please specify only one of --ollama or --vllm")
        raise click.Abort()

    if ollama:
        _start_ollama(port, num_parallel)
    elif vllm:
        if not model:
            error("--model is required for vLLM")
            raise click.Abort()
        _start_vllm(model, port, gpu_memory_utilization, tensor_parallel_size, None)


def _start_ollama(port: Optional[int], num_parallel: Optional[int] = None) -> None:
    """Start Ollama server."""
    actual_port = port or 11434

    info(f"Starting Ollama server on port {actual_port}...")

    try:
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error("Ollama not found. Install from https://ollama.ai")
            raise click.Abort()

        env = {"OLLAMA_HOST": f"0.0.0.0:{actual_port}"}
        if num_parallel is not None:
            env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
        subprocess.Popen(
            ["ollama", "serve"],
            env={**subprocess.os.environ, **env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        success(f"Ollama server started on http://localhost:{actual_port}")
        if num_parallel is not None:
            info(f"OLLAMA_NUM_PARALLEL={num_parallel}")
        info("Use 'ollama pull <model>' to download models")

    except FileNotFoundError:
        error("Ollama not found. Install from https://ollama.ai")
        raise click.Abort()


def _start_vllm(
    model: str,
    port: Optional[int],
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    extra_args: Optional[dict] = None,
) -> None:
    """Start vLLM server.

    Args:
        extra_args: Additional vLLM flags from a preset (e.g. reasoning_parser,
            tool_call_parser, language_model_only).
    """
    import os

    actual_port = port or 8000

    info(f"Starting vLLM server with model {model} on port {actual_port}...")

    try:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(actual_port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
        ]

        if extra_args:
            ea = dict(extra_args)
            # Pop keys already handled above
            ea.pop("tensor_parallel_size", None)

            tool_parser = ea.pop("tool_call_parser", None)
            if tool_parser:
                cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])

            for key, value in ea.items():
                if value is None:
                    continue
                flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        env = os.environ.copy()
        env.pop("RUST_LOG", None)
        env.pop("LD_LIBRARY_PATH", None)

        subprocess.Popen(
            cmd,
            env=env,
            stdout=None,
            stderr=None,
            preexec_fn=os.setsid,
        )
        success(f"vLLM server started on http://localhost:{actual_port}")
        info(f"Model: {model}")
        info(f"Tensor parallel size: {tensor_parallel_size}")
        info("OpenAI-compatible API available at /v1/")

    except Exception as e:
        error(f"Failed to start vLLM: {e}")
        raise click.Abort()


@servers.command()
@click.option("--ollama", is_flag=True, help="Launch Ollama server")
@click.option("--vllm", is_flag=True, help="Launch vLLM server")
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model to load (required for vLLM, optional for Ollama to pre-pull)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to run server on (default: 11434 for Ollama, 8000 for vLLM)",
)
@click.option(
    "--preset",
    type=str,
    default=None,
    help="Model preset name (e.g. glm-4.7-flash, qwen3-30b-a3b)",
)
@click.option(
    "--num-parallel",
    type=int,
    default=None,
    help="Ollama: number of requests to process in parallel (OLLAMA_NUM_PARALLEL)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (default: 0.9)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="Number of GPUs for tensor parallelism (auto from preset or default: 1)",
)
@click.option(
    "--wait-timeout",
    type=int,
    default=DEFAULT_WAIT_TIMEOUT,
    help=f"Timeout in seconds waiting for server to be ready (default: {DEFAULT_WAIT_TIMEOUT})",
)
def launch(
    ollama: bool,
    vllm: bool,
    model: Optional[str],
    port: Optional[int],
    preset: Optional[str],
    num_parallel: Optional[int],
    gpu_memory_utilization: float,
    tensor_parallel_size: Optional[int],
    wait_timeout: int,
) -> None:
    """Launch inference server and wait until ready.

    This command starts the server and blocks until it's ready to accept
    requests. Recommended for use before running benchmarks to ensure
    server warmup costs are excluded from measurements.

    Examples:
        ipw servers launch --ollama
        ipw servers launch --ollama --num-parallel 8
        ipw servers launch --vllm --model Qwen/Qwen3-4B
        ipw servers launch --vllm --preset glm-4.7-flash
        ipw servers launch --vllm --model openai/gpt-oss-120b --tensor-parallel-size 4
    """
    if not ollama and not vllm:
        error("Please specify --ollama or --vllm")
        raise click.Abort()

    if ollama and vllm:
        error("Please specify only one of --ollama or --vllm")
        raise click.Abort()

    # Resolve preset if provided
    preset_extra_args: Optional[dict] = None
    if preset:
        from ipw.cli.model_presets import resolve_preset
        try:
            preset_config = resolve_preset(preset)
        except KeyError as exc:
            error(str(exc))
            raise click.Abort()
        if model:
            warning(f"--model overrides preset model ({preset_config['model_id']})")
        else:
            model = preset_config["model_id"]
        vllm_args = preset_config.get("vllm_args", {})
        if tensor_parallel_size is None:
            tensor_parallel_size = vllm_args.get("tensor_parallel_size", 1)
        # Collect remaining vllm_args (excluding tensor_parallel_size) for _start_vllm
        preset_extra_args = {k: v for k, v in vllm_args.items() if k != "tensor_parallel_size"}

    if tensor_parallel_size is None:
        tensor_parallel_size = 1

    if ollama:
        _launch_ollama(port, model, wait_timeout, num_parallel)
    elif vllm:
        if not model:
            error("--model is required for vLLM (or use --preset)")
            raise click.Abort()
        _launch_vllm(model, port, gpu_memory_utilization, tensor_parallel_size, wait_timeout, preset_extra_args)


def _launch_ollama(port: Optional[int], model: Optional[str], timeout: int, num_parallel: Optional[int] = None) -> None:
    """Launch Ollama server and wait for it to be ready."""
    actual_port = port or 11434

    if _check_ollama_status():
        info("Ollama server already running")
    else:
        _start_ollama(port, num_parallel)
        info("Waiting for Ollama to be ready...")

        if not _wait_for_server("ollama", actual_port, timeout):
            error(f"Ollama server not ready after {timeout}s")
            raise click.Abort()

    success(f"Ollama server ready at http://localhost:{actual_port}")

    if model:
        info(f"Pulling model {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                success(f"Model {model} ready")
                info("Running warmup inference...")
                subprocess.run(
                    ["ollama", "run", model, "Hello"],
                    capture_output=True,
                    timeout=60,
                )
                success("Warmup complete")
            else:
                warning(f"Failed to pull model: {result.stderr}")
        except subprocess.TimeoutExpired:
            warning("Model pull timed out")
        except Exception as e:
            warning(f"Failed to pull model: {e}")


def _launch_vllm(
    model: str,
    port: Optional[int],
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    timeout: int,
    extra_args: Optional[dict] = None,
) -> None:
    """Launch vLLM server and wait for it to be ready."""
    actual_port = port or 8000

    if _check_vllm_status(actual_port):
        info("vLLM server already running")
        success(f"vLLM server ready at http://localhost:{actual_port}")
        return

    _start_vllm(model, port, gpu_memory_utilization, tensor_parallel_size, extra_args)
    info(f"Waiting for vLLM to load model {model} (this may take a while)...")

    if not _wait_for_server("vllm", actual_port, timeout):
        error(f"vLLM server not ready after {timeout}s")
        info("Tip: Try increasing --wait-timeout for larger models")
        raise click.Abort()

    success(f"vLLM server ready at http://localhost:{actual_port}")
    info("OpenAI-compatible API available at /v1/")


def _wait_for_server(server_type: str, port: int, timeout: int) -> bool:
    """Wait for server to become ready."""
    import urllib.error
    import urllib.request

    if server_type == "ollama":
        url = f"http://localhost:{port}/api/version"
    else:
        url = f"http://localhost:{port}/v1/models"

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, Exception):
            pass
        time.sleep(POLL_INTERVAL)

    return False


@servers.command()
@click.option("--ollama", is_flag=True, help="Stop Ollama server")
@click.option("--vllm", is_flag=True, help="Stop vLLM server")
@click.option("--all", "stop_all", is_flag=True, help="Stop all managed inference servers")
def stop(ollama: bool, vllm: bool, stop_all: bool) -> None:
    """Stop inference server(s).

    Examples:
        ipw servers stop --ollama
        ipw servers stop --vllm
        ipw servers stop --all
    """
    if not ollama and not vllm and not stop_all:
        error("Please specify --ollama, --vllm, or --all")
        raise click.Abort()

    if stop_all or ollama:
        _stop_ollama()

    if stop_all or vllm:
        _stop_vllm()


def _stop_ollama() -> None:
    """Stop Ollama server."""
    info("Stopping Ollama server...")
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
        success("Ollama server stopped")
    except Exception as e:
        warning(f"Could not stop Ollama: {e}")


def _stop_vllm() -> None:
    """Stop vLLM server."""
    from ipw.cli.vllm_lifecycle import VLLMServerRegistry

    info("Stopping vLLM server...")
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
        )
        success("vLLM server stopped")
    except Exception as e:
        warning(f"Could not stop vLLM: {e}")

    # Clean up any lock files
    try:
        registry = VLLMServerRegistry()
        cleaned = registry.cleanup_stale_locks()
        if cleaned:
            info(f"Cleaned up {len(cleaned)} stale lock file(s)")
    except Exception:
        pass


@servers.command()
def status() -> None:
    """Show status of inference servers."""
    from ipw.cli.vllm_lifecycle import VLLMServerRegistry

    info("Checking inference server status...\n")

    # Check Ollama
    ollama_status = _check_ollama_status()
    if ollama_status:
        success(f"Ollama: Running on {ollama_status}")
    else:
        warning("Ollama: Not running")

    # Check vLLM
    vllm_status = _check_vllm_status()
    if vllm_status:
        success(f"vLLM: Running on {vllm_status}")
        # Show loaded model
        try:
            from ipw.cli.vllm_lifecycle import VLLMProcessDetector
            detector = VLLMProcessDetector()
            model = detector.query_loaded_model(8000)
            if model:
                info(f"  Model: {model}")
        except Exception:
            pass
    else:
        warning("vLLM: Not running")

    # Show lock files
    try:
        registry = VLLMServerRegistry()
        locks = registry.list_locks()
        if locks:
            info("\nRegistered servers:")
            for port, lock_info in locks.items():
                info(f"  Port {port}: {lock_info.model_id} (PID {lock_info.pid}, owner: {lock_info.owner})")
    except Exception:
        pass


def _check_ollama_status() -> Optional[str]:
    """Check if Ollama is running and return endpoint if so."""
    try:
        import urllib.error
        import urllib.request

        url = "http://localhost:11434/api/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return "http://localhost:11434"
    except (Exception,):
        pass
    return None


def _check_vllm_status(port: int = 8000) -> Optional[str]:
    """Check if vLLM is running and return endpoint if so."""
    try:
        import urllib.error
        import urllib.request

        url = f"http://localhost:{port}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return f"http://localhost:{port}"
    except (Exception,):
        pass
    return None


__all__ = ["servers"]

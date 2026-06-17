"""vLLM MCP server for large open-source models.

This server connects to a vLLM inference server (OpenAI-compatible API)
to run large models like Qwen3-32B, Llama-70B, and specialist models
(math, code) that require tensor parallelism across multiple GPUs.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, Optional

import httpx

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult

# Module-level counter for retry warnings (to reduce noise)
_retry_warn_count = 0
_REQUEST_TIMEOUT_SECONDS = 240.0


def _parse_vllm_context_budget_error(text: str) -> tuple[int, int] | None:
    """Return (context_limit, actual_input_tokens) from vLLM validation text."""
    lowered = text.lower()
    limit_match = re.search(r"maximum context length is\s+(\d+)", lowered)
    input_match = (
        re.search(r"your request has\s+(\d+)\s+input", lowered)
        or re.search(r"\((\d+)\s+in the messages?", lowered)
    )
    if not limit_match or not input_match:
        return None
    return int(limit_match.group(1)), int(input_match.group(1))


class VLLMMCPServer(BaseMCPServer):
    """MCP server for vLLM-served models.

    vLLM provides an OpenAI-compatible API for serving large open-source
    models with optimizations like PagedAttention, continuous batching,
    and tensor parallelism.

    Supported model categories:
    - General: Qwen3-32B, Qwen3-8B, Llama-3.3-70B-Instruct
    - Math specialist: Qwen2.5-Math-72B, Qwen2.5-Math-7B
    - Code specialist: Qwen2.5-Coder-32B, DeepSeek-Coder-V2

    Example:
        # Start vLLM server externally:
        # vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000

        server = VLLMMCPServer(model_name="qwen3-32b")
        result = server.execute("Explain quantum computing")
    """

    # Model name aliases to full HuggingFace paths
    SUPPORTED_MODELS: Dict[str, str] = {
        # General purpose
        "qwen3-32b": "Qwen/Qwen3-32B",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
        # Math specialists
        "glm-4.7": "THUDM/glm-4-9b-chat",
        "qwen-math-7b": "Qwen/Qwen2.5-Math-7B-Instruct",
        "qwen-math-1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        # Code specialists
        "qwen3-coder-plus": "Qwen/Qwen3-Coder-Plus",
        "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        # MoE models
        "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    }

    # Estimated costs per 1M tokens (local compute, GPU rental approximation)
    MODEL_COSTS: Dict[str, Dict[str, float]] = {
        "qwen3-32b": {"prompt": 0.50, "completion": 0.50},
        "qwen3-8b": {"prompt": 0.10, "completion": 0.10},
        "llama-70b": {"prompt": 1.00, "completion": 1.00},
        "llama-8b": {"prompt": 0.10, "completion": 0.10},
        "glm-4.7": {"prompt": 1.00, "completion": 1.00},
        "qwen-math-7b": {"prompt": 0.10, "completion": 0.10},
        "qwen-math-1.5b": {"prompt": 0.02, "completion": 0.02},
        "qwen3-coder-plus": {"prompt": 0.50, "completion": 0.50},
        "qwen-coder-7b": {"prompt": 0.10, "completion": 0.10},
        "glm-4.7-flash": {"prompt": 0.30, "completion": 0.30},
    }

    def __init__(
        self,
        model_name: str,
        vllm_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        **vllm_params: Any,
    ):
        """Initialize vLLM server connection.

        Args:
            model_name: Model alias (e.g., 'qwen3-32b') or full HF path
            vllm_url: URL of the vLLM server (default: localhost:8000)
            api_key: Optional API key for authenticated endpoints
            telemetry_collector: Energy monitor collector
            event_recorder: EventRecorder for per-action tracking
            **vllm_params: Default parameters (max_tokens, temperature, top_p, etc.)
        """
        super().__init__(
            name=f"vllm:{model_name}",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )

        self.model_name = model_name
        self.model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.vllm_url = vllm_url.rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")
        self.vllm_params = vllm_params

        # Cost estimation (per 1M tokens)
        self.cost_per_1m = self.MODEL_COSTS.get(
            model_name,
            {"prompt": 0.0, "completion": 0.0}
        )

        # Query server's actual max_model_len to handle validation
        self._server_max_model_len: Optional[int] = None

    def _get_server_max_model_len(self) -> Optional[int]:
        """Query server's actual max_model_len from /v1/models endpoint."""
        if self._server_max_model_len is not None:
            return self._server_max_model_len

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.vllm_url}/v1/models")
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    for model in models:
                        if model.get("id") == self.model_path or model.get("id") == self.model_name:
                            self._server_max_model_len = model.get("max_model_len")
                            return self._server_max_model_len
        except Exception:
            pass
        return None

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute inference via vLLM's OpenAI-compatible API."""
        global _retry_warn_count

        # Merge default params with per-request params
        merged_params = {**self.vllm_params, **params}

        max_tokens = merged_params.get("max_tokens", 8192)
        temperature = merged_params.get("temperature", 0.7)
        top_p = merged_params.get("top_p", 0.9)
        system_prompt = merged_params.get("system_prompt")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        original_max_tokens = max_tokens

        payload = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            # Extract response
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Calculate cost
            cost_usd = (
                (prompt_tokens / 1_000_000) * self.cost_per_1m["prompt"] +
                (completion_tokens / 1_000_000) * self.cost_per_1m["completion"]
            )

            return MCPToolResult(
                content=content,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                cost_usd=cost_usd,
                metadata={
                    "model": self.model_path,
                    "server": "vllm",
                    "finish_reason": data["choices"][0].get("finish_reason"),
                },
            )

        except httpx.ConnectError:
            return MCPToolResult(
                content=f"Error: Cannot connect to vLLM server at {self.vllm_url}. "
                        f"Please ensure vLLM is running with the model {self.model_path}.",
                usage={},
                cost_usd=0.0,
                metadata={"error": "connection_failed"},
            )
        except httpx.ReadTimeout:
            return MCPToolResult(
                content=(
                    "Error: vLLM request timed out after "
                    f"{_REQUEST_TIMEOUT_SECONDS}s. "
                    "The model may be overloaded or the response is too long."
                ),
                usage={},
                cost_usd=0.0,
                metadata={"error": "timeout"},
            )
        except httpx.HTTPStatusError as e:
            # Handle max_tokens validation error - try with capped value
            error_text = e.response.text
            error_text_lower = error_text.lower()

            if e.response.status_code == 400 and ("max_tokens" in error_text_lower or "max_completion_tokens" in error_text_lower) and "too large" in error_text_lower:
                parsed_budget = _parse_vllm_context_budget_error(error_text)

                if parsed_budget is not None:
                    validation_limit, actual_input_tokens = parsed_budget
                    capped_max_tokens = max(1, int(validation_limit - actual_input_tokens - 1))

                    if capped_max_tokens < max_tokens and capped_max_tokens > 0:
                        _retry_warn_count += 1
                        if _retry_warn_count == 1 or _retry_warn_count % 100 == 0:
                            warnings.warn(
                                f"vLLM validation rejected max_tokens={max_tokens} "
                                f"(validation uses model config limit: {validation_limit}). "
                                f"Retrying with max_tokens={capped_max_tokens}. "
                                f"(Total retries: {_retry_warn_count})",
                                UserWarning,
                                stacklevel=2,
                            )

                        payload["max_tokens"] = capped_max_tokens
                        try:
                            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                                retry_response = client.post(
                                    f"{self.vllm_url}/v1/chat/completions",
                                    headers=headers,
                                    json=payload,
                                )
                                retry_response.raise_for_status()
                                retry_data = retry_response.json()

                            content = retry_data["choices"][0]["message"]["content"]
                            usage = retry_data.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            cost_usd = (
                                (prompt_tokens / 1_000_000) * self.cost_per_1m["prompt"] +
                                (completion_tokens / 1_000_000) * self.cost_per_1m["completion"]
                            )
                            return MCPToolResult(
                                content=content,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                },
                                cost_usd=cost_usd,
                                metadata={
                                    "model": self.model_path,
                                    "server": "vllm",
                                    "finish_reason": retry_data["choices"][0].get("finish_reason"),
                                    "max_tokens_capped": True,
                                    "original_max_tokens": original_max_tokens,
                                },
                            )
                        except httpx.HTTPStatusError as retry_e:
                            return MCPToolResult(
                                content=f"Error: vLLM server returned {retry_e.response.status_code}: {retry_e.response.text}",
                                usage={},
                                cost_usd=0.0,
                                metadata={"error": f"http_{retry_e.response.status_code}", "retry_failed": True},
                            )
                        except Exception as retry_e:
                            return MCPToolResult(
                                content=f"Error during retry: {type(retry_e).__name__}: {retry_e}",
                                usage={},
                                cost_usd=0.0,
                                metadata={"error": "retry_exception"},
                            )

            return MCPToolResult(
                content=f"Error: vLLM server returned {e.response.status_code}: {e.response.text}",
                usage={},
                cost_usd=0.0,
                metadata={"error": f"http_{e.response.status_code}"},
            )
        except Exception as e:
            return MCPToolResult(
                content=f"Error: {type(e).__name__}: {e}",
                usage={},
                cost_usd=0.0,
                metadata={"error": str(e)},
            )

    def health_check(self) -> bool:
        """Check if vLLM server is running and model is loaded."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.vllm_url}/v1/models")
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    model_ids = [m.get("id", "") for m in models]
                    return self.model_path in model_ids or self.model_name in model_ids
            return False
        except Exception:
            return False

    @classmethod
    def list_supported_models(cls) -> Dict[str, str]:
        """Return mapping of model aliases to HuggingFace paths."""
        return cls.SUPPORTED_MODELS.copy()

"""Ollama MCP server for local models."""

from __future__ import annotations

import time
from typing import Any, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult


def _normalize_base_url(base_url: str) -> str:
    """Normalize Ollama URL (add http:// if missing)."""
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


class OllamaMCPServer(BaseMCPServer):
    """MCP server for local models via Ollama.

    Supports any model available in Ollama (Llama, Qwen, DeepSeek, etc.)

    Example:
        server = OllamaMCPServer(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434"
        )

        result = server.execute("What is 2+2?")
        print(result.content)  # "4"
        print(result.cost_usd)  # 0.0 (local model, no cost)
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        **ollama_params: Any,
    ):
        """Initialize Ollama MCP server.

        Args:
            model_name: Ollama model name (e.g., "llama3.2:1b", "qwen2.5:0.5b")
            base_url: Ollama server URL (default: http://127.0.0.1:11434)
            telemetry_collector: Energy monitor collector
            event_recorder: EventRecorder for per-action tracking
            **ollama_params: Additional Ollama parameters (temperature, etc.)
        """
        super().__init__(
            name=f"ollama:{model_name}",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )

        self.model_name = model_name
        self.ollama_params = ollama_params

        # Lazy import: ollama is optional
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "ollama package is required for OllamaMCPServer. "
                "Install with: pip install ollama"
            )

        # Initialize Ollama client
        host = _normalize_base_url(base_url or self.DEFAULT_BASE_URL)
        self._client = Client(host=host)

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute model inference via Ollama."""
        from ollama import ResponseError

        # Merge default params with request params
        payload = {**self.ollama_params, **params}
        payload["model"] = self.model_name
        payload["prompt"] = prompt
        payload["stream"] = True

        # Call Ollama API
        start = time.perf_counter()
        try:
            stream = self._client.generate(**payload)
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error for {self.model_name}: {exc}") from exc

        # Consume stream and collect response
        content_chunks: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        ttft_ms: Optional[float] = None

        for chunk in stream:
            text = getattr(chunk, "response", None)
            if text:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                content_chunks.append(text)

            # Final chunk contains token counts
            if getattr(chunk, "done", False):
                if getattr(chunk, "prompt_eval_count", None) is not None:
                    prompt_tokens = int(chunk.prompt_eval_count)
                if getattr(chunk, "eval_count", None) is not None:
                    completion_tokens = int(chunk.eval_count)

        content = "".join(content_chunks)
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        return MCPToolResult(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost_usd=0.0,  # Local model, no API cost
            ttft_seconds=(ttft_ms / 1000.0) if ttft_ms else None,
            metadata={"model": self.model_name, "backend": "ollama"},
        )

    def health_check(self) -> bool:
        """Check if Ollama server is available and model is loaded."""
        try:
            models = self._client.list()
            available_models = [str(m.model) for m in models.models]
            return any(self.model_name in m for m in available_models)
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """List all models available in Ollama."""
        from ollama import ResponseError
        try:
            response = self._client.list()
            return [str(model.model) for model in response.models]
        except ResponseError as exc:
            raise RuntimeError(f"Failed to list Ollama models: {exc}") from exc

"""OpenAI MCP server with cost tracking."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult
from ipw.cost.pricing import OPENAI_PRICING, calculate_cost


class OpenAIMCPServer(BaseMCPServer):
    """MCP server for OpenAI models with automatic cost tracking.

    Tracks API costs based on token usage and current pricing.

    Example:
        server = OpenAIMCPServer(
            model_name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        result = server.execute("Explain quantum computing")
        print(result.content)
        print(f"Cost: ${result.cost_usd:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        **openai_params: Any,
    ):
        """Initialize OpenAI MCP server.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-5-mini-2025-08-07")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            telemetry_collector: Energy monitor collector
            event_recorder: EventRecorder for per-action tracking
            **openai_params: Additional OpenAI parameters (temperature, max_tokens, etc.)
        """
        super().__init__(
            name=f"openai:{model_name}",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )

        self.model_name = model_name
        self.openai_params = openai_params

        # Lazy import: openai is optional
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIMCPServer. "
                "Install with: pip install openai"
            )

        # Initialize OpenAI client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

        # Get pricing for this model
        self.pricing = OPENAI_PRICING.get(model_name)
        if not self.pricing:
            print(f"Warning: No pricing info for {model_name}, using gpt-4o rates")
            self.pricing = OPENAI_PRICING["gpt-4o"]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute OpenAI API call with cost tracking."""
        from openai import OpenAIError

        # Merge default params with request params
        payload = {**self.openai_params, **params}
        payload["model"] = self.model_name
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = True

        # GPT-5+ models use max_completion_tokens instead of max_tokens
        if "max_tokens" in payload and self.model_name.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        # Call OpenAI API
        start = time.perf_counter()
        try:
            stream = self._client.chat.completions.create(**payload)
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI error for {self.model_name}: {exc}") from exc

        # Consume stream and collect response
        content_chunks: list[str] = []
        ttft_ms: Optional[float] = None
        usage = None

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    content_chunks.append(delta.content)

            # Last chunk contains usage
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

        content = "".join(content_chunks)

        # Extract token counts
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(content.split())
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        cost_usd = calculate_cost("openai", self.model_name, prompt_tokens, completion_tokens)

        return MCPToolResult(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost_usd=cost_usd,
            ttft_seconds=(ttft_ms / 1000.0) if ttft_ms else None,
            metadata={
                "model": self.model_name,
                "backend": "openai",
                "pricing_input_per_1m": self.pricing["input"],
                "pricing_output_per_1m": self.pricing["output"],
            },
        )

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return response is not None
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """List all available OpenAI models."""
        from openai import OpenAIError
        try:
            models = self._client.models.list()
            return [model.id for model in models.data if model.id.startswith("gpt")]
        except OpenAIError as exc:
            raise RuntimeError(f"Failed to list OpenAI models: {exc}") from exc

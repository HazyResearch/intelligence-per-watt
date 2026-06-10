"""Anthropic MCP server with cost tracking."""

from __future__ import annotations

import time
from typing import Any, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult
from ipw.cost.pricing import ANTHROPIC_PRICING, calculate_cost


class AnthropicMCPServer(BaseMCPServer):
    """MCP server for Anthropic Claude models with cost tracking.

    Example:
        server = AnthropicMCPServer(
            model_name="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        result = server.execute("Write a haiku about AI")
        print(result.content)
        print(f"Cost: ${result.cost_usd:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        **anthropic_params: Any,
    ):
        """Initialize Anthropic MCP server.

        Args:
            model_name: Claude model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            telemetry_collector: Energy monitor collector
            **anthropic_params: Additional params (temperature, max_tokens, etc.)
        """
        super().__init__(
            name=f"anthropic:{model_name}",
            telemetry_collector=telemetry_collector,
        )

        self.model_name = model_name
        self.anthropic_params = anthropic_params

        # Lazy import: anthropic is optional
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicMCPServer. "
                "Install with: pip install anthropic"
            )

        # Initialize Anthropic client
        self._client = Anthropic(api_key=api_key)

        # Get pricing for this model
        self.pricing = ANTHROPIC_PRICING.get(model_name)
        if not self.pricing:
            print(f"Warning: No pricing info for {model_name}, using Sonnet 4.5 rates")
            self.pricing = ANTHROPIC_PRICING["claude-sonnet-4-5-20250929"]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute Anthropic API call with cost tracking."""
        from anthropic import AnthropicError

        # Merge default params with request params
        payload = {**self.anthropic_params, **params}
        payload["model"] = self.model_name
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["max_tokens"] = payload.get("max_tokens", 4096)
        payload["stream"] = True

        # Call Anthropic API
        start = time.perf_counter()
        try:
            stream = self._client.messages.create(**payload)
        except AnthropicError as exc:
            raise RuntimeError(f"Anthropic error for {self.model_name}: {exc}") from exc

        # Consume stream and collect response
        content_chunks: list[str] = []
        ttft_ms: Optional[float] = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        with stream as event_stream:
            for event in event_stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        content_chunks.append(event.delta.text)

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens

        content = "".join(content_chunks)
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )

        # Calculate cost
        cost_usd = (
            calculate_cost("anthropic", self.model_name, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )

        return MCPToolResult(
            content=content,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            cost_usd=cost_usd,
            ttft_seconds=(ttft_ms / 1000.0) if ttft_ms else None,
            metadata={
                "model": self.model_name,
                "backend": "anthropic",
                "pricing_input_per_1m": self.pricing["input"],
                "pricing_output_per_1m": self.pricing["output"],
            },
        )

    def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            response = self._client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return response is not None
        except Exception:
            return False

"""OpenRouter MCP server with cost tracking.

OpenRouter provides access to many models through a unified API:
- Google (Gemini)
- Anthropic (Claude)
- OpenAI (GPT)
- Meta (Llama)
- Qwen
- DeepSeek
- Mistral
- And many more
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult


class OpenRouterMCPServer(BaseMCPServer):
    """MCP server for OpenRouter with automatic cost tracking.

    OpenRouter provides a unified API for accessing many LLM providers.
    Uses OpenAI-compatible API format.

    Example:
        server = OpenRouterMCPServer(
            model_name="google/gemini-2.5-flash",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        result = server.execute("Explain quantum computing")
        print(result.content)
        print(f"Cost: ${result.cost_usd:.6f}")
    """

    # Approximate pricing per 1M tokens (varies by model)
    # Source: https://openrouter.ai/models
    PRICING = {
        # Meta Llama
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.40, "output": 0.40},
        "meta-llama/llama-3.1-405b-instruct": {"input": 2.00, "output": 2.00},
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.40, "output": 0.40},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.05},
        # Qwen - General
        "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.40},
        "qwen/qwen-2.5-32b-instruct": {"input": 0.15, "output": 0.15},
        "qwen/qwq-32b": {"input": 0.15, "output": 0.15},
        "qwen/qwen-2.5-coder-32b-instruct": {"input": 0.15, "output": 0.15},
        # Qwen3
        "qwen/qwen3-32b": {"input": 0.08, "output": 0.24},
        "qwen/qwen3-coder-next": {"input": 0.20, "output": 1.50},
        "qwen/qwen3-coder-plus": {"input": 1.00, "output": 5.00},
        "qwen/qwen3-max": {"input": 1.20, "output": 6.00},
        "qwen/qwen3-next-80b-a3b-instruct": {"input": 0.09, "output": 1.10},
        # Math specialist
        "z-ai/glm-4.7": {"input": 0.40, "output": 1.50},
        "qwen/qwen2.5-math-72b-instruct": {"input": 0.40, "output": 0.40},
        "qwen/qwen2.5-coder-32b-instruct": {"input": 0.15, "output": 0.15},
        # DeepSeek
        "deepseek/deepseek-r1": {"input": 0.70, "output": 2.50},
        "deepseek/deepseek-r1-0528": {"input": 0.40, "output": 1.75},
        "deepseek/deepseek-v3.2": {"input": 0.25, "output": 0.38},
        "deepseek/deepseek-v3.1-terminus": {"input": 0.21, "output": 0.79},
        "deepseek/deepseek-chat-v3-0324": {"input": 0.14, "output": 0.28},
        "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek/deepseek-coder-v2": {"input": 0.14, "output": 0.28},
        # Mistral
        "mistralai/mistral-large-2411": {"input": 2.00, "output": 6.00},
        "mistralai/mistral-small-2501": {"input": 0.10, "output": 0.30},
        "mistralai/codestral-2501": {"input": 0.30, "output": 0.90},
        # Google Gemini (via OpenRouter)
        "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
        "z-ai/glm-4.7-flash": {"input": 0.07, "output": 0.40},
        # Small/cheap models
        "openai/gpt-oss-20b": {"input": 0.02, "output": 0.10},
        "deepseek/deepseek-r1-distill-qwen-1.5b": {"input": 0.02, "output": 0.05},
        "google/gemma-3-4b-it": {"input": 0.02, "output": 0.07},
    }

    # Default fallback pricing
    DEFAULT_PRICING = {"input": 1.00, "output": 3.00}

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        **openai_params: Any,
    ):
        """Initialize OpenRouter MCP server.

        Args:
            model_name: Model identifier (e.g., "google/gemini-2.5-flash")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            telemetry_collector: Energy monitor collector
            site_url: Your site URL for OpenRouter rankings
            app_name: Your app name for OpenRouter rankings
            **openai_params: Additional parameters (temperature, max_tokens, etc.)
        """
        super().__init__(
            name=f"openrouter:{model_name}",
            telemetry_collector=telemetry_collector,
        )

        self.model_name = model_name
        self.openai_params = openai_params

        # Get API key
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

        # Lazy import: openai is optional
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenRouterMCPServer. "
                "Install with: pip install openai"
            )

        # Initialize OpenAI-compatible client with OpenRouter base URL
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url or "https://github.com/ipw",
                "X-Title": app_name or "IPW Orchestrator",
            },
        )

        # Get pricing for this model
        self.pricing = self.PRICING.get(model_name, self.DEFAULT_PRICING)
        if model_name not in self.PRICING:
            print(f"Warning: No pricing info for {model_name}, using default rates")

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute OpenRouter API call with cost tracking."""
        from openai import OpenAIError

        # Merge default params with request params
        payload = {**self.openai_params, **params}
        payload["model"] = self.model_name
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = True

        # Call OpenRouter API
        start = time.perf_counter()
        try:
            stream = self._client.chat.completions.create(**payload)
        except OpenAIError as exc:
            raise RuntimeError(f"OpenRouter error for {self.model_name}: {exc}") from exc

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

            # Last chunk may contain usage
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

        content = "".join(content_chunks)

        # Extract token counts
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        else:
            prompt_tokens = int(len(prompt.split()) * 1.3)
            completion_tokens = int(len(content.split()) * 1.3)
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost based on token usage
        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)

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
                "backend": "openrouter",
                "pricing_input_per_1m": self.pricing["input"],
                "pricing_output_per_1m": self.pricing["output"],
            },
        )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API cost in USD."""
        input_cost = (prompt_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost

    def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return response is not None
        except Exception:
            return False

    @classmethod
    def list_popular_models(cls) -> list[str]:
        """List popular models available on OpenRouter."""
        return list(cls.PRICING.keys())

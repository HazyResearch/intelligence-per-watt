"""Google Gemini MCP server with cost tracking.

Uses the google-genai SDK for Gemini 3.0 Flash and other Gemini models.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult
from ipw.cost.pricing import GEMINI_PRICING, calculate_cost

# Lazy import to avoid requiring google-genai if not using Gemini
_genai_client = None


def _get_genai_client(api_key: Optional[str] = None):
    """Lazily initialize the google-genai client."""
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        # Client picks up GEMINI_API_KEY from env if not provided
        if api_key:
            _genai_client = genai.Client(api_key=api_key)
        else:
            _genai_client = genai.Client()

    return _genai_client


class GeminiMCPServer(BaseMCPServer):
    """MCP server for Google Gemini models with automatic cost tracking.

    Example:
        server = GeminiMCPServer(
            model_name="gemini-3-flash-preview",
            api_key=os.getenv("GEMINI_API_KEY")
        )

        result = server.execute("Explain quantum computing")
        print(result.content)
        print(f"Cost: ${result.cost_usd:.4f}")
    """

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        **genai_params: Any,
    ):
        """Initialize Gemini MCP server.

        Args:
            model_name: Gemini model name (e.g., "gemini-3-flash-preview")
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            telemetry_collector: Energy monitor collector
            **genai_params: Additional parameters (temperature, max_tokens, etc.)
        """
        super().__init__(
            name=f"gemini:{model_name}",
            telemetry_collector=telemetry_collector,
        )

        self.model_name = model_name
        self.api_key = api_key
        self.genai_params = genai_params

        # Get pricing for this model
        self.pricing = GEMINI_PRICING.get(model_name)
        if not self.pricing:
            print(f"Warning: No pricing info for {model_name}, using gemini-3-flash-preview rates")
            self.pricing = GEMINI_PRICING["gemini-3-flash-preview"]

    def _get_client(self):
        """Get or create the genai client."""
        return _get_genai_client(self.api_key)

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute Gemini API call with cost tracking."""
        client = self._get_client()

        # Merge default params with request params
        config = {**self.genai_params, **params}

        # Call Gemini API
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config if config else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini error for {self.model_name}: {exc}") from exc

        end = time.perf_counter()

        # Extract content
        content = response.text if hasattr(response, 'text') else str(response)

        # Extract token counts from usage_metadata
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_meta = response.usage_metadata
            prompt_tokens = getattr(usage_meta, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(usage_meta, 'candidates_token_count', 0) or 0
            total_tokens = getattr(usage_meta, 'total_token_count', 0) or (prompt_tokens + completion_tokens)

        # Fallback: estimate from content length if no usage metadata
        if total_tokens == 0:
            prompt_tokens = len(prompt.split()) * 2
            completion_tokens = len(content.split()) * 2
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        cost_usd = calculate_cost("gemini", self.model_name, prompt_tokens, completion_tokens)

        return MCPToolResult(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost_usd=cost_usd,
            ttft_seconds=end - start,
            metadata={
                "model": self.model_name,
                "backend": "gemini",
                "pricing_input_per_1m": self.pricing["input"],
                "pricing_output_per_1m": self.pricing["output"],
            },
        )

    def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model_name,
                contents="test",
            )
            return response is not None
        except Exception:
            return False

    def generate_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        **params: Any,
    ) -> MCPToolResult:
        """Generate response with a system prompt."""
        combined_prompt = f"{system_prompt}\n\nUser: {prompt}"
        return self.execute(combined_prompt, **params)

# Adding Inference Clients

Inference clients are adapters that connect IPW to different LLM inference servers. To add a new client, subclass `InferenceClient` and register it with `ClientRegistry`.

## Step 1: Create the Client File

Create a new file in `intelligence-per-watt/src/ipw/clients/`:

```python
# ipw/clients/my_service.py
from __future__ import annotations

from typing import Any, Sequence

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient


@ClientRegistry.register("my-service")
class MyServiceClient(InferenceClient):
    """Client for MyService inference API."""

    client_id = "my-service"
    client_name = "MyService"

    def __init__(self, base_url: str, **config: Any) -> None:
        super().__init__(base_url, **config)
        # Initialize your client library
        # Use lazy imports for optional dependencies
        try:
            import my_service_sdk
        except ImportError:
            raise ImportError(
                "my-service-sdk is required. Install with: pip install my-service-sdk"
            )
        self._client = my_service_sdk.Client(base_url=base_url)

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Run a streamed chat completion."""
        import time

        request_start = time.time()
        first_token_time = None
        content_parts = []

        # Stream the response
        for chunk in self._client.stream(model=model, prompt=prompt):
            if first_token_time is None:
                first_token_time = time.time()
            content_parts.append(chunk.text)

        request_end = time.time()
        content = "".join(content_parts)

        # Build the response
        ttft_ms = (
            (first_token_time - request_start) * 1000
            if first_token_time
            else 0.0
        )

        return Response(
            content=content,
            usage=ChatUsage(
                prompt_tokens=chunk.usage.input_tokens,
                completion_tokens=chunk.usage.output_tokens,
                total_tokens=chunk.usage.total_tokens,
            ),
            time_to_first_token_ms=ttft_ms,
            first_token_time=first_token_time,
            request_start_time=request_start,
            request_end_time=request_end,
        )

    def list_models(self) -> Sequence[str]:
        """Return available models."""
        return [m.id for m in self._client.list_models()]

    def health(self) -> bool:
        """Check if the service is reachable."""
        try:
            self._client.health()
            return True
        except Exception:
            return False
```

## Step 2: Register the Import

Add your module to the client initialization code so it gets imported when clients are registered. If your client has optional dependencies, wrap the import:

```python
# In ipw/clients/__init__.py
try:
    from . import my_service  # noqa: F401
except ImportError:
    pass
```

## Step 3: Add Optional Dependency

If your client requires an external library, add it as an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
my-service = ["my-service-sdk>=1.0"]
```

## Step 4: Test

```bash
# Install your extra
uv pip install -e 'intelligence-per-watt[my-service]'

# Verify it appears in the registry
ipw list clients

# Run a profile
ipw profile --client my-service --model my-model \
  --client-base-url http://localhost:8080
```

## Key Requirements

### `stream_chat_completion()`

This is the core method. Requirements:

- **Must stream**: Consume the streaming API to measure time-to-first-token accurately.
- **Must return `Response`**: Include content, token usage, and timing information.
- **Must record timestamps**: `request_start_time` and `request_end_time` are used to window telemetry readings.
- **Must report tokens**: `ChatUsage` is needed for FLOPs estimation and cost calculation.

### `list_models()`

Return the list of model IDs available on the server. Used by `ipw list clients` for discovery.

### `health()`

Return `True` if the server is reachable. Called before profiling starts to fail fast.

### `chat()` (Optional)

The `chat()` method is used for evaluation (LLM judge). If your client can serve as an evaluation judge, implement it:

```python
def chat(
    self,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Synchronous chat completion for evaluation."""
    response = self._client.chat(
        system=system_prompt,
        user=user_prompt,
        temperature=temperature,
    )
    return response.text
```

### `prepare()` (Optional)

Called before the first query. Use it for model warmup:

```python
def prepare(self, model: str) -> None:
    """Warm up the model."""
    self._client.load_model(model)
```

## Existing Clients

Study these implementations for reference:

- `ipw/clients/ollama.py` -- Ollama client with streaming
- `ipw/clients/vllm.py` -- vLLM offline client
- `ipw/clients/openai.py` -- OpenAI-compatible client (used for judge evaluation)

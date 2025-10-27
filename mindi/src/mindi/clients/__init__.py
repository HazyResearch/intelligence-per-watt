"""Inference client implementations.

Clients register themselves with ``mindi.core.ClientRegistry``.
"""

from .ollama import OllamaClient
from .vllm import VLLMClient

__all__ = ["OllamaClient", "VLLMClient"]

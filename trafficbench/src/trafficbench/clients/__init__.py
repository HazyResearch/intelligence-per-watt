"""Inference client implementations.

Clients register themselves with ``trafficbench.core.ClientRegistry``.
"""

from .base import InferenceClient
from .ollama import OllamaClient
from .vllm import VLLMClient

__all__ = ["InferenceClient", "OllamaClient", "VLLMClient"]

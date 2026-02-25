"""Model configuration presets for common vLLM-served models."""

from __future__ import annotations

from typing import Any, Dict

# Each preset maps a short name to model_id + vLLM launch arguments.
# tensor_parallel_size is set based on the model's memory requirements.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "glm-4.7-flash": {
        "model_id": "zai-org/GLM-4.7-FP8",
        "vllm_args": {"tensor_parallel_size": 1},
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        "vllm_args": {"tensor_parallel_size": 4},
    },
    "qwen3-30b-a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "vllm_args": {"tensor_parallel_size": 1},
    },
    "minimax-m2.5": {
        "model_id": "MiniMaxAI/MiniMax-M2.5",
        "vllm_args": {"tensor_parallel_size": 2},
    },
    "kimi-k2.5": {
        "model_id": "moonshotai/Kimi-K2.5",
        "vllm_args": {"tensor_parallel_size": 4},
    },
    "qwen35-35b-a3b": {
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "vllm_args": {
            "tensor_parallel_size": 1,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "language_model_only": True,
        },
    },
    "qwen35-27b": {
        "model_id": "Qwen/Qwen3.5-27B",
        "vllm_args": {
            "tensor_parallel_size": 1,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "language_model_only": True,
        },
    },
}


def resolve_preset(name: str) -> Dict[str, Any]:
    """Look up a model preset by name.

    Args:
        name: Preset short name (e.g. "glm-4.7-flash").

    Returns:
        Dict with ``model_id`` and ``vllm_args`` keys.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return MODEL_PRESETS[name]


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(MODEL_PRESETS)


__all__ = ["MODEL_PRESETS", "resolve_preset", "list_presets"]

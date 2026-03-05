"""Model configuration presets for common vLLM-served models."""

from __future__ import annotations

from typing import Any, Dict

# Each preset maps a short name to model_id + vLLM launch arguments.
# tensor_parallel_size is set based on the model's memory requirements.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "glm-4.7-flash": {
        "model_id": "zai-org/GLM-4.7-FP8",
        "vllm_args": {"tensor_parallel_size": 8},
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        "vllm_args": {"tensor_parallel_size": 8},
    },
    "qwen3-30b-a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "vllm_args": {"tensor_parallel_size": 1},
    },
    "minimax-m2.5": {
        "model_id": "MiniMaxAI/MiniMax-M2.5",
        "vllm_args": {"tensor_parallel_size": 4, "trust_remote_code": True, "max_model_len": 32768, "enforce_eager": True},
    },
    "kimi-k2.5": {
        "model_id": "moonshotai/Kimi-K2.5",
        "vllm_args": {"tensor_parallel_size": 8, "trust_remote_code": True, "max_model_len": 8192, "enforce_eager": True},
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
    "qwen35-397b-a17b-fp8": {
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "vllm_args": {
            "tensor_parallel_size": 8,
            "trust_remote_code": True,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "language_model_only": True,
        },
    },
    "qwen35-122b-a10b-fp8": {
        "model_id": "Qwen/Qwen3.5-122B-A10B-FP8",
        "vllm_args": {
            "tensor_parallel_size": 4,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "language_model_only": True,
        },
    },
    "qwen3-235b-a22b-fp8": {
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "vllm_args": {
            "tensor_parallel_size": 8,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        },
    },
    "qwen3-30b-a3b-fp8": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "vllm_args": {
            "tensor_parallel_size": 1,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        },
    },
    "glm-5-fp8": {
        "model_id": "zai-org/GLM-5-FP8",
        "vllm_args": {"tensor_parallel_size": 8, "trust_remote_code": True},
    },
    "glm-5-nvfp4": {
        "model_id": "lukealonso/GLM-5-NVFP4",
        "vllm_args": {"tensor_parallel_size": 8, "trust_remote_code": True, "max_model_len": 8192, "enforce_eager": True},
    },
    "kimi-k2.5-nvfp4": {
        "model_id": "nvidia/Kimi-K2.5-NVFP4",
        "vllm_args": {"tensor_parallel_size": 8, "trust_remote_code": True, "max_model_len": 8192, "enforce_eager": True, "gpu_memory_utilization": 0.95},
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

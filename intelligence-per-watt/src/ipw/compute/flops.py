"""FLOPs estimation for LLM inference."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Known model parameter counts (billions)
MODEL_PARAMS: dict[str, float] = {
    # Llama family
    "llama-3.2-1b": 1.24,
    "llama-3.2-3b": 3.21,
    "llama-3.1-8b": 8.03,
    "llama-3.1-70b": 70.6,
    "llama-3.1-405b": 405.0,
    # Qwen family
    "qwen-2.5-0.5b": 0.49,
    "qwen-2.5-1.5b": 1.54,
    "qwen-2.5-3b": 3.09,
    "qwen-2.5-7b": 7.62,
    "qwen-2.5-14b": 14.8,
    "qwen-2.5-32b": 32.5,
    "qwen-2.5-72b": 72.7,
    # Mistral
    "mistral-7b": 7.24,
    "mixtral-8x7b": 46.7,
    "mixtral-8x22b": 141.0,
    # Qwen 3
    "qwen-3-0.6b": 0.6,
    "qwen-3-1.7b": 1.7,
    "qwen-3-4b": 4.0,
    "qwen-3-8b": 8.0,
    "qwen-3-14b": 14.0,
    "qwen-3-32b": 32.0,
    "qwen-3-30b-a-3b": 3.0,
    "qwen-3-235b-a-22b": 22.0,
    "qwen-3-235b": 235.0,
    # Qwen 3.5
    "qwen-3.5-397b-a-17b": 17.0,
    "qwen-3.5-122b-a-10b": 10.0,
    "qwen-3.5-35b-a-3b": 3.0,
    "qwen-3.5-27b": 27.0,
    # DeepSeek
    "deepseek-r-1": 671.0,
    "deepseek-v-3": 671.0,
    "deepseek-v-2.5": 236.0,
    "deepseek-coder-v-2": 236.0,
    # Phi
    "phi-3-mini": 3.82,
    "phi-3-small": 7.39,
    "phi-3-medium": 14.0,
    "phi-4": 14.0,
    "phi-4-mini": 3.8,
    # Gemma
    "gemma-2-2b": 2.61,
    "gemma-2-9b": 9.24,
    "gemma-2-27b": 27.2,
    # Gemma 3
    "gemma-3-1b": 1.0,
    "gemma-3-4b": 4.0,
    "gemma-3-12b": 12.0,
    "gemma-3-27b": 27.0,
    # Llama 4
    "llama-4-scout": 17.0,
    "llama-4-maverick": 17.0,
    # Apple Foundation Models (on-device, AFM 3). "Core" is a dense ~3B model.
    # "Core Advanced" is a 20B sparse MoE that activates only 1-4B parameters per
    # request via Instruction-Following Pruning, so the nominal 4B upper bound is
    # used here -- FLOPs and flops_per_joule for that label are an estimate of
    # active compute, not a measured figure. Apple has not published exact
    # parameter counts. Core Advanced is listed first because `lookup_params`
    # falls back to a substring scan in insertion order and "afm-3" is a
    # substring of the longer labels.
    "afm-3-core-advanced": 4.0,
    "afm-3-core": 3.0,
    "afm-3": 3.0,
}


def normalize_model_name(model: str) -> str:
    """Normalize model name for parameter lookup.

    Handles common naming patterns like:
    - 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama-3.1-8b'
    - 'llama3.2:1b' (ollama format) -> 'llama-3.2-1b'
    - 'qwen2.5-7b-instruct' -> 'qwen-2.5-7b'
    """
    name = model.lower()
    # Remove common suffixes
    for suffix in ["-instruct", "-chat", "-it", "-base", ":latest"]:
        name = name.replace(suffix, "")
    # Remove org prefix
    if "/" in name:
        name = name.split("/")[-1]
    # Normalize separators
    name = name.replace(":", "-").replace("_", "-")
    # Insert hyphen between letters and digits (e.g., llama3 -> llama-3, qwen2 -> qwen-2)
    # but not after 'x' to preserve patterns like '8x7b'
    name = re.sub(r"([a-wyz])(\d)", r"\1-\2", name)
    # Remove extra qualifiers
    for q in ["-fp16", "-fp32", "-fp-8", "-bf16", "-awq", "-gptq", "-gguf", "-q4", "-q8"]:
        name = name.replace(q, "")
    # Collapse any double-hyphens left after qualifier removal
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-")


def lookup_params(model: str) -> float | None:
    """Look up parameter count (in billions) for a model.

    Returns None if the model is not in the known list.
    """
    normalized = normalize_model_name(model)
    # Try exact match first
    if normalized in MODEL_PARAMS:
        return MODEL_PARAMS[normalized]
    # Try partial match
    for key, params in MODEL_PARAMS.items():
        if key in normalized or normalized in key:
            return params
    return None


def estimate_flops_fallback(
    params_billions: float,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float]:
    """Estimate FLOPs using the 2*P*T approximation.

    For transformer inference:
    - Prefill: ~2 * P * T_input (matrix multiplications)
    - Decode: ~2 * P * T_output (autoregressive generation)
    - Total: ~2 * P * (T_input + T_output)

    Args:
        params_billions: Model parameter count in billions
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (total_flops, flops_per_token)
    """
    params = params_billions * 1e9
    total_tokens = input_tokens + output_tokens
    total_flops = 2.0 * params * total_tokens
    flops_per_token = 2.0 * params if total_tokens > 0 else 0.0
    return total_flops, flops_per_token


def estimate_flops_calflops(
    model_name_or_path: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float] | None:
    """Estimate FLOPs using the calflops library (optional dependency).

    Returns None if calflops is not installed or estimation fails.
    """
    try:
        from calflops import calculate_flops  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("calflops not installed, skipping detailed FLOPs estimation")
        return None

    try:
        # calflops can estimate FLOPs for HuggingFace models
        flops, macs, params = calculate_flops(
            model_name=model_name_or_path,
            input_shape=(1, input_tokens),
            output_as_string=False,
        )
        total_flops = float(flops) if flops else 0.0
        total_tokens = input_tokens + output_tokens
        flops_per_token = total_flops / total_tokens if total_tokens > 0 else 0.0
        return total_flops, flops_per_token
    except Exception as e:
        logger.warning(f"calflops estimation failed: {e}")
        return None


def estimate_flops(
    model: str,
    input_tokens: int,
    output_tokens: int,
    use_calflops: bool = False,
) -> tuple[float, float]:
    """Estimate FLOPs for a model inference.

    Strategy:
    1. If use_calflops=True, try calflops library first
    2. Fall back to 2*P*T formula using known parameter counts
    3. Return (0, 0) if model is unknown

    Args:
        model: Model name or path
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_calflops: Whether to try calflops library

    Returns:
        Tuple of (total_flops, flops_per_token)
    """
    # Try calflops first if requested
    if use_calflops:
        result = estimate_flops_calflops(model, input_tokens, output_tokens)
        if result is not None:
            return result

    # Fall back to 2*P*T formula
    params = lookup_params(model)
    if params is not None:
        return estimate_flops_fallback(params, input_tokens, output_tokens)

    logger.debug(f"Unknown model '{model}', cannot estimate FLOPs")
    return 0.0, 0.0

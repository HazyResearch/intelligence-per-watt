"""Tests for compute/flops.py — FLOPs estimation."""

from __future__ import annotations

import pytest

from ipw.compute.flops import (
    MODEL_PARAMS,
    estimate_flops,
    estimate_flops_fallback,
    lookup_params,
    normalize_model_name,
)


class TestNormalizeModelName:
    """Test normalize_model_name with various input formats."""

    def test_ollama_format(self) -> None:
        assert normalize_model_name("llama3.2:1b") == "llama-3.2-1b"

    def test_huggingface_format(self) -> None:
        result = normalize_model_name("meta-llama/Llama-3.1-8B-Instruct")
        assert result == "llama-3.1-8b"

    def test_removes_instruct_suffix(self) -> None:
        result = normalize_model_name("qwen2.5-7b-instruct")
        assert result == "qwen-2.5-7b"

    def test_removes_chat_suffix(self) -> None:
        result = normalize_model_name("model-7b-chat")
        assert result == "model-7b"

    def test_removes_quantization_suffixes(self) -> None:
        result = normalize_model_name("llama-3.1-8b-awq")
        assert result == "llama-3.1-8b"

    def test_lowercase(self) -> None:
        result = normalize_model_name("Qwen2.5-7B")
        assert result == "qwen-2.5-7b"

    def test_preserves_mixture_of_experts_pattern(self) -> None:
        result = normalize_model_name("mixtral-8x7b")
        assert "8x7b" in result

    def test_removes_latest_tag(self) -> None:
        result = normalize_model_name("llama3.2:1b:latest")
        assert "latest" not in result


class TestLookupParams:
    """Test lookup_params for known models."""

    def test_known_model_exact(self) -> None:
        params = lookup_params("llama-3.1-8b")
        assert params == pytest.approx(8.03)

    def test_known_model_via_normalization(self) -> None:
        params = lookup_params("meta-llama/Llama-3.1-8B-Instruct")
        assert params == pytest.approx(8.03)

    def test_unknown_model_returns_none(self) -> None:
        params = lookup_params("completely-unknown-model-xyz")
        assert params is None

    def test_ollama_format_lookup(self) -> None:
        params = lookup_params("llama3.2:1b")
        assert params == pytest.approx(1.24)

    def test_qwen_model(self) -> None:
        params = lookup_params("qwen2.5-7b-instruct")
        assert params == pytest.approx(7.62)


class TestEstimateFlopssFallback:
    """Test the 2*P*T fallback formula."""

    def test_basic_calculation(self) -> None:
        # 1B params, 100 input + 50 output tokens
        total_flops, flops_per_token = estimate_flops_fallback(1.0, 100, 50)
        expected_total = 2.0 * 1e9 * 150
        expected_per_token = 2.0 * 1e9
        assert total_flops == pytest.approx(expected_total)
        assert flops_per_token == pytest.approx(expected_per_token)

    def test_zero_tokens(self) -> None:
        total_flops, flops_per_token = estimate_flops_fallback(8.0, 0, 0)
        assert total_flops == 0.0
        assert flops_per_token == 0.0

    def test_large_model(self) -> None:
        total_flops, flops_per_token = estimate_flops_fallback(405.0, 1000, 500)
        expected = 2.0 * 405e9 * 1500
        assert total_flops == pytest.approx(expected)


class TestEstimateFlops:
    """Test the main estimate_flops function."""

    def test_known_model(self) -> None:
        total, per_token = estimate_flops("llama-3.1-8b", 100, 50)
        assert total > 0
        assert per_token > 0

    def test_unknown_model_returns_zeros(self) -> None:
        total, per_token = estimate_flops("unknown-model-xyz", 100, 50)
        assert total == 0.0
        assert per_token == 0.0

    def test_via_normalization(self) -> None:
        total, per_token = estimate_flops("llama3.2:1b", 100, 50)
        assert total > 0

    def test_uses_2pt_formula(self) -> None:
        params = lookup_params("llama-3.1-8b")
        assert params is not None
        total, _ = estimate_flops("llama-3.1-8b", 100, 50)
        expected = 2.0 * params * 1e9 * 150
        assert total == pytest.approx(expected)


class TestQwen35Models:
    """Test Qwen 3.5 model lookups."""

    def test_qwen35_397b(self) -> None:
        total, per_token = estimate_flops("Qwen/Qwen3.5-397B-A17B-Instruct", 100, 200)
        assert total > 0
        assert per_token > 0

    def test_qwen35_122b(self) -> None:
        total, per_token = estimate_flops("Qwen/Qwen3.5-122B-A10B", 100, 200)
        assert total > 0

    def test_qwen35_35b(self) -> None:
        total, per_token = estimate_flops("qwen3.5-35b-a3b", 100, 200)
        assert total > 0

    def test_qwen35_27b(self) -> None:
        total, per_token = estimate_flops("Qwen/Qwen3.5-27B-Instruct", 100, 200)
        assert total > 0

    def test_qwen3_moe_30b_a3b(self) -> None:
        """Verify the fixed MoE entry matches after normalization."""
        params = lookup_params("Qwen/Qwen3-30B-A3B")
        assert params is not None
        assert params == pytest.approx(3.0)

    def test_qwen3_235b_a22b(self) -> None:
        total, per_token = estimate_flops("Qwen/Qwen3-235B-A22B-Instruct", 100, 200)
        assert total > 0

    def test_fp8_normalization(self) -> None:
        """fp8 is correctly stripped during normalization."""
        result = normalize_model_name("qwen2.5-72b-instruct-fp8")
        assert "fp" not in result
        assert result == "qwen-2.5-72b"


class TestModelParams:
    """Test MODEL_PARAMS dictionary."""

    def test_has_llama_models(self) -> None:
        assert "llama-3.2-1b" in MODEL_PARAMS
        assert "llama-3.1-8b" in MODEL_PARAMS

    def test_has_qwen_models(self) -> None:
        assert "qwen-2.5-7b" in MODEL_PARAMS

    def test_has_mistral_models(self) -> None:
        assert "mistral-7b" in MODEL_PARAMS

    def test_all_values_positive(self) -> None:
        for model, params in MODEL_PARAMS.items():
            assert params > 0, f"{model} has non-positive params: {params}"

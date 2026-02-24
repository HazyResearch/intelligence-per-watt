"""Tests for inference server manager."""

from __future__ import annotations

import pytest

from ipw.cli.server_manager import (
    ServerConfig,
    build_server_configs,
    parse_submodel_spec,
)


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_defaults(self) -> None:
        config = ServerConfig(model_id="test/model", alias="test", backend="vllm")
        assert config.port == 8000
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 32768
        assert config.gpu_ids == []
        assert config.extra_args == {}
        assert config.env_vars == {}

    def test_custom_values(self) -> None:
        config = ServerConfig(
            model_id="test/model",
            alias="test",
            backend="vllm",
            port=9000,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
        )
        assert config.port == 9000
        assert config.gpu_ids == [0, 1]
        assert config.tensor_parallel_size == 2


class TestParseSubmodelSpec:
    """Tests for submodel spec parsing."""

    def test_vllm_spec(self) -> None:
        config = parse_submodel_spec("math:vllm:Qwen/Qwen2.5-Math-72B")
        assert config.alias == "math"
        assert config.backend == "vllm"
        assert config.model_id == "Qwen/Qwen2.5-Math-72B"
        assert config.port == 8000

    def test_ollama_spec(self) -> None:
        config = parse_submodel_spec("small:ollama:llama3.2:1b")
        assert config.alias == "small"
        assert config.backend == "ollama"
        assert config.model_id == "llama3.2:1b"
        assert config.port == 11434

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid submodel spec"):
            parse_submodel_spec("invalid-spec")

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid backend"):
            parse_submodel_spec("test:invalid:model")


class TestBuildServerConfigs:
    """Tests for server config building."""

    def test_single_model(self) -> None:
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=[],
        )
        assert len(configs) == 1
        assert configs[0].model_id == "Qwen/Qwen3-4B"
        assert configs[0].alias == "main"
        assert configs[0].backend == "vllm"
        assert configs[0].port == 8000

    def test_with_submodels(self) -> None:
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=["math:vllm:Qwen/Qwen2.5-Math-72B"],
        )
        assert len(configs) == 2
        assert configs[0].port == 8000
        assert configs[1].port == 8001
        assert configs[1].alias == "math"

    def test_custom_base_port(self) -> None:
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=["math:vllm:Qwen/Math"],
            base_port=9000,
        )
        assert configs[0].port == 9000
        assert configs[1].port == 9001

    def test_ollama_main_backend(self) -> None:
        configs = build_server_configs(
            main_model="llama3.2:1b",
            main_alias="main",
            submodel_specs=[],
            main_backend="ollama",
        )
        assert configs[0].backend == "ollama"
        assert configs[0].port == 11434

    def test_mixed_backends(self) -> None:
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=["small:ollama:llama3.2:1b", "code:vllm:Qwen/Code"],
            base_port=8000,
        )
        assert len(configs) == 3
        # Main vLLM on 8000
        assert configs[0].port == 8000
        assert configs[0].backend == "vllm"
        # Ollama on 11434
        assert configs[1].port == 11434
        assert configs[1].backend == "ollama"
        # Second vLLM on 8001
        assert configs[2].port == 8001
        assert configs[2].backend == "vllm"

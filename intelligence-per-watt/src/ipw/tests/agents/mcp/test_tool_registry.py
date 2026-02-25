"""Tests for agents/mcp/tool_registry.py — ToolRegistry and ToolSpec."""

from __future__ import annotations

import pytest

from ipw.agents.mcp.tool_registry import (
    ToolCategory,
    ToolRegistry,
    ToolSpec,
    get_registry,
    reset_registry,
)


class TestToolSpec:
    """Test ToolSpec dataclass validation."""

    def test_minimal_spec(self) -> None:
        spec = ToolSpec(
            name="test_tool",
            category=ToolCategory.UTILITY,
            description="A test tool",
        )
        assert spec.name == "test_tool"
        assert spec.category == ToolCategory.UTILITY
        assert spec.estimated_cost_usd == 0.0
        assert spec.capabilities == []
        assert spec.adp_domains == []

    def test_all_fields(self) -> None:
        spec = ToolSpec(
            name="web_search",
            category=ToolCategory.SEARCH,
            description="Search the web",
            estimated_latency_ms=500.0,
            estimated_cost_usd=0.01,
            requires_api_key="TAVILY_API_KEY",
            capabilities=["search", "retrieval"],
            adp_domains=["agenttuning_webshop"],
        )
        assert spec.estimated_latency_ms == 500.0
        assert spec.requires_api_key == "TAVILY_API_KEY"
        assert "search" in spec.capabilities


class TestToolCategory:
    """Test ToolCategory enum."""

    def test_utility_category(self) -> None:
        assert ToolCategory.UTILITY.value == "utility"

    def test_code_category(self) -> None:
        assert ToolCategory.CODE.value == "code"

    def test_llm_categories_exist(self) -> None:
        assert ToolCategory.LLM_SMALL.value == "llm_small"
        assert ToolCategory.LLM_MEDIUM.value == "llm_medium"
        assert ToolCategory.LLM_LARGE.value == "llm_large"
        assert ToolCategory.LLM_CLOUD.value == "llm_cloud"

    def test_adp_categories(self) -> None:
        assert ToolCategory.ADP_CODEACT.value == "adp_codeact"
        assert ToolCategory.ADP_ALFWORLD.value == "adp_alfworld"


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return ToolRegistry()

    def test_builtin_calculator_registered(self, registry: ToolRegistry) -> None:
        spec = registry.get_spec("calculator")
        assert spec is not None
        assert spec.category == ToolCategory.UTILITY

    def test_builtin_think_registered(self, registry: ToolRegistry) -> None:
        spec = registry.get_spec("think")
        assert spec is not None

    def test_builtin_code_interpreter_registered(self, registry: ToolRegistry) -> None:
        spec = registry.get_spec("code_interpreter")
        assert spec is not None
        assert spec.category == ToolCategory.CODE

    def test_builtin_web_search_registered(self, registry: ToolRegistry) -> None:
        spec = registry.get_spec("web_search")
        assert spec is not None
        assert spec.category == ToolCategory.SEARCH

    def test_get_all_specs_nonempty(self, registry: ToolRegistry) -> None:
        specs = registry.get_all_specs()
        assert len(specs) > 10

    def test_get_specs_by_category(self, registry: ToolRegistry) -> None:
        utility_specs = registry.get_specs_by_category(ToolCategory.UTILITY)
        assert len(utility_specs) >= 2  # calculator, think
        for spec in utility_specs:
            assert spec.category == ToolCategory.UTILITY

    def test_get_specs_by_capability(self, registry: ToolRegistry) -> None:
        math_specs = registry.get_specs_by_capability("math")
        assert len(math_specs) >= 1
        for spec in math_specs:
            assert "math" in spec.capabilities

    def test_register_custom_tool(self, registry: ToolRegistry) -> None:
        custom = ToolSpec(
            name="custom_tool",
            category=ToolCategory.UTILITY,
            description="Custom tool",
        )
        registry.register(custom)
        assert registry.get_spec("custom_tool") is custom

    def test_nonexistent_spec_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get_spec("nonexistent_tool") is None

    def test_cloud_llms_registered(self, registry: ToolRegistry) -> None:
        cloud_specs = registry.get_specs_by_category(ToolCategory.LLM_CLOUD)
        assert len(cloud_specs) > 0
        names = [s.name for s in cloud_specs]
        assert any("openai:" in n for n in names)

    def test_get_tool_descriptions(self, registry: ToolRegistry) -> None:
        descriptions = registry.get_tool_descriptions(["calculator", "think"])
        assert "calculator" in descriptions
        assert "think" in descriptions


class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_registry_returns_registry(self) -> None:
        reset_registry()
        reg = get_registry()
        assert isinstance(reg, ToolRegistry)

    def test_get_registry_returns_same_instance(self) -> None:
        reset_registry()
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_reset_registry_creates_new_instance(self) -> None:
        reset_registry()
        reg1 = get_registry()
        reset_registry()
        reg2 = get_registry()
        assert reg1 is not reg2

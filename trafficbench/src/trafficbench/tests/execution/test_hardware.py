"""Tests for hardware label derivation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from trafficbench.execution.hardware import derive_hardware_label


@dataclass
class FakeSystemInfo:
    """Mock SystemInfo for testing."""
    cpu_brand: str


@dataclass
class FakeGpuInfo:
    """Mock GpuInfo for testing."""
    name: str


class TestDeriveHardwareLabel:
    """Test hardware label derivation."""

    def test_returns_gpu_name_when_available(self) -> None:
        gpu_info = FakeGpuInfo(name="NVIDIA RTX3090")
        label = derive_hardware_label(None, gpu_info)
        # Should extract alphanumeric token
        assert "RTX3090" in label or "RTX" in label or "3090" in label

    def test_returns_gpu_name_from_dict(self) -> None:
        gpu_info = {"name": "NVIDIA RTX4090"}
        label = derive_hardware_label(None, gpu_info)
        assert "RTX4090" in label or "RTX" in label or "4090" in label

    def test_prefers_gpu_over_cpu(self) -> None:
        system_info = FakeSystemInfo(cpu_brand="Intel Core i9")
        gpu_info = FakeGpuInfo(name="NVIDIA RTX3090")
        label = derive_hardware_label(system_info, gpu_info)
        # Should prefer GPU-related token
        assert "RTX" in label or "NVIDIA" in label or "3090" in label

    def test_falls_back_to_cpu_when_no_gpu(self) -> None:
        system_info = FakeSystemInfo(cpu_brand="Intel Core i9")
        label = derive_hardware_label(system_info, None)
        # Should use CPU info
        assert "I9" in label or "INTEL" in label or "CORE" in label

    def test_returns_unknown_when_no_info(self) -> None:
        label = derive_hardware_label(None, None)
        assert label == "UNKNOWN_HW"

    def test_sanitizes_special_characters(self) -> None:
        gpu_info = {"name": "NVIDIA-RTX-3090"}
        label = derive_hardware_label(None, gpu_info)
        # Should contain extracted tokens without special chars
        assert "-" not in label

    def test_handles_empty_strings(self) -> None:
        system_info = {"cpu_brand": ""}
        gpu_info = {"name": ""}
        label = derive_hardware_label(system_info, gpu_info)
        assert label == "UNKNOWN_HW"

    def test_uppercases_result(self) -> None:
        gpu_info = {"name": "nvidia rtx 3090"}
        label = derive_hardware_label(None, gpu_info)
        assert label.isupper()

    def test_handles_none_attributes(self) -> None:
        system_info = FakeSystemInfo(cpu_brand=None)
        label = derive_hardware_label(system_info, None)
        assert label == "UNKNOWN_HW"

    def test_handles_apple_silicon(self) -> None:
        system_info = {"cpu_brand": "Apple M2 Pro"}
        label = derive_hardware_label(system_info, None)
        # Should extract M2 (alphanumeric)
        assert "M2" in label

    def test_extracts_alphanumeric_tokens_preferentially(self) -> None:
        # GPU name with clear alphanumeric token
        gpu_info = {"name": "Tesla V100"}
        label = derive_hardware_label(None, gpu_info)
        # V100 is alphanumeric and should be preferred
        assert "V100" in label

    def test_returns_consistent_format(self) -> None:
        # Result should be uppercase and alphanumeric (no spaces)
        gpu_info = {"name": "NVIDIA GeForce RTX3090"}
        label = derive_hardware_label(None, gpu_info)
        assert label.isupper()
        assert " " not in label

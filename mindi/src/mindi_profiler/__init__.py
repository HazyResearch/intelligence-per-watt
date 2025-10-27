"""
Core package for the Mindi profiler implementation.

This module exposes the primary extension points so downstream packages can
import from a single namespace.
"""

from .core.client import InferenceClient
from .core.collector import HardwareCollector
from .core.dataset import (
    DatasetProvider,
    get_dataset_provider,
    list_dataset_providers,
    register_dataset,
)
from .core.registry import ClientRegistry
from .core.types import (
    ChatUsage,
    DatasetRecord,
    GpuInfo,
    HardwareInfo,
    Platform,
    ProfilerConfig,
    Response,
    SystemInfo,
    TelemetryReading,
)
from .clients import OllamaClient
from .telemetry_collectors import EnergyMonitorCollector
from .datasets import TrafficBenchDataset

__all__ = [
    "InferenceClient",
    "HardwareCollector",
    "DatasetProvider",
    "register_dataset",
    "get_dataset_provider",
    "list_dataset_providers",
    "ClientRegistry",
    "ChatUsage",
    "GpuInfo",
    "HardwareInfo",
    "Platform",
    "ProfilerConfig",
    "Response",
    "SystemInfo",
    "TelemetryReading",
    "DatasetRecord",
    "OllamaClient",
    "EnergyMonitorCollector",
    "TrafficBenchDataset",
]

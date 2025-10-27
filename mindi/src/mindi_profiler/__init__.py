"""
Core package for the Mindi profiler implementation.

This module exposes the primary extension points so downstream packages can
import from a single namespace.
"""

from .core.client import InferenceClient
from .core.collector import HardwareCollector
from .core.dataset import DatasetProvider
from .core.registry import ClientRegistry, DatasetRegistry
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
from .telemetry_collectors import MindiEnergyMonitorCollector
from .datasets import TrafficBenchDataset

__all__ = [
    "InferenceClient",
    "HardwareCollector",
    "DatasetProvider",
    "ClientRegistry",
    "DatasetRegistry",
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
    "MindiEnergyMonitorCollector",
    "TrafficBenchDataset",
]

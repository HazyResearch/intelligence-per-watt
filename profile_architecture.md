# Mindi Profiler: Extensible Architecture Proposal

**Version:** 1.0  
**Date:** October 24, 2025  
**Author:** Architecture Team

---

## Executive Summary

This document proposes a modular, extensible architecture for the Mindi Profiler—a standalone performance profiling package for inference services. The design prioritizes **extensibility** as the primary goal, enabling contributors to easily add support for new inference clients (vLLM, Ollama, MLX, SGLang), hardware platforms (NVIDIA, AMD, Apple Silicon, Intel), and model configurations without modifying core code.

### Key Design Principles

1. **Plugin-based Architecture** - Registry pattern for clients and collectors
2. **Clear Extension Points** - Abstract base classes with minimal requirements
3. **Configuration-Driven** - Models defined in YAML, no code changes needed
4. **Auto-Discovery** - Automatic detection of available components
5. **Service Agnostic** - Works with any OpenAI-compatible API
6. **Minimal Dependencies** - Core requires only HTTP client, numpy, click

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Extensible Client System](#extensible-client-system)
4. [Extensible Hardware Collectors](#extensible-hardware-collectors)
5. [Configuration-Driven Model Registry](#configuration-driven-model-registry)
6. [Execution Engine](#execution-engine)
7. [Analysis Pipeline](#analysis-pipeline)
8. [Extension Guide](#extension-guide)
9. [Implementation Plan](#implementation-plan)
10. [Appendix: Code Examples](#appendix-code-examples)

---

## 1. Architecture Overview

### 1.1 Package Structure

```
mindi-profiler/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── src/
│   └── mindi_profiler/
│       ├── core/                      # Core abstractions
│       │   ├── client.py              # Base client interface
│       │   ├── collector.py           # Base hardware collector
│       │   ├── registry.py            # Plugin registry
│       │   └── types.py               # Shared types
│       │
│       ├── clients/                   # Inference client plugins
│       │   ├── litellm.py
│       │   ├── vllm.py
│       │   ├── ollama.py
│       │   
│       │   
│       │
│       ├── collectors/   # This will just bind to rust 
collector plugins
│       │   
│       │   
│       │   
│       │   
│       │   
│       │
│       │
│       ├── execution/                 # Profiling engine
│       ├── analysis/                  # Statistical analysis
│       ├── datasets/                  # Dataset management
│       ├── visualization/             # Plotting
│       └── cli/                       # Command-line interface
│
├── examples/                          # Extension examples
│   ├── custom_client.py
│   ├── custom_collector.py
│   └── custom_model.py
│
└── tests/
```

### 1.2 Component Interaction

```
User CLI Command
     ↓
┌────────────────────────────────────────────────────┐
│  Profiling Runner (Orchestration)                  │
└────────────────────────────────────────────────────┘
     ↓                    ↓                    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │  Hardware   │    │   Dataset   │
│  Registry   │    │  Collector  │    │   Loader    │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                    ↓                    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  vLLM/      │    │  NVIDIA/    │    │    Our dataset of 
│  Ollama/    │    │  Apple/     │    │ 1k queries│
│  MLX/etc    │    │  AMD/etc    │    │     │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                    ↓                    ↓
     └──────────────┬─────────────────────────┘
                    ↓
            ┌──────────────────┐
            │  Telemetry       │
            │  Collection      │
            └──────────────────┘
                    ↓
            ┌──────────────────┐
            │  Regression      │
            │  Analysis        │
            └──────────────────┘
                    ↓
            ┌──────────────────┐
            │  Results         │
            │  (Dataset +      │
            │   Summary JSON)  │
            └──────────────────┘
```
### Assume we are using the 1000 queries
---

## 2. Core Components

### 2.1 Core Abstractions

The `core/` module provides base interfaces that all plugins must implement:

#### InferenceClient (Base Interface)

**Purpose:** Abstract interface for inference services

**Required Methods:**
- `__init__(base_url, **config)` - Initialize client
- `stream_chat_completion()` - Streaming inference
- `list_models()` - List available models
- `health()` - Health check (optional)
- `get_metadata()` -> returns metadata (below)

**Metadata:**
- `client_id` - Unique identifier
- `client_name` - Human-readable name
- `client_version` - Version string



#### HardwareCollector (Base Interface)

**Purpose:** Abstract interface for hardware telemetry

**Required Methods:**
- `is_available()` - Check if hardware/drivers present
- `detect_hardware()` - Return hardware information

**Optional Methods:**
- `collect_telemetry()` - Real-time telemetry readings
- `start_monitoring()` - Start continuous monitoring
- `stop_monitoring()` - Stop and return aggregated stats

**Metadata:**
- `collector_id` - Unique identifier
- `collector_name` - Human-readable name
- `supported_platforms` - List of platforms




### 2.2 Registry Pattern

**ClientRegistry** and **CollectorRegistry** provide:
- Component registration via decorators
- Factory methods for instantiation
- Auto-discovery of available components
- Metadata querying

**Registration Example:**
```python
@ClientRegistry.register("vllm")
class VLLMClient(InferenceClient):
    client_id = "vllm"
    # Implementation...
```

**Usage Example:**
```python
# List available
clients = ClientRegistry.list_clients()

# Create instance
client = ClientRegistry.create("vllm", "http://localhost:8000")

# Auto-detect hardware
collector = CollectorRegistry.auto_detect()
```

---

## 3. Extensible Client System

### 3.1 Design Goals

1. **Minimal Implementation Burden** - Only 2 methods required
2. **Standard Interface** - All clients expose same API
3. **Capability Declaration** - Clients declare features
4. **Configuration Validation** - Optional config validation
5. **Easy Registration** - Single decorator for registration

### 3.3 Supported Clients

| Client | Status | Streaming | Telemetry | Special Features |
|--------|--------|-----------|-----------|------------------|
| **vLLM** | ✅ Built-in | Yes | No | OpenAI-compatible |
| **Ollama** | ✅ Built-in | Yes | No | Local model mgmt |
| **LiteLLM** | 📖 Example | - | - | Extension example |

### 3.4 Client Implementation

Each client implementation is ~50-100 lines of code:

stream_chat_completions returns a single Response type that is shared across all of the clients.


```python
@ClientRegistry.register("vllm")
class VLLMClient(InferenceClient):
    client_id = "vllm"
    client_name = "vLLM"
    
    def __init__(self, base_url: str, **config):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, ...)
    
    def stream_chat_completion(self, model, messages, **params):
        response = self._client.post("/v1/chat/completions", 
                                     json={"model": model, 
                                           "messages": messages})
        return <Responce object>
```

---

## 4. Extensible Hardware Collectors

### 4.1 Design Goals

1. **Platform Detection** - Auto-detect available hardware
2. **Graceful Degradation** - Work without telemetry
3. **Permission Awareness** - Document required permissions
4. **Unified Interface** - Standard telemetry format
5. **Optional Telemetry** - Not all collectors need real-time data






### 4.2 Hardware Information

```python
@dataclass
class HardwareInfo:
    platform: Platform              # nvidia, amd, apple, etc
    device_name: str                # GPU/CPU name
    device_id: Optional[str]        # Unique device ID
    vendor: Optional[str]           # NVIDIA, AMD, Apple
    memory_total_gb: Optional[float]
    compute_capability: Optional[str]
    driver_version: Optional[str]
    architecture: Optional[str]     # x86_64, arm64
```

### 4.3 Telemetry Readings

```python
@dataclass
class TelemetryReading:
    timestamp_ms: float
    power_watts: Optional[float]
    energy_joules: Optional[float]
    temperature_c: Optional[float]
    memory_used_gb: Optional[float]
    utilization_percent: Optional[float]
```

### 4.4 Supported Platforms

| Platform | Status | Telemetry | Requirements |
|----------|--------|-----------|--------------|
| **NVIDIA** | ✅ Built-in | Yes | nvidia-ml-py (pynvml) |
| **Apple Silicon** | ✅ Built-in | Yes | macOS, sudo for powermetrics |
| **AMD** | ✅ Built-in | Yes | pyrsmi or rocm-smi |
| **CPU Only** | ✅ Built-in | No | Always available (fallback) |

### 4.5 Auto-Detection Flow

```
CollectorRegistry.auto_detect()
    ↓
Try each registered collector:
    ├─→ NVIDIACollector.is_available()
    │   ├─→ Check for nvidia-smi
    │   └─→ Try importing pynvml
    │
    ├─→ AppleCollector.is_available()
    │   ├─→ Check if macOS
    │   └─→ Check for Apple Silicon
    │
    ├─→ AMDCollector.is_available()
    │   └─→ Check for ROCm
    │
    └─→ CPUCollector.is_available()
        └─→ Always returns True (fallback)
```

Python will determine system that we are running on and run the corresponding rust binary.

---


## 6. Execution Engine

### 6.1 Profiling Runner

The `ProfilingRunner` orchestrates the entire profiling workflow:

**Responsibilities:**
1. Load dataset (bundled or custom)
2. Create inference client
3. Detect hardware
4. Execute queries with batching/concurrency
5. Collect telemetry per query
6. Save incrementally (every N queries)
7. Generate summary statistics

**Key Features:**
- Resume from partial runs
- Automatic retries
- Progress tracking (tqdm)
- Incremental saves

### 6.2 Execution Flow

```
1. Initialize
   ├─→ Load dataset
   ├─→ Create client
   ├─→ Detect hardware
   └─→ Setup collectors

2. For each query:
   ├─→ Send request to inference service
   ├─→ Collect response + telemetry
   ├─→ Parse telemetry (energy, latency, tokens)
   ├─→ Store in records
   └─→ Save every N queries

3. Finalize
   ├─→ Save complete dataset
   ├─→ Compute regressions
   ├─→ Generate summary.json
   └─→ Return results path
```

### 6.3 Telemetry Collection

For each query, collect:

**Token Metrics:**
- Prompt tokens
- Completion tokens
- Total tokens

**Latency Metrics:**
- Time to first token (TTFT)
- Total latency
- Per-token latency
- Throughput (tokens/sec)

**Energy/Power Metrics** (if available):
- Energy per query (Joules)
- Power draw (Watts)
- Temperature (Celsius)

**Memory Metrics** (if available):
- GPU memory peak/median
- CPU memory usage

### 6.4 Dataset Format

Results saved as HuggingFace Dataset with schema:

```python
{
    "problem": str,              # Input prompt
    "answer": str,               # Expected answer (if available)
    "model_answers": {           # Responses by model
        "model_name": str
    },
    "model_metrics": {           # Telemetry by model
        "model_name": {
            "latency_metrics": {...},
            "energy_metrics": {...},
            "power_metrics": {...},
            "memory_metrics": {...},
            "token_metrics": {...},
            "system_info": {...},
            "gpu_info": {...}
        }
    }
}
```

---

## 7. Analysis Pipeline

### 7.1 Regression Analysis

Compute linear regressions for key relationships:

1. **Input tokens vs TTFT** - How prompt length affects first token latency
2. **Total tokens vs Energy** - Energy consumption per token
3. **Total tokens vs Latency** - Overall latency scaling
4. **Total tokens vs Power** - Power consumption patterns
5. **Total tokens vs Power (log)** - Log-scale for exponential relationships

### 7.2 Regression Output

```python
@dataclass
class RegressionResult:
    slope: Optional[float]           # Coefficient
    intercept: Optional[float]       # Y-intercept
    r_squared: Optional[float]       # Goodness of fit
    sample_count: int                # Number of samples
    avg_y: Optional[float]           # Mean Y value
```

### 7.3 Summary Generation

Output `summary.json`:

```json
{
  "model": "llama-3.1-70b",
  "hardware": "A100_80GB",
  "total_queries": 1000,
  "regressions": {
    "input_tokens_vs_ttft": {
      "slope": 0.0023,
      "intercept": 0.15,
      "r2": 0.94,
      "avg_y": 0.35,
      "count": 998
    },
    "total_tokens_vs_energy": {
      "slope": 2.45,
      "intercept": 15.3,
      "r2": 0.89,
      "avg_y": 245.6,
      "count": 987
    },
    ...
  },
  "timestamp": 1729785600,
  "node_metadata": {...}
}
```

### 7.4 Visualization

Generate plots:
- **Scatter plots** with regression lines
- **KDE plots** for token distributions
- **Hardware-labeled** with model info
- **PNG output** at 280 DPI

---

## 8. Extension Guide

### 8.1 Adding a New Client

**Step-by-Step:**

1. Create file: `src/mindi_profiler/clients/myclient.py`

2. Implement client:
```python
from mindi_profiler.core.client import InferenceClient
from mindi_profiler.core.registry import ClientRegistry

@ClientRegistry.register("myclient")
class MyClient(InferenceClient):
    client_id = "myclient"
    client_name = "My Client"
    
    def __init__(self, base_url, **config):
        self.base_url = base_url
        # Your initialization
    
    def stream_chat_completion(self, model, messages, **params):
        # Your implementation
        return {...}  # OpenAI-compatible format
    
    @classmethod
    def capabilities(cls):
        return ClientCapabilities(streaming=True)
```

3. Add tests: `tests/test_clients/test_myclient.py`

4. Update README with supported client

5. Submit PR!

**Minimal Requirements:**
- Inherit from `InferenceClient`
- Implement `__init__` and `chat_completion`
- Register with decorator
- Return OpenAI-compatible response format

**Optional:**
- Override `stream_chat_completion()` for streaming
- Override `list_models()` for model discovery
- Override `capabilities()` for feature declaration
- Override `validate_config()` for config validation

### 8.2 Adding a New Hardware Collector
## Refactor to show how to do it in rust

**Step-by-Step:**

1. Create file: `src/mindi_profiler/collectors/mycollector.py`

2. Implement collector:
```python
from mindi_profiler.core.collector import HardwareCollector
from mindi_profiler.core.registry import CollectorRegistry

@CollectorRegistry.register("mycollector")
class MyCollector(HardwareCollector):
    collector_id = "mycollector"
    collector_name = "My Hardware"
    
    def is_available(self) -> bool:
        # Check if hardware/drivers present
        return True
    
    def detect_hardware(self) -> HardwareInfo:
        # Query hardware
        return HardwareInfo(
            platform=Platform.UNKNOWN,
            device_name="My Device",
            ...
        )
    
    def collect_telemetry(self) -> TelemetryReading:
        # Optional: real-time telemetry
        return TelemetryReading(...)
    
    @classmethod
    def supports_telemetry(cls) -> bool:
        return True  # If you implement collect_telemetry
```

3. Add tests: `tests/test_collectors/test_mycollector.py`

4. Update README

5. Submit PR!

**Minimal Requirements:**
- Inherit from `HardwareCollector`
- Implement `is_available()` and `detect_hardware()`
- Register with decorator

**Optional:**
- Implement `collect_telemetry()` for real-time metrics
- Override `supports_telemetry()` to return True
- Override `required_permissions()` to document needs

---


### A.3 CLI Usage Examples

```bash
# List available components
mindi-profiler list-clients
mindi-profiler list-collectors
mindi-profiler list-models

# Profile vLLM service
mindi-profiler profile \
    --client vllm \
    --url http://localhost:8000 \
    --model llama-3.1-70b \
    --output ./results \
    --batch-size 4

# Profile Ollama service
mindi-profiler profile \
    --client ollama \
    --url http://localhost:11434 \
    --model mistral \
    --dataset ./custom_dataset \
    --max-queries 500

# Profile MLX (Apple Silicon)
mindi-profiler profile \
    --client mlx \
    --url http://localhost:8080 \
    --model llama-3.2-1b \
    --collector apple

# Analyze existing results
mindi-profiler analyze ./results

# Generate plots
mindi-profiler plot ./results --output-dir ./plots
```

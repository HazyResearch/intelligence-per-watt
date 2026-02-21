# Prerequisites

IPW's energy telemetry relies on platform-specific APIs. This page covers what is needed on each platform.

## All Platforms

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | >= 3.13 | Runtime |
| uv | latest | Package management |
| Rust | stable | Building the energy monitor |
| protoc | >= 3.0 | Protocol Buffer compiler for gRPC |

## macOS (Apple Silicon)

The macOS collector uses Apple's `powermetrics` utility for GPU, CPU, and ANE (Apple Neural Engine) power readings.

**Requirements:**

- macOS 13+ on Apple Silicon (M1/M2/M3/M4)
- `sudo` access -- `powermetrics` requires root privileges

**What you get:**

- GPU power (watts) and energy (joules)
- CPU power (watts) and energy (joules)
- ANE power (watts) and energy (joules)
- CPU memory usage

**Limitations:**

- No GPU memory usage reporting (Apple Unified Memory is shared)
- No GPU utilization percentage
- Requires entering your password or configuring passwordless sudo for `powermetrics`

## Linux with NVIDIA GPU

The NVIDIA collector uses NVML (NVIDIA Management Library) for GPU telemetry, with optional RAPL for CPU energy.

**Requirements:**

- NVIDIA GPU with driver >= 525
- NVML library (ships with the driver)
- Optional: Read access to `/sys/class/powercap/intel-rapl/` for CPU energy (RAPL)

**What you get:**

- GPU power (watts) and energy (joules)
- GPU temperature (Celsius)
- GPU memory usage and total (MB)
- GPU compute utilization (%)
- GPU memory bandwidth utilization (%)
- GPU tensor core utilization (%) -- requires Ampere or newer
- CPU energy via RAPL (if accessible)
- CPU memory usage

**Enabling RAPL access:**

RAPL energy counters require read permissions. On most distributions:

```bash
# Check if RAPL is available
ls /sys/class/powercap/intel-rapl/

# Grant read access (as root)
chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

Some systems require loading the `intel_rapl_common` kernel module:

```bash
sudo modprobe intel_rapl_common
```

## Linux with AMD GPU

The AMD collector uses ROCm SMI for GPU telemetry.

**Requirements:**

- AMD GPU with ROCm >= 5.0 installed
- `rocm-smi` library accessible
- Optional: RAPL for CPU energy (same as NVIDIA section above)

**What you get:**

- GPU power (watts) and energy (joules)
- GPU temperature (Celsius)
- GPU memory usage and total (MB)
- GPU compute utilization (%)
- GPU memory bandwidth utilization (%)
- CPU energy via RAPL (if accessible)
- CPU memory usage

## Linux (CPU-Only)

If no GPU is detected, the energy monitor falls back to a RAPL-only collector for CPU energy or a null collector that reports system memory only.

**What you get with RAPL:**

- CPU package energy (joules)
- CPU core energy (joules)
- CPU memory usage

**What you get without RAPL (null collector):**

- CPU memory usage only
- All power/energy/temperature fields report -1

## Windows

Windows support is limited. The energy monitor attempts NVML-based collection if an NVIDIA GPU is available, otherwise falls back to the null collector.

**What you get:**

- Same as Linux NVIDIA (if NVIDIA GPU present)
- Null collector otherwise (memory only)

## Docker Requirements (Terminus Agent)

The Terminus agent runs tasks inside Docker containers:

- Docker Engine installed and running
- Current user in the `docker` group (or use `sudo`)
- Internet access for pulling the base image (`ubuntu:22.04`)

```bash
# Verify Docker is accessible
docker run --rm hello-world
```

## Inference Runtimes

You need at least one inference backend:

### Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:1b

# Start the server (default port 11434)
ollama serve
```

### vLLM

```bash
# Install vLLM (requires NVIDIA GPU)
pip install vllm

# Start the server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### OpenAI API

No local setup needed. Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

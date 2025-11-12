# energy-monitor

Cross-platform gRPC service providing unified energy and power telemetry for Intelligence Per Watt.

## Overview

energy-monitor runs as a standalone service that streams hardware telemetry data via gRPC. It abstracts platform-specific power monitoring APIs behind a unified interface, enabling Intelligence Per Watt to collect consistent energy metrics across different hardware configurations.

## Architecture

The service implements a collector pattern with automatic platform detection:

- **gRPC Server**: Streams telemetry data to clients at configurable intervals
- **Collectors**: Platform-specific implementations of the `TelemetryCollector` trait
- **Auto-detection**: Automatically selects the appropriate collector at runtime

Each collector samples hardware metrics and returns:
- Power consumption (watts)
- Energy usage (joules)
- GPU temperature
- Memory usage
- GPU information

## Supported Platforms

- **macOS**: Uses `powermetrics` for Apple Silicon and Intel Macs
- **Linux/Windows + NVIDIA**: Uses NVML (NVIDIA Management Library)
- **Linux + AMD**: Uses ROCm SMI library

Unsupported platforms fall back to a null collector.

## Building

Build the binary using Cargo:

```bash
cd energy-monitor
cargo build --release
```

The binary will be located at `target/release/energy-monitor`.

For Intelligence Per Watt integration, use the provided build script:

```bash
# run from the repository root
uv run scripts/build_energy_monitor.py
```

This compiles the binary and installs it to `intelligence-per-watt/src/ipw/telemetry/bin/`.

## Running

Start the gRPC server:

```bash
./energy-monitor --bind-address 127.0.0.1 --port 50053
```

Options:
- `--bind-address`: IP address to bind (default: 127.0.0.1)
- `--port`: Port number (default: 50053)
- `--collection-interval-ms`: Sample interval in milliseconds (default: 50)

macOS requires `sudo` privileges for `powermetrics`.

## Adding Custom Collectors

### 1. Create Collector Module

Create a new file in `src/collectors/` for your platform:

```rust
// src/collectors/custom.rs
use anyhow::Result;
use async_trait::async_trait;
use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

pub struct CustomCollector {
    // Platform-specific state
}

impl CustomCollector {
    pub fn new() -> Result<Self> {
        // Initialize platform-specific APIs
        Ok(Self {})
    }
}

#[async_trait]
impl TelemetryCollector for CustomCollector {
    fn platform_name(&self) -> &str {
        "custom"
    }

    async fn is_available(&self) -> bool {
        // Check if platform APIs are accessible
        true
    }

    async fn collect(&self) -> Result<CollectorSample> {
        // Sample hardware metrics
        // Note: Use -1.0 for metrics that are unavailable or unsupported
        Ok(CollectorSample {
            power_watts: 0.0,
            energy_joules: 0.0,
            temperature_celsius: 0.0,
            gpu_memory_usage_mb: 0.0,
            cpu_memory_usage_mb: 0.0,
            platform: self.platform_name().to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as i64,
            gpu_info: Some(GpuInfo {
                name: String::new(),
                vendor: String::new(),
                device_id: 0,
                device_type: String::new(),
                backend: String::new(),
            }),
        })
    }
}
```

### 2. Register Collector

Add conditional compilation and registration in `src/collectors/mod.rs`:

```rust
#[cfg(target_os = "your_platform")]
mod custom;

#[cfg(target_os = "your_platform")]
use custom::CustomCollector;

pub async fn create_collector(config: Arc<Config>) -> Arc<dyn TelemetryCollector> {
    #[cfg(target_os = "your_platform")]
    {
        if let Ok(collector) = CustomCollector::new() {
            debug!("Auto-detected custom platform");
            return Arc::new(collector);
        }
    }
    
    // Existing platform checks...
    
    Arc::new(NullCollector::new())
}
```

### 3. Add Dependencies

If your collector requires platform-specific crates, add them to `Cargo.toml`:

```toml
[target.'cfg(target_os = "your_platform")'.dependencies]
platform-api = "1.0"
```

### 4. Test

Build and verify your collector is detected:

```bash
cargo build --release
./target/release/energy-monitor
```

The service should log "Auto-detected custom platform" on startup.

## Integration with Intelligence Per Watt

Intelligence Per Watt launches energy-monitor as a subprocess via `ipw.telemetry.launcher`. The launcher:

1. Starts the energy-monitor binary
2. Connects via gRPC client
3. Streams telemetry during profiling runs
4. Terminates the service on completion

# Adding Platform Collectors

Platform collectors are Rust modules that interface with hardware-specific APIs to read power, energy, temperature, and utilization metrics. They run inside the energy monitor gRPC service.

## Architecture

Collectors implement the `TelemetryCollector` trait and are selected at compile time via `#[cfg(target_os)]` attributes. The collector hierarchy is:

```
energy-monitor/src/collectors/
    mod.rs          # TelemetryCollector trait + collector selection logic
    nvidia.rs       # NVIDIA NVML collector
    amd.rs          # AMD ROCm SMI collector
    macos.rs        # macOS powermetrics collector
    linux_rapl.rs   # Linux RAPL CPU energy collector
```

## Step 1: Create the Collector

Create a new file in `energy-monitor/src/collectors/`:

```rust
// energy-monitor/src/collectors/my_platform.rs
use super::{Reading, TelemetryCollector};

pub struct MyPlatformCollector {
    // Platform-specific state
    handle: MyLibraryHandle,
    baseline_energy: f64,
}

impl MyPlatformCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize platform-specific library
        let handle = my_library::init()?;
        let baseline_energy = handle.get_total_energy()?;

        Ok(Self {
            handle,
            baseline_energy,
        })
    }
}

impl TelemetryCollector for MyPlatformCollector {
    fn collect(&mut self) -> Reading {
        // Read current metrics from hardware
        let power = self.handle.get_power_watts()
            .unwrap_or(-1.0);

        let total_energy = self.handle.get_total_energy()
            .unwrap_or(-1.0);
        let energy = if total_energy >= 0.0 {
            total_energy - self.baseline_energy
        } else {
            -1.0
        };

        let temperature = self.handle.get_temperature()
            .unwrap_or(-1.0);

        let gpu_memory_usage = self.handle.get_memory_used_mb()
            .unwrap_or(-1.0);

        let gpu_memory_total = self.handle.get_memory_total_mb()
            .unwrap_or(-1.0);

        Reading {
            power_watts: power,
            energy_joules: energy,
            temperature_celsius: temperature,
            gpu_memory_usage_mb: gpu_memory_usage,
            gpu_memory_total_mb: gpu_memory_total,
            cpu_memory_usage_mb: get_system_memory_mb(),
            cpu_power_watts: -1.0,
            cpu_energy_joules: -1.0,
            ane_power_watts: -1.0,
            ane_energy_joules: -1.0,
            gpu_compute_utilization_pct: self.handle
                .get_utilization().unwrap_or(-1.0),
            gpu_memory_bandwidth_utilization_pct: -1.0,
            gpu_tensor_core_utilization_pct: -1.0,
            platform: "my-platform".to_string(),
            // GPU and system info populated separately
            ..Default::default()
        }
    }

    fn platform(&self) -> &str {
        "my-platform"
    }
}
```

## Step 2: Register in mod.rs

Add your collector to the detection chain in `energy-monitor/src/collectors/mod.rs`:

```rust
#[cfg(target_os = "linux")]
mod my_platform;

pub fn create_collector() -> Box<dyn TelemetryCollector> {
    // Try collectors in order of preference
    #[cfg(target_os = "linux")]
    {
        // Try NVIDIA first
        if let Ok(collector) = nvidia::NvidiaCollector::new() {
            return Box::new(collector);
        }
        // Try AMD
        if let Ok(collector) = amd::AmdCollector::new() {
            return Box::new(collector);
        }
        // Try your new platform
        if let Ok(collector) = my_platform::MyPlatformCollector::new() {
            return Box::new(collector);
        }
        // Fall back to RAPL
        if let Ok(collector) = linux_rapl::RaplCollector::new() {
            return Box::new(collector);
        }
    }

    // Null collector as final fallback
    Box::new(NullCollector::new())
}
```

## Step 3: Add Dependencies

If your collector needs external crate dependencies, add them to `energy-monitor/Cargo.toml`:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
my-library-sys = "0.1"
```

## Step 4: Build and Test

```bash
# Rebuild the energy monitor
uv run scripts/build_energy_monitor.py

# Test the collector
uv run scripts/test_energy_monitor.py
```

Verify the output shows your platform name and metrics.

## The Reading Struct

Each collector returns a `Reading` with these fields:

| Field | Type | Sentinel | Description |
|-------|------|----------|-------------|
| `power_watts` | f64 | -1.0 | GPU power draw |
| `energy_joules` | f64 | -1.0 | GPU energy since baseline |
| `temperature_celsius` | f64 | -1.0 | GPU temperature |
| `gpu_memory_usage_mb` | f64 | -1.0 | GPU memory used |
| `gpu_memory_total_mb` | f64 | -1.0 | Total GPU memory |
| `cpu_memory_usage_mb` | f64 | -1.0 | System memory used |
| `cpu_power_watts` | f64 | -1.0 | CPU power draw |
| `cpu_energy_joules` | f64 | -1.0 | CPU energy since baseline |
| `ane_power_watts` | f64 | -1.0 | ANE power (macOS only) |
| `ane_energy_joules` | f64 | -1.0 | ANE energy (macOS only) |
| `gpu_compute_utilization_pct` | f64 | -1.0 | Compute utilization % |
| `gpu_memory_bandwidth_utilization_pct` | f64 | -1.0 | Memory BW utilization % |
| `gpu_tensor_core_utilization_pct` | f64 | -1.0 | Tensor core utilization % |
| `platform` | String | -- | Platform identifier |

**Convention**: Use `-1.0` for any metric that your collector cannot provide. The Python side checks for this with `math.isfinite()`.

## Energy Calculation Pattern

Most hardware APIs provide a cumulative energy counter. The standard pattern is:

1. Record the baseline energy at initialization.
2. On each `collect()` call, read the current cumulative value.
3. Return `current - baseline` as the energy since monitoring started.

```rust
let energy = if total_energy >= 0.0 {
    total_energy - self.baseline_energy
} else {
    -1.0  // Counter unavailable
};
```

## Existing Collectors for Reference

- `nvidia.rs` -- NVML queries, energy counter differencing, utilization rates
- `amd.rs` -- ROCm SMI, similar pattern to NVIDIA
- `macos.rs` -- `powermetrics` subprocess parsing, CPU/GPU/ANE breakdown
- `linux_rapl.rs` -- sysfs file reading, microjoule to joule conversion

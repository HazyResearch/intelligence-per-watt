// amd_collector.rs
#![allow(dead_code)]
#[cfg(not(target_os = "macos"))]
use anyhow::Result;
#[cfg(not(target_os = "macos"))]
use async_trait::async_trait;
#[cfg(not(target_os = "macos"))]
use rocm_smi_lib::{RocmSmi, RocmSmiDevice, RsmiTemperatureMetric, RsmiTemperatureType};
#[cfg(not(target_os = "macos"))]
use std::{
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[cfg(not(target_os = "macos"))]
use tracing::info;

#[cfg(not(target_os = "macos"))]
use super::{TelemetryCollector, get_system_info};
#[cfg(not(target_os = "macos"))]
use crate::energy::{GpuInfo, SystemInfo, TelemetryReading};

/// AMD telemetry collector using ROC-SMI
#[cfg(not(target_os = "macos"))]
pub struct AmdCollector {
    rsmi: Arc<Mutex<Option<RocmSmi>>>,
    devices: Arc<Mutex<Vec<RocmSmiDevice>>>,
    last_timestamp: Arc<Mutex<Option<Instant>>>,
    accumulated_energy_j: Arc<Mutex<f64>>,
    system_info: Arc<Mutex<SystemInfo>>,
    gpu_info: Arc<Mutex<GpuInfo>>,
}

#[cfg(not(target_os = "macos"))]
impl AmdCollector {
    pub fn new(_config: Arc<crate::config::Config>) -> Result<Self> {
        // Initialize ROC-SMI
        let mut rsmi =
            RocmSmi::init().map_err(|e| anyhow::anyhow!("Failed to init ROC-SMI: {:?}", e))?;

        let count = rsmi.get_device_count();
        if count == 0 {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

        // Open all available GPUs
        let mut devices: Vec<RocmSmiDevice> = Vec::new();
        for i in 0..count {
            if let Ok(dev) = RocmSmiDevice::new(i) {
                devices.push(dev);
            }
        }
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

        // Build aggregated GPU info similar to NVIDIA collector
        let mut names: Vec<String> = Vec::with_capacity(devices.len());
        for d in devices.iter_mut() {
            let name = match d.get_identifiers() {
                Ok(id) => id.name.unwrap_or_else(|_| "Unknown GPU".to_string()),
                Err(_) => "Unknown GPU".to_string(),
            };
            names.push(name);
        }
        let aggregated_name = if names.iter().all(|n| *n == names[0]) {
            format!("{} x{}", names[0], devices.len())
        } else {
            format!("AMD ({} GPUs)", devices.len())
        };
        let gpu_info = GpuInfo {
            name: aggregated_name,
            vendor: "AMD".into(),
            device_id: 0,
            device_type: "GPU".into(),
            backend: "AMDSMI".into(),
        };

        info!("AMD GPUs detected for energy monitoring: {}", gpu_info.name);

        // ROC-SMI doesn't expose a cumulative energy counter; integrate from power
        Ok(Self {
            rsmi: Arc::new(Mutex::new(Some(rsmi))),
            devices: Arc::new(Mutex::new(devices)),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j: Arc::new(Mutex::new(0.0)),
            system_info: Arc::new(Mutex::new(get_system_info())),
            gpu_info: Arc::new(Mutex::new(gpu_info)),
        })
    }
}

#[cfg(not(target_os = "macos"))]
#[async_trait]
impl TelemetryCollector for AmdCollector {
    fn platform_name(&self) -> &str {
        "amd"
    }

    async fn is_available(&self) -> bool {
        self.devices.lock().unwrap().len() > 0
    }

    async fn collect(&self) -> Result<TelemetryReading> {
        let mut reading = TelemetryReading {
            power_watts: -1.0,
            energy_joules: -1.0,
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            platform: "amd".to_string(),
            timestamp_nanos: SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as i64,
            system_info: Some(self.system_info.lock().unwrap().clone()),
            gpu_info: Some(self.gpu_info.lock().unwrap().clone()),
        };

        let mut power_sum_w: f64 = 0.0;
        let mut any_power_ok = false;
        let mut temp_sum_c: f64 = 0.0;
        let mut temp_count: usize = 0;
        let mut mem_sum_mb: f64 = 0.0;
        let mut any_mem_ok = false;

        if let Ok(mut guard) = self.devices.lock() {
            for dev in guard.iter_mut() {
                // Power (already in W per AMD SMI docs)
                if let Ok(power) = dev.get_power_data() {
                    power_sum_w += power.current_power as f64;
                    any_power_ok = true;
                }

                // Temperature - prefer Junction, fallback to Edge; normalize units
                if let Ok(temp) = dev.get_temperature_metric(
                    RsmiTemperatureType::Junction,
                    RsmiTemperatureMetric::Current,
                ) {
                    let t = if temp > 1000.0 { temp / 1000.0 } else { temp };
                    temp_sum_c += t;
                    temp_count += 1;
                } else if let Ok(temp) = dev.get_temperature_metric(
                    RsmiTemperatureType::Edge,
                    RsmiTemperatureMetric::Current,
                ) {
                    let t = if temp > 1000.0 { temp / 1000.0 } else { temp };
                    temp_sum_c += t;
                    temp_count += 1;
                }

                // GPU Memory
                if let Ok(mem) = dev.get_memory_data() {
                    let used_mb = mem.vram_used as f64 / (1024.0 * 1024.0);
                    mem_sum_mb += used_mb;
                    any_mem_ok = true;
                }
            }
        }

        // Fallback: sum hwmon power across all AMD GPU hwmon entries
        if !any_power_ok {
            if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
                for entry in entries.flatten() {
                    let hwmon_path = entry.path();
                    let name_file = hwmon_path.join("name");
                    if let Ok(name) = std::fs::read_to_string(&name_file) {
                        let name = name.trim();
                        if name.contains("amdgpu") || name.contains("radeon") {
                            let power_file = hwmon_path.join("power1_input");
                            if power_file.exists() {
                                if let Ok(power_str) = std::fs::read_to_string(&power_file) {
                                    if let Ok(power_microwatts) = power_str.trim().parse::<u64>() {
                                        power_sum_w += power_microwatts as f64 / 1_000_000.0;
                                        any_power_ok = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if any_power_ok {
            reading.power_watts = power_sum_w;
            // Integrate energy from aggregated power
            let now = Instant::now();
            let mut ts = self.last_timestamp.lock().unwrap();
            if let Some(last) = *ts {
                let dt = now.duration_since(last).as_secs_f64();
                *self.accumulated_energy_j.lock().unwrap() += power_sum_w * dt;
            }
            *ts = Some(now);
            reading.energy_joules = *self.accumulated_energy_j.lock().unwrap();
        }

        if temp_count > 0 {
            reading.temperature_celsius = temp_sum_c / (temp_count as f64);
        }

        if any_mem_ok {
            reading.gpu_memory_usage_mb = mem_sum_mb;
        }

        // CPU memory (system used)
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        reading.cpu_memory_usage_mb = (sys.used_memory() as f64) / (1024.0 * 1024.0);

        Ok(reading)
    }

    async fn reset_baseline(&self) -> Result<()> {
        *self.accumulated_energy_j.lock().unwrap() = 0.0;
        *self.last_timestamp.lock().unwrap() = None;
        Ok(())
    }

    // per-query memory tracking removed from collector; computed client-side
}

// Stub for macOS
#[cfg(target_os = "macos")]
pub struct AmdCollector;

#[cfg(target_os = "macos")]
impl AmdCollector {
    pub fn new(_: Arc<crate::config::Config>) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!("AMD collector not available on macOS"))
    }
}

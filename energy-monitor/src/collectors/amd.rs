#![cfg(all(not(target_os = "macos"), not(target_os = "windows")))]

use anyhow::Result;
use async_trait::async_trait;
use rocm_smi_lib::{RocmSmi, RocmSmiDevice, RsmiTemperatureMetric, RsmiTemperatureType};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

/// AMD telemetry collector using ROC-SMI
pub struct AmdCollector {
    rsmi: Arc<Mutex<Option<RocmSmi>>>,
    devices: Arc<Mutex<Vec<RocmSmiDevice>>>,
    last_timestamp: Arc<Mutex<Option<Instant>>>,
    accumulated_energy_j: Arc<Mutex<f64>>,
    gpu_info: Arc<Mutex<GpuInfo>>,
}

impl AmdCollector {
    pub fn new(_config: Arc<crate::config::Config>) -> Result<Self> {
        let mut rsmi =
            RocmSmi::init().map_err(|e| anyhow::anyhow!("Failed to init ROC-SMI: {:?}", e))?;

        let count = rsmi.get_device_count();
        if count == 0 {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

        let mut devices: Vec<RocmSmiDevice> = Vec::new();
        for i in 0..count {
            if let Ok(dev) = RocmSmiDevice::new(i) {
                devices.push(dev);
            }
        }
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

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

        Ok(Self {
            rsmi: Arc::new(Mutex::new(Some(rsmi))),
            devices: Arc::new(Mutex::new(devices)),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j: Arc::new(Mutex::new(0.0)),
            gpu_info: Arc::new(Mutex::new(gpu_info)),
        })
    }
}

#[async_trait]
impl TelemetryCollector for AmdCollector {
    fn platform_name(&self) -> &str {
        "amd"
    }

    async fn is_available(&self) -> bool {
        self.devices.lock().unwrap().len() > 0
    }

    async fn collect(&self) -> Result<CollectorSample> {
        let mut sample = CollectorSample {
            power_watts: -1.0,
            energy_joules: -1.0,
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            platform: "amd".to_string(),
            timestamp_nanos: SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as i64,
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
                if let Ok(power) = dev.get_power_data() {
                    power_sum_w += power.current_power as f64 / 1_000_000.0;
                    any_power_ok = true;
                }

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

                if let Ok(mem) = dev.get_memory_data() {
                    let used_mb = mem.vram_used as f64 / (1024.0 * 1024.0);
                    mem_sum_mb += used_mb;
                    any_mem_ok = true;
                }
            }
        }

        if any_power_ok {
            sample.power_watts = power_sum_w;
            let now = Instant::now();
            let mut ts = self.last_timestamp.lock().unwrap();
            if let Some(last) = *ts {
                let dt = now.duration_since(last).as_secs_f64();
                *self.accumulated_energy_j.lock().unwrap() += power_sum_w * dt;
            }
            *ts = Some(now);
            sample.energy_joules = *self.accumulated_energy_j.lock().unwrap();
        } else {
            warn!("ROC-SMI did not provide AMD GPU power metrics; returning sentinel values");
        }

        if temp_count > 0 {
            sample.temperature_celsius = temp_sum_c / (temp_count as f64);
        }

        if any_mem_ok {
            sample.gpu_memory_usage_mb = mem_sum_mb;
        }

        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        sample.cpu_memory_usage_mb = (sys.used_memory() as f64) / (1024.0 * 1024.0);

        Ok(sample)
    }
}

#![cfg(all(not(target_os = "macos"), not(target_os = "windows")))]

use anyhow::Result;
use async_trait::async_trait;
use rocm_smi_lib::{RocmSmi, RocmSmiDevice, RsmiTemperatureMetric, RsmiTemperatureType};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

const MICROWATTS_PER_WATT: f64 = 1_000_000.0;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// AMD telemetry collector using ROC-SMI
pub struct AmdCollector {
    rsmi: Arc<Mutex<Option<RocmSmi>>>,
    devices: Arc<Mutex<Vec<(u32, RocmSmiDevice)>>>,
    last_timestamp: Arc<Mutex<Option<Instant>>>,
    accumulated_energy_j: Arc<Mutex<f64>>,
    gpu_info: Arc<Mutex<GpuInfo>>,
    #[cfg(target_os = "linux")]
    rapl_reader: Mutex<Option<super::linux_rapl::RaplReader>>,
}

impl AmdCollector {
    pub fn new(_config: Arc<crate::config::Config>) -> Result<Self> {
        let mut rsmi =
            RocmSmi::init().map_err(|e| anyhow::anyhow!("Failed to init ROC-SMI: {:?}", e))?;

        let count = rsmi.get_device_count();
        if count == 0 {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

        let mut devices: Vec<(u32, RocmSmiDevice)> = Vec::new();
        for i in 0..count {
            if let Ok(dev) = RocmSmiDevice::new(i) {
                devices.push((i, dev));
            }
        }
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No AMD GPUs found"));
        }

        let mut names: Vec<String> = Vec::with_capacity(devices.len());
        for (_, d) in devices.iter_mut() {
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

        // Initialize RAPL reader for CPU energy on Linux
        #[cfg(target_os = "linux")]
        let rapl_reader = {
            let reader = super::linux_rapl::RaplReader::new();
            if reader.is_some() {
                info!("RAPL reader initialized for CPU energy monitoring");
            }
            Mutex::new(reader)
        };

        Ok(Self {
            rsmi: Arc::new(Mutex::new(Some(rsmi))),
            devices: Arc::new(Mutex::new(devices)),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j: Arc::new(Mutex::new(0.0)),
            gpu_info: Arc::new(Mutex::new(gpu_info)),
            #[cfg(target_os = "linux")]
            rapl_reader,
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
            gpu_memory_total_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            cpu_power_watts: -1.0,
            cpu_energy_joules: -1.0,
            ane_power_watts: -1.0,
            ane_energy_joules: -1.0,
            gpu_compute_utilization_pct: -1.0,
            gpu_memory_bandwidth_utilization_pct: -1.0,
            gpu_tensor_core_utilization_pct: -1.0,
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
        let mut mem_total_sum_mb: f64 = 0.0;
        let mut any_mem_total_ok = false;
        let mut compute_util_sum: f64 = 0.0;
        let mut compute_util_count: usize = 0;
        let mut memory_util_sum: f64 = 0.0;
        let mut memory_util_count: usize = 0;

        if let Ok(mut guard) = self.devices.lock() {
            for (device_index, dev) in guard.iter_mut() {
                // Prefer current socket power: it's supported on MI300X VFs
                // even when other power telemetry calls are not.
                let mut socket_power_uw: u64 = 0;
                let power_status = unsafe {
                    rocm_smi_lib::rsmi_dev_current_socket_power_get(
                        *device_index,
                        &mut socket_power_uw,
                    )
                };
                if power_status == 0 {
                    power_sum_w += socket_power_uw as f64 / MICROWATTS_PER_WATT;
                    any_power_ok = true;
                } else if let Ok(power) = dev.get_power_data() {
                    // Fallback within ROCm SMI for older devices.
                    power_sum_w += power.current_power as f64 / MICROWATTS_PER_WATT;
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
                    let used_mb = mem.vram_used as f64 / BYTES_PER_MIB;
                    mem_sum_mb += used_mb;
                    any_mem_ok = true;
                    let total_mb = mem.vram_total as f64 / BYTES_PER_MIB;
                    mem_total_sum_mb += total_mb;
                    any_mem_total_ok = true;
                }

                // GPU utilization (compute)
                let mut busy_pct: u32 = 0;
                let status = unsafe {
                    rocm_smi_lib::rsmi_dev_busy_percent_get(*device_index, &mut busy_pct)
                };
                if status == 0 {
                    compute_util_sum += busy_pct as f64;
                    compute_util_count += 1;
                }

                // Memory utilization (approx bandwidth utilization)
                let mut mem_busy_pct: u32 = 0;
                let status = unsafe {
                    rocm_smi_lib::rsmi_dev_memory_busy_percent_get(
                        *device_index,
                        &mut mem_busy_pct,
                    )
                };
                if status == 0 {
                    memory_util_sum += mem_busy_pct as f64;
                    memory_util_count += 1;
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
        if any_mem_total_ok {
            sample.gpu_memory_total_mb = mem_total_sum_mb;
        }
        if compute_util_count > 0 {
            sample.gpu_compute_utilization_pct =
                compute_util_sum / (compute_util_count as f64);
        }
        if memory_util_count > 0 {
            sample.gpu_memory_bandwidth_utilization_pct =
                memory_util_sum / (memory_util_count as f64);
        }

        // Fill CPU energy using RAPL on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(mut rapl_guard) = self.rapl_reader.lock() {
                if let Some(ref mut rapl) = *rapl_guard {
                    if let Some((cpu_power, cpu_energy)) = rapl.read() {
                        sample.cpu_power_watts = cpu_power;
                        sample.cpu_energy_joules = cpu_energy;
                    }
                }
            }
        }

        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        sample.cpu_memory_usage_mb = (sys.used_memory() as f64) / (1024.0 * 1024.0);

        Ok(sample)
    }
}

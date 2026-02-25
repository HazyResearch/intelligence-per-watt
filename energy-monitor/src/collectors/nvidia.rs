#[cfg(not(target_os = "macos"))]
use anyhow::Result;
#[cfg(not(target_os = "macos"))]
use async_trait::async_trait;
// no VecDeque needed after removing local memory sampling
#[cfg(not(target_os = "macos"))]
use std::sync::Arc;
#[cfg(not(target_os = "macos"))]
use std::sync::Mutex;
#[cfg(not(target_os = "macos"))]
use tracing::{debug, trace};

#[cfg(not(target_os = "macos"))]
use super::{CollectorSample, TelemetryCollector};
#[cfg(not(target_os = "macos"))]
use crate::energy::GpuInfo;
#[cfg(not(target_os = "macos"))]
use sysinfo::System;

const MICROWATTS_PER_WATT: f64 = 1_000_000.0;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// NVIDIA telemetry collector using NVML
#[cfg(not(target_os = "macos"))]
pub struct NvidiaCollector {
    nvml_devices: Arc<Mutex<Vec<nvml_wrapper::Device<'static>>>>,
    energy_baselines: Arc<Mutex<Vec<u64>>>,
    last_energy_readings: Arc<Mutex<Vec<u64>>>,
    last_timestamp: Arc<Mutex<Option<std::time::Instant>>>,
    accumulated_energy_j_per_gpu: Arc<Mutex<Vec<f64>>>,
    gpu_info: Arc<Mutex<GpuInfo>>,
    #[cfg(target_os = "linux")]
    rapl_reader: Mutex<Option<super::linux_rapl::RaplReader>>,
    /// Cached sysinfo System object for memory queries (avoids expensive new_all() each cycle)
    sysinfo: Mutex<System>,
}

#[cfg(not(target_os = "macos"))]
impl NvidiaCollector {
    pub fn new(_config: Arc<crate::config::Config>) -> anyhow::Result<Self> {
        // Initialize NVML and leak the handle to obtain 'static device references
        let nvml = nvml_wrapper::Nvml::init()
            .map_err(|e| anyhow::anyhow!("Failed to initialize NVML: {}", e))?;
        debug!("NVML initialized successfully for energy monitoring");
        let nvml_static: &'static nvml_wrapper::Nvml = Box::leak(Box::new(nvml));

        // Enumerate visible devices
        let count = nvml_static
            .device_count()
            .map_err(|e| anyhow::anyhow!("Failed to get NVIDIA device count: {}", e))?;
        debug!("NVML device count detected: {}", count);

        let selectors = visible_device_filter_from_env();
        let mut devices: Vec<nvml_wrapper::Device<'static>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        match selectors {
            VisibleDeviceFilter::List(filters) => {
                debug!(
                    "GPU visibility filter resolved to {} selector(s)",
                    filters.len()
                );
                for selector in filters {
                    match selector {
                        VisibleDeviceSelector::Index(idx) => {
                            if idx >= count {
                                debug!(
                                    "Skipping NVIDIA GPU index {} (out of bounds for count {})",
                                    idx, count
                                );
                                continue;
                            }
                            match nvml_static.device_by_index(idx) {
                                Ok(device) => {
                                    let index = device.index().ok();
                                    let name =
                                        device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                                    debug!(index, "Using NVIDIA GPU {}", name);
                                    names.push(name);
                                    devices.push(device);
                                }
                                Err(e) => {
                                    debug!(
                                        "Skipping NVIDIA GPU at filtered index {} due to error: {}",
                                        idx, e
                                    );
                                }
                            }
                        }
                        VisibleDeviceSelector::Uuid(uuid) => {
                            match nvml_static.device_by_uuid(uuid.as_str()) {
                                Ok(device) => {
                                    let index = device.index().ok();
                                    let name =
                                        device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                                    debug!(index, "Using NVIDIA GPU {}", name);
                                    names.push(name);
                                    devices.push(device);
                                }
                                Err(e) => {
                                    debug!("Skipping NVIDIA GPU UUID {} due to error: {}", uuid, e);
                                }
                            }
                        }
                    }
                }

                if devices.is_empty() {
                    debug!(
                        "CUDA_VISIBLE_DEVICES filter matched no GPUs; falling back to all visible GPUs"
                    );
                    enumerate_all_gpus(nvml_static, count, &mut devices, &mut names);
                }
            }
            VisibleDeviceFilter::All => {
                enumerate_all_gpus(nvml_static, count, &mut devices, &mut names);
            }
            VisibleDeviceFilter::None => {
                // Intentionally leave devices empty; user explicitly requested none.
            }
        }

        if devices.is_empty() {
            return Err(anyhow::anyhow!("No NVIDIA GPU found"));
        }

        // Aggregated GPU info
        let aggregated_name = if names.iter().all(|n| *n == names[0]) {
            format!("{} x{}", names[0], devices.len())
        } else {
            format!("NVIDIA ({} GPUs)", devices.len())
        };
        let gpu_info = GpuInfo {
            name: aggregated_name,
            vendor: "NVIDIA".to_string(),
            device_id: 0,
            device_type: "GPU".to_string(),
            backend: "NVML".to_string(),
        };
        debug!("Aggregated GPU info: {}", gpu_info.name);

        // Initialize per-GPU energy state
        let mut energy_baselines: Vec<u64> = Vec::with_capacity(devices.len());
        let mut last_energy_readings: Vec<u64> = Vec::with_capacity(devices.len());
        let accumulated_energy_j_per_gpu: Vec<f64> = vec![0.0; devices.len()];
        for (i, d) in devices.iter().enumerate() {
            let initial_energy = d.total_energy_consumption().ok().unwrap_or(0);
            energy_baselines.push(initial_energy);
            last_energy_readings.push(initial_energy);
            debug!(
                "Initial energy baseline for GPU {} set to: {} mJ",
                i, initial_energy
            );
        }

        // Initialize RAPL reader for CPU energy on Linux
        #[cfg(target_os = "linux")]
        let rapl_reader = {
            let reader = super::linux_rapl::RaplReader::new();
            if reader.is_some() {
                debug!("RAPL reader initialized for CPU energy monitoring");
            } else {
                debug!("RAPL not available; CPU energy will not be collected");
            }
            Mutex::new(reader)
        };

        // Initialize sysinfo once (new_all is expensive, avoid calling per-cycle)
        let sysinfo = System::new_all();
        debug!("sysinfo System initialized for memory monitoring");

        let collector = Self {
            nvml_devices: Arc::new(Mutex::new(devices)),
            energy_baselines: Arc::new(Mutex::new(energy_baselines)),
            last_energy_readings: Arc::new(Mutex::new(last_energy_readings)),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j_per_gpu: Arc::new(Mutex::new(accumulated_energy_j_per_gpu)),
            gpu_info: Arc::new(Mutex::new(gpu_info)),
            #[cfg(target_os = "linux")]
            rapl_reader,
            sysinfo: Mutex::new(sysinfo),
        };

        Ok(collector)
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
enum VisibleDeviceSelector {
    Index(u32),
    Uuid(String),
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
enum VisibleDeviceFilter {
    All,
    None,
    List(Vec<VisibleDeviceSelector>),
}

#[cfg(not(target_os = "macos"))]
fn visible_device_filter_from_env() -> VisibleDeviceFilter {
    let Ok(raw_value) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return VisibleDeviceFilter::All;
    };

    let trimmed = raw_value.trim();
    if trimmed.is_empty() {
        debug!("CUDA_VISIBLE_DEVICES is empty; no GPUs will be visible");
        return VisibleDeviceFilter::None;
    }
    if trimmed.eq_ignore_ascii_case("all") {
        debug!("CUDA_VISIBLE_DEVICES specifies all GPUs");
        return VisibleDeviceFilter::All;
    }
    if trimmed.eq_ignore_ascii_case("none")
        || trimmed.eq_ignore_ascii_case("void")
        || trimmed.eq_ignore_ascii_case("nodevfiles")
    {
        debug!("CUDA_VISIBLE_DEVICES requested no visible GPUs");
        return VisibleDeviceFilter::None;
    }

    let mut selectors = Vec::new();
    for token in trimmed.split(',') {
        let token_trimmed = token.trim();
        if token_trimmed.is_empty() {
            continue;
        }
        match token_trimmed.parse::<u32>() {
            Ok(idx) => selectors.push(VisibleDeviceSelector::Index(idx)),
            Err(_) => selectors.push(VisibleDeviceSelector::Uuid(token_trimmed.to_string())),
        }
    }

    if selectors.is_empty() {
        debug!(
            "No valid entries in CUDA_VISIBLE_DEVICES='{}'; falling back to all GPUs",
            raw_value
        );
        VisibleDeviceFilter::All
    } else {
        VisibleDeviceFilter::List(selectors)
    }
}

#[cfg(not(target_os = "macos"))]
fn enumerate_all_gpus(
    nvml: &'static nvml_wrapper::Nvml,
    count: u32,
    devices: &mut Vec<nvml_wrapper::Device<'static>>,
    names: &mut Vec<String>,
) {
    for i in 0..count {
        match nvml.device_by_index(i) {
            Ok(device) => {
                let index = device.index().ok();
                let name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                debug!(index, "Using NVIDIA GPU {}", name);
                names.push(name);
                devices.push(device);
            }
            Err(e) => {
                debug!("Skipping NVIDIA GPU at index {} due to error: {}", i, e);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[async_trait]
impl TelemetryCollector for NvidiaCollector {
    fn platform_name(&self) -> &str {
        "nvidia"
    }

    async fn is_available(&self) -> bool {
        if let Ok(devices_guard) = self.nvml_devices.lock() {
            !devices_guard.is_empty()
        } else {
            false
        }
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
            platform: "nvidia".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            gpu_info: Some(self.gpu_info.lock().unwrap().clone()),
        };

        if let Ok(devices_guard) = self.nvml_devices.lock() {
            let gpu_count = devices_guard.len();
            trace!("Collecting telemetry for {} NVIDIA GPUs", gpu_count);

            // Determine dt for this cycle
            let now = std::time::Instant::now();
            let dt_s = {
                let mut ts_guard = self.last_timestamp.lock().unwrap();
                let dt = if let Some(last_ts) = *ts_guard {
                    now.duration_since(last_ts).as_secs_f64()
                } else {
                    0.0
                };
                *ts_guard = Some(now);
                dt
            };

            let mut power_sum_w: f64 = 0.0;
            let mut any_power_ok = false;
            let mut energy_total_j: f64 = 0.0;
            let mut any_energy_ok = false;
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
            let mut tensor_util_sum: f64 = 0.0;
            let mut tensor_util_count: usize = 0;

            let mut last_energy_readings = self.last_energy_readings.lock().unwrap();
            let mut energy_baselines = self.energy_baselines.lock().unwrap();
            let mut acc_energy = self.accumulated_energy_j_per_gpu.lock().unwrap();

            if last_energy_readings.len() != gpu_count {
                last_energy_readings.resize(gpu_count, 0);
            }
            if acc_energy.len() != gpu_count {
                acc_energy.resize(gpu_count, 0.0);
            }
            if energy_baselines.len() != gpu_count {
                energy_baselines.resize(gpu_count, 0);
            }

            for (i, device) in devices_guard.iter().enumerate() {
                // Power (direct)
                let mut power_w_i: Option<f64> = None;
                if let Ok(power_mw) = device.power_usage() {
                    let p = (power_mw as f64) / 1000.0;
                    power_w_i = Some(p);
                    power_sum_w += p;
                    any_power_ok = true;
                    trace!("GPU {} power (direct): {:.3} W", i, p);
                }

                // Energy
                match device.total_energy_consumption() {
                    Ok(current_energy_mj) => {
                        // Use energy since baseline, handling wraparound
                        let baseline_mj = energy_baselines[i];
                        let delta_since_baseline_mj = if current_energy_mj >= baseline_mj {
                            current_energy_mj - baseline_mj
                        } else {
                            (u64::MAX - baseline_mj) + current_energy_mj
                        };
                        energy_total_j += (delta_since_baseline_mj as f64) / 1000.0;
                        any_energy_ok = true;

                        // Derive power if needed
                        if dt_s > 0.0 && power_w_i.is_none() {
                            let last_mj = last_energy_readings[i];
                            let delta_mj = if current_energy_mj >= last_mj {
                                current_energy_mj - last_mj
                            } else {
                                (u64::MAX - last_mj) + current_energy_mj
                            };
                            let p = (delta_mj as f64 / 1000.0) / dt_s;
                            power_sum_w += p;
                            any_power_ok = true;
                            trace!("GPU {} power (derived from energy): {:.3} W", i, p);
                        }

                        // Large jump logging (compute against previous before updating)
                        let last_mj_prev = last_energy_readings[i];
                        let jump_mj = if current_energy_mj >= last_mj_prev {
                            current_energy_mj - last_mj_prev
                        } else {
                            (u64::MAX - last_mj_prev) + current_energy_mj
                        };
                        if jump_mj > 1_000_000 {
                            trace!("GPU {} large energy jump detected: {} mJ", i, jump_mj);
                        }

                        // Update last energy reading
                        last_energy_readings[i] = current_energy_mj;
                    }
                    Err(_) => {
                        // No hardware energy counter; integrate with power if available
                        if dt_s > 0.0 {
                            if let Some(p) = power_w_i {
                                acc_energy[i] += p * dt_s;
                                trace!(
                                    "GPU {} energy (integrated fallback) += {:.6} J (total {:.6} J)",
                                    i,
                                    p * dt_s,
                                    acc_energy[i]
                                );
                            } else {
                                trace!(
                                    "GPU {}: no power available; cannot integrate energy this cycle",
                                    i
                                );
                            }
                        } else {
                            trace!(
                                "GPU {}: dt=0; skipping energy integration on first cycle",
                                i
                            );
                        }

                        // Only report fallback energy if we have accumulated > 0.0 J for this GPU
                        if acc_energy[i] > 0.0 {
                            energy_total_j += acc_energy[i];
                            any_energy_ok = true;
                        } else {
                            trace!("GPU {}: no usable energy reading this cycle (fallback)", i);
                        }
                    }
                }

                // Temperature
                if let Ok(temp_c) =
                    device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                {
                    temp_sum_c += temp_c as f64;
                    temp_count += 1;
                    trace!("GPU {} temperature: {} C", i, temp_c);
                }

                // Memory (used)
                if let Ok(mem_info) = device.memory_info() {
                    let used_mb = mem_info.used as f64 / BYTES_PER_MIB;
                    let total_mb = mem_info.total as f64 / BYTES_PER_MIB;
                    mem_sum_mb += used_mb;
                    any_mem_ok = true;
                    mem_total_sum_mb += total_mb;
                    any_mem_total_ok = true;
                    trace!("GPU {} memory used: {:.2} MB", i, used_mb);
                }

                // Utilization (SM + memory controller)
                if let Ok(util) = device.utilization_rates() {
                    compute_util_sum += util.gpu as f64;
                    compute_util_count += 1;
                    memory_util_sum += util.memory as f64;
                    memory_util_count += 1;
                }

                // Tensor core utilization is not exposed via nvml-wrapper on all platforms.
                // Keep a placeholder for future NVML field-based support.
            }

            if any_power_ok {
                sample.power_watts = power_sum_w;
            }
            if any_energy_ok {
                sample.energy_joules = energy_total_j;
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
                // NVML reports memory controller utilization; use as bandwidth utilization proxy.
                sample.gpu_memory_bandwidth_utilization_pct =
                    memory_util_sum / (memory_util_count as f64);
            }
            if tensor_util_count > 0 {
                sample.gpu_tensor_core_utilization_pct =
                    tensor_util_sum / (tensor_util_count as f64);
            }

            trace!(
                "Aggregated: power={:.3} W, energy={:.6} J, temp_avg={:.2} C, gpu_mem_sum={:.2} MB",
                sample.power_watts,
                sample.energy_joules,
                sample.temperature_celsius,
                sample.gpu_memory_usage_mb
            );

            trace!(
                "Valid counts this cycle: power_ok={} energy_ok={} temp_ok={} mem_ok={}",
                any_power_ok, any_energy_ok, temp_count, any_mem_ok
            );
        }

        // Fill CPU energy using RAPL on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(mut rapl_guard) = self.rapl_reader.lock() {
                if let Some(ref mut rapl) = *rapl_guard {
                    if let Some((cpu_power, cpu_energy)) = rapl.read() {
                        sample.cpu_power_watts = cpu_power;
                        sample.cpu_energy_joules = cpu_energy;
                        trace!(
                            "RAPL CPU: power={:.3} W, energy={:.6} J",
                            cpu_power,
                            cpu_energy
                        );
                    }
                }
            }
        }

        // Fill cpu_memory_usage_mb using cached sysinfo (avoids expensive new_all per cycle)
        if let Ok(mut sys) = self.sysinfo.lock() {
            sys.refresh_memory();
            let used_mb = (sys.used_memory() as f64) / BYTES_PER_MIB;
            sample.cpu_memory_usage_mb = used_mb;
        }
        Ok(sample)
    }

    // per-query memory tracking removed from collector; computed client-side
}

// Empty stub for macOS builds
#[cfg(target_os = "macos")]
pub struct NvidiaCollector;

#[cfg(target_os = "macos")]
impl NvidiaCollector {
    pub fn new(_config: Arc<crate::config::Config>) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!("NVIDIA collector not available on macOS"))
    }
}

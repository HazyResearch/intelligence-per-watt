#![cfg(target_os = "macos")]

use anyhow::Result;
use async_process::{Command, Stdio};
use async_trait::async_trait;
use futures::io::AsyncReadExt;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

/// macOS telemetry collector using powermetrics
///
/// Uses monotonic energy tracking: stores baseline at init, reports delta from baseline.
/// This matches the NVIDIA collector behavior and prevents energy undercounting from
/// missed samples.
pub struct MacOSCollector {
    child: Arc<Mutex<Option<async_process::Child>>>,
    // GPU metrics - baseline and last reading for monotonic tracking
    gpu_energy_baseline_j: Arc<Mutex<Option<f64>>>,
    gpu_energy_accumulated_j: Arc<Mutex<f64>>,
    last_gpu_power_w: Arc<Mutex<f64>>,
    // CPU metrics - baseline and last reading for monotonic tracking
    cpu_energy_baseline_j: Arc<Mutex<Option<f64>>>,
    cpu_energy_accumulated_j: Arc<Mutex<f64>>,
    last_cpu_power_w: Arc<Mutex<f64>>,
    // ANE metrics - baseline and last reading for monotonic tracking
    ane_energy_baseline_j: Arc<Mutex<Option<f64>>>,
    ane_energy_accumulated_j: Arc<Mutex<f64>>,
    last_ane_power_w: Arc<Mutex<f64>>,
    available: Arc<Mutex<bool>>,
    // Debug: track if we've logged processor keys
    logged_keys: Arc<Mutex<bool>>,
}

impl MacOSCollector {
    pub async fn new() -> Result<Self> {
        info!("Initializing macOS powermetrics collector");

        // Use 50ms sample rate to match the server's collection interval
        let mut child = match Command::new("sudo")
            .args([
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
                "--sample-rate",
                "50",
                "--format",
                "plist",
                "--hide-cpu-duty-cycle",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                warn!(
                    "Failed to spawn powermetrics: {}. Energy monitoring will be unavailable.",
                    e
                );
                return Ok(Self::unavailable());
            }
        };

        match child.try_status() {
            Ok(Some(status)) => {
                warn!("powermetrics exited immediately with status: {:?}", status);
                return Ok(Self::unavailable());
            }
            Ok(None) => info!("powermetrics started successfully (50ms sample rate)"),
            Err(e) => warn!("Failed to check powermetrics status: {}", e),
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            gpu_energy_baseline_j: Arc::new(Mutex::new(None)),
            gpu_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_gpu_power_w: Arc::new(Mutex::new(0.0)),
            cpu_energy_baseline_j: Arc::new(Mutex::new(None)),
            cpu_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_cpu_power_w: Arc::new(Mutex::new(0.0)),
            ane_energy_baseline_j: Arc::new(Mutex::new(None)),
            ane_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_ane_power_w: Arc::new(Mutex::new(0.0)),
            available: Arc::new(Mutex::new(true)),
            logged_keys: Arc::new(Mutex::new(false)),
        })
    }

    fn unavailable() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
            gpu_energy_baseline_j: Arc::new(Mutex::new(None)),
            gpu_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_gpu_power_w: Arc::new(Mutex::new(0.0)),
            cpu_energy_baseline_j: Arc::new(Mutex::new(None)),
            cpu_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_cpu_power_w: Arc::new(Mutex::new(0.0)),
            ane_energy_baseline_j: Arc::new(Mutex::new(None)),
            ane_energy_accumulated_j: Arc::new(Mutex::new(0.0)),
            last_ane_power_w: Arc::new(Mutex::new(0.0)),
            available: Arc::new(Mutex::new(false)),
            logged_keys: Arc::new(Mutex::new(false)),
        }
    }

    fn extract_gpu_metrics(plist_value: &plist::Value) -> Option<(f64, f64)> {
        let dict = plist_value.as_dictionary()?;
        let processor = dict.get("processor")?.as_dictionary()?;
        let power_mw = processor.get("gpu_power").and_then(Self::value_as_f64)?;
        let energy_mj = processor.get("gpu_energy").and_then(Self::value_as_f64)?;
        Some((power_mw / 1000.0, energy_mj / 1000.0))
    }

    fn extract_cpu_metrics(plist_value: &plist::Value) -> Option<(f64, f64)> {
        let dict = plist_value.as_dictionary()?;
        let processor = dict.get("processor")?.as_dictionary()?;

        let power_mw = processor.get("cpu_power").and_then(Self::value_as_f64)?;
        let energy_mj = processor.get("cpu_energy").and_then(Self::value_as_f64)?;

        Some((power_mw / 1000.0, energy_mj / 1000.0))
    }

    fn extract_ane_metrics(plist_value: &plist::Value) -> Option<(f64, f64)> {
        let dict = plist_value.as_dictionary()?;
        let processor = dict.get("processor")?.as_dictionary()?;

        let power_mw = processor.get("ane_power").and_then(Self::value_as_f64);
        let energy_mj = processor.get("ane_energy").and_then(Self::value_as_f64);

        match (power_mw, energy_mj) {
            (Some(p), Some(e)) => {
                debug!("ANE metrics: power={}mW, energy={}mJ", p, e);
                Some((p / 1000.0, e / 1000.0))
            }
            _ => {
                debug!(
                    "ANE metrics unavailable: power={:?}, energy={:?}",
                    power_mw, energy_mj
                );
                None
            }
        }
    }

    fn log_processor_keys(plist_value: &plist::Value, logged: &Mutex<bool>) {
        let mut logged_guard = logged.lock().unwrap();
        if *logged_guard {
            return;
        }
        *logged_guard = true;

        if let Some(dict) = plist_value.as_dictionary() {
            if let Some(processor) = dict.get("processor").and_then(|v| v.as_dictionary()) {
                let keys: Vec<&String> = processor.keys().collect();
                info!("powermetrics processor keys: {:?}", keys);

                // Log specific values for debugging
                for key in &["ane_power", "ane_energy", "cpu_power", "cpu_energy", "gpu_power", "gpu_energy"] {
                    if let Some(val) = processor.get(*key) {
                        info!("  {}: {:?}", key, val);
                    }
                }
            }
        }
    }

    fn value_as_f64(value: &plist::Value) -> Option<f64> {
        if let Some(real) = value.as_real() {
            Some(real)
        } else if let Some(integer) = value.as_signed_integer() {
            Some(integer as f64)
        } else if let Some(uinteger) = value.as_unsigned_integer() {
            Some(uinteger as f64)
        } else {
            debug!("value_as_f64 failed for value: {:?}", value);
            None
        }
    }

    /// Update energy tracking with monotonic baseline approach.
    ///
    /// On first reading, sets the baseline. Subsequent readings report
    /// delta from baseline (accumulated_total - baseline).
    /// Also accumulates per-sample energy for robustness.
    fn update_energy_monotonic(
        baseline: &Mutex<Option<f64>>,
        accumulated: &Mutex<f64>,
        sample_energy_j: f64,
    ) -> f64 {
        let mut baseline_guard = baseline.lock().unwrap();
        let mut accumulated_guard = accumulated.lock().unwrap();

        // Accumulate per-sample energy
        *accumulated_guard += sample_energy_j;

        // Set baseline on first reading
        if baseline_guard.is_none() {
            *baseline_guard = Some(*accumulated_guard - sample_energy_j);
            debug!("Set energy baseline to {} J", baseline_guard.unwrap());
        }

        // Return energy since baseline
        let baseline_val = baseline_guard.unwrap_or(0.0);
        *accumulated_guard - baseline_val
    }

    async fn measure_power(&self) -> Result<()> {
        let stdout_option = {
            let mut child_guard = self.child.lock().unwrap();
            if let Some(ref mut child) = *child_guard {
                child.stdout.take()
            } else {
                None
            }
        };

        if let Some(mut stdout) = stdout_option {
            let mut buffer = Vec::new();
            let mut byte = [0u8; 1];
            let mut found_start = false;

            loop {
                match stdout.read_exact(&mut byte).await {
                    Ok(_) => {
                        if !found_start {
                            if byte[0] == b'<' {
                                buffer.push(byte[0]);
                                if let Ok(_) = stdout.read_exact(&mut byte).await {
                                    buffer.push(byte[0]);
                                    if byte[0] == b'?' {
                                        found_start = true;
                                    } else {
                                        buffer.clear();
                                    }
                                }
                            }
                        } else {
                            if byte[0] == 0 {
                                break;
                            }
                            buffer.push(byte[0]);
                        }
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof {
                            return Err(anyhow::anyhow!(
                                "Error reading powermetrics output: {}",
                                e
                            ));
                        }
                        break;
                    }
                }

                if buffer.len() > 1_000_000 {
                    warn!("powermetrics buffer exceeded 1MB, discarding");
                    buffer.clear();
                    found_start = false;
                }
            }

            if !buffer.is_empty() && found_start {
                match plist::Value::from_reader_xml(&buffer[..]) {
                    Ok(plist_value) => {
                        // Log available keys on first successful parse
                        Self::log_processor_keys(&plist_value, &self.logged_keys);

                        // Extract GPU metrics with monotonic energy tracking
                        if let Some((power_watts, energy_joules)) =
                            Self::extract_gpu_metrics(&plist_value)
                        {
                            *self.last_gpu_power_w.lock().unwrap() = power_watts;
                            Self::update_energy_monotonic(
                                &self.gpu_energy_baseline_j,
                                &self.gpu_energy_accumulated_j,
                                energy_joules,
                            );
                        }

                        // Extract CPU metrics with monotonic energy tracking
                        if let Some((cpu_power_watts, cpu_energy_joules)) =
                            Self::extract_cpu_metrics(&plist_value)
                        {
                            *self.last_cpu_power_w.lock().unwrap() = cpu_power_watts;
                            Self::update_energy_monotonic(
                                &self.cpu_energy_baseline_j,
                                &self.cpu_energy_accumulated_j,
                                cpu_energy_joules,
                            );
                        }

                        // Extract ANE metrics with monotonic energy tracking
                        if let Some((ane_power_watts, ane_energy_joules)) =
                            Self::extract_ane_metrics(&plist_value)
                        {
                            *self.last_ane_power_w.lock().unwrap() = ane_power_watts;
                            Self::update_energy_monotonic(
                                &self.ane_energy_baseline_j,
                                &self.ane_energy_accumulated_j,
                                ane_energy_joules,
                            );
                        }
                    }
                    Err(e) => {
                        debug!("Failed to parse plist XML: {}", e);
                    }
                }
            }

            {
                let mut child_guard = self.child.lock().unwrap();
                if let Some(ref mut child) = *child_guard {
                    child.stdout = Some(stdout);
                }
            }
        }

        Ok(())
    }

    /// Get monotonic energy since baseline for a component.
    fn get_energy_since_baseline(baseline: &Mutex<Option<f64>>, accumulated: &Mutex<f64>) -> f64 {
        let baseline_guard = baseline.lock().unwrap();
        let accumulated_guard = accumulated.lock().unwrap();

        if let Some(baseline_val) = *baseline_guard {
            *accumulated_guard - baseline_val
        } else {
            // No baseline yet, return 0
            0.0
        }
    }
}

#[async_trait]
impl TelemetryCollector for MacOSCollector {
    fn platform_name(&self) -> &str {
        "macos"
    }

    async fn is_available(&self) -> bool {
        *self.available.lock().unwrap()
    }

    async fn collect(&self) -> Result<CollectorSample> {
        // measure_power() updates all metrics internally
        if let Err(e) = self.measure_power().await {
            debug!("Failed to measure power: {}", e);
        }

        let gpu_power_watts = *self.last_gpu_power_w.lock().unwrap();
        let gpu_energy_joules = Self::get_energy_since_baseline(
            &self.gpu_energy_baseline_j,
            &self.gpu_energy_accumulated_j,
        );

        let cpu_power_watts = *self.last_cpu_power_w.lock().unwrap();
        let cpu_energy_joules = Self::get_energy_since_baseline(
            &self.cpu_energy_baseline_j,
            &self.cpu_energy_accumulated_j,
        );

        let ane_power_watts = *self.last_ane_power_w.lock().unwrap();
        let ane_energy_joules = Self::get_energy_since_baseline(
            &self.ane_energy_baseline_j,
            &self.ane_energy_accumulated_j,
        );

        let cpu_memory_usage_mb = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let used_bytes = sys.total_memory().saturating_sub(sys.available_memory());
            (used_bytes as f64) / 1_048_576.0
        };

        Ok(CollectorSample {
            power_watts: if gpu_power_watts >= 0.0 { gpu_power_watts } else { -1.0 },
            energy_joules: if gpu_energy_joules >= 0.0 { gpu_energy_joules } else { -1.0 },
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            gpu_memory_total_mb: -1.0,
            cpu_memory_usage_mb,
            cpu_power_watts: if cpu_power_watts >= 0.0 { cpu_power_watts } else { -1.0 },
            cpu_energy_joules: if cpu_energy_joules >= 0.0 { cpu_energy_joules } else { -1.0 },
            ane_power_watts: if ane_power_watts >= 0.0 { ane_power_watts } else { -1.0 },
            ane_energy_joules: if ane_energy_joules >= 0.0 { ane_energy_joules } else { -1.0 },
            gpu_compute_utilization_pct: -1.0,
            gpu_memory_bandwidth_utilization_pct: -1.0,
            gpu_tensor_core_utilization_pct: -1.0,
            platform: "macos".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            gpu_info: Some(GpuInfo {
                name: "Apple GPU".to_string(),
                vendor: "Apple".to_string(),
                device_id: 0,
                device_type: "Integrated GPU".to_string(),
                backend: "powermetrics".to_string(),
            }),
        })
    }
}

impl Drop for MacOSCollector {
    fn drop(&mut self) {
        if let Ok(mut child_guard) = self.child.lock() {
            if let Some(mut child) = child_guard.take() {
                let _ = child.kill();
            }
        }
    }
}

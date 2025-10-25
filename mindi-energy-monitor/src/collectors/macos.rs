#[cfg(target_os = "macos")]
use anyhow::Result;
#[cfg(target_os = "macos")]
use async_process::{Command, Stdio};
#[cfg(target_os = "macos")]
use async_trait::async_trait;
#[cfg(target_os = "macos")]
use futures::io::AsyncReadExt;
#[cfg(target_os = "macos")]
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::time::Instant;
#[cfg(target_os = "macos")]
use tracing::{debug, info, warn};

#[cfg(target_os = "macos")]
use super::{TelemetryCollector, get_system_info};
#[cfg(target_os = "macos")]
use crate::energy::{GpuInfo, SystemInfo, TelemetryReading};

/// macOS telemetry collector using powermetrics
#[cfg(target_os = "macos")]
pub struct MacOSCollector {
    child: Arc<Mutex<Option<async_process::Child>>>,
    last_timestamp: Arc<Mutex<Option<Instant>>>,
    accumulated_energy_j: Arc<Mutex<f64>>,
    last_power_w: Arc<Mutex<f64>>,
    system_info: Arc<Mutex<SystemInfo>>,
    available: Arc<Mutex<bool>>,
}

#[cfg(target_os = "macos")]
impl MacOSCollector {
    pub async fn new() -> Result<Self> {
        info!("Initializing macOS powermetrics collector");

        // Spawn powermetrics under sudo
        let mut child = match Command::new("sudo")
            .args([
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
                "--sample-rate",
                "200",
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
                return Ok(Self {
                    child: Arc::new(Mutex::new(None)),
                    last_timestamp: Arc::new(Mutex::new(None)),
                    accumulated_energy_j: Arc::new(Mutex::new(0.0)),
                    last_power_w: Arc::new(Mutex::new(0.0)),
                    system_info: Arc::new(Mutex::new(get_system_info())),
                    available: Arc::new(Mutex::new(false)),
                });
            }
        };

        // Verify it started successfully
        match child.try_status() {
            Ok(Some(status)) => {
                warn!("powermetrics exited immediately with status: {:?}", status);
                return Ok(Self {
                    child: Arc::new(Mutex::new(None)),
                    last_timestamp: Arc::new(Mutex::new(None)),
                    accumulated_energy_j: Arc::new(Mutex::new(0.0)),
                    last_power_w: Arc::new(Mutex::new(0.0)),
                    system_info: Arc::new(Mutex::new(get_system_info())),
                    available: Arc::new(Mutex::new(false)),
                });
            }
            Ok(None) => {
                info!("powermetrics started successfully");
            }
            Err(e) => {
                warn!("Failed to check powermetrics status: {}", e);
            }
        }

        // Wait a bit for powermetrics to start generating data
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j: Arc::new(Mutex::new(0.0)),
            last_power_w: Arc::new(Mutex::new(0.0)),
            system_info: Arc::new(Mutex::new(get_system_info())),
            available: Arc::new(Mutex::new(true)),
        })
    }

    fn extract_power_watts(plist_value: &plist::Value) -> Option<f64> {
        if let Some(dict) = plist_value.as_dictionary() {
            // Try Apple Silicon path first: processor_power.actual
            if let Some(processor_power) = dict.get("processor_power") {
                if let Some(processor_dict) = processor_power.as_dictionary() {
                    if let Some(actual) = processor_dict.get("actual") {
                        if let Some(power_mw) = actual.as_real() {
                            return Some(power_mw / 1000.0); // Convert mW to W
                        }
                    }
                }
            }

            // Try Intel fallback: processor.combined_power
            if let Some(processor) = dict.get("processor") {
                if let Some(processor_dict) = processor.as_dictionary() {
                    if let Some(combined_power) = processor_dict.get("combined_power") {
                        if let Some(power_mw) = combined_power.as_real() {
                            return Some(power_mw / 1000.0); // Convert mW to W
                        }
                    }
                }
            }
        }

        None
    }

    async fn measure_power(&self) -> Result<f64> {
        // Extract stdout from the child process to avoid holding the lock across await
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

            // Skip any partial data and find the start of a plist
            loop {
                match stdout.read_exact(&mut byte).await {
                    Ok(_) => {
                        if !found_start {
                            // Look for <?xml which marks the start of a plist
                            if byte[0] == b'<' {
                                buffer.push(byte[0]);
                                // Check if this is the start of <?xml
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
                            // We're in a plist, read until NUL
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

                // Safety check to prevent infinite loops
                if buffer.len() > 1_000_000 {
                    buffer.clear();
                    found_start = false;
                }
            }

            if !buffer.is_empty() && found_start {
                // Parse the plist
                if let Ok(plist_value) = plist::Value::from_reader_xml(&buffer[..]) {
                    if let Some(power_watts) = Self::extract_power_watts(&plist_value) {
                        // Update last power reading
                        *self.last_power_w.lock().unwrap() = power_watts;

                        let now = Instant::now();
                        let mut timestamp_guard = self.last_timestamp.lock().unwrap();
                        let mut energy_guard = self.accumulated_energy_j.lock().unwrap();

                        if let Some(last_ts) = *timestamp_guard {
                            let duration_secs = now.duration_since(last_ts).as_secs_f64();
                            let energy_delta = power_watts * duration_secs;
                            *energy_guard += energy_delta;
                        }

                        *timestamp_guard = Some(now);

                        // Restore stdout back to the child process
                        {
                            let mut child_guard = self.child.lock().unwrap();
                            if let Some(ref mut child) = *child_guard {
                                child.stdout = Some(stdout);
                            }
                        }
                        return Ok(power_watts);
                    }
                }
            }

            // Restore stdout back to the child process even if we didn't get power reading
            {
                let mut child_guard = self.child.lock().unwrap();
                if let Some(ref mut child) = *child_guard {
                    child.stdout = Some(stdout);
                }
            }
        }

        Ok(*self.last_power_w.lock().unwrap())
    }
}

#[cfg(target_os = "macos")]
#[async_trait]
impl TelemetryCollector for MacOSCollector {
    fn platform_name(&self) -> &str {
        "macos"
    }

    async fn is_available(&self) -> bool {
        *self.available.lock().unwrap()
    }

    async fn collect(&self) -> Result<TelemetryReading> {
        let power_watts = match self.measure_power().await {
            Ok(p) => p,
            Err(e) => {
                debug!("Failed to measure power: {}", e);
                -1.0
            }
        };

        let energy_joules = *self.accumulated_energy_j.lock().unwrap();

        // System-wide RAM usage via sysinfo
        let cpu_memory_usage_mb = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            // sysinfo 0.36 returns memory in bytes for total/available
            let used_bytes = sys.total_memory().saturating_sub(sys.available_memory());
            (used_bytes as f64) / 1_048_576.0
        };

        Ok(TelemetryReading {
            power_watts,
            energy_joules,
            temperature_celsius: 0.0, // Apple Silicon exposes qualitative thermal state only
            gpu_memory_usage_mb: 0.0, // Unknown/stub for Apple GPU unified memory
            cpu_memory_usage_mb,
            platform: "macos".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            system_info: Some(self.system_info.lock().unwrap().clone()),
            gpu_info: Some(GpuInfo {
                name: "Apple GPU".to_string(),
                vendor: "Apple".to_string(),
                device_id: 0,
                device_type: "Integrated GPU".to_string(),
                backend: "powermetrics".to_string(),
            }),
        })
    }

    async fn reset_baseline(&self) -> Result<()> {
        *self.accumulated_energy_j.lock().unwrap() = 0.0;
        *self.last_timestamp.lock().unwrap() = None;
        debug!("Reset energy baseline");
        Ok(())
    }

    // Tracking hooks removed from trait; no additional methods here
}

#[cfg(target_os = "macos")]
impl Drop for MacOSCollector {
    fn drop(&mut self) {
        if let Ok(mut child_guard) = self.child.lock() {
            if let Some(mut child) = child_guard.take() {
                let _ = child.kill();
            }
        }
    }
}

// Empty stub for non-macOS builds
#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub struct MacOSCollector;

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
impl MacOSCollector {
    pub async fn new() -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "macOS collector not available on this platform"
        ))
    }
}

#![cfg(target_os = "macos")]

use anyhow::Result;
use async_process::{Command, Stdio};
use async_trait::async_trait;
use futures::io::AsyncReadExt;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info, warn};

use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

/// macOS telemetry collector using powermetrics
pub struct MacOSCollector {
    child: Arc<Mutex<Option<async_process::Child>>>,
    last_timestamp: Arc<Mutex<Option<Instant>>>,
    accumulated_energy_j: Arc<Mutex<f64>>,
    last_power_w: Arc<Mutex<f64>>,
    available: Arc<Mutex<bool>>,
}

impl MacOSCollector {
    pub async fn new() -> Result<Self> {
        info!("Initializing macOS powermetrics collector");

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
                    available: Arc::new(Mutex::new(false)),
                });
            }
        };

        match child.try_status() {
            Ok(Some(status)) => {
                warn!("powermetrics exited immediately with status: {:?}", status);
                return Ok(Self {
                    child: Arc::new(Mutex::new(None)),
                    last_timestamp: Arc::new(Mutex::new(None)),
                    accumulated_energy_j: Arc::new(Mutex::new(0.0)),
                    last_power_w: Arc::new(Mutex::new(0.0)),
                    available: Arc::new(Mutex::new(false)),
                });
            }
            Ok(None) => info!("powermetrics started successfully"),
            Err(e) => warn!("Failed to check powermetrics status: {}", e),
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            last_timestamp: Arc::new(Mutex::new(None)),
            accumulated_energy_j: Arc::new(Mutex::new(0.0)),
            last_power_w: Arc::new(Mutex::new(0.0)),
            available: Arc::new(Mutex::new(true)),
        })
    }

    fn extract_power_watts(plist_value: &plist::Value) -> Option<f64> {
        if let Some(dict) = plist_value.as_dictionary() {
            if let Some(processor_power) = dict.get("processor_power") {
                if let Some(processor_dict) = processor_power.as_dictionary() {
                    if let Some(actual) = processor_dict.get("actual") {
                        if let Some(power_mw) = actual.as_real() {
                            return Some(power_mw / 1000.0);
                        }
                    }
                }
            }

            if let Some(processor) = dict.get("processor") {
                if let Some(processor_dict) = processor.as_dictionary() {
                    if let Some(combined_power) = processor_dict.get("combined_power") {
                        if let Some(power_mw) = combined_power.as_real() {
                            return Some(power_mw / 1000.0);
                        }
                    }
                }
            }
        }

        None
    }

    async fn measure_power(&self) -> Result<f64> {
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
                    buffer.clear();
                    found_start = false;
                }
            }

            if !buffer.is_empty() && found_start {
                if let Ok(plist_value) = plist::Value::from_reader_xml(&buffer[..]) {
                    if let Some(power_watts) = Self::extract_power_watts(&plist_value) {
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

#[async_trait]
impl TelemetryCollector for MacOSCollector {
    fn platform_name(&self) -> &str {
        "macos"
    }

    async fn is_available(&self) -> bool {
        *self.available.lock().unwrap()
    }

    async fn collect(&self) -> Result<CollectorSample> {
        let power_watts = match self.measure_power().await {
            Ok(p) => p,
            Err(e) => {
                debug!("Failed to measure power: {}", e);
                -1.0
            }
        };

        let energy_joules = *self.accumulated_energy_j.lock().unwrap();

        let cpu_memory_usage_mb = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let used_bytes = sys.total_memory().saturating_sub(sys.available_memory());
            (used_bytes as f64) / 1_048_576.0
        };

        Ok(CollectorSample {
            power_watts,
            energy_joules,
            temperature_celsius: 0.0,
            gpu_memory_usage_mb: 0.0,
            cpu_memory_usage_mb,
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

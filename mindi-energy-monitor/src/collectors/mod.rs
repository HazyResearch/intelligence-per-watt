use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[cfg(all(not(target_os = "macos"), not(target_os = "windows"), feature = "amd"))]
pub mod amd;
pub mod macos;
#[cfg(not(target_os = "macos"))]
pub mod nvidia;

pub use crate::energy::TelemetryReading;
use crate::energy::{GpuInfo, SystemInfo};

/// Trait for telemetry collectors
#[async_trait]
pub trait TelemetryCollector: Send + Sync {
    /// Get the platform name
    fn platform_name(&self) -> &str;

    /// Get current telemetry reading
    async fn collect(&self) -> Result<TelemetryReading>;

    /// Check if the collector is available
    async fn is_available(&self) -> bool;

    /// Reset energy baseline
    async fn reset_baseline(&self) -> Result<()>;
}

/// Null collector for when no hardware is available
pub struct NullCollector {
    system_info: Arc<Mutex<SystemInfo>>,
}

impl Default for NullCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl NullCollector {
    pub fn new() -> Self {
        Self {
            system_info: Arc::new(Mutex::new(get_system_info())),
        }
    }
}

#[async_trait]
impl TelemetryCollector for NullCollector {
    fn platform_name(&self) -> &str {
        "null"
    }

    async fn is_available(&self) -> bool {
        false
    }

    async fn collect(&self) -> Result<TelemetryReading> {
        Ok(TelemetryReading {
            power_watts: -1.0,
            energy_joules: -1.0,
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            platform: "null".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            system_info: Some(self.system_info.lock().unwrap().clone()),
            gpu_info: Some(GpuInfo {
                name: String::new(),
                vendor: String::new(),
                device_id: 0,
                device_type: String::new(),
                backend: String::new(),
            }),
        })
    }

    async fn reset_baseline(&self) -> Result<()> {
        Ok(())
    }
}

/// Get system information
pub fn get_system_info() -> SystemInfo {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let os_name = System::name().unwrap_or_else(|| String::from(std::env::consts::OS));
    let os_version = System::os_version().unwrap_or_else(|| String::from("Unknown"));
    let kernel_version = System::kernel_version().unwrap_or_else(|| String::from("Unknown"));
    let host_name = System::host_name().unwrap_or_else(|| String::from("localhost"));
    let cpu_count = num_cpus::get();
    let cpu_brand = String::from("Unknown CPU");

    SystemInfo {
        os_name,
        os_version,
        kernel_version,
        host_name,
        cpu_count: cpu_count as u32,
        cpu_brand,
    }
}

/// Create appropriate collector based on platform
pub async fn create_collector(
    #[cfg(not(target_os = "macos"))] config: Arc<crate::config::Config>,
    #[cfg(target_os = "macos")] _config: Arc<crate::config::Config>,
) -> Arc<dyn TelemetryCollector> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(collector) = macos::MacOSCollector::new().await {
            tracing::debug!("Auto-detected macOS platform");
            return Arc::new(collector);
        } else {
            tracing::warn!("Failed to create macOS collector; falling back to null collector");
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(collector) = nvidia::NvidiaCollector::new(config.clone()) {
            tracing::debug!("Auto-detected NVIDIA platform");
            return Arc::new(collector);
        } else {
            tracing::debug!("NVIDIA collector unavailable or failed to initialize");
        }

        #[cfg(all(not(target_os = "macos"), not(target_os = "windows"), feature = "amd"))]
        {
            if let Ok(collector) = amd::AmdCollector::new(config.clone()) {
                tracing::debug!("Auto-detected AMD platform");
                return Arc::new(collector);
            } else {
                tracing::debug!("AMD collector unavailable or failed to initialize");
            }
        }
    }

    tracing::info!("Using null collector (no hardware support)");
    Arc::new(NullCollector::new())
}

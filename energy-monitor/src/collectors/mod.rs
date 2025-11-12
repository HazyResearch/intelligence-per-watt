use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

#[cfg(all(feature = "amd", not(target_os = "macos"), not(target_os = "windows")))]
mod amd;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(target_os = "macos"))]
mod nvidia;

#[cfg(all(feature = "amd", not(target_os = "macos"), not(target_os = "windows")))]
use amd::AmdCollector;
#[cfg(target_os = "macos")]
use macos::MacOSCollector;
#[cfg(not(target_os = "macos"))]
use nvidia::NvidiaCollector;

use crate::config::Config;
use crate::energy::GpuInfo;

/// Trait for telemetry collectors
#[async_trait]
pub trait TelemetryCollector: Send + Sync {
    fn platform_name(&self) -> &str;
    async fn collect(&self) -> Result<CollectorSample>;
    async fn is_available(&self) -> bool;
}

/// GPU-centric telemetry sample emitted by collectors.
#[derive(Clone, Debug)]
pub struct CollectorSample {
    pub power_watts: f64,
    pub energy_joules: f64,
    pub temperature_celsius: f64,
    pub gpu_memory_usage_mb: f64,
    pub cpu_memory_usage_mb: f64,
    pub platform: String,
    pub timestamp_nanos: i64,
    pub gpu_info: Option<GpuInfo>,
}

pub struct NullCollector;

impl Default for NullCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl NullCollector {
    pub fn new() -> Self {
        Self
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

    async fn collect(&self) -> Result<CollectorSample> {
        Ok(CollectorSample {
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

#[cfg_attr(target_os = "macos", allow(unused_variables))]
pub async fn create_collector(config: Arc<Config>) -> Arc<dyn TelemetryCollector> {
    #[cfg(target_os = "macos")]
    {
        match MacOSCollector::new().await {
            Ok(collector) => {
                debug!("Auto-detected macOS platform");
                return Arc::new(collector);
            }
            Err(err) => tracing::warn!("Failed to create macOS collector; falling back: {}", err),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(collector) = NvidiaCollector::new(config.clone()) {
            debug!("Auto-detected NVIDIA platform");
            return Arc::new(collector);
        } else {
            debug!("NVIDIA collector unavailable or failed to initialize");
        }

        #[cfg(all(feature = "amd", not(target_os = "windows")))]
        {
            if let Ok(collector) = AmdCollector::new(config.clone()) {
                debug!("Auto-detected AMD platform");
                return Arc::new(collector);
            } else {
                debug!("AMD collector unavailable or failed to initialize");
            }
        }
    }

    info!("Using null collector (no hardware support)");
    Arc::new(NullCollector::new())
}

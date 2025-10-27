//! Mindi Energy Monitor

pub mod collectors;
pub mod config;
pub mod host;
pub mod server;

// Re-export the generated proto code
pub mod energy {
    tonic::include_proto!("mindi.energy");
}

use std::sync::Arc;
use tracing::{debug, info};

// Export main types
pub use collectors::{CollectorSample, TelemetryCollector};
pub use config::Config;
pub use energy::TelemetryReading;

/// Initializes and returns a telemetry collector based on the provided configuration.
///
/// This function is the main entry point for using the energy monitor as a library.
/// It detects the platform, creates the appropriate collector, and returns it.
pub async fn initialize_collector(config: Config) -> Arc<dyn TelemetryCollector> {
    let config = Arc::new(config);
    debug!(
        "Initializing energy monitor collector with config: {:?}",
        config
    );

    // Create collector based on platform
    let collector = collectors::create_collector(config.clone()).await;

    let is_available = collector.is_available().await;

    if !is_available {
        info!(
            "Energy monitoring is not available on this platform. The monitor will return default values."
        );
    }

    collector
}

use anyhow::Result;
use std::sync::Arc;
use tracing::{info, warn};
mod collectors;
mod config;
mod host;
mod server;

// Include the generated proto code
pub mod energy {
    tonic::include_proto!("energy");
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    info!("Starting Energy Monitor");

    // Parse configuration from CLI
    let config = config::Config::parse();
    let config = Arc::new(config);
    info!("Configuration loaded: {:?}", config);

    let system_info = Arc::new(host::get_system_info());
    info!("System info: {:?}", system_info);

    // Create collector based on platform
    let collector = collectors::create_collector(config.clone()).await;

    match collector.collect().await {
        Ok(sample) => match sample.gpu_info {
            Some(gpu_info)
                if !(gpu_info.name.is_empty()
                    && gpu_info.vendor.is_empty()
                    && gpu_info.device_id == 0
                    && gpu_info.device_type.is_empty()
                    && gpu_info.backend.is_empty()) =>
            {
                info!("GPU info: {:?}", gpu_info);
            }
            _ => info!("GPU info: unavailable"),
        },
        Err(err) => warn!("Failed to gather initial GPU info: {}", err),
    }

    // Run the gRPC server
    server::run_server(
        config.bind_address.clone(),
        config.port,
        collector,
        config,
        system_info,
    )
    .await?;

    Ok(())
}

use anyhow::Result;
use std::sync::Arc;
use tracing::info;
mod collectors;
mod config;
mod host;
mod server;

// Include the generated proto code
pub mod energy {
    tonic::include_proto!("mindi.energy");
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    info!("Starting Mindi Energy Monitor");

    // Parse configuration from CLI
    let config = config::Config::parse();
    let config = Arc::new(config);
    info!("Configuration loaded: {:?}", config);

    // Create collector based on platform
    let collector = collectors::create_collector(config.clone()).await;
    info!(
        "Using {} collector (available: {})",
        collector.platform_name(),
        collector.is_available().await
    );

    // Run the gRPC server
    server::run_server(config.bind_address.clone(), config.port, collector, config).await?;

    Ok(())
}

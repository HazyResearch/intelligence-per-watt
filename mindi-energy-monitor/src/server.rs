//! gRPC server implementation for energy monitoring

use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{error, info};

use crate::collectors::TelemetryCollector;
use crate::config::Config;
use crate::energy::{
    HealthRequest, HealthResponse, ResetRequest, ResetResponse, StreamRequest,
    energy_monitor_server::{EnergyMonitor, EnergyMonitorServer},
};

pub struct EnergyMonitorService {
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
}

impl EnergyMonitorService {
    pub fn new(collector: Arc<dyn TelemetryCollector>, config: Arc<Config>) -> Self {
        Self { collector, config }
    }
}

#[tonic::async_trait]
impl EnergyMonitor for EnergyMonitorService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let healthy = self.collector.is_available().await;
        let platform = self.collector.platform_name().to_string();

        Ok(Response::new(HealthResponse { healthy, platform }))
    }

    type StreamTelemetryStream = Pin<
        Box<
            dyn tokio_stream::Stream<Item = Result<crate::energy::TelemetryReading, Status>> + Send,
        >,
    >;

    async fn stream_telemetry(
        &self,
        _request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        // Create an unbounded channel for streaming
        let (tx, rx) = mpsc::unbounded_channel();
        let collector = self.collector.clone();
        let collection_interval_ms = self.config.collection_interval_ms;

        // Spawn background task to send telemetry based on configured interval
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(collection_interval_ms));

            loop {
                interval.tick().await;

                // Collect telemetry
                match collector.collect().await {
                    Ok(reading) => {
                        if tx.send(Ok(reading)).is_err() {
                            // Client disconnected
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect telemetry: {}", e);
                        // Send error but continue streaming
                        if tx
                            .send(Err(Status::internal("Failed to collect telemetry")))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }

            info!("Telemetry streaming stopped");
        });

        Ok(Response::new(Box::pin(UnboundedReceiverStream::new(rx))))
    }

    async fn reset_energy_baseline(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        match self.collector.reset_baseline().await {
            Ok(()) => Ok(Response::new(ResetResponse { success: true })),
            Err(e) => {
                error!("Failed to reset energy baseline: {}", e);
                Ok(Response::new(ResetResponse { success: false }))
            }
        }
    }
}

pub async fn run_server(
    bind_address: String,
    port: u16,
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
) -> Result<()> {
    let addr = format!("{}:{}", bind_address, port).parse()?;
    let service = EnergyMonitorService::new(collector, config);

    info!("Starting energy monitor gRPC server on {}", addr);

    Server::builder()
        .add_service(EnergyMonitorServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

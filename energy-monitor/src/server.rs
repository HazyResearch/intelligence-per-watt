//! gRPC server implementation for energy monitoring

use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{error, info};

use crate::collectors::{CollectorSample, TelemetryCollector};
use crate::config::Config;
use crate::energy::{
    HealthRequest, HealthResponse, StreamRequest, SystemInfo,
    energy_monitor_server::{EnergyMonitor, EnergyMonitorServer},
};

pub struct EnergyMonitorService {
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
    system_info: Arc<SystemInfo>,
}

impl EnergyMonitorService {
    pub fn new(
        collector: Arc<dyn TelemetryCollector>,
        config: Arc<Config>,
        system_info: Arc<SystemInfo>,
    ) -> Self {
        Self {
            collector,
            config,
            system_info,
        }
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
        let system_info = self.system_info.clone();

        // Spawn background task to send telemetry based on configured interval
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(collection_interval_ms));

            loop {
                interval.tick().await;

                // Collect telemetry
                match collector.collect().await {
                    Ok(sample) => {
                        let reading = assemble_reading(sample, &system_info);
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
}

pub async fn run_server(
    bind_address: String,
    port: u16,
    collector: Arc<dyn TelemetryCollector>,
    config: Arc<Config>,
    system_info: Arc<SystemInfo>,
) -> Result<()> {
    let addr = format!("{}:{}", bind_address, port).parse()?;
    let service = EnergyMonitorService::new(collector, config, system_info);

    info!("Starting energy monitor gRPC server on {}", addr);

    Server::builder()
        .add_service(EnergyMonitorServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

fn assemble_reading(
    sample: CollectorSample,
    system_info: &Arc<SystemInfo>,
) -> crate::energy::TelemetryReading {
    crate::energy::TelemetryReading {
        power_watts: sample.power_watts,
        energy_joules: sample.energy_joules,
        temperature_celsius: sample.temperature_celsius,
        gpu_memory_usage_mb: sample.gpu_memory_usage_mb,
        cpu_memory_usage_mb: sample.cpu_memory_usage_mb,
        platform: sample.platform,
        timestamp_nanos: sample.timestamp_nanos,
        system_info: Some((**system_info).clone()),
        gpu_info: sample.gpu_info,
    }
}

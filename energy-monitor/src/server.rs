//! gRPC server implementation for energy monitoring

use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{debug, error, info};

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
            let mut sample_count: u64 = 0;
            let start_time = std::time::Instant::now();
            let mut last_log_time = start_time;

            loop {
                interval.tick().await;

                let collect_start = std::time::Instant::now();
                // Collect telemetry
                match collector.collect().await {
                    Ok(sample) => {
                        let collect_duration = collect_start.elapsed();
                        sample_count += 1;

                        // Log every 10 seconds
                        let now = std::time::Instant::now();
                        if now.duration_since(last_log_time).as_secs() >= 10 {
                            let elapsed = now.duration_since(start_time).as_secs_f64();
                            let rate = sample_count as f64 / elapsed;
                            debug!(
                                "Telemetry streaming: {} samples in {:.1}s ({:.1}/s), last collect took {:?}",
                                sample_count, elapsed, rate, collect_duration
                            );
                            last_log_time = now;
                        }

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

            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = sample_count as f64 / elapsed;
            info!("Telemetry streaming stopped: {} samples in {:.1}s ({:.1}/s)", sample_count, elapsed, rate);
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
        gpu_memory_total_mb: sample.gpu_memory_total_mb,
        cpu_memory_usage_mb: sample.cpu_memory_usage_mb,
        cpu_power_watts: sample.cpu_power_watts,
        cpu_energy_joules: sample.cpu_energy_joules,
        ane_power_watts: sample.ane_power_watts,
        ane_energy_joules: sample.ane_energy_joules,
        gpu_compute_utilization_pct: sample.gpu_compute_utilization_pct,
        gpu_memory_bandwidth_utilization_pct: sample.gpu_memory_bandwidth_utilization_pct,
        gpu_tensor_core_utilization_pct: sample.gpu_tensor_core_utilization_pct,
        platform: sample.platform,
        timestamp_nanos: sample.timestamp_nanos,
        system_info: Some((**system_info).clone()),
        gpu_info: sample.gpu_info,
    }
}

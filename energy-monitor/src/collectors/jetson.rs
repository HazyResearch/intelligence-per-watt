//! NVIDIA Jetson (Tegra) telemetry collector.
//!
//! NVML initializes on Jetson and reports a device, but its power, energy,
//! temperature, utilization and memory queries all return `NotSupported`.
//! Jetson modules instead carry INA3221 power monitors exposed through hwmon:
//!
//! ```text
//! /sys/class/hwmon/hwmonN/name          -> ina3221
//! /sys/class/hwmon/hwmonN/in{i}_label   -> rail name (e.g. VDD_IN)
//! /sys/class/hwmon/hwmonN/in{i}_input   -> bus voltage (mV)
//! /sys/class/hwmon/hwmonN/curr{i}_input -> current (mA)
//! ```
//!
//! When a `VDD_IN` rail exists (Orin Nano / NX) it is module input power and is
//! reported as the total. Otherwise (AGX Orin, two chips, no `VDD_IN`) the
//! rails are summed. Energy is integrated from power between samples.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use anyhow::Result;
use async_trait::async_trait;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use sysinfo::System;
use tracing::{debug, trace};

use super::{CollectorSample, TelemetryCollector};
use crate::energy::GpuInfo;

const HWMON_ROOT: &str = "/sys/class/hwmon";
const THERMAL_ROOT: &str = "/sys/class/thermal";
const TEGRA_RELEASE_PATH: &str = "/etc/nv_tegra_release";
const DEVICE_TREE_MODEL_PATH: &str = "/proc/device-tree/model";
const INA3221_NAME: &str = "ina3221";
const TOTAL_RAIL_LABEL: &str = "VDD_IN";
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// One INA3221 channel.
#[derive(Clone, Debug, PartialEq)]
struct PowerRail {
    label: String,
    voltage_path: PathBuf,
    current_path: PathBuf,
}

impl PowerRail {
    fn read_watts(&self) -> Option<f64> {
        let millivolts = read_f64_file(&self.voltage_path)?;
        let milliamps = read_f64_file(&self.current_path)?;
        Some(millivolts * milliamps / 1_000_000.0)
    }
}

/// Rails discovered on the module and which one (if any) is the total.
#[derive(Debug, PartialEq)]
struct RailSet {
    rails: Vec<PowerRail>,
    total_index: Option<usize>,
}

impl RailSet {
    /// Returns (total watts, per-rail watts). Total is `VDD_IN` when present,
    /// otherwise the sum of every readable rail.
    fn read(&self) -> Option<(f64, Vec<Option<f64>>)> {
        let per_rail: Vec<Option<f64>> = self.rails.iter().map(PowerRail::read_watts).collect();
        let total = match self.total_index {
            Some(i) => per_rail[i]?,
            None => {
                let readable: Vec<f64> = per_rail.iter().flatten().copied().collect();
                if readable.is_empty() {
                    return None;
                }
                readable.iter().sum()
            }
        };
        Some((total, per_rail))
    }
}

struct EnergyState {
    last_timestamp: Option<Instant>,
    last_power_w: f64,
    accumulated_j: f64,
}

pub struct JetsonCollector {
    rails: RailSet,
    gpu_thermal_path: Option<PathBuf>,
    gpu_info: GpuInfo,
    energy: Mutex<EnergyState>,
    /// Cached sysinfo System object for memory queries (avoids expensive new_all() each cycle)
    sysinfo: Mutex<System>,
}

impl JetsonCollector {
    pub fn new() -> Result<Self> {
        if !is_tegra(Path::new(TEGRA_RELEASE_PATH)) {
            return Err(anyhow::anyhow!(
                "{} not found; not a Jetson/Tegra system",
                TEGRA_RELEASE_PATH
            ));
        }

        let rails = discover_rails(Path::new(HWMON_ROOT));
        if rails.rails.is_empty() {
            return Err(anyhow::anyhow!(
                "Tegra system detected but no readable INA3221 rails under {}",
                HWMON_ROOT
            ));
        }
        for rail in &rails.rails {
            debug!(
                "Jetson power rail {} ({:?}, {:?})",
                rail.label, rail.voltage_path, rail.current_path
            );
        }
        match rails.total_index {
            Some(i) => debug!("Reporting {} as total module power", rails.rails[i].label),
            None => debug!(
                "No {} rail; reporting the sum of {} rails as total power",
                TOTAL_RAIL_LABEL,
                rails.rails.len()
            ),
        }

        let gpu_thermal_path = find_gpu_thermal_zone(Path::new(THERMAL_ROOT));
        debug!("Jetson GPU thermal zone: {:?}", gpu_thermal_path);

        let name = read_device_tree_model(Path::new(DEVICE_TREE_MODEL_PATH))
            .unwrap_or_else(|| "NVIDIA Jetson".to_string());
        let gpu_info = GpuInfo {
            name,
            vendor: "NVIDIA".to_string(),
            device_id: 0,
            device_type: "SoC".to_string(),
            backend: "sysfs/ina3221".to_string(),
        };

        Ok(Self {
            rails,
            gpu_thermal_path,
            gpu_info,
            energy: Mutex::new(EnergyState {
                last_timestamp: None,
                last_power_w: 0.0,
                accumulated_j: 0.0,
            }),
            sysinfo: Mutex::new(System::new_all()),
        })
    }
}

#[async_trait]
impl TelemetryCollector for JetsonCollector {
    fn platform_name(&self) -> &str {
        "jetson"
    }

    async fn is_available(&self) -> bool {
        !self.rails.rails.is_empty()
    }

    async fn collect(&self) -> Result<CollectorSample> {
        let mut sample = CollectorSample {
            power_watts: -1.0,
            energy_joules: -1.0,
            temperature_celsius: -1.0,
            gpu_memory_usage_mb: -1.0,
            gpu_memory_total_mb: -1.0,
            cpu_memory_usage_mb: -1.0,
            cpu_power_watts: -1.0,
            cpu_energy_joules: -1.0,
            ane_power_watts: -1.0,
            ane_energy_joules: -1.0,
            gpu_compute_utilization_pct: -1.0,
            gpu_memory_bandwidth_utilization_pct: -1.0,
            gpu_tensor_core_utilization_pct: -1.0,
            platform: "jetson".to_string(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            gpu_info: Some(self.gpu_info.clone()),
        };

        if let Some((power_w, per_rail)) = self.rails.read() {
            let now = Instant::now();
            let mut state = self.energy.lock().unwrap();
            if let Some(last_ts) = state.last_timestamp {
                // Trapezoidal integration between consecutive samples
                let dt_s = now.duration_since(last_ts).as_secs_f64();
                state.accumulated_j += 0.5 * (state.last_power_w + power_w) * dt_s;
            }
            state.last_timestamp = Some(now);
            state.last_power_w = power_w;

            sample.power_watts = power_w;
            sample.energy_joules = state.accumulated_j;

            for (rail, watts) in self.rails.rails.iter().zip(&per_rail) {
                trace!("Jetson rail {}: {:?} W", rail.label, watts);
            }
            trace!(
                "Jetson total: power={:.3} W, energy={:.6} J",
                power_w, state.accumulated_j
            );
        } else {
            trace!("Jetson: no readable power rails this cycle");
        }

        if let Some(millidegrees) = self.gpu_thermal_path.as_deref().and_then(read_f64_file) {
            sample.temperature_celsius = millidegrees / 1000.0;
        }

        // Jetson memory is unified; report it as CPU memory like the NVIDIA collector does.
        if let Ok(mut sys) = self.sysinfo.lock() {
            sys.refresh_memory();
            sample.cpu_memory_usage_mb = (sys.used_memory() as f64) / BYTES_PER_MIB;
        }

        Ok(sample)
    }
}

fn is_tegra(release_path: &Path) -> bool {
    release_path.exists()
}

/// Find every INA3221 channel under `hwmon_root` that has readable voltage and
/// current inputs. Channels labelled `NC` (not connected) are skipped.
fn discover_rails(hwmon_root: &Path) -> RailSet {
    let mut hwmon_dirs: Vec<PathBuf> = match fs::read_dir(hwmon_root) {
        Ok(entries) => entries.flatten().map(|e| e.path()).collect(),
        Err(e) => {
            debug!("Cannot read {:?}: {}", hwmon_root, e);
            Vec::new()
        }
    };
    // Deterministic rail order across runs
    hwmon_dirs.sort();

    let mut rails = Vec::new();
    for dir in hwmon_dirs {
        let is_ina3221 = fs::read_to_string(dir.join("name"))
            .map(|n| n.trim() == INA3221_NAME)
            .unwrap_or(false);
        if !is_ina3221 {
            continue;
        }

        // INA3221 has three channels, numbered from 1
        for channel in 1..=3 {
            let voltage_path = dir.join(format!("in{}_input", channel));
            let current_path = dir.join(format!("curr{}_input", channel));
            if read_f64_file(&voltage_path).is_none() || read_f64_file(&current_path).is_none() {
                continue;
            }
            let label = fs::read_to_string(dir.join(format!("in{}_label", channel)))
                .map(|l| l.trim().to_string())
                .unwrap_or_else(|_| format!("{}:in{}", dir.display(), channel));
            if label.eq_ignore_ascii_case("NC") {
                continue;
            }
            rails.push(PowerRail {
                label,
                voltage_path,
                current_path,
            });
        }
    }

    let total_index = rails.iter().position(|r| r.label == TOTAL_RAIL_LABEL);
    RailSet { rails, total_index }
}

/// Find the thermal zone whose type names the GPU (`gpu-thermal` on Orin,
/// `GPU-therm` on older modules).
fn find_gpu_thermal_zone(thermal_root: &Path) -> Option<PathBuf> {
    let mut zones: Vec<PathBuf> = fs::read_dir(thermal_root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("thermal_zone"))
        })
        .collect();
    zones.sort();
    zones.into_iter().find_map(|zone| {
        let zone_type = fs::read_to_string(zone.join("type")).ok()?;
        if zone_type.trim().to_ascii_lowercase().starts_with("gpu") {
            Some(zone.join("temp"))
        } else {
            None
        }
    })
}

/// Module name from the device tree, e.g. "NVIDIA Jetson Orin Nano Engineering
/// Reference Developer Kit Super". The file is NUL-terminated.
fn read_device_tree_model(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let model = raw.trim_end_matches('\0').trim();
    (!model.is_empty()).then(|| model.to_string())
}

fn read_f64_file(path: &Path) -> Option<f64> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Temporary directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "energy-monitor-jetson-{}-{}",
                name,
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    /// Write a hwmon device with channels given as (label, mV, mA).
    fn write_hwmon(root: &Path, dir: &str, name: &str, channels: &[(&str, u32, u32)]) {
        let dir = root.join(dir);
        write(&dir.join("name"), &format!("{}\n", name));
        for (i, (label, mv, ma)) in channels.iter().enumerate() {
            let ch = i + 1;
            write(
                &dir.join(format!("in{}_label", ch)),
                &format!("{}\n", label),
            );
            write(&dir.join(format!("in{}_input", ch)), &format!("{}\n", mv));
            write(&dir.join(format!("curr{}_input", ch)), &format!("{}\n", ma));
        }
    }

    fn labels(set: &RailSet) -> Vec<&str> {
        set.rails.iter().map(|r| r.label.as_str()).collect()
    }

    #[test]
    fn orin_nano_reports_vdd_in_as_total() {
        let tmp = TempDir::new("nano");
        write_hwmon(&tmp.0, "hwmon0", "some-other-sensor", &[("X", 1000, 1000)]);
        write_hwmon(
            &tmp.0,
            "hwmon1",
            "ina3221",
            &[
                ("VDD_IN", 5000, 1200),
                ("VDD_CPU_GPU_CV", 5000, 400),
                ("VDD_SOC", 5000, 300),
            ],
        );

        let set = discover_rails(&tmp.0);
        assert_eq!(labels(&set), ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]);
        assert_eq!(set.total_index, Some(0));

        let (total, per_rail) = set.read().unwrap();
        assert!(
            (total - 6.0).abs() < 1e-9,
            "VDD_IN = 5 V * 1.2 A, got {}",
            total
        );
        assert_eq!(per_rail, vec![Some(6.0), Some(2.0), Some(1.5)]);
    }

    #[test]
    fn agx_orin_sums_rails_across_chips_and_skips_nc() {
        let tmp = TempDir::new("agx");
        write_hwmon(
            &tmp.0,
            "hwmon1",
            "ina3221",
            &[
                ("VDD_GPU_SOC", 5000, 1000),
                ("VDD_CPU_CV", 5000, 400),
                ("VIN_SYS_5V0", 5000, 200),
            ],
        );
        write_hwmon(
            &tmp.0,
            "hwmon2",
            "ina3221",
            &[("NC", 0, 0), ("VDDQ_VDD2_1V8AO", 5000, 100)],
        );

        let set = discover_rails(&tmp.0);
        assert_eq!(
            labels(&set),
            [
                "VDD_GPU_SOC",
                "VDD_CPU_CV",
                "VIN_SYS_5V0",
                "VDDQ_VDD2_1V8AO"
            ]
        );
        assert_eq!(set.total_index, None);

        let (total, _) = set.read().unwrap();
        assert!(
            (total - 8.5).abs() < 1e-9,
            "expected 5+2+1+0.5 W, got {}",
            total
        );
    }

    #[test]
    fn channels_without_readable_inputs_are_skipped() {
        let tmp = TempDir::new("partial");
        write_hwmon(&tmp.0, "hwmon0", "ina3221", &[("VDD_IN", 5000, 1000)]);
        // Channel 2 has a label but no inputs
        write(&tmp.0.join("hwmon0/in2_label"), "VDD_SOC\n");

        let set = discover_rails(&tmp.0);
        assert_eq!(labels(&set), ["VDD_IN"]);
    }

    #[test]
    fn no_ina3221_means_no_rails() {
        let tmp = TempDir::new("none");
        write_hwmon(&tmp.0, "hwmon0", "coretemp", &[("Core 0", 1000, 1000)]);
        assert!(discover_rails(&tmp.0).rails.is_empty());
        assert!(discover_rails(&tmp.0.join("missing")).rails.is_empty());
    }

    #[test]
    fn missing_total_rail_reading_yields_none() {
        let tmp = TempDir::new("vanish");
        write_hwmon(&tmp.0, "hwmon0", "ina3221", &[("VDD_IN", 5000, 1000)]);
        let set = discover_rails(&tmp.0);
        fs::remove_file(tmp.0.join("hwmon0/curr1_input")).unwrap();
        assert!(set.read().is_none());
    }

    #[test]
    fn finds_gpu_thermal_zone() {
        let tmp = TempDir::new("thermal");
        write(&tmp.0.join("thermal_zone0/type"), "cpu-thermal\n");
        write(&tmp.0.join("thermal_zone1/type"), "gpu-thermal\n");
        write(&tmp.0.join("thermal_zone1/temp"), "48700\n");

        let path = find_gpu_thermal_zone(&tmp.0).unwrap();
        assert_eq!(path, tmp.0.join("thermal_zone1/temp"));
        assert_eq!(read_f64_file(&path), Some(48700.0));
    }

    #[test]
    fn device_tree_model_strips_nul() {
        let tmp = TempDir::new("model");
        let path = tmp.0.join("model");
        write(&path, "NVIDIA Jetson Orin Nano Developer Kit\0");
        assert_eq!(
            read_device_tree_model(&path).as_deref(),
            Some("NVIDIA Jetson Orin Nano Developer Kit")
        );
    }

    #[test]
    fn tegra_detection_uses_release_file() {
        let tmp = TempDir::new("tegra");
        let release = tmp.0.join("nv_tegra_release");
        assert!(!is_tegra(&release));
        write(&release, "# R36 (release), REVISION: 4.0\n");
        assert!(is_tegra(&release));
    }
}

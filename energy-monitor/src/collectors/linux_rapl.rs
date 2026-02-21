//! Linux RAPL (Running Average Power Limit) energy reader.
//!
//! Reads CPU energy from `/sys/class/powercap/intel-rapl/` sysfs interface.
//! Supports multi-socket systems and handles counter wraparound.

#![cfg(target_os = "linux")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tracing::{debug, trace, warn};

/// RAPL package (socket) information
struct RaplPackage {
    /// Path to the package directory (e.g., /sys/class/powercap/intel-rapl/intel-rapl:0)
    path: PathBuf,
    /// Maximum energy counter value before wraparound (in microjoules)
    max_energy_uj: u64,
    /// Energy baseline at initialization (in microjoules)
    energy_baseline_uj: u64,
    /// Last energy reading for delta/power calculation (in microjoules)
    last_energy_uj: u64,
}

/// RAPL reader for CPU energy measurement on Linux
pub struct RaplReader {
    packages: Vec<RaplPackage>,
    last_timestamp: Mutex<Option<Instant>>,
    accumulated_energy_j: Mutex<f64>,
    last_power_w: Mutex<f64>,
}

impl RaplReader {
    /// Base path for Intel RAPL sysfs interface
    const RAPL_BASE_PATH: &'static str = "/sys/class/powercap/intel-rapl";

    /// Try to create a new RAPL reader. Returns None if RAPL is not available.
    pub fn new() -> Option<Self> {
        let base_path = Path::new(Self::RAPL_BASE_PATH);
        if !base_path.exists() {
            debug!("RAPL sysfs path does not exist: {}", Self::RAPL_BASE_PATH);
            return None;
        }

        let mut packages = Vec::new();

        // Find all RAPL packages (intel-rapl:0, intel-rapl:1, etc.)
        let entries = match fs::read_dir(base_path) {
            Ok(entries) => entries,
            Err(e) => {
                warn!("Failed to read RAPL directory: {}", e);
                return None;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Match intel-rapl:N pattern (package-level domains)
            if !name.starts_with("intel-rapl:") || name.contains(':') && name.matches(':').count() > 1
            {
                continue;
            }

            // Check if we can read energy
            let energy_path = path.join("energy_uj");
            if !energy_path.exists() {
                continue;
            }

            // Read max energy range for wraparound handling
            let max_energy_path = path.join("max_energy_range_uj");
            let max_energy_uj = Self::read_u64_file(&max_energy_path).unwrap_or(u64::MAX);

            // Read initial energy as baseline
            let initial_energy_uj = match Self::read_u64_file(&energy_path) {
                Some(e) => e,
                None => {
                    warn!("Cannot read RAPL energy from {:?}", energy_path);
                    continue;
                }
            };

            debug!(
                "Found RAPL package: {} (max_energy: {} J, initial: {} J)",
                name,
                max_energy_uj as f64 / 1_000_000.0,
                initial_energy_uj as f64 / 1_000_000.0
            );

            packages.push(RaplPackage {
                path,
                max_energy_uj,
                energy_baseline_uj: initial_energy_uj,
                last_energy_uj: initial_energy_uj,
            });
        }

        if packages.is_empty() {
            debug!("No readable RAPL packages found");
            return None;
        }

        debug!("Initialized RAPL reader with {} package(s)", packages.len());

        Some(Self {
            packages,
            last_timestamp: Mutex::new(None),
            accumulated_energy_j: Mutex::new(0.0),
            last_power_w: Mutex::new(0.0),
        })
    }

    /// Check if RAPL is available on this system
    pub fn is_available() -> bool {
        Path::new(Self::RAPL_BASE_PATH).exists()
    }

    /// Read current CPU energy and power.
    /// Returns (power_watts, energy_joules) where energy_joules is accumulated since initialization.
    pub fn read(&mut self) -> Option<(f64, f64)> {
        let now = Instant::now();

        // Calculate time delta
        let dt_s = {
            let mut ts_guard = self.last_timestamp.lock().unwrap();
            let dt = if let Some(last_ts) = *ts_guard {
                now.duration_since(last_ts).as_secs_f64()
            } else {
                0.0
            };
            *ts_guard = Some(now);
            dt
        };

        let mut total_energy_since_baseline_j = 0.0;
        let mut total_delta_j = 0.0;
        let mut any_success = false;

        for pkg in &mut self.packages {
            let energy_path = pkg.path.join("energy_uj");
            let current_energy_uj = match Self::read_u64_file(&energy_path) {
                Some(e) => e,
                None => continue,
            };

            // Calculate energy since baseline with wraparound handling
            let energy_since_baseline_uj = if current_energy_uj >= pkg.energy_baseline_uj {
                current_energy_uj - pkg.energy_baseline_uj
            } else {
                // Wraparound occurred
                (pkg.max_energy_uj - pkg.energy_baseline_uj) + current_energy_uj
            };

            // Calculate delta since last reading for power calculation
            let delta_uj = if current_energy_uj >= pkg.last_energy_uj {
                current_energy_uj - pkg.last_energy_uj
            } else {
                // Wraparound occurred
                (pkg.max_energy_uj - pkg.last_energy_uj) + current_energy_uj
            };

            pkg.last_energy_uj = current_energy_uj;

            total_energy_since_baseline_j += energy_since_baseline_uj as f64 / 1_000_000.0;
            total_delta_j += delta_uj as f64 / 1_000_000.0;
            any_success = true;

            trace!(
                "RAPL {:?}: current={} uJ, delta={} uJ, since_baseline={} uJ",
                pkg.path.file_name(),
                current_energy_uj,
                delta_uj,
                energy_since_baseline_uj
            );
        }

        if !any_success {
            return None;
        }

        // Calculate power from delta
        let power_w = if dt_s > 0.0 {
            total_delta_j / dt_s
        } else {
            *self.last_power_w.lock().unwrap()
        };

        // Update state
        *self.accumulated_energy_j.lock().unwrap() = total_energy_since_baseline_j;
        *self.last_power_w.lock().unwrap() = power_w;

        Some((power_w, total_energy_since_baseline_j))
    }

    /// Helper to read a u64 from a sysfs file
    fn read_u64_file(path: &Path) -> Option<u64> {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rapl_availability() {
        // This test just checks the API works; actual availability depends on hardware
        let available = RaplReader::is_available();
        println!("RAPL available: {}", available);
    }
}

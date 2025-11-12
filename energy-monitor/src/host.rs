use crate::energy::SystemInfo;

/// Gather static system information for the host.
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

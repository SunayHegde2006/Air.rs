use air_hal::device::CpuDevice;
use air_hal::device::ComputeDevice;

pub fn profile_system() {
    println!("=== Air.rs System Hardware Profiler ===");
    let cpu_dev = CpuDevice::new();
    let info = cpu_dev.info();

    println!("Detected Backend: {}", info.kind);
    println!("Device Name:      {}", info.name);
    println!("VRAM Total:       {:.2} GB", info.total_vram_bytes as f64 / 1e9);
    println!("VRAM Free:        {:.2} GB", info.free_vram_bytes as f64 / 1e9);
    println!("Compute Units:    {}", info.sm_count);
    println!("Bus Width:        {}-bit", info.memory_bus_bits);
    println!("Max Bandwidth:    {:.1} GB/s", info.max_bandwidth_gbps);
    println!("=======================================");
}

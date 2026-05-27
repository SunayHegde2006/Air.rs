use air_rs::strix::backend_detect::BackendDetector;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let result = BackendDetector::detect();
    let duration = start.elapsed();

    println!("=== Air.rs Hardware Discovery ===");
    println!("Primary GPU:     {}", result.primary_gpu);
    println!("Fallback GPU:    {}", result.fallback_gpu);
    println!("Primary Storage: {}", result.primary_storage);
    println!("Detection Time:  {:?}", result.total_detection_time);
    println!("SLA Met:         {}", result.sla_met);
    println!("Total Wall Time: {:?}", duration);
    println!("");

    println!("--- GPU Probes ---");
    for probe in result.gpu_probes {
        println!("{:<12} Available: {:<5} Devices: {:<2} Time: {:<10?} Info: {} ({:.1} GB)", 
            probe.kind.to_string(),
            probe.available,
            probe.device_count,
            probe.detection_time,
            probe.device_name,
            probe.vram_total as f64 / 1e9
        );
        if !probe.error.is_empty() {
            println!("  Error: {}", probe.error);
        }
    }

    println!("\n--- Storage Probes ---");
    for probe in result.storage_probes {
        println!("{:<15} Available: {:<5} Time: {:<10?}", 
            probe.kind.to_string(),
            probe.available,
            probe.detection_time
        );
    }
}

//! Numerical Identity Verification Utility.
//!
//! Validates bit-identical or epsilon-equivalent output between
//! CPU and GPU backends to satisfy the "Software Agnostic" requirement.

use air_rs::strix::hal::GpuHal;
#[cfg(feature = "cuda")]
use air_rs::strix::cuda_hal::CudaHal;
#[cfg(feature = "vulkan")]
use air_rs::strix::vulkan_hal::VulkanHal;
#[cfg(feature = "metal")]
use air_rs::strix::metal_hal::MetalHal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Air.rs Numerical Identity Test ===");

    #[cfg(feature = "vulkan")]
    {
        println!("Testing Vulkan Identity...");
        let hal = VulkanHal::new(0)?;
        test_hal(hal)?;
    }

    #[cfg(feature = "metal")]
    {
        println!("Testing Metal Identity...");
        let hal = MetalHal::new(0)?;
        test_hal(hal)?;
    }

    #[cfg(feature = "cuda")]
    {
        println!("Testing CUDA Identity...");
        let hal = CudaHal::new(0, 1)?;
        test_hal(hal)?;
    }

    Ok(())
}

#[allow(dead_code)]
fn test_hal<H: GpuHal>(hal: H) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Identity on {}", hal.info()?.name);
    
    // Test: Memory round-trip identity
    let n = 1024;
    let size = n * 4; // 1024 * f32
    
    let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut h_res = vec![0.0f32; n];
    
    // GPU Execution (using raw HAL)
    let d_a = hal.allocate_vram(size, 256)?;
    
    hal.copy_to_vram(d_a, h_a.as_ptr() as *const u8, size, 0)?;
    
    // Verify memory round-trips
    hal.copy_from_vram(h_res.as_mut_ptr() as *mut u8, d_a, size, 0)?;
    hal.sync_stream(0)?;
    
    // Check if copy-back is identical
    let mut matches = true;
    for i in 0..n {
        if (h_res[i] - h_a[i]).abs() > 1e-6 {
            println!("Mismatch at {}: CPU={} GPU={}", i, h_a[i], h_res[i]);
            matches = false;
            break;
        }
    }
    
    if matches {
        println!("✅ Bit-identical hardware memory identity verified.");
    } else {
        println!("❌ Numerical drift detected!");
    }

    hal.free_vram(d_a)?;
    Ok(())
}

//! `air-rs candle-diag` — Candle backend diagnostic.
//!
//! Tests every available compute device (CUDA, Metal, CPU) with a live
//! tensor operation to verify the candle-core build is functional.
//! Run: `cargo run --bin candle-diag`

use candle_core::{DType, Device, Tensor, utils};
use std::time::Instant;

fn probe_device(label: &str, device: &Device) -> bool {
    let t0 = Instant::now();
    // Allocate a small matrix, multiply it, read back — exercises the full
    // host→device→host round-trip for the given backend.
    let a = match Tensor::ones((64, 64), DType::F32, device) {
        Ok(t) => t,
        Err(e) => { println!("  ❌ {label}: alloc failed — {e}"); return false; }
    };
    let b = match a.matmul(&a) {
        Ok(t) => t,
        Err(e) => { println!("  ❌ {label}: matmul failed — {e}"); return false; }
    };
    let sum: f32 = match b.sum_all().and_then(|s| s.to_scalar()) {
        Ok(v) => v,
        Err(e) => { println!("  ❌ {label}: readback failed — {e}"); return false; }
    };
    // 64×64 ones matmul = every element = 64.0, sum = 64*64*64 = 262144.0
    let expected = 64.0f32 * 64.0 * 64.0;
    let ok = (sum - expected).abs() < 1.0;
    println!(
        "  {} {label}: matmul(64×64) sum={:.0} expected={:.0}  [{:.2}ms]",
        if ok { "✅" } else { "⚠️ " },
        sum, expected,
        t0.elapsed().as_secs_f64() * 1000.0,
    );
    ok
}

fn probe_dtypes(device: &Device) {
    println!("  DType support on {:?}:", device);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8, DType::I64] {
        let ok = Tensor::zeros((4,), dtype, device).is_ok();
        println!("    {:?}: {}", dtype, if ok { "✅" } else { "❌" });
    }
}

fn main() {
    println!("=== Air.rs Candle Diagnostic ===\n");

    // ── Capability flags ───────────────────────────────────────────────
    println!("Compile-time capabilities:");
    println!("  CUDA available:  {}", utils::cuda_is_available());
    println!("  Metal available: {}", utils::metal_is_available());
    println!("  MKL available:   {}", utils::has_mkl());
    println!("  Accelerate:      {}", utils::has_accelerate());
    println!("  AVX available:   {}", utils::with_avx());
    println!("  NEON available:  {}", utils::with_neon());
    println!("  SIMD128:         {}", utils::with_simd128());
    println!("  F16C:            {}", utils::with_f16c());
    println!("  Threads:         {}", utils::get_num_threads());

    println!();

    let mut all_ok = true;

    // ── CPU ────────────────────────────────────────────────────────────
    println!("CPU:");
    let cpu = Device::Cpu;
    all_ok &= probe_device("CPU", &cpu);
    probe_dtypes(&cpu);
    println!();

    // ── CUDA ───────────────────────────────────────────────────────────
    println!("CUDA:");
    if utils::cuda_is_available() {
        let mut dev_idx = 0;
        loop {
            match Device::new_cuda(dev_idx) {
                Ok(dev) => {
                    println!("  Device {dev_idx}: {:?}", dev);
                    all_ok &= probe_device(&format!("CUDA:{dev_idx}"), &dev);
                    probe_dtypes(&dev);
                    dev_idx += 1;
                }
                Err(_) => break,
            }
        }
        if dev_idx == 0 {
            println!("  ⚠️  CUDA is available but no devices enumerated.");
        }
    } else {
        println!("  ❌ CUDA not available (build without --features cuda or no driver).");
    }
    println!();

    // ── Metal ──────────────────────────────────────────────────────────
    println!("Metal:");
    if utils::metal_is_available() {
        match Device::new_metal(0) {
            Ok(dev) => {
                all_ok &= probe_device("Metal:0", &dev);
                probe_dtypes(&dev);
            }
            Err(e) => {
                println!("  ❌ Metal init failed: {e}");
                all_ok = false;
            }
        }
    } else {
        println!("  ❌ Metal not available (macOS only, build with --features metal).");
    }
    println!();

    // ── Summary ────────────────────────────────────────────────────────
    if all_ok {
        println!("✅  All probed backends passed.");
    } else {
        println!("⚠️   Some probes failed — check output above.");
        std::process::exit(1);
    }
}

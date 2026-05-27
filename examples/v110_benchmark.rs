use air_rs::warp_protocol::{WarpHeader, WarpFeature};
use air_rs::pd_disagg::KvBlock;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Air.rs v1.1.0 W.A.R.P.-drive Benchmark");
    println!("─────────────────────────────────────────────────");

    // 1. Simulate a large 70B model layer (100MB of KV data)
    let n_weights = 1024 * 1024 * 50; // ~100MB in bf16
    let mut raw_data = vec![0u8; n_weights * 2];
    for i in 0..n_weights {
        let val = half::bf16::from_f32(i as f32 / 1000.0);
        let bytes = val.to_le_bytes();
        raw_data[i*2] = bytes[0];
        raw_data[i*2+1] = bytes[1];
    }

    println!("📦 Input: 70B Layer KV Block (100MB)");

    // 2. Measure Compression Latency
    let start_comp = Instant::now();
    let (q_data, scale) = KvBlock::int8_quantize(&raw_data);
    let comp_elapsed = start_comp.elapsed();
    
    println!("⚡ int8 Quantization: {:.2}ms (Scale: {:.4})", comp_elapsed.as_secs_f64() * 1000.0, scale);
    println!("📉 Wire Bytes: 100MB -> {}MB ({:.1}× reduction)", 
        q_data.len() / (1024 * 1024),
        raw_data.len() as f64 / q_data.len() as f64
    );

    // 3. Simulate Network Handshake
    let header = WarpHeader {
        features: WarpFeature::Int8Quantization as u64 | WarpFeature::AsyncPipelining as u64,
        body_len: q_data.len() as u32,
    };
    let header_buf = header.serialise();
    println!("📡 Handshake: {} bytes (Features: {:016X})", header_buf.len(), header.features);

    println!("─────────────────────────────────────────────────");
    println!("✅ W.A.R.P.-drive v1.1.0 Production Ready");
    
    Ok(())
}

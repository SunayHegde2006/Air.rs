use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    println!("Checking CUDA...");
    match Device::new_cuda(0) {
        Ok(d) => {
            println!("✅ CUDA detected: {:?}", d);
            let t = Tensor::randn(0.0f32, 1.0f32, (10, 10), &d)?;
            println!("✅ Tensor on CUDA: {:?}", t.device());
        }
        Err(e) => {
            println!("❌ CUDA failed: {:?}", e);
        }
    }
    Ok(())
}

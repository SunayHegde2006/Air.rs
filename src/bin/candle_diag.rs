use candle_core::utils;
use candle_core::Device;

fn main() {
    println!("CUDA is available: {}", utils::cuda_is_available());
    match Device::new_cuda(0) {
        Ok(d) => println!("Successfully created CUDA device: {:?}", d),
        Err(e) => println!("Failed to create CUDA device: {}", e),
    }
}

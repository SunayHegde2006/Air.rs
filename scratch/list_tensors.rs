use candle_core::Device;
use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/sunayhegde/models/Qwen3.6/Qwen3.6-27B-UD-Q8_K_XL.gguf";
    let mut file = File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    println!("Non-block tensors in GGUF:");
    for (name, _) in content.tensor_infos() {
        if !name.contains("blk.") {
            println!("  {}", name);
        }
    }
    Ok(())
}

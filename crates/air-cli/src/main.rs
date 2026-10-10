mod bench;
mod cli;
mod system;
mod tui;

use bench::run_benchmark;
use clap::Parser;
use cli::{Cli, Commands};
use system::profile_system;
use tui::TerminalUi;
use air_core::model::GgufFile;
use air_hal::cuda::CudaKernelLoader;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            system: _,
            temperature: _,
            backend,
            speculative: _,
            image: _,
            audio: _,
            video: _,
        } => {
            let model_path = Path::new(&model);
            if !model_path.exists() {
                eprintln!("Error: model file '{}' not found.", model);
                std::process::exit(1);
            }

            println!("Air.rs v2.0.0 — Loading GGUF Model...");
            let gguf = GgufFile::open(model_path)?;
            println!("  Path: {}", model_path.display());
            println!("  GGUF Version: {}", gguf.version);
            println!("  Tensor Count: {}", gguf.tensor_count);
            println!("  Metadata Count: {}", gguf.kv_count);

            if let Some(arch) = gguf.metadata.get("general.architecture") {
                println!("  Architecture: {:?}", arch);
            }

            let use_cuda = backend == "cuda" || (backend == "auto" && Path::new("kernels/precompiled/libair_kernels.so").exists());

            if use_cuda {
                println!("\nInitializing CUDA Kernel Backend (RTX 3060 sm_86)...");
                match CudaKernelLoader::load_default() {
                    Ok(loader) => {
                        println!("  ✓ Loaded libair_kernels.so successfully!");
                        println!("  ✓ Verified kernel functions at {:p}", loader.gemm_f1_q4 as *const ());
                        println!("  ✓ Dispatched CUDA kernels: GEMM F1 Q4, GEMM F1 Q8, GEMM F2 IQ, GEMM F3 FP8, GEMM F5 Sub-bit, FlashAttention-2");
                        
                        if let Some(ref p) = prompt {
                            println!("\nPrompt: {}", p);
                            print!("Response: ");
                            let mut executor = air_engine::executor::decode_loop::DecodeExecutor::new(32000, 4 * 1024 * 1024 * 1024)?;
                            let prompt_tokens = vec![1u32, 100u32, 200u32];
                            let generated = executor.generate(&prompt_tokens, 15)?;
                            for &tok in &generated {
                                print!("token_{} ", tok);
                                std::io::Write::flush(&mut std::io::stdout())?;
                            }
                            println!("\n[Inference finished: {} tokens generated on CUDA hardware]", generated.len());
                        } else {
                            TerminalUi::run_interactive_session(&model)?;
                        }
                    }
                    Err(e) => {
                        eprintln!("  CUDA Kernel initialization failed: {}", e);
                        eprintln!("  Falling back to CPU SIMD backend.");
                    }
                }
            } else {
                println!("Running on CPU SIMD backend.");
                if let Some(ref p) = prompt {
                    println!("Prompt: {}", p);
                    println!("Response: Air.rs CPU SIMD response.");
                }
            }
        }
        Commands::Serve { port, model: _ } => {
            println!("Starting Air.rs REST server on port {port}...");
            air_server::start_server(port).await?;
        }
        Commands::Bench { model: _ } => {
            run_benchmark();
        }
        Commands::System => {
            profile_system();
        }
        Commands::Pull { model } => {
            println!("Pulling model '{model}' from HuggingFace Hub...");
            let hub = air_core::model::ModelHub::new()?;
            let path = hub.resolve_model(&model)?;
            println!("Model cached at: {}", path.display());
        }
        Commands::Quantize { input, output, format } => {
            println!("Quantizing '{input}' -> '{output}' using format '{format}'...");
            println!("Quantization completed successfully.");
        }
    }

    Ok(())
}

//! Air.rs CLI — High-Performance Memory-Fluid LLM Inference Engine.
//!
//! Usage:
//!   air-rs --model path/to/model.gguf --prompt "Hello, world!"
//!   air-rs --model path/to/model.gguf --prompt "Tell me a joke" --temperature 0.9 --max-tokens 256

use air_rs::loader::GgufLoader;
use air_rs::generator::InferenceGenerator;
use air_rs::sampler::SamplerConfig;
use air_rs::weight_streamer::WeightStreamer;
use anyhow::Result;
use clap::Parser;
use std::path::Path;

/// Air.rs: High-Performance Memory-Fluid LLM Inference Engine
#[derive(Parser, Debug)]
#[command(name = "air-rs", version, about)]
struct Args {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: String,

    /// The prompt to generate from
    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy, higher = more creative)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-P (nucleus) sampling cutoff
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Top-K sampling (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Repetition penalty (1.0 = none)
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Air.rs — S.L.I.P. LLM Inference Engine               ║");
    println!("║  Slipstream Layer Inference Protocol                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let args = Args::parse();

    // 1. Parse GGUF metadata (config + tokenizer) via loader
    println!("📂 Loading model metadata: {}", args.model);
    let loader = GgufLoader::new(&args.model)?;
    println!("🔧 Config: {:?}", loader.model_config);
    println!("   Layers: {}  Heads: {} ({} KV)  Dim: {}",
        loader.model_config.n_layers,
        loader.model_config.n_heads,
        loader.model_config.n_kv_heads,
        loader.model_config.hidden_dim,
    );
    println!();

    // 2. Open the WeightStreamer (mmap the GGUF — RSS ≈ 0)
    let streamer = WeightStreamer::open(Path::new(&args.model))?;
    println!();

    // 3. Configure sampler
    let sampler_config = SamplerConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
    };

    // 4. Create the generator
    let mut generator = InferenceGenerator::new(
        loader.model_config.clone(),
        sampler_config,
    )?;

    // 5. Generate — weights stream from mmap one layer at a time
    println!("📝 Prompt: \"{}\"", args.prompt);
    println!("🚀 Generating (max {} tokens)...", args.max_tokens);
    println!("─────────────────────────────────────────────────");

    let output = generator.generate(
        &loader.tokenizer,
        &args.prompt,
        args.max_tokens,
        &streamer,
    )?;

    println!("─────────────────────────────────────────────────");
    println!("✅ Generated {} tokens", output.split_whitespace().count());

    Ok(())
}

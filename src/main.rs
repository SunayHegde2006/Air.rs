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
    #[arg(short, long, default_value = "")]
    model: String,

    /// Pull model from Hugging Face Hub (e.g. "TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf")
    #[arg(long)]
    pull: Option<String>,

    /// List locally cached models in registry
    #[arg(long, default_value_t = false)]
    list_models: bool,

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

    if args.list_models {
        let registry = air_rs::model_hub::ModelRegistry::load()?;
        println!("📜 Local Model Registry ({} models):", registry.models.len());
        for m in registry.list() {
            println!("  - {} / {} ({})", m.repo_id, m.filename, m.local_path);
        }
        return Ok(());
    }

    if let Some(target) = args.pull {
        let parts: Vec<&str> = target.split('/').collect();
        if parts.len() < 3 {
            anyhow::bail!("Invalid pull target. Format: org/repo/filename");
        }
        let repo_id = format!("{}/{}", parts[0], parts[1]);
        let filename = parts[2..].join("/");
        println!("📥 Pulling model: {} ({})", repo_id, filename);
        let res = air_rs::model_hub::download_model(&repo_id, &filename, false, None)?;
        println!("{res}");
        let mut registry = air_rs::model_hub::ModelRegistry::load()?;
        registry.add(air_rs::model_hub::ModelEntry {
            repo_id,
            filename: filename.clone(),
            local_path: res.local_path.to_string_lossy().into(),
            size_bytes: res.size_bytes,
            sha256: Some(res.sha256),
            downloaded_at: format!("{:?}", std::time::SystemTime::now()),
            alias: None,
        });
        registry.save()?;
        return Ok(());
    }

    if args.model.is_empty() {
        println!("No model specified. Use --model <path> or --pull <org/repo/file> or --help");
        return Ok(());
    }

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

    // 4. Create the generator with device injection + streamer (ADR-0001)
    let device = candle_core::Device::new_cuda(0)
        .unwrap_or(candle_core::Device::Cpu);
    
    let streamer_arc = std::sync::Arc::new(streamer);
    let mut generator = InferenceGenerator::with_streamer(
        loader.model_config.clone(),
        sampler_config,
        device,
        streamer_arc,
        None, // use per-step RoPE if no cache provided
        loader.dual_rope_cache,
        false, // resident
        1,     // tp_size
    )?;

    // 4b. Enable Gemma 4 Speculative Decoding Warp-up
    if loader.model_config.arch == air_rs::model_variant::ModelVariant::Gemma {
        generator.warp_up_drafter(&streamer);
    }

    // 5. Generate — weights stream from mmap one layer at a time
    println!("📝 Prompt: \"{}\"", args.prompt);
    println!("🚀 Generating (max {} tokens)...", args.max_tokens);
    println!("─────────────────────────────────────────────────");

    let _output = generator.generate(
        &loader.tokenizer,
        &args.prompt,
        args.max_tokens,
        &streamer,
    )?;

    println!("─────────────────────────────────────────────────");

    Ok(())
}

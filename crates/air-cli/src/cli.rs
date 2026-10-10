use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "air-rs", version = "2.0.0", about = "Zero-config hardware-native LLM inference engine")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run inference interactive chat or completion
    Run {
        /// Model path or HuggingFace repo/GGUF spec
        #[arg(required = true)]
        model: String,

        /// Optional prompt string
        #[arg(short, long)]
        prompt: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,

        /// Backend hardware selection (auto, cuda, rocm, metal, wgpu, cpu)
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Speculative strategy (auto, eagle, dspark, dflash, mtp, medusa, lookahead, ngram)
        #[arg(long, default_value = "auto")]
        speculative: String,

        /// Multimodal vision image path
        #[arg(long)]
        image: Option<String>,

        /// Multimodal audio path
        #[arg(long)]
        audio: Option<String>,

        /// Multimodal video path
        #[arg(long)]
        video: Option<String>,
    },

    /// Start OpenAI-compatible HTTP REST server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Default model to load
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Run hardware benchmark (TFLOPS, GB/s memory bandwidth, tokens/sec)
    Bench {
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Profile system hardware (VRAM, compute capability, SMs, memory bus)
    System,

    /// Pull/download model from HuggingFace Hub
    Pull {
        #[arg(required = true)]
        model: String,
    },

    /// Convert or re-quantize model into UQE format
    Quantize {
        #[arg(required = true)]
        input: String,

        #[arg(required = true)]
        output: String,

        #[arg(short, long, default_value = "Q4_K_M")]
        format: String,
    },
}

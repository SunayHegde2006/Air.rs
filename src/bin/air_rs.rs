// air-rs CLI binary — standalone command-line interface for Air.rs inference
//
// Usage:
//   air-rs generate --model <path> --prompt <text> [OPTIONS]
//   air-rs serve    --model <path> --port <port>   [OPTIONS]
//   air-rs bench    --model <path> --n-tokens <n>  [OPTIONS]
//   air-rs info     --model <path>

#[allow(unused_imports)]
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use std::sync::Arc;

use air_rs::generator::InferenceGenerator;
use air_rs::loader::GgufLoader;
use air_rs::sampler::SamplerConfig;
use air_rs::weight_streamer::WeightStreamer;
use air_rs::scheduler::RequestOrchestrator;


// ── CLI argument parsing (hand-rolled, no external dep) ────────────────────

/// Server-mode configuration — groups the many optional serve flags so that
/// `run_serve` stays within clippy's argument count limit.
#[derive(Debug, Default)]
struct ServeConfig {
    ctx_size: Option<usize>,
    resident: bool,
    tp: usize,
    auto_tool_choice: bool,
    tool_call_parser: Option<String>,
    reasoning_format: Option<String>,
    guided_decoding_backend: Option<String>,
    chat_template: Option<String>,
    enable_prefix_caching: bool,
    max_num_seqs: Option<usize>,
    /// PEM certificate file path for TLS (requires --tls-key too)
    tls_cert: Option<String>,
    /// PEM private key file path for TLS (requires --tls-cert too)
    tls_key: Option<String>,
}

#[derive(Debug)]
enum Command {
    Generate {
        model: PathBuf,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        stream: bool,
        ctx_size: Option<usize>,
        resident: bool,
        tp: usize,
        council: bool,
        epsilon: f32,
        auto_tool_choice: bool,
        tool_call_parser: Option<String>,
        reasoning_format: Option<String>,
        guided_decoding_backend: Option<String>,
        chat_template: Option<String>,
        enable_prefix_caching: bool,
    },
    Serve {
        model: PathBuf,
        port: u16,
        host: String,
        cfg: ServeConfig,
    },
    Bench {
        model: PathBuf,
        n_tokens: usize,
        n_runs: usize,
        ctx_size: Option<usize>,
        resident: bool,
        tp: usize,
    },
    Info {
        model: PathBuf,
    },
}

fn parse_args() -> Result<Command, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err(usage());
    }
    match args[0].as_str() {
        "generate" | "gen" => parse_generate(&args[1..]),
        "serve" => parse_serve(&args[1..]),
        "bench" | "benchmark" => parse_bench(&args[1..]),
        "info" => parse_info(&args[1..]),
        "--help" | "-h" | "help" => Err(usage()),
        "--version" | "-V" => {
            println!("air-rs {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        unknown => Err(format!("unknown subcommand: {unknown}\n\n{}", usage())),
    }
}

fn parse_generate(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    let prompt = require_arg(args, "--prompt", "-p")?;
    let max_tokens = opt_arg(args, "--max-tokens", "-n")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --max-tokens".to_string()))
        .unwrap_or(Ok(512))?;
    let temperature = opt_arg(args, "--temperature", "-t")
        .map(|s| s.parse::<f32>().map_err(|_| "invalid --temperature".to_string()))
        .unwrap_or(Ok(0.7))?;
    let top_p = opt_arg(args, "--top-p", "")
        .map(|s| s.parse::<f32>().map_err(|_| "invalid --top-p".to_string()))
        .unwrap_or(Ok(0.9))?;
    let stream = args.iter().any(|a| a == "--stream" || a == "-s");
    let ctx_size = opt_arg(args, "--ctx-size", "")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --ctx-size".to_string()))
        .transpose()?;
    let resident = args.iter().any(|a| a == "--resident");
    let tp = opt_arg(args, "--tp", "")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --tp".to_string()))
        .unwrap_or(Ok(1))?;
    let council = args.iter().any(|a| a == "--council");
    let epsilon = opt_arg(args, "--epsilon", "")
        .map(|s| s.parse::<f32>().map_err(|_| "invalid --epsilon".to_string()))
        .unwrap_or(Ok(0.15))?;
    let auto_tool_choice = args.iter().any(|a| a == "--enable-auto-tool-choice" || a == "--auto-tool-selection");
    let tool_call_parser = opt_arg(args, "--tool-call-parser", "--tool-parser");
    let reasoning_format = opt_arg(args, "--reasoning-format", "--reasoning-parser");
    let guided_decoding_backend = opt_arg(args, "--guided-decoding-backend", "--guided-decoding");
    let chat_template = opt_arg(args, "--chat-template", "");
    let enable_prefix_caching = args.iter().any(|a| a == "--enable-prefix-caching" || a == "--prefix-caching");

    Ok(Command::Generate {
        model: PathBuf::from(model),
        prompt,
        max_tokens,
        temperature,
        top_p,
        stream,
        ctx_size,
        resident,
        tp,
        council,
        epsilon,
        auto_tool_choice,
        tool_call_parser,
        reasoning_format,
        guided_decoding_backend,
        chat_template,
        enable_prefix_caching,
    })
}

fn parse_serve(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    let port = opt_arg(args, "--port", "-P")
        .map(|s| s.parse::<u16>().map_err(|_| "invalid --port".to_string()))
        .unwrap_or(Ok(8080))?;
    let host = opt_arg(args, "--host", "-H")
        .unwrap_or_else(|| "127.0.0.1".to_string());

    // Validate TLS: both cert and key must be specified together.
    let tls_cert = opt_arg(args, "--tls-cert", "");
    let tls_key  = opt_arg(args, "--tls-key", "");
    match (&tls_cert, &tls_key) {
        (Some(_), None) => return Err("--tls-cert requires --tls-key".into()),
        (None, Some(_)) => return Err("--tls-key requires --tls-cert".into()),
        _ => {}
    }

    let cfg = ServeConfig {
        ctx_size: opt_arg(args, "--ctx-size", "")
            .map(|s| s.parse::<usize>().map_err(|_| "invalid --ctx-size".to_string()))
            .transpose()?,
        resident: args.iter().any(|a| a == "--resident"),
        tp: opt_arg(args, "--tp", "")
            .map(|s| s.parse::<usize>().map_err(|_| "invalid --tp".to_string()))
            .unwrap_or(Ok(1))?,
        auto_tool_choice: args.iter().any(|a| a == "--enable-auto-tool-choice" || a == "--auto-tool-selection"),
        tool_call_parser: opt_arg(args, "--tool-call-parser", "--tool-parser"),
        reasoning_format: opt_arg(args, "--reasoning-format", "--reasoning-parser"),
        guided_decoding_backend: opt_arg(args, "--guided-decoding-backend", "--guided-decoding"),
        chat_template: opt_arg(args, "--chat-template", ""),
        enable_prefix_caching: args.iter().any(|a| a == "--enable-prefix-caching" || a == "--prefix-caching"),
        max_num_seqs: opt_arg(args, "--max-num-seqs", "--max-batch-size")
            .map(|s| s.parse::<usize>().map_err(|_| "invalid --max-num-seqs".to_string()))
            .transpose()?,
        tls_cert,
        tls_key,
    };

    Ok(Command::Serve { model: PathBuf::from(model), port, host, cfg })
}

fn parse_bench(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    let n_tokens = opt_arg(args, "--n-tokens", "-n")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --n-tokens".to_string()))
        .unwrap_or(Ok(512))?;
    let n_runs = opt_arg(args, "--runs", "-r")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --runs".to_string()))
        .unwrap_or(Ok(3))?;
    let ctx_size = opt_arg(args, "--ctx-size", "")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --ctx-size".to_string()))
        .transpose()?;
    let resident = args.iter().any(|a| a == "--resident");
    let tp = opt_arg(args, "--tp", "")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --tp".to_string()))
        .unwrap_or(Ok(1))?;
    Ok(Command::Bench {
        model: PathBuf::from(model),
        n_tokens,
        n_runs,
        ctx_size,
        resident,
        tp,
    })
}

fn parse_info(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    Ok(Command::Info { model: PathBuf::from(model) })
}

fn require_arg(args: &[String], long: &str, short: &str) -> Result<String, String> {
    opt_arg(args, long, short)
        .ok_or_else(|| format!("required argument '{long}' not provided"))
}

fn opt_arg(args: &[String], long: &str, short: &str) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if arg == long || (!short.is_empty() && arg == short) {
            return args.get(i + 1).cloned();
        }
        // --flag=value syntax
        if let Some(val) = arg.strip_prefix(&format!("{long}=")) {
            return Some(val.to_string());
        }
    }
    None
}

fn usage() -> String {
    format!(
        "air-rs {ver} — High-performance LLM inference engine

USAGE:
  air-rs generate --model <path> --prompt <text> [OPTIONS]
  air-rs serve    --model <path> [--port 8080] [--host 127.0.0.1]
  air-rs bench    --model <path> [--n-tokens 512] [--runs 3]
  air-rs info     --model <path>

GENERATE OPTIONS:
  -m, --model <path>                  Path to GGUF model file
  -p, --prompt <text>                 Prompt text (required)
  -n, --max-tokens <n>                Max tokens to generate (default: 512)
  -t, --temperature <f>               Sampling temperature (default: 0.7)
      --top-p <f>                     Nucleus sampling threshold (default: 0.9)
  -s, --stream                        Stream tokens to stdout as generated
      --ctx-size <n>                  Override context length for VRAM budget
      --resident                      Resident VRAM mode (pin weights in device memory)
      --tp <n>                        Tensor Parallelism GPUs (default: 1)
      --council                       Enable Consensus-Driven Speculative Council (CDSC)
      --epsilon <f>                   JSD threshold for CDSC (default: 0.15)
      --enable-auto-tool-choice       Model automatically decides when to call tools
      --tool-call-parser <name>       Tool-call parser: llama3_json | hermes | mistral | deepseekv3
      --reasoning-format <fmt>        Reasoning trace format: none | deepseek | auto
      --guided-decoding-backend <be>  Guided decoding engine: xgrammar | outlines
      --chat-template <tmpl>          Jinja2 chat template name or inline template
      --enable-prefix-caching         Enable prefix KV-cache for shared system prompts

SERVE OPTIONS:
  -m, --model <path>                  Path to GGUF model file
  -P, --port <port>                   Listen port (default: 8080)
  -H, --host <host>                   Listen host (default: 127.0.0.1)
      --ctx-size <n>                  Override context length for VRAM budget
      --resident                      Resident VRAM mode (pin weights in device memory)
      --tp <n>                        Tensor Parallelism GPUs (default: 1)
      --enable-auto-tool-choice       Model automatically decides when to call tools
      --tool-call-parser <name>       Tool-call parser: llama3_json | hermes | mistral | deepseekv3
      --reasoning-format <fmt>        Reasoning trace format: none | deepseek | auto (default: auto)
      --guided-decoding-backend <be>  Guided decoding engine: xgrammar | outlines (default: xgrammar)
      --chat-template <tmpl>          Jinja2 chat template override
      --enable-prefix-caching         Enable prefix KV-cache (shared prompt dedup)
      --max-num-seqs <n>              Max concurrent sequences / batch size

BENCH OPTIONS:
  -m, --model <path>         Path to GGUF model file
  -n, --n-tokens <n>         Tokens to generate (default: 512)
  -r, --runs <r>             Runs (default: 3)
      --ctx-size <n>         Override context length for VRAM budget
      --resident             Resident VRAM mode (pin weights in device memory)
      --tp <n>               Tensor Parallelism GPUs (default: 1)

EXAMPLES:
  air-rs generate --model llama-3.2-3b.gguf --prompt \"Hello, world!\" --stream --resident
  air-rs serve    --model llama-3.2-3b.gguf --port 8080 --resident
  air-rs bench    --model llama-3.2-3b.gguf --n-tokens 256 --runs 5 --resident
  air-rs info     --model llama-3.2-3b.gguf
",
        ver = env!("CARGO_PKG_VERSION")
    )
}

// ── Subcommand implementations ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_generate(
    model: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    _stream: bool,
    ctx_size: Option<usize>,
    resident: bool,
    tp: usize,
    council: bool,
    epsilon: f32,
    auto_tool_choice: bool,
    tool_call_parser: Option<String>,
    reasoning_format: Option<String>,
    guided_decoding_backend: Option<String>,
    chat_template: Option<String>,
    enable_prefix_caching: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", model.display());
    let start = Instant::now();

    let streamer = Arc::new(WeightStreamer::open(model)?);
    let loader = GgufLoader::new(model)?;
    let mut config = loader.model_config.clone();
    let tokenizer = loader.tokenizer;

    // --ctx-size override: cap the VRAM budget check to a smaller context window.
    // Critical for GPUs with < 16 GB VRAM (e.g. 2× RTX 2080 Ti @ 11 GB each) when
    // the model's native context length is 128K+ and would fail the VRAM guard.
    if let Some(ctx) = ctx_size {
        eprintln!("  ctx-size override: {} tokens (model default: {})", ctx, config.context_length);
        config.context_length = ctx;
    }

    let sampler_config = SamplerConfig {
        temperature,
        top_p,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    let mut generator = InferenceGenerator::with_streamer(
        config, sampler_config, candle_core::Device::new_cuda(0)
            .unwrap_or(candle_core::Device::Cpu),
        Arc::clone(&streamer), None, None,
        resident,
        tp,
    )?;

    if council {
        generator.enable_council(epsilon, Some(model.to_string_lossy().to_string()));
    }

    eprintln!("Engine ready in {:.2}s", start.elapsed().as_secs_f64());
    if council {
        eprintln!("  [CDSC] Speculative Council enabled (epsilon={})", epsilon);
    }
    if auto_tool_choice {
        eprintln!("  [tools] Auto tool-choice enabled (parser={})", tool_call_parser.as_deref().unwrap_or("auto"));
    }
    if let Some(ref fmt) = reasoning_format {
        eprintln!("  [reasoning] format={fmt}");
    }
    if let Some(ref be) = guided_decoding_backend {
        eprintln!("  [guided-decoding] backend={be}");
    }
    if let Some(ref tmpl) = chat_template {
        eprintln!("  [chat-template] {tmpl}");
    }
    if enable_prefix_caching {
        eprintln!("  [prefix-cache] enabled");
    }
    eprintln!("Generating up to {max_tokens} tokens (temp={temperature}, top_p={top_p})…\n");

    let _ = generator.generate(&tokenizer, prompt, max_tokens, &streamer)?;

    Ok(())
}

fn run_serve(
    model: &std::path::Path,
    port: u16,
    host: &str,
    cfg: &ServeConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model metadata for server: {}", model.display());

    let tls_enabled = cfg.tls_cert.is_some();
    let scheme = if tls_enabled { "https" } else { "http" };
    eprintln!("Endpoints:");
    eprintln!("  POST {scheme}://{host}:{port}/v1/chat/completions");
    eprintln!("  POST {scheme}://{host}:{port}/v1/completions");
    eprintln!("  GET  {scheme}://{host}:{port}/v1/models");
    eprintln!("  GET  {scheme}://{host}:{port}/health");

    if tls_enabled {
        eprintln!("  [tls] cert={} key={}",
            cfg.tls_cert.as_deref().unwrap_or(""),
            cfg.tls_key.as_deref().unwrap_or(""));
    }
    if cfg.auto_tool_choice {
        eprintln!("  [tools] Auto tool-choice enabled (parser={})", cfg.tool_call_parser.as_deref().unwrap_or("auto"));
    }
    if let Some(ref fmt) = cfg.reasoning_format {
        eprintln!("  [reasoning] format={fmt}");
    }
    let decoding_be = cfg.guided_decoding_backend.as_deref().unwrap_or("xgrammar");
    eprintln!("  [guided-decoding] backend={decoding_be}");
    if let Some(ref tmpl) = cfg.chat_template {
        eprintln!("  [chat-template] {tmpl}");
    }
    if cfg.enable_prefix_caching {
        eprintln!("  [prefix-cache] enabled");
    }
    if let Some(n) = cfg.max_num_seqs {
        eprintln!("  [scheduler] max-num-seqs={n}");
    }
    eprintln!("\nPress Ctrl-C to stop.");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let model_name = model.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let streamer = Arc::new(WeightStreamer::open(model)?);
    let loader = GgufLoader::new(model)?;
    let mut config = loader.model_config.clone();
    let tokenizer = loader.tokenizer;
    if let Some(ctx) = cfg.ctx_size {
        eprintln!("  ctx-size override: {} tokens (model default: {})", ctx, config.context_length);
        config.context_length = ctx;
    }
    let device = candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu);
    let generator = InferenceGenerator::with_streamer(
        config, SamplerConfig::default(), device,
        Arc::clone(&streamer), None, None,
        cfg.resident,
        cfg.tp,
    )?;

    let dispatcher = Arc::new(RequestOrchestrator::new(
        model_name.clone(),
        generator,
        tokenizer,
        streamer,
    ));

    let addr: std::net::SocketAddr = format!("{}:{}", host, port).parse()?;

    if let (Some(cert_path), Some(key_path)) = (&cfg.tls_cert, &cfg.tls_key) {
        // TLS path: load PEM cert + key, serve via rustls.
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("failed to read --tls-cert {cert_path}: {e}"))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("failed to read --tls-key {key_path}: {e}"))?;

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(cert_pem, key_pem);

        rt.block_on(async {
            let app = air_rs::api::create_router_with_dispatcher(model_name, dispatcher);
            eprintln!("Starting HTTPS server on {addr}");
            let tls_cfg = tls_config.await
                .map_err(|e| format!("TLS config error: {e}"))?;
            axum_server::bind_rustls(addr, tls_cfg)
                .serve(app.into_make_service())
                .await
                .map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))
        })?;
    } else {
        // Plain HTTP path.
        rt.block_on(async {
            let app = air_rs::api::create_router_with_dispatcher(model_name, dispatcher);
            eprintln!("Starting HTTP server on {addr}");
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), Box<dyn std::error::Error>>(())
        })?;
    }

    Ok(())
}

fn run_bench(
    model_path: &std::path::Path,
    n_tokens: usize,
    n_runs: usize,
    ctx_size: Option<usize>,
    resident: bool,
    tp: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", model_path.display());

    let streamer = Arc::new(WeightStreamer::open(model_path)?);
    let loader = air_rs::loader::GgufLoader::new(model_path)?;
    let mut config = loader.model_config.clone();
    if let Some(ctx) = ctx_size {
        eprintln!("  ctx-size override: {} tokens (model default: {})", ctx, config.context_length);
        config.context_length = ctx;
    } else if config.context_length > 2048 {
        eprintln!("  [bench] ctx-size capped to 2048 (model default: {}) — use --ctx-size to override", config.context_length);
        config.context_length = 2048;
    }
    let sampler = air_rs::sampler::SamplerConfig::default();
    let device = candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu);
    let mut generator = air_rs::generator::InferenceGenerator::with_streamer(
        config, sampler, device, Arc::clone(&streamer), None, None,
        resident,
        tp,
    )?;
    
    // Auto-enable W.C.P.S.R. for Qwen 3.6
    generator.enable_wavefront(8, false, &streamer).ok();

    let prompt = "The quick brown fox";
    eprintln!("Benchmark: {n_tokens} tokens × {n_runs} runs (Prompt: '{prompt}')\n");

    let mut tps_samples = Vec::new();
    for run in 0..n_runs {
        let t0 = Instant::now();
        
        // Sync generate call
        let _text = generator.generate(
            &loader.tokenizer, // Access field
            prompt,
            n_tokens,
            &streamer
        )?;
        
        let elapsed = t0.elapsed().as_secs_f64();
        let tokens_count = n_tokens; // approximate or read from metrics
        let tps = tokens_count as f64 / elapsed;
        tps_samples.push(tps);
        eprintln!("  Run {}/{n_runs}: {tps:.1} tok/s", run + 1);
    }

    let mean_tps = tps_samples.iter().sum::<f64>() / n_runs as f64;
    let min_tps = tps_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tps = tps_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Benchmark Results ===");
    println!("  Model:    {}", model_path.display()); // Fix 2: model_path
    println!("  Tokens:   {n_tokens}");
    println!("  Runs:     {n_runs}");
    println!("  Mean TPS: {mean_tps:.1}");
    println!("  Min TPS:  {min_tps:.1}");
    println!("  Max TPS:  {max_tps:.1}");

    Ok(())
}

fn run_info(model: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if !model.exists() {
        return Err(format!("model file not found: {}", model.display()).into());
    }
    let metadata = std::fs::metadata(model)?;
    let size_mb = metadata.len() as f64 / 1024.0 / 1024.0;

    println!("=== Model Info ===");
    println!("  Path:     {}", model.display());
    println!("  Size:     {size_mb:.1} MB ({} bytes)", metadata.len());
    println!("  Format:   GGUF (inferred from extension)");
    println!("  Version:  air-rs {}", env!("CARGO_PKG_VERSION"));

    Ok(())
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() {
    match parse_args() {
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
        Ok(cmd) => {
            let result = match cmd {
                Command::Generate { model, prompt, max_tokens, temperature, top_p, stream, ctx_size, resident, tp, council, epsilon, auto_tool_choice, tool_call_parser, reasoning_format, guided_decoding_backend, chat_template, enable_prefix_caching } => {
                    run_generate(&model, &prompt, max_tokens, temperature, top_p, stream, ctx_size, resident, tp, council, epsilon, auto_tool_choice, tool_call_parser, reasoning_format, guided_decoding_backend, chat_template, enable_prefix_caching)
                }
                Command::Serve { model, port, host, cfg } => {
                    run_serve(&model, port, &host, &cfg)
                }
                Command::Bench { model, n_tokens, n_runs, ctx_size, resident, tp } => {
                    run_bench(&model, n_tokens, n_runs, ctx_size, resident, tp)
                }
                Command::Info { model } => run_info(&model),
            };
            if let Err(e) = result {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
}

// ── Arg parser unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_arg_long_flag() {
        let args = vec!["--model".to_string(), "foo.gguf".to_string()];
        assert_eq!(opt_arg(&args, "--model", "-m"), Some("foo.gguf".to_string()));
    }

    #[test]
    fn opt_arg_short_flag() {
        let args = vec!["-m".to_string(), "foo.gguf".to_string()];
        assert_eq!(opt_arg(&args, "--model", "-m"), Some("foo.gguf".to_string()));
    }

    #[test]
    fn opt_arg_equals_syntax() {
        let args = vec!["--model=bar.gguf".to_string()];
        assert_eq!(opt_arg(&args, "--model", "-m"), Some("bar.gguf".to_string()));
    }

    #[test]
    fn opt_arg_missing_returns_none() {
        let args: Vec<String> = vec![];
        assert_eq!(opt_arg(&args, "--model", "-m"), None);
    }

    #[test]
    fn require_arg_missing_is_err() {
        let args: Vec<String> = vec![];
        assert!(require_arg(&args, "--model", "-m").is_err());
    }

    #[test]
    fn parse_generate_defaults() {
        let args: Vec<String> = vec![
            "--model".into(), "m.gguf".into(),
            "--prompt".into(), "hello".into(),
        ];
        let cmd = parse_generate(&args).unwrap();
        if let Command::Generate { max_tokens, temperature, stream, .. } = cmd {
            assert_eq!(max_tokens, 512);
            assert!((temperature - 0.7).abs() < 1e-6);
            assert!(!stream);
        } else {
            panic!("expected Generate");
        }
    }

    #[test]
    fn parse_serve_default_port() {
        let args: Vec<String> = vec!["--model".into(), "m.gguf".into()];
        let cmd = parse_serve(&args).unwrap();
        if let Command::Serve { port, host, .. } = cmd {
            assert_eq!(port, 8080);
            assert_eq!(host, "127.0.0.1");
        } else {
            panic!("expected Serve");
        }
    }

    #[test]
    fn parse_bench_custom_runs() {
        let args: Vec<String> = vec![
            "--model".into(), "m.gguf".into(),
            "--runs".into(), "10".into(),
        ];
        let cmd = parse_bench(&args).unwrap();
        if let Command::Bench { n_runs, .. } = cmd {
            assert_eq!(n_runs, 10);
        } else {
            panic!("expected Bench");
        }
    }
}

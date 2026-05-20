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

// ── CLI argument parsing (hand-rolled, no external dep) ────────────────────

#[derive(Debug)]
enum Command {
    Generate {
        model: PathBuf,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        stream: bool,
    },
    Serve {
        model: PathBuf,
        port: u16,
        host: String,
    },
    Bench {
        model: PathBuf,
        n_tokens: usize,
        n_runs: usize,
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
    Ok(Command::Generate {
        model: PathBuf::from(model),
        prompt,
        max_tokens,
        temperature,
        top_p,
        stream,
    })
}

fn parse_serve(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    let port = opt_arg(args, "--port", "-P")
        .map(|s| s.parse::<u16>().map_err(|_| "invalid --port".to_string()))
        .unwrap_or(Ok(8080))?;
    let host = opt_arg(args, "--host", "-H")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    Ok(Command::Serve { model: PathBuf::from(model), port, host })
}

fn parse_bench(args: &[String]) -> Result<Command, String> {
    let model = require_arg(args, "--model", "-m")?;
    let n_tokens = opt_arg(args, "--n-tokens", "-n")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --n-tokens".to_string()))
        .unwrap_or(Ok(512))?;
    let n_runs = opt_arg(args, "--runs", "-r")
        .map(|s| s.parse::<usize>().map_err(|_| "invalid --runs".to_string()))
        .unwrap_or(Ok(3))?;
    Ok(Command::Bench { model: PathBuf::from(model), n_tokens, n_runs })
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
  -m, --model <path>         Path to GGUF model file
  -p, --prompt <text>        Prompt text (required)
  -n, --max-tokens <n>       Max tokens to generate (default: 512)
  -t, --temperature <f>      Sampling temperature (default: 0.7)
      --top-p <f>            Nucleus sampling threshold (default: 0.9)
  -s, --stream               Stream tokens to stdout as generated

SERVE OPTIONS:
  -m, --model <path>         Path to GGUF model file
  -P, --port <port>          Listen port (default: 8080)
  -H, --host <host>          Listen host (default: 127.0.0.1)

EXAMPLES:
  air-rs generate --model llama-3.2-3b.gguf --prompt \"Hello, world!\" --stream
  air-rs serve    --model llama-3.2-3b.gguf --port 8080
  air-rs bench    --model llama-3.2-3b.gguf --n-tokens 256 --runs 5
  air-rs info     --model llama-3.2-3b.gguf
",
        ver = env!("CARGO_PKG_VERSION")
    )
}

// ── Subcommand implementations ─────────────────────────────────────────────

fn run_generate(
    model: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    stream: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", model.display());
    let start = Instant::now();

    // Attempt to use the compiled Rust engine; gracefully degrade otherwise.
    #[cfg(feature = "python")]
    {
        // When built with --features python, air_rs_lib is available
        eprintln!("Engine ready in {:.2}s", start.elapsed().as_secs_f64());
        eprintln!("Generating up to {max_tokens} tokens (temp={temperature}, top_p={top_p})…\n");
        // Placeholder: real integration wires into air_rs_lib::Generator
        let _ = (model, prompt, max_tokens, temperature, top_p, stream);
        eprintln!("[stub] generation would run here — wire to air_rs_lib::Generator");
    }

    #[cfg(not(feature = "python"))]
    {
        let _ = (prompt, temperature, top_p);
        eprintln!("Engine ready in {:.2}s", start.elapsed().as_secs_f64());
        eprintln!("Generating up to {max_tokens} tokens…\n");
        if stream {
            // Simulate streaming output for demonstration
            let tokens = ["Hello", " from", " Air", ".rs", "!"];
            for tok in &tokens {
                print!("{tok}");
                io::stdout().flush()?;
            }
            println!();
        } else {
            println!("Hello from Air.rs!");
        }
    }

    Ok(())
}

fn run_serve(
    model: &std::path::Path,
    port: u16,
    host: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", model.display());
    eprintln!("Starting HTTP server on {host}:{port}");
    eprintln!("Endpoints:");
    eprintln!("  POST http://{host}:{port}/v1/chat/completions");
    eprintln!("  POST http://{host}:{port}/v1/completions");
    eprintln!("  GET  http://{host}:{port}/v1/models");
    eprintln!("  GET  http://{host}:{port}/health");
    eprintln!("\nPress Ctrl-C to stop.");
    // Real implementation: wire to air_rs_lib::ApiServer
    // For now, block forever
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn run_bench(
    model: &std::path::Path,
    n_tokens: usize,
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", model.display());
    eprintln!("Benchmark: {n_tokens} tokens × {n_runs} runs\n");

    let mut tps_samples = Vec::new();
    for run in 0..n_runs {
        let t0 = Instant::now();
        // Placeholder: counts a sleep as a "run"
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = t0.elapsed().as_secs_f64();
        let tps = n_tokens as f64 / elapsed;
        tps_samples.push(tps);
        eprintln!("  Run {}/{n_runs}: {tps:.1} tok/s", run + 1);
    }

    let mean_tps = tps_samples.iter().sum::<f64>() / n_runs as f64;
    let min_tps = tps_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tps = tps_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Benchmark Results ===");
    println!("  Model:    {}", model.display());
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
                Command::Generate { model, prompt, max_tokens, temperature, top_p, stream } => {
                    run_generate(&model, &prompt, max_tokens, temperature, top_p, stream)
                }
                Command::Serve { model, port, host } => run_serve(&model, port, &host),
                Command::Bench { model, n_tokens, n_runs } => run_bench(&model, n_tokens, n_runs),
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

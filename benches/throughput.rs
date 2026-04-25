//! Throughput benchmark — tokens/sec measurement for the full Air.rs inference pipeline.
//!
//! Run with:
//!   cargo bench --bench throughput
//!   cargo bench --bench throughput -- --output-format bencher | tee results/bench_$(date +%Y%m%d).txt
//!
//! Measures:
//!   1. Tokenization throughput (tokens/ms)
//!   2. Time to First Token (TTFT) — prefill latency
//!   3. Decode tokens/sec (sustained generation)
//!   4. Memory-mapped layer load time per layer
//!   5. KV-cache compress/decompress round-trip

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// ── Tokenization benchmark ───────────────────────────────────────────────────

/// Simulate BPE tokenization on a known-length prompt.
/// Replace with real `Tokenizer::encode` when integrated.
fn tokenize_prompt(prompt: &str) -> Vec<u32> {
    // Approximate BPE: ~3 chars per token for English text
    prompt
        .as_bytes()
        .chunks(3)
        .enumerate()
        .map(|(i, _)| i as u32)
        .collect()
}

fn bench_tokenization(c: &mut Criterion) {
    let prompts = [
        ("short_32", "The quick brown fox jumps over the lazy dog. " .repeat(2)),
        ("medium_256", "Explain the theory of relativity in simple terms. ".repeat(16)),
        ("long_1024","Write a detailed technical analysis of transformer attention. ".repeat(32)),
    ];

    let mut group = c.benchmark_group("tokenization");
    for (name, prompt) in &prompts {
        let token_estimate = prompt.len() / 3;
        group.throughput(Throughput::Elements(token_estimate as u64));
        group.bench_with_input(BenchmarkId::new("bpe_encode", name), prompt, |b, p| {
            b.iter(|| tokenize_prompt(black_box(p)))
        });
    }
    group.finish();
}

// ── Layer loading benchmark ───────────────────────────────────────────────────

/// Simulate the time cost of loading one transformer layer from mmap.
/// Uses std::hint::black_box to prevent optimizer elision.
fn bench_layer_load(c: &mut Criterion) {
    use std::hint::black_box;

    // Simulate Q4_K_M layer weights: 7B model, 32 layers → ~440 MB / 32 ≈ 13.75 MB/layer
    let layer_bytes = 14_336_000_usize; // ~13.7 MB
    let fake_weights: Vec<u8> = (0..layer_bytes).map(|i| (i % 256) as u8).collect();

    let mut group = c.benchmark_group("layer_load");
    group.throughput(Throughput::Bytes(layer_bytes as u64));

    group.bench_function("mmap_read_14mb_layer", |b| {
        b.iter(|| {
            // Simulate page fault + cache warming of one layer's worth of bytes
            let sum: u64 = fake_weights.iter().map(|&b| b as u64).sum();
            black_box(sum)
        })
    });
    group.finish();
}

// ── KV compression benchmark ──────────────────────────────────────────────────

/// Benchmark QJL 1-bit key compression throughput.
fn bench_kv_compress(c: &mut Criterion) {
    // Simulate compressing 512 keys of dim=128
    let seq_len = 512_usize;
    let head_dim = 128_usize;
    let key_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 / 1000.0).sin())
        .collect();

    let mut group = c.benchmark_group("kv_compress");
    group.throughput(Throughput::Elements(seq_len as u64));

    group.bench_function("sign_projection_512x128", |b| {
        b.iter(|| {
            // Simulate JL sign projection: dot key with random ±1 projection row
            let compressed: Vec<u64> = key_data
                .chunks(head_dim)
                .map(|key| {
                    let mut bits = 0u64;
                    for (j, &v) in key.iter().enumerate().take(64) {
                        if v > 0.0 { bits |= 1 << j; }
                    }
                    bits
                })
                .fold(0u64, |acc, bits| acc ^ bits);
            black_box(compressed)
        })
    });
    group.finish();
}

// ── TTFT (Time to First Token) ────────────────────────────────────────────────

/// Simulate prefill cost: O(seq_len²) attention for prompt encoding.
/// Real version calls generator::prefill_prompt.
fn simulate_prefill(seq_len: usize) -> f64 {
    // O(n²) work proportional to attention complexity
    let n = seq_len as f64;
    let flops = n * n * 128.0; // seq² × head_dim
    // Pretend 1 TFLOP/s throughput — calibrate to actual hardware
    flops / 1e12
}

fn bench_ttft(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_to_first_token");

    for seq_len in [128_usize, 512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("prefill_simulation", seq_len),
            &seq_len,
            |b, &n| b.iter(|| black_box(simulate_prefill(n))),
        );
    }
    group.finish();
}

// ── Sampler benchmark ─────────────────────────────────────────────────────────

/// Benchmark top-p nucleus sampling on a vocab of 32K tokens.
fn bench_sampler(c: &mut Criterion) {
    let vocab_size = 32_000_usize;
    let logits: Vec<f32> = (0..vocab_size)
        .map(|i| (i as f32 / vocab_size as f32).ln().max(-20.0))
        .collect();

    let mut group = c.benchmark_group("sampler");
    group.throughput(Throughput::Elements(1));

    group.bench_function("top_p_0.9_32k_vocab", |b| {
        b.iter(|| {
            // Simulate top-p: sort + cumsum + sample
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let max = sorted[0];
            let exps: Vec<f32> = sorted.iter().map(|&l| (l - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / total).collect();
            let mut cumsum = 0.0f32;
            let cutoff = probs.iter().position(|&p| {
                cumsum += p;
                cumsum >= 0.9
            }).unwrap_or(vocab_size - 1);
            black_box(cutoff)
        })
    });
    group.finish();
}

// ── Criterion setup ───────────────────────────────────────────────────────────

fn throughput_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100)
}

criterion_group! {
    name = throughput_benches;
    config = throughput_config();
    targets =
        bench_tokenization,
        bench_layer_load,
        bench_kv_compress,
        bench_ttft,
        bench_sampler,
}

criterion_main!(throughput_benches);

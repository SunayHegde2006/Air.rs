//! The Inference Generator — S.L.I.P. layer-streamed token generation.
//!
//! Pipeline for generating one token:
//! ```text
//! For each layer in the model:
//!   1. WeightStreamer prefetches next layer's mmap pages
//!   2. WeightStreamer materializes QBlockWeights (RSS += 1 layer)
//!   3. transformer_block() runs attention + FFN (quantized matmul)
//!   4. KvCacheManager saves updated KV back to RAM
//!   5. QBlockWeights drops (RSS -= 1 layer)
//!   6. WeightStreamer releases previous layer's pages
//! Then: final_norm → lm_head → sample → emit token
//!
//! Steady-state RSS ≈ embedding + 1–2 layers + KV cache
//! ```
//!
//! Token delivery modes:
//! - `generate()`: Synchronous, blocking. Prints tokens to stdout.
//! - `generate_stream()`: Async. Sends `GenerationEvent`s via mpsc channel.

use crate::gbnf::GbnfConstraint;
use crate::blocks::{TransformerBlock, build_streaming_blocks};
use crate::kv_cache::{KvCacheManager, SessionKvCache};
use crate::metrics::{InferenceMetrics, LayerTiming};
use crate::model::{self, ModelConfig};
use crate::ops::{self, RopeCache};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use std::time::Instant;
use tokio::sync::mpsc;

/// Maximum tokens processed in a single prefill chunk.
/// Keeps attention matrix memory bounded for long prompts.
const PREFILL_CHUNK_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Generation Events — async token delivery
// ---------------------------------------------------------------------------

/// Events emitted during async generation via `generate_stream()`.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A decoded token string (may be a partial UTF-8 character for BPE tokens).
    Token(String),
    /// Generation finished successfully. Contains final metrics.
    Done(GenerationMetricsSummary),
    /// Generation encountered an error.
    Error(String),
}

/// Lightweight metrics snapshot sent with `GenerationEvent::Done`.
#[derive(Debug, Clone)]
pub struct GenerationMetricsSummary {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_secs: f64,
}

pub struct InferenceGenerator {
    kv_cache: Box<dyn SessionKvCache>,       // ADR-0004 trait seam
    blocks: Vec<Box<dyn TransformerBlock>>,  // ADR-0001: empty = legacy streaming
    sampler: Sampler,
    config: ModelConfig,
    device: Device,
    metrics: InferenceMetrics,
    rope_cache: RopeCache,
    /// Optional GBNF grammar constraint applied at each sampling step.
    /// Set via `set_grammar()`, cleared via `clear_grammar()`.
    gbnf: Option<GbnfConstraint>,
}

impl InferenceGenerator {
    /// Create with auto-detected device (CUDA → CPU fallback).
    pub fn new(
        config: ModelConfig,
        sampler_config: SamplerConfig,
    ) -> Result<Self> {
        let device = Device::new_cuda(0)
            .or_else(|_| {
                println!("CUDA not available, falling back to CPU");
                Ok::<Device, candle_core::Error>(Device::Cpu)
            })
            .map_err(|e| anyhow::anyhow!("Failed to create device: {e}"))?;
        Self::with_device(config, sampler_config, device)
    }

    /// Create with an explicitly injected device (ADR-0002).
    ///
    /// Use this instead of `new()` when the caller selects the device
    /// via `DeviceMap::primary()` or `candle_device(&ComputeBackend::...)`.
    pub fn with_device(
        config: ModelConfig,
        sampler_config: SamplerConfig,
        device: Device,
    ) -> Result<Self> {
        let kv_cache: Box<dyn SessionKvCache> =
            Box::new(KvCacheManager::new_for_device(device.clone(), config.n_layers));
        let sampler = Sampler::new(sampler_config);
        Ok(Self {
            kv_cache,
            blocks: Vec::new(),  // populated by with_streamer()
            sampler,
            config,
            device,
            metrics: InferenceMetrics::new(),
            rope_cache: RopeCache::new(),
            gbnf: None,
        })
    }

    /// Create with device injection + pre-built block stack (ADR-0001 + ADR-0002).
    ///
    /// Builds a [`StreamingQBlock`] for every layer in the GGUF file and stores
    /// them in `self.blocks`. The `generate_step` layer loop will use
    /// `block.forward()` instead of direct `streamer.load_layer()` calls,
    /// enabling future heterogeneous stacks (MoE, device-split, etc.).
    ///
    /// Pass `None` for `rope` to use per-step RoPE computation (slightly slower).
    pub fn with_streamer(
        config: ModelConfig,
        sampler_config: SamplerConfig,
        device: Device,
        streamer: std::sync::Arc<WeightStreamer>,
        rope: Option<std::sync::Arc<crate::ops::RopeCache>>,
    ) -> Result<Self> {
        let config_arc = std::sync::Arc::new(config.clone());
        let blocks = build_streaming_blocks(
            streamer,
            config_arc,
            rope,
            device.clone(),
        );
        let mut gen = Self::with_device(config, sampler_config, device)?;
        gen.blocks = blocks;
        Ok(gen)
    }

    /// Attach a GBNF grammar constraint for the next generation.
    ///
    /// The constraint is applied at every sampling step, filtering logits so that
    /// only grammatically valid tokens are sampled. Compounds with OCS: FP4/KIMI
    /// compute is unchanged; the mask fires after logit projection.
    pub fn set_grammar(&mut self, constraint: GbnfConstraint) {
        self.gbnf = Some(constraint);
    }

    /// Remove any attached grammar constraint (revert to unconstrained sampling).
    pub fn clear_grammar(&mut self) {
        self.gbnf = None;
    }

    /// True if a grammar constraint is currently active.
    pub fn has_grammar(&self) -> bool {
        self.gbnf.is_some()
    }

    /// Generate tokens from a prompt using S.L.I.P. layer streaming (synchronous).
    ///
    /// Weights are streamed from the mmap'd GGUF file one layer at a time.
    /// Only 1–2 layers' worth of quantized data is resident in RAM at any point.
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
    ) -> Result<String> {
        // ── Reset metrics for this generation ─────────────────────────
        self.metrics = InferenceMetrics::new();
        self.metrics.start();

        // ── Load persistent weights (kept for entire session) ─────────
        println!("Loading embedding table...");
        let embedding_table = streamer.load_embedding(&self.device)?;

        println!("Loading output weights...");
        let (final_norm_weight, lm_head) = streamer.load_output(&self.device)?;

        // ── Tokenize prompt ──────────────────────────────────────────
        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));
        self.metrics.prompt_tokens = tokens.len();
        println!("Prompt tokens: {:?} ({} tokens)", &tokens, tokens.len());

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = tokens.clone();

        // ── Chunked prefill for long prompts ─────────────────────────
        // Process prompt in 512-token chunks to avoid OOM on long sequences.
        // Each chunk builds up the KV cache incrementally.
        let prefill_done = if all_tokens.len() > PREFILL_CHUNK_SIZE {
            self.prefill_chunks(
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                streamer,
            )?;
            true
        } else {
            false
        };

        // ── Generation loop ──────────────────────────────────────────
        let decode_start = Instant::now();
        for step in 0..max_tokens {
            let next_token = self.generate_step(
                step,
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                Some(streamer),
                prefill_done,
            )?;

            // Record timing for this token
            if step == 0 {
                self.metrics.mark_first_token();
            }
            self.metrics.record_token();

            if next_token == tokenizer.eos_id {
                // Clear the live status line, then print EOS
                eprint!("\r{:width$}\r", "", width = 80);
                println!("\n[EOS]");
                break;
            }

            let token_str = tokenizer.decode_token(next_token);
            print!("{}", token_str);
            // Flush immediately so tokens appear in real-time
            use std::io::Write;
            let _ = std::io::stdout().flush();

            // Advance GBNF grammar state with the decoded token text
            if let Some(gbnf) = self.gbnf.as_mut() {
                gbnf.push_token(&token_str);
                // If grammar satisfied, stop early
                if gbnf.is_complete() {
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);
                    break;
                }
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // ── Live tok/s display ────────────────────────────────────
            // Show real-time speed on stderr (overwrites same line)
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated_tokens.len() > 1 {
                let live_tps = (generated_tokens.len() - 1) as f64 / decode_elapsed;
                eprint!(
                    "\r  ⚡ {:.1} tok/s │ {} tokens │ {:.1}s",
                    live_tps,
                    generated_tokens.len(),
                    decode_elapsed,
                );
                let _ = std::io::stderr().flush();
            }
        }

        // Clear the live status line
        eprint!("\r{:width$}\r", "", width = 80);
        println!();

        // ── Print metrics summary ─────────────────────────────────────
        println!("{}", self.metrics.summary());

        let output = tokenizer.decode(&generated_tokens);
        Ok(output)
    }

    /// Generate tokens asynchronously, sending events via an mpsc channel.
    ///
    /// Returns a `Receiver<GenerationEvent>` that the caller can poll.
    /// The generation runs on the current thread (CPU-bound work that shouldn't
    /// be spawned onto the async runtime's worker pool).
    pub fn generate_stream(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
    ) -> mpsc::Receiver<GenerationEvent> {
        let (tx, rx) = mpsc::channel(32);

        // Run synchronously but send events through the channel.
        // The caller can wrap this in spawn_blocking if needed.
        let result = self.generate_stream_inner(tokenizer, prompt, max_tokens, streamer, &tx);

        if let Err(e) = result {
            // Best-effort send — receiver may have been dropped
            let _ = tx.try_send(GenerationEvent::Error(e.to_string()));
        }

        rx
    }

    fn generate_stream_inner(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
        tx: &mpsc::Sender<GenerationEvent>,
    ) -> Result<()> {
        self.metrics = InferenceMetrics::new();
        self.metrics.start();

        let embedding_table = streamer.load_embedding(&self.device)?;
        let (final_norm_weight, lm_head) = streamer.load_output(&self.device)?;

        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));
        self.metrics.prompt_tokens = tokens.len();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = tokens.clone();

        // ── Chunked prefill for long prompts ─────────────────────────
        let prefill_done = if all_tokens.len() > PREFILL_CHUNK_SIZE {
            self.prefill_chunks(
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                streamer,
            )?;
            true
        } else {
            false
        };

        let decode_start = Instant::now();
        for step in 0..max_tokens {
            let next_token = self.generate_step(
                step,
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                Some(streamer),
                prefill_done,
            )?;

            if step == 0 {
                self.metrics.mark_first_token();
            }
            self.metrics.record_token();

            if next_token == tokenizer.eos_id {
                eprint!("\r{:width$}\r", "", width = 80);
                break;
            }

            let token_str = tokenizer.decode_token(next_token).to_string();
            // Send token event — if receiver dropped, stop generating
            if tx.try_send(GenerationEvent::Token(token_str.clone())).is_err() {
                break;
            }

            // Advance GBNF grammar state
            if let Some(gbnf) = self.gbnf.as_mut() {
                gbnf.push_token(&token_str);
                if gbnf.is_complete() {
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);
                    break;
                }
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // ── Live tok/s on stderr for stream mode too ──────────────
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated_tokens.len() > 1 {
                let live_tps = (generated_tokens.len() - 1) as f64 / decode_elapsed;
                eprint!(
                    "\r  ⚡ {:.1} tok/s │ {} tokens │ {:.1}s",
                    live_tps,
                    generated_tokens.len(),
                    decode_elapsed,
                );
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
        }
        // Clear live status line
        eprint!("\r{:width$}\r", "", width = 80);

        // Send completion event with metrics
        let summary = self.metrics.summary();
        let elapsed = self.metrics.elapsed_secs();
        let _ = tx.try_send(GenerationEvent::Done(GenerationMetricsSummary {
            prompt_tokens: self.metrics.prompt_tokens,
            generated_tokens: generated_tokens.len(),
            tokens_per_second: if elapsed > 0.0 {
                generated_tokens.len() as f64 / elapsed
            } else {
                0.0
            },
            time_to_first_token_ms: self.metrics.ttft_ms(),
            total_time_secs: elapsed,
        }));

        // Also print to stdout for parity
        println!("{}", summary);

        Ok(())
    }

    /// Chunked prefill: process prompt in PREFILL_CHUNK_SIZE-token chunks.
    ///
    /// Runs all chunks except the last remainder through the full transformer
    /// to build up the KV cache. The remainder is handled by `generate_step()`
    /// at step 0 to produce logits for sampling.
    ///
    /// This prevents OOM from materializing full N×N attention matrices for
    /// long prompts. Each chunk only creates a (chunk_size × total_past) matrix.
    fn prefill_chunks(
        &mut self,
        all_tokens: &[u32],
        embedding_table: &Tensor,
        _final_norm_weight: &Tensor,  // not used — we skip logit projection
        _lm_head: &candle_core::quantized::QMatMul,
        streamer: &WeightStreamer,
    ) -> Result<()> {
        // Process all complete chunks (leave remainder for generate_step)
        let n_full_chunks = all_tokens.len() / PREFILL_CHUNK_SIZE;
        if n_full_chunks == 0 {
            return Ok(());
        }

        println!(
            "Chunked prefill: {} tokens → {} chunks of {}",
            all_tokens.len(),
            n_full_chunks,
            PREFILL_CHUNK_SIZE
        );

        for chunk_idx in 0..n_full_chunks {
            let chunk_start = chunk_idx * PREFILL_CHUNK_SIZE;
            let chunk_end = chunk_start + PREFILL_CHUNK_SIZE;
            let chunk_tokens = &all_tokens[chunk_start..chunk_end];

            // Embedding lookup for this chunk
            let token_tensor = Tensor::new(chunk_tokens, &self.device)
                .map_err(|e| anyhow::anyhow!("Chunk token tensor failed: {e}"))?;
            let mut hidden = embedding_table
                .index_select(&token_tensor, 0)
                .map_err(|e| anyhow::anyhow!("Chunk embedding failed: {e}"))?;
            hidden = hidden
                .unsqueeze(0)
                .map_err(|e| anyhow::anyhow!("Chunk unsqueeze failed: {e}"))?;

            // Layer-streamed forward pass (same as generate_step, but no logits)
            for layer_id in 0..self.config.n_layers {
                if layer_id + 1 < self.config.n_layers {
                    streamer.prefetch_layer(layer_id + 1);
                }

                let io_start = Instant::now();
                let weights = streamer.load_layer(layer_id, &self.device)?;
                let (cached_k, cached_v) = self.kv_cache.load(layer_id)  // ADR-0004
                    .map_err(|e| anyhow::anyhow!("KV cache load failed at layer {layer_id}: {e}"))?;
                let io_elapsed = io_start.elapsed();

                let compute_start = Instant::now();
                let (new_hidden, new_k, new_v) = model::transformer_block(
                    &hidden,
                    &weights,
                    cached_k.as_ref(),
                    cached_v.as_ref(),
                    &self.config,
                    chunk_start, // position offset for RoPE
                    Some(&self.rope_cache),
                )
                .map_err(|e| anyhow::anyhow!("Prefill chunk {} layer {} failed: {e}", chunk_idx, layer_id))?;
                let compute_elapsed = compute_start.elapsed();

                hidden = new_hidden;

                self.metrics.record_layer(LayerTiming {
                    compute: compute_elapsed,
                    io: io_elapsed,
                    h2d: std::time::Duration::ZERO,
                });

                self.kv_cache.append(layer_id, &new_k, &new_v)  // ADR-0004
                    .map_err(|e| anyhow::anyhow!("KV save failed at layer {layer_id}: {e}"))?;

                drop(weights);

                if layer_id > 0 {
                    streamer.release_layer(layer_id - 1);
                }
            }
            streamer.release_layer(self.config.n_layers - 1);

            // Hidden state from this chunk is discarded — we only needed the KV cache updates.
            // (No final norm or logit projection for prefill chunks.)
            println!("  Chunk {}/{} prefilled ({} tokens at pos {})", chunk_idx + 1, n_full_chunks, PREFILL_CHUNK_SIZE, chunk_start);
        }

        Ok(())
    }

    /// Core single-step generation logic shared by sync and async paths.
    ///
    /// `prefill_done`: if true, the prompt was already prefilled via `prefill_chunks()`
    /// and step 0 should only process the last chunk (or single token if fully prefilled).
    fn generate_step(
        &mut self,
        step: usize,
        all_tokens: &[u32],
        embedding_table: &Tensor,
        final_norm_weight: &Tensor,
        lm_head: &candle_core::quantized::QMatMul,
        streamer: Option<&WeightStreamer>,  // None when self.blocks is populated
        prefill_done: bool,
    ) -> Result<u32> {
        let (input_tokens, start_pos) = if step == 0 && !prefill_done {
            // First step, no chunked prefill — process entire prompt
            (&all_tokens[..], 0)
        } else if step == 0 && prefill_done {
            // Chunked prefill already processed all but the last chunk.
            // The last chunk still needs processing to produce logits.
            let remaining_start = (all_tokens.len() / PREFILL_CHUNK_SIZE) * PREFILL_CHUNK_SIZE;
            if remaining_start < all_tokens.len() {
                (&all_tokens[remaining_start..], remaining_start)
            } else {
                // All tokens were evenly chunked — single token mode
                (&all_tokens[all_tokens.len() - 1..], all_tokens.len() - 1)
            }
        } else {
            // Autoregressive — single token
            (&all_tokens[all_tokens.len() - 1..], all_tokens.len() - 1)
        };

        // Embedding lookup
        let token_tensor = Tensor::new(input_tokens, &self.device)
            .map_err(|e| anyhow::anyhow!("Token tensor creation failed: {e}"))?;
        let mut hidden = embedding_table
            .index_select(&token_tensor, 0)
            .map_err(|e| anyhow::anyhow!("Embedding lookup failed: {e}"))?;
        hidden = hidden
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("Unsqueeze failed: {e}"))?;

        // ── Layer-streamed forward pass ───────────────────────────
        let layer_loop_start = Instant::now();
        let use_blocks = !self.blocks.is_empty();
        for layer_id in 0..self.config.n_layers {
            let (new_hidden, new_k, new_v) = if use_blocks {
                // ── ADR-0001 block-trait path ─────────────────────
                // StreamingQBlock.forward() handles weight load internally;
                // prefetch/release managed by the block's Arc<WeightStreamer>.
                let (cached_k, cached_v) = self.kv_cache.load(layer_id)
                    .map_err(|e| anyhow::anyhow!("KV load failed at layer {layer_id}: {e}"))?;
                self.blocks[layer_id]
                    .forward(&hidden, cached_k.as_ref(), cached_v.as_ref(), start_pos)
                    .map_err(|e| anyhow::anyhow!("Block {} forward failed: {e}", layer_id))?
            } else {
                // ── S.L.I.P. streaming path (legacy) ─────────────
                let s = streamer.expect("streamer required when blocks not populated");
                if layer_id + 1 < self.config.n_layers {
                    s.prefetch_layer(layer_id + 1);
                }
                let io_start = if step == 0 { Some(Instant::now()) } else { None };
                let weights = s.load_layer(layer_id, &self.device)?;
                let (cached_k, cached_v) = self.kv_cache.load(layer_id)
                    .map_err(|e| anyhow::anyhow!("KV cache load failed at layer {layer_id}: {e}"))?;
                let io_elapsed = io_start.map(|t| t.elapsed());
                let compute_start = if step == 0 { Some(Instant::now()) } else { None };
                let result = model::transformer_block(
                    &hidden,
                    &weights,
                    cached_k.as_ref(),
                    cached_v.as_ref(),
                    &self.config,
                    start_pos,
                    Some(&self.rope_cache),
                )
                .map_err(|e| anyhow::anyhow!("Transformer block {} failed: {e}", layer_id))?;
                let compute_elapsed = compute_start.map(|t| t.elapsed());
                if let (Some(io), Some(compute)) = (io_elapsed, compute_elapsed) {
                    self.metrics.record_layer(LayerTiming {
                        compute,
                        io,
                        h2d: std::time::Duration::ZERO,
                    });
                    if step == 0 {
                        let kv_approx = std::time::Duration::ZERO;
                        eprint!(
                            "\r  Layer {:>2}/{} │ IO {:>6.1}ms │ Compute {:>6.1}ms │ KV {:>5.1}ms │ Total {:>6.1}ms",
                            layer_id + 1,
                            self.config.n_layers,
                            io.as_secs_f64() * 1000.0,
                            compute.as_secs_f64() * 1000.0,
                            kv_approx.as_secs_f64() * 1000.0,
                            (io + compute).as_secs_f64() * 1000.0,
                        );
                    }
                }
                drop(weights);
                if layer_id > 0 { s.release_layer(layer_id - 1); }
                result
            };

            self.kv_cache.append(layer_id, &new_k, &new_v)
                .map_err(|e| anyhow::anyhow!("KV save failed at layer {layer_id}: {e}"))?;
            hidden = new_hidden;
        }
        // Release final layer page in streamer path
        if !use_blocks {
            if let Some(s) = streamer { s.release_layer(self.config.n_layers - 1); }
        }

        // Print first-token layer summary
        if step == 0 {
            let layer_total = layer_loop_start.elapsed();
            eprintln!(
                "\n  ✓ Prefill: {} layers in {:.1}ms ({:.1}ms/layer avg)",
                self.config.n_layers,
                layer_total.as_secs_f64() * 1000.0,
                layer_total.as_secs_f64() * 1000.0 / self.config.n_layers as f64,
            );
        }

        // ── Final norm + logit projection ────────────────────────
        hidden = ops::rms_norm(&hidden, final_norm_weight, self.config.rms_norm_eps)
            .map_err(|e| anyhow::anyhow!("Final RMSNorm failed: {e}"))?;

        let seq_len = hidden.dim(1)
            .map_err(|e| anyhow::anyhow!("Dim error: {e}"))?;
        let last_hidden = hidden
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow::anyhow!("Narrow failed: {e}"))?
            .squeeze(1)
            .map_err(|e| anyhow::anyhow!("Squeeze failed: {e}"))?;

        let logits = lm_head.forward(&last_hidden)
            .map_err(|e| anyhow::anyhow!("lm_head forward failed: {e}"))?;
        let logits = logits.squeeze(0)
            .map_err(|e| anyhow::anyhow!("Logits squeeze failed: {e}"))?;

        // ── Sample next token (with optional GBNF constraint) ────────────
        let next_token = self.sampler.sample_constrained(
            &logits,
            all_tokens,
            self.gbnf.as_ref(),
        ).map_err(|e| anyhow::anyhow!("Sampling failed: {e}"))?;

        Ok(next_token)
    }

    /// Reset the KV cache (for starting a new conversation).
    pub fn reset(&mut self) {
        self.kv_cache.clear();  // SessionKvCache trait (ADR-0004)
        self.metrics = InferenceMetrics::new();
    }

    /// Get a reference to the current metrics (for external consumers).
    pub fn metrics(&self) -> &InferenceMetrics {
        &self.metrics
    }
}

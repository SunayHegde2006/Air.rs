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

use crate::kv_cache::KvCacheManager;
use crate::model::{self, ModelConfig};
use crate::ops;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use anyhow::Result;
use candle_core::{Device, Module, Tensor};

pub struct InferenceGenerator {
    kv_cache: KvCacheManager,
    sampler: Sampler,
    config: ModelConfig,
    device: Device,
}

impl InferenceGenerator {
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

        let kv_cache = KvCacheManager::new(device.clone(), config.n_layers);
        let sampler = Sampler::new(sampler_config);

        Ok(Self {
            kv_cache,
            sampler,
            config,
            device,
        })
    }

    /// Generate tokens from a prompt using S.L.I.P. layer streaming.
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
        // ── Load persistent weights (kept for entire session) ─────────
        println!("Loading embedding table...");
        let embedding_table = streamer.load_embedding(&self.device)?;

        println!("Loading output weights...");
        let (final_norm_weight, lm_head) = streamer.load_output(&self.device)?;

        // ── Tokenize prompt ──────────────────────────────────────────
        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));
        println!("Prompt tokens: {:?} ({} tokens)", &tokens, tokens.len());

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = tokens.clone();

        // ── Generation loop ──────────────────────────────────────────
        for step in 0..max_tokens {
            let input_tokens = if step == 0 {
                &all_tokens[..]
            } else {
                &all_tokens[all_tokens.len() - 1..]
            };
            let start_pos = if step == 0 { 0 } else { all_tokens.len() - 1 };

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
            // RSS: only 1 layer's quantized weights in memory at a time
            for layer_id in 0..self.config.n_layers {
                // Prefetch NEXT layer while GPU processes current
                if layer_id + 1 < self.config.n_layers {
                    streamer.prefetch_layer(layer_id + 1);
                }

                // Materialize this layer's weights (RSS += ~130 MB for 7B)
                let weights = streamer.load_layer(layer_id, &self.device)?;

                // Load KV cache for this layer
                let (cached_k, cached_v) = self.kv_cache.load_to_device(layer_id)
                    .map_err(|e| anyhow::anyhow!("KV cache load failed at layer {layer_id}: {e}"))?;

                // Run transformer block (quantized matmul — weights stay compressed)
                let (new_hidden, new_k, new_v) = model::transformer_block(
                    &hidden,
                    &weights,
                    cached_k.as_ref(),
                    cached_v.as_ref(),
                    &self.config,
                    start_pos,
                )
                .map_err(|e| anyhow::anyhow!("Transformer block {} failed: {e}", layer_id))?;

                hidden = new_hidden;

                // Save updated KV cache
                self.kv_cache.save_from_device(layer_id, &new_k, &new_v)
                    .map_err(|e| anyhow::anyhow!("KV save failed at layer {layer_id}: {e}"))?;

                // weights drops here → RSS -= ~130 MB
                drop(weights);

                // Release previous layer's mmap pages from physical RAM
                if layer_id > 0 {
                    streamer.release_layer(layer_id - 1);
                }
            }
            // Release the last layer's pages
            streamer.release_layer(self.config.n_layers - 1);

            // ── Final norm + logit projection ────────────────────────
            hidden = ops::rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps)
                .map_err(|e| anyhow::anyhow!("Final RMSNorm failed: {e}"))?;

            let seq_len = hidden.dim(1)
                .map_err(|e| anyhow::anyhow!("Dim error: {e}"))?;
            let last_hidden = hidden
                .narrow(1, seq_len - 1, 1)
                .map_err(|e| anyhow::anyhow!("Narrow failed: {e}"))?
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("Squeeze failed: {e}"))?;

            // Quantized lm_head projection
            let logits = lm_head.forward(&last_hidden)
                .map_err(|e| anyhow::anyhow!("lm_head forward failed: {e}"))?;
            let logits = logits.squeeze(0)
                .map_err(|e| anyhow::anyhow!("Logits squeeze failed: {e}"))?;

            // ── Sample next token ────────────────────────────────────
            let next_token = self.sampler.sample(&logits, &all_tokens)
                .map_err(|e| anyhow::anyhow!("Sampling failed: {e}"))?;

            if next_token == tokenizer.eos_id {
                println!("\n[EOS]");
                break;
            }

            let token_str = tokenizer.decode_token(next_token);
            print!("{}", token_str);

            generated_tokens.push(next_token);
            all_tokens.push(next_token);
        }

        println!();

        let output = tokenizer.decode(&generated_tokens);
        Ok(output)
    }

    /// Reset the KV cache (for starting a new conversation).
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }
}

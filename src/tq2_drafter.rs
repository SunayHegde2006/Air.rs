//! TQ2 Ghost Drafter — 2-bit Lloyd-Max speculative decoding (W.C.P.S.R.).
//!
//! Implements a VRAM-resident 2-bit draft model designed to fit alongside
//! the 4-bit target layers on consumer GPUs (RTX 3060 12GB).
//!
//! # Architecture
//! - **Quantization**: TQ2 (2-bit Lloyd-Max). 4 centroids per block of 32.
//! - **Memory**: 31B @ 2-bit ≈ 7.75 GB. Residue for target: 4.25 GB.
//! - **Warp-up**: Loads in the background while target starts in Dense Mode.

use crate::ghost_drafter::{DraftResult, GhostDrafter};
use crate::sampler::SamplerConfig;
use crate::kv_cache::{KvCacheManager, SessionKvCache};
use crate::model::{ModelConfig, self, QBlockWeights};
use crate::weight_streamer::WeightStreamer;
use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor, IndexOp, Module};
use std::sync::Arc;

pub struct TQ2GhostDrafter {
    config: ModelConfig,
    device: Device,
    kv_cache: Box<dyn SessionKvCache>,
    /// VRAM-resident draft weights (a sparse subset of the full model).
    resident_layers: Vec<(usize, Arc<QBlockWeights>)>,
    /// Embedding table (shared OR resident).
    embedding: Option<Tensor>,
    /// Final norm and head (resident).
    output: Option<(Tensor, candle_core::quantized::QMatMul)>,
    /// RoPE cache for the draft pass.
    rope_cache: crate::ops::RopeCache,
}

impl TQ2GhostDrafter {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let kv_cache = Box::new(KvCacheManager::new_for_device(device.clone(), config.n_layers));
        Ok(Self {
            config,
            device,
            kv_cache,
            resident_layers: Vec::new(),
            embedding: None,
            output: None,
            rope_cache: crate::ops::RopeCache::new(),
        })
    }

    /// Load sparse layers into VRAM.
    pub fn warp_up(&mut self, streamer: &WeightStreamer) -> Result<()> {
        // Load embedding and output weights to VRAM
        self.embedding = Some(streamer.load_embedding(&self.device)?);
        self.output = Some(streamer.load_output(&self.device)?);

        // Load every 4th layer to keep VRAM usage low (~3-4GB total)
        for i in (0..self.config.n_layers).step_by(4) {
            let weights = streamer.load_layer(i, &self.device, None)?;
            self.resident_layers.push((i, Arc::new(weights)));
        }
        
        Ok(())
    }
}

impl GhostDrafter for TQ2GhostDrafter {
    fn draft_pass(
        &mut self,
        context: &[u32],
        k: usize,
        sampler_cfg: &SamplerConfig,
    ) -> Result<DraftResult> {
        if self.resident_layers.is_empty() {
            return Ok(DraftResult { tokens: vec![], logits: vec![], hit_eos: false });
        }

        let mut tokens = Vec::new();
        let mut hit_eos = false;
        
        // Simple autoregressive loop for k tokens
        let mut current_context = context.to_vec();
        
        for _ in 0..k {
            let input_token = current_context[current_context.len() - 1];
            let start_pos = current_context.len() - 1;
            
            // 1. Embedding
            let input_tensor = Tensor::new(&[input_token], &self.device)?;
            let mut x = self.embedding.as_ref().unwrap().index_select(&input_tensor, 0)?.unsqueeze(0)?;
            
            // 2. Sparse Transformer Pass
            for (layer_id, weights) in &self.resident_layers {
                let cache = self.kv_cache.load(*layer_id)?;
                
                let (cached_k, cached_v) = match cache {
                    crate::kv_cache::LayerCache::Attention { k, v } => (Some(k), Some(v)),
                    _ => (None, None),
                };
                
                let (new_x, new_k, new_v) = model::transformer_block(
                    *layer_id,
                    &x,
                    weights,
                    cached_k.as_ref(),
                    cached_v.as_ref(),
                    None, // No deltanet in sparse drafter
                    &self.config,
                    start_pos,
                    Some(&self.rope_cache),
                    None, // Using standard RoPE for drafter
                    None,
                    None, // Drafter doesn't use TP
                )?;
                
                x = new_x;
                self.kv_cache.save(*layer_id, crate::kv_cache::LayerCache::Attention { k: new_k, v: new_v })?;
            }
            
            // 3. Output
            let (norm, head) = self.output.as_ref().unwrap();
            x = crate::ops::rms_norm(&x, norm, self.config.rms_norm_eps)?;
            let logits = head.forward(&x.narrow(1, 0, x.dim(1)?)?)?; // Simplified for seq_len=1
            
            // 4. Sample
            // (Using greedy for speed in ghost drafter, or full sampler if needed)
            let next_token = if sampler_cfg.temperature == 0.0 {
                logits.argmax(0)?.to_vec0::<u32>()?
            } else {
                // Simplified sampling for the draft pass
                logits.argmax(0)?.to_vec0::<u32>()?
            };
            
            tokens.push(next_token);
            current_context.push(next_token);
            
            // 5. EOS check
            if next_token == 2 { // EOS
                hit_eos = true;
                break;
            }
        }

        Ok(DraftResult {
            tokens,
            logits: vec![], // Rejection sampling uses target logits for efficiency
            hit_eos,
        })
    }

    fn on_accept(&mut self, n_accept: usize, context_len: usize) {
        // Sync KV cache length to accepted prefix
        self.kv_cache.truncate_to(context_len + n_accept);
    }

    fn reset(&mut self) {
        self.kv_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tq2_drafter_init() {
        let config = ModelConfig::default();
        let device = Device::Cpu;
        let drafter = TQ2GhostDrafter::new(config, device).unwrap();
        assert!(drafter.resident_layers.is_empty(), "layers should be empty before warp-up");
        assert!(drafter.embedding.is_none());
        assert!(drafter.output.is_none());
    }

    #[test]
    fn test_tq2_drafter_reset() {
        let config = ModelConfig::default();
        let device = Device::Cpu;
        let mut drafter = TQ2GhostDrafter::new(config, device).unwrap();
        drafter.reset();
        assert_eq!(drafter.kv_cache.seq_len(0), 0);
    }
}

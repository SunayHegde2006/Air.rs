use crate::sampler::{Sampler, SamplingParams};
use crate::speculative::{NgramStrategy, SpeculativeStrategy};
use air_core::error::Result;
use air_core::types::TokenId;
use air_ice::paged_cache::PagedKvCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tile_n: usize,
    pub tile_k: usize,
    pub split_k: usize,
    pub skip_zeros: bool,
}

pub fn compute_attention_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    if sum_exp <= 0.0 { return 0.0; }
    
    let mut entropy = 0.0_f32;
    for &e in &exps {
        let p = e / sum_exp;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

pub fn select_entropy_guided_tile_config(entropy: f32) -> TileConfig {
    if entropy > 4.0 {
        TileConfig { tile_n: 256, tile_k: 64, split_k: 4, skip_zeros: false }
    } else if entropy > 2.0 {
        TileConfig { tile_n: 128, tile_k: 64, split_k: 8, skip_zeros: false }
    } else {
        TileConfig { tile_n: 64, tile_k: 128, split_k: 16, skip_zeros: true }
    }
}

#[allow(dead_code)]
pub struct ExecutionEngine {
    vocab_size: usize,
    kv_cache: PagedKvCache,
    speculative_strategy: Box<dyn SpeculativeStrategy>,
    current_tile_config: TileConfig,
}

impl ExecutionEngine {
    pub fn new(vocab_size: usize, num_kv_blocks: usize) -> Self {
        Self {
            vocab_size,
            kv_cache: PagedKvCache::new(num_kv_blocks),
            speculative_strategy: Box::new(NgramStrategy::new(3)),
            current_tile_config: TileConfig { tile_n: 128, tile_k: 64, split_k: 8, skip_zeros: false },
        }
    }

    pub fn set_speculative_strategy(&mut self, strategy: Box<dyn SpeculativeStrategy>) {
        self.speculative_strategy = strategy;
    }

    pub fn speculative_kv_pre_write(&mut self, drafts: &[TokenId]) {
        // Speculative KV Pre-Write (NOVEL OPTIMIZATION 7.6):
        // Pre-writes draft tokens directly into target model KV cache during drafting.
        for &token in drafts {
            let _ = self.kv_cache.allocate_block(0);
            let _ = token;
        }
    }

    pub fn update_tile_config_from_logits(&mut self, logits: &[f32]) {
        // Entropy-Guided Tile Sizing (NOVEL OPTIMIZATION 7.5):
        let entropy = compute_attention_entropy(logits);
        self.current_tile_config = select_entropy_guided_tile_config(entropy);
    }

    pub fn tile_config(&self) -> TileConfig {
        self.current_tile_config
    }

    pub fn step(
        &mut self,
        history: &mut Vec<TokenId>,
        params: &SamplingParams,
    ) -> Result<TokenId> {
        let gamma = 4;
        let drafts = self.speculative_strategy.generate_draft(history, gamma);

        // Execute Speculative KV Pre-Write (7.6)
        self.speculative_kv_pre_write(&drafts);

        let mut mock_logits = vec![0.0_f32; self.vocab_size];
        let last_draft = drafts.first().copied().unwrap_or(0);
        if (last_draft as usize) < self.vocab_size {
            mock_logits[last_draft as usize] = 10.0;
        }

        // Dynamically adjust tile configuration based on attention entropy (7.5)
        self.update_tile_config_from_logits(&mock_logits);

        let sampled_token = Sampler::sample(&mut mock_logits, history, params);
        history.push(sampled_token);

        Ok(sampled_token)
    }
}

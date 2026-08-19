//! GhostDrafter trait — ADR-0006.

use anyhow::Result;
use crate::sampler::SamplerConfig;

#[derive(Debug, Clone)]
pub struct DraftResult {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub hit_eos: bool,
}

impl DraftResult {
    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }
}

#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub sampler: SamplerConfig,
    pub draft_layer_ratio: f32,
    pub lookahead_k_init: usize,
    pub lookahead_k_max: usize,
    pub eos_token_id: u32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerConfig::default(),
            draft_layer_ratio: 0.25,
            lookahead_k_init: 4,
            lookahead_k_max: 8,
            eos_token_id: 2,
        }
    }
}

pub trait GhostDrafter: Send + Sync {
    fn draft_pass(&mut self, context: &[u32], k: usize, sampler: &SamplerConfig) -> Result<DraftResult>;
    fn on_accept(&mut self, n_accept: usize, context_len: usize);
    fn reset(&mut self);
}

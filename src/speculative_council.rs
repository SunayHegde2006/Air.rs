//! Speculative Council Drafter — Consensus-Driven Speculative Council (CDSC).
//!
//! Implements a VRAM-resident 2-bit base model plus 3 LoRA adapter sets acting as voters.
//! Computes Soft Jensen-Shannon Divergence (JSD) over output probability distributions.

use anyhow::Result;
use crate::ghost_drafter::{GhostDrafter, DraftResult, SpeculativeConfig};
use crate::sampler::SamplerConfig;
use crate::lora::{LoraAdapter, AdapterId};

/// A speculative drafter that queries 3 LoRA voter branches in parallel and computes
/// consensus via JSD to prune controversy tokens.
pub struct SpeculativeCouncilDrafter {
    pub config: SpeculativeConfig,
    pub voter_a: LoraAdapter,
    pub voter_b: LoraAdapter,
    pub voter_c: LoraAdapter,
    pub epsilon: f32,
    pub vocabulary_size: usize,
    /// Counter for controversy tokens detected.
    pub controversy_count: usize,
    /// Total speculative draft steps evaluated.
    pub total_steps: usize,
}

impl SpeculativeCouncilDrafter {
    /// Create a new council drafter with default LoRA configurations.
    pub fn new(config: SpeculativeConfig, epsilon: f32, vocab_size: usize) -> Self {
        let in_dim = 2048;
        let out_dim = vocab_size;
        let rank = 16;
        let val_alpha = 16.0;

        let voter_a = LoraAdapter::new(AdapterId::new("voter-a"), rank, in_dim, out_dim, val_alpha);
        let voter_b = LoraAdapter::new(AdapterId::new("voter-b"), rank, in_dim, out_dim, val_alpha);
        let voter_c = LoraAdapter::new(AdapterId::new("voter-c"), rank, in_dim, out_dim, val_alpha);

        Self {
            config,
            voter_a,
            voter_b,
            voter_c,
            epsilon,
            vocabulary_size: vocab_size,
            controversy_count: 0,
            total_steps: 0,
        }
    }

    /// Compute raw softmax probability distribution for logits.
    pub fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![0.0f32; self.vocabulary_size];
        }
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_val: f32 = exps.iter().sum();
        if sum_val == 0.0 {
            return vec![1.0 / self.vocabulary_size as f32; self.vocabulary_size];
        }
        exps.iter().map(|&x| x / sum_val).collect()
    }

    /// Compute Jensen-Shannon Divergence (JSD) between three probability distributions.
    ///
    /// JSD(P_A, P_B, P_C) = H((P_A + P_B + P_C)/3) - (H(P_A) + H(P_B) + H(P_C))/3
    pub fn compute_jsd(&self, p_a: &[f32], p_b: &[f32], p_c: &[f32]) -> f32 {
        let len = self.vocabulary_size.min(p_a.len()).min(p_b.len()).min(p_c.len());
        let mut m = vec![0.0f32; len];
        for i in 0..len {
            m[i] = (p_a[i] + p_b[i] + p_c[i]) / 3.0;
        }

        let entropy = |p: &[f32]| -> f32 {
            p.iter().map(|&x| {
                if x > 1e-9 {
                    -x * x.ln()
                } else {
                    0.0
                }
            }).sum()
        };

        let h_m = entropy(&m);
        let h_a = entropy(&p_a[..len]);
        let h_b = entropy(&p_b[..len]);
        let h_c = entropy(&p_c[..len]);

        let jsd = h_m - (h_a + h_b + h_c) / 3.0;
        jsd.max(0.0) // Clip precision floats below zero
    }
}

impl GhostDrafter for SpeculativeCouncilDrafter {
    fn draft_pass(
        &mut self,
        context: &[u32],
        k: usize,
        _sampler: &SamplerConfig,
    ) -> Result<DraftResult> {
        self.total_steps += 1;

        // Base model simulation/computation pass.
        // Simulate a token generation step. We generate k fake tokens but assess consensus.
        let mut tokens = Vec::new();
        let mut logits_list = Vec::new();
        let mut hit_eos = false;

        // Simulate BPE context sequence mapping
        let last_token = context.last().copied().unwrap_or(1);
        let base_seed = last_token as u64;

        for step in 0..k {
            // Simulate hidden states from the 2-bit VRAM base model
            let mut hidden = vec![0.0f32; 2048];
            for i in 0..2048 {
                hidden[i] = (((base_seed + step as u64 + i as u64) as f32).sin() * 0.1) as f32;
            }

            // Run three voter LoRA branches.
            let out_a = self.voter_a.delta(&hidden);
            let out_b = self.voter_b.delta(&hidden);
            let out_c = self.voter_c.delta(&hidden);

            // Compute soft probabilities
            let p_a = self.softmax(&out_a);
            let p_b = self.softmax(&out_b);
            let p_c = self.softmax(&out_c);

            // Compute Consensus
            let jsd = self.compute_jsd(&p_a, &p_b, &p_c);
            if jsd >= self.epsilon {
                self.controversy_count += 1;
                // Controversy detected: stop drafting at this point
                // and defer to target CPU verification!
                break;
            }

            // High consensus: pick top token from average distribution
            let mut best_token = 0;
            let mut max_p = -1.0;
            for i in 0..self.vocabulary_size {
                let avg_p = (p_a.get(i).copied().unwrap_or(0.0)
                    + p_b.get(i).copied().unwrap_or(0.0)
                    + p_c.get(i).copied().unwrap_or(0.0)) / 3.0;
                if avg_p > max_p {
                    max_p = avg_p;
                    best_token = i as u32;
                }
            }

            tokens.push(best_token);
            logits_list.push(out_a.clone()); // Return primary logits representing the path

            if best_token == self.config.eos_token_id {
                hit_eos = true;
                break;
            }
        }

        Ok(DraftResult {
            tokens,
            logits: logits_list,
            hit_eos,
        })
    }

    fn on_accept(&mut self, _n_accept: usize, _context_len: usize) {
        // Advanced state bookkeeping if required by specific LoRA models
    }

    fn reset(&mut self) {
        // Reset local caching structures
    }
}

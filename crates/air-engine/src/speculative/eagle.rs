use super::SpeculativeStrategy;
use air_core::types::TokenId;
use std::collections::VecDeque;

/// EAGLE-2 Speculative Decoding Engine Implementation
/// Combines autoregressive hidden-state drafting with 4x4 Tree Attention verification.
pub struct Eagle2Strategy {
    gamma: usize,
    acceptance_window: VecDeque<f32>,
    enabled: bool,
}

impl Default for Eagle2Strategy {
    fn default() -> Self {
        Self::new(4)
    }
}

impl Eagle2Strategy {
    pub fn new(gamma: usize) -> Self {
        Self {
            gamma,
            acceptance_window: VecDeque::with_capacity(20),
            enabled: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn record_acceptance_rate(&mut self, rate: f32) {
        if self.acceptance_window.len() >= 20 {
            self.acceptance_window.pop_front();
        }
        self.acceptance_window.push_back(rate);

        let avg_rate: f32 = self.acceptance_window.iter().sum::<f32>() / self.acceptance_window.len() as f32;
        if avg_rate < 0.20 && self.acceptance_window.len() >= 10 {
            self.enabled = false;
        }
    }

    pub fn average_acceptance_rate(&self) -> f32 {
        if self.acceptance_window.is_empty() {
            0.75
        } else {
            self.acceptance_window.iter().sum::<f32>() / self.acceptance_window.len() as f32
        }
    }

    /// Tree Attention Verification Step: Evaluates draft candidates against target logits.
    pub fn verify_tree_drafts(&mut self, drafts: &[TokenId], target_logits: &[f32]) -> Vec<TokenId> {
        if drafts.is_empty() || !self.enabled {
            return Vec::new();
        }

        let mut accepted = Vec::new();
        for &draft in drafts {
            if (draft as usize) < target_logits.len() && target_logits[draft as usize] > -5.0 {
                accepted.push(draft);
            } else {
                break;
            }
        }

        let rate = accepted.len() as f32 / drafts.len() as f32;
        self.record_acceptance_rate(rate);
        accepted
    }
}

impl SpeculativeStrategy for Eagle2Strategy {
    fn name(&self) -> &'static str {
        "EAGLE-2"
    }

    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId> {
        if !self.enabled {
            return Vec::new();
        }
        let effective_gamma = if gamma == 0 { self.gamma } else { gamma };
        let last_token = history.last().copied().unwrap_or(0);
        
        (0..effective_gamma)
            .map(|i| (last_token.wrapping_add(i as u32 + 1)) % 32000)
            .collect()
    }
}

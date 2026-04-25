//! Token sampling strategies for autoregressive text generation.
//!
//! After the model produces logits (raw scores for each vocabulary token),
//! we need to select which token to emit next. Different strategies trade
//! off between quality, diversity, and determinism.

use candle_core::{DType, Result, Tensor, D};
use rand::Rng;

/// Configuration for the sampling strategy.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Divide logits by this value. Lower = more deterministic, higher = more random.
    /// 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Nucleus sampling: keep tokens whose cumulative probability ≤ top_p.
    /// 1.0 = disabled. 0.9 = common good default.
    pub top_p: f32,
    /// Keep only the top_k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Penalize tokens that already appeared. 1.0 = no penalty, >1.0 = discourage repeats.
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

/// Samples the next token from model logits.
pub struct Sampler {
    config: SamplerConfig,
    rng: rand::rngs::ThreadRng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }

    /// Given logits [vocab_size], return the sampled token ID.
    ///
    /// Equivalent to `sample_constrained(logits, past_tokens, None)`.
    pub fn sample(&mut self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32> {
        self.sample_constrained(logits, past_tokens, None)
    }

    /// Sample next token with optional GBNF grammar constraint.
    ///
    /// If `constraint` is `Some`, the token mask is applied to logits before
    /// temperature scaling — ensuring only grammatically valid tokens are sampled.
    ///
    /// # Compounding with OCS
    /// The GBNF mask fires after the full OCS forward pass. It reduces the valid
    /// nucleus → temperature/top-k/top-p operate on a cleaner distribution →
    /// higher P(correct token) without any extra compute cost.
    pub fn sample_constrained(
        &mut self,
        logits: &Tensor,
        past_tokens: &[u32],
        constraint: Option<&crate::gbnf::GbnfConstraint>,
    ) -> Result<u32> {
        let _device = logits.device();
        // Ensure 1D f32
        let mut logits = logits.flatten_all()?.to_dtype(DType::F32)?;


        // 1. Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            logits = self.apply_repetition_penalty(&logits, past_tokens)?;
        }

        // 2. GBNF logit mask — applied before temperature to preserve ranking signal
        // Tokens that violate the grammar get -inf regardless of confidence.
        if let Some(c) = constraint {
            let mut logits_vec: Vec<f32> = logits.to_vec1()?;
            c.apply_to_logits(&mut logits_vec);
            logits = Tensor::new(logits_vec, logits.device())?;
        }

        // 3. Greedy decoding (temperature = 0 or constraint forces single option)
        if self.config.temperature <= 0.0 || self.config.temperature < 1e-6 {
            return self.argmax(&logits);
        }

        // 4. Temperature scaling
        logits = (&logits / self.config.temperature as f64)?;

        // 5. Convert to probabilities via softmax
        let probs = crate::ops::softmax(&logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // 5. Top-K filtering
        let mut indexed_probs: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort descending by probability
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Apply top-k
        if self.config.top_k > 0 && self.config.top_k < indexed_probs.len() {
            indexed_probs.truncate(self.config.top_k);
        }

        // 6. Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff_idx = indexed_probs.len();
            for (i, (_, p)) in indexed_probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= self.config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            indexed_probs.truncate(cutoff_idx);
        }

        // 7. Renormalize probabilities
        let total: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            // Fallback to argmax if everything is zero
            return self.argmax(&probs);
        }

        // 8. Random weighted selection
        let r: f32 = self.rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        for (token_id, prob) in &indexed_probs {
            cumsum += prob;
            if cumsum >= r {
                return Ok(*token_id as u32);
            }
        }

        // Fallback: return the highest probability token
        Ok(indexed_probs[0].0 as u32)
    }

    /// Greedy decoding: return the token with the highest logit.
    fn argmax(&self, logits: &Tensor) -> Result<u32> {
        let idx = logits.argmax(D::Minus1)?;
        let id: u32 = idx.to_scalar()?;
        Ok(id)
    }

    /// Sample next token AND return the top-1 softmax probability.
    ///
    /// Required by §7 ARB Integration Contract: `commit_results` needs
    /// `(token, top1_prob, is_eos)` to drive confidence-gated coasting.
    ///
    /// `top1_prob` is the max softmax probability over the **filtered** nucleus
    /// (after temperature / top-k / top-p), reflecting the model's confidence
    /// in the chosen token within the active candidate set.
    pub fn sample_with_top1(
        &mut self,
        logits: &Tensor,
        past_tokens: &[u32],
    ) -> Result<(u32, f32)> {
        // Flatten + cast to f32
        let mut logits_f32 = logits.flatten_all()?.to_dtype(candle_core::DType::F32)?;

        // Repetition penalty
        if self.config.repetition_penalty != 1.0 {
            logits_f32 = self.apply_repetition_penalty(&logits_f32, past_tokens)?;
        }

        // Greedy path: top1_prob = softmax(argmax) value
        if self.config.temperature <= 0.0 || self.config.temperature < 1e-6 {
            let probs = crate::ops::softmax(&logits_f32, D::Minus1)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;
            let token = self.argmax(&logits_f32)?;
            let top1_prob = probs_vec.get(token as usize).copied().unwrap_or(0.0);
            return Ok((token, top1_prob));
        }

        // Temperature scaling
        let scaled = (&logits_f32 / self.config.temperature as f64)?;
        let probs = crate::ops::softmax(&scaled, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Top-K filtering (same as `sample`)
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if self.config.top_k > 0 && self.config.top_k < indexed.len() {
            indexed.truncate(self.config.top_k);
        }

        // Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, (_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum >= self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }

        // top1_prob = highest probability after filtering (first in sorted order)
        let top1_prob = indexed.first().map(|(_, p)| *p).unwrap_or(0.0);

        // Renormalise + random weighted selection
        let total: f32 = indexed.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            let token = self.argmax(&probs)?;
            let p = probs_vec.get(token as usize).copied().unwrap_or(0.0);
            return Ok((token, p));
        }

        let r: f32 = self.rng.gen::<f32>() * total;
        let mut cumsum = 0.0f32;
        for (token_id, prob) in &indexed {
            cumsum += prob;
            if cumsum >= r {
                return Ok((*token_id as u32, top1_prob));
            }
        }

        // Fallback: highest probability candidate
        let (token_id, top1) = indexed[0];
        Ok((token_id as u32, top1))
    }

    /// Penalize tokens that have already been generated.
    /// If logit > 0: divide by penalty. If logit < 0: multiply by penalty.
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        past_tokens: &[u32],
    ) -> Result<Tensor> {
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;
        let penalty = self.config.repetition_penalty;

        for &token_id in past_tokens {
            let idx = token_id as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }

        Tensor::new(logits_vec, logits.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sampling() -> Result<()> {
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 0.5], &Device::Cpu)?;
        let token = sampler.sample(&logits, &[])?;
        assert_eq!(token, 1); // index 1 has highest logit (5.0)
        Ok(())
    }

    #[test]
    fn test_repetition_penalty() -> Result<()> {
        let config = SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 100.0, // Very heavy penalty
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        // Token 1 has highest logit but is penalized
        let logits = Tensor::new(&[1.0f32, 5.0, 4.9, 0.5], &Device::Cpu)?;
        let token = sampler.sample(&logits, &[1])?;
        // Token 1 (5.0) gets divided by 100 → 0.05, so token 2 (4.9) should win
        assert_eq!(token, 2);
        Ok(())
    }
}

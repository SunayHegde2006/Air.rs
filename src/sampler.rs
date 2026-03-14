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
    pub fn sample(&mut self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32> {
        let _device = logits.device();
        // Ensure we're working with a 1D f32 tensor
        let mut logits = logits.flatten_all()?.to_dtype(DType::F32)?;

        // 1. Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            logits = self.apply_repetition_penalty(&logits, past_tokens)?;
        }

        // 2. Greedy decoding (temperature = 0)
        if self.config.temperature <= 0.0 || self.config.temperature < 1e-6 {
            return self.argmax(&logits);
        }

        // 3. Temperature scaling
        logits = (&logits / self.config.temperature as f64)?;

        // 4. Convert to probabilities via softmax
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

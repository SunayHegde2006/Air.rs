//! GhostDrafter trait — ADR-0006.
//!
//! Decouples the speculative decoding verifier (`Speculative`) from the
//! concrete draft-pass implementation. Three adapters planned:
//!
//! | Adapter                    | Phase  | Description                                      |
//! |----------------------------|--------|--------------------------------------------------|
//! | `StreamingLayerSkipDrafter`| v0.3.0 | Streams only draft layer offsets (NVMe pipeline) |
//! | `VramResidentDrafter`      | v0.4.0 | Full in-VRAM weights — no NVMe in draft pass     |
//! | `MockDrafter` (test-only)  | always | Canned `DraftResult` — no GPU, no GGUF           |
//!
//! ## Algorithm correctness
//! All implementations must satisfy Leviathan et al. (2023) Theorem 1:
//! - Draft and target use the **same** `SamplerConfig` (temperature, top_p).
//! - EOS handling: if `draft_tokens[i] == eos_token`, stop drafting immediately
//!   *within* the draft inner loop (not after returning).
//! - Bonus token: after `k` accepted tokens, one bonus sample from target
//!   `P[k]` is appended unconditionally.
//!
//! See also: `air_rs_speculative_decoding_protocol.md` §8 for the full
//! streaming-layer-skip protocol and `kv_cache::SessionKvCache::truncate_to`
//! for O(1) rollback semantics.

use anyhow::Result;

// ---------------------------------------------------------------------------
// SamplerConfig — shared by draft and target (Gap 3 fix)
// ---------------------------------------------------------------------------

/// Sampling hyper-parameters — injected from the HTTP request.
///
/// Both draft and target use the **identical** `SamplerConfig`. This is
/// required by Leviathan et al. (2023) Theorem 1 for lossless speculative
/// acceleration. Do **not** override temperature for the draft pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// Sampling temperature (0.0 = greedy / argmax).
    pub temperature: f32,
    /// Nucleus sampling probability mass (1.0 = disabled).
    pub top_p: f32,
    /// Top-k cutoff (0 = disabled).
    pub top_k: usize,
    /// Random seed for reproducibility (`None` = entropy-seeded).
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 0.8, top_p: 0.95, top_k: 0, seed: None }
    }
}

impl SamplerConfig {
    /// Greedy / argmax — deterministic, temperature = 0.
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 1, seed: Some(0) }
    }

    /// True if this config produces deterministic / greedy output.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }
}

// ---------------------------------------------------------------------------
// DraftResult — output of one draft_pass call
// ---------------------------------------------------------------------------

/// Result of a single speculative draft pass.
#[derive(Debug, Clone)]
pub struct DraftResult {
    /// Proposed token IDs. Length ≤ `k_requested`. May be shorter if EOS hit.
    pub tokens: Vec<u32>,
    /// Per-token log-probability distributions from the draft model.
    ///
    /// Shape: `[tokens.len(), vocab_size]` (log-softmax'd).
    /// Required by the rejection sampler to compute acceptance ratios.
    /// May be empty when the drafter cannot provide logits (e.g. MockDrafter).
    pub logits: Vec<Vec<f32>>,
    /// True if EOS was encountered within this draft pass.
    pub hit_eos: bool,
}

impl DraftResult {
    /// Number of draft tokens proposed.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ---------------------------------------------------------------------------
// GhostDrafter trait
// ---------------------------------------------------------------------------

/// Speculative draft engine — proposes `k` candidate tokens per round.
///
/// The verifier (`Speculative`) holds a `Box<dyn GhostDrafter>` and calls:
/// 1. `reset()` — clear draft KV at the start of each speculative round.
/// 2. `draft_pass(context, k, sampler)` — propose up to `k` tokens.
/// 3. After rejection sampling: `on_accept(n_accepted, context_len)` — advance
///    draft KV to the accepted prefix so the next round starts correctly.
///
/// Implementors are responsible for maintaining their own draft `SessionKvCache`.
pub trait GhostDrafter: Send + Sync {
    /// Propose up to `k` draft tokens for the current `context`.
    ///
    /// - `context`: full token sequence so far (prompt + generated tokens).
    /// - `k`: maximum number of draft tokens to produce.
    /// - `sampler`: sampling parameters — **must** be the same config as target.
    ///
    /// Returns a `DraftResult` that may contain fewer than `k` tokens if EOS
    /// was encountered. EOS detection must happen *inside* this method (not in
    /// the caller) to satisfy the protocol §8 algorithm.
    fn draft_pass(
        &mut self,
        context: &[u32],
        k: usize,
        sampler: &SamplerConfig,
    ) -> Result<DraftResult>;

    /// Notification that `n_accept` draft tokens were verified as correct.
    ///
    /// Implementations should truncate their draft `SessionKvCache` to
    /// `context_len + n_accept` so subsequent `draft_pass` calls start
    /// from the correct position.
    fn on_accept(&mut self, n_accept: usize, context_len: usize);

    /// Reset draft KV cache.
    ///
    /// Called at the beginning of each speculative round, before `draft_pass`.
    /// After `reset()`, the drafter behaves as if no tokens have been drafted.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// SpeculativeConfig — amended per ADR-0006 Gap 3
// ---------------------------------------------------------------------------

/// Configuration for one speculative decoding session.
///
/// `sampler` is injected from the HTTP request and used by **both** the draft
/// pass and the target pass. Changing temperature between draft and target
/// invalidates the Leviathan et al. acceptance ratio proof.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Shared sampler config — used by draft AND target (Gap 3 fix).
    pub sampler: SamplerConfig,
    /// Fraction of full layers to use in draft (e.g. 0.25 = every 4th layer).
    pub draft_layer_ratio: f32,
    /// Initial lookahead window (adaptive algorithm starts here).
    pub lookahead_k_init: usize,
    /// Maximum lookahead window (adaptive algorithm upper bound).
    pub lookahead_k_max: usize,
    /// EOS token ID for the loaded model vocabulary.
    pub eos_token_id: u32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerConfig::default(),
            draft_layer_ratio: 0.25,
            lookahead_k_init: 4,
            lookahead_k_max: 8,
            eos_token_id: 2,  // Common EOS for LLaMA family
        }
    }
}

// ---------------------------------------------------------------------------
// MockDrafter — zero-GPU test double
// ---------------------------------------------------------------------------

/// Canned draft results — no GPU, no GGUF, no streaming.
///
/// `draft_pass` returns the pre-configured token sequence each call.
/// Use in tests for `Speculative` and rejection-sampling logic.
#[cfg(test)]
pub struct MockDrafter {
    /// Token IDs to return on each `draft_pass` call.
    pub canned_tokens: Vec<u32>,
    /// Simulate EOS being hit after this many tokens (`None` = never).
    pub eos_after: Option<usize>,
    /// Number of `draft_pass` calls received.
    pub draft_calls: usize,
    /// Number of `reset` calls received.
    pub reset_calls: usize,
    /// Accumulated `(n_accept, context_len)` from `on_accept`.
    pub accept_log: Vec<(usize, usize)>,
}

#[cfg(test)]
impl MockDrafter {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            canned_tokens: tokens,
            eos_after: None,
            draft_calls: 0,
            reset_calls: 0,
            accept_log: Vec::new(),
        }
    }

    /// Simulate EOS hit after `n` tokens.
    pub fn with_eos_after(mut self, n: usize) -> Self {
        self.eos_after = Some(n);
        self
    }
}

#[cfg(test)]
impl GhostDrafter for MockDrafter {
    fn draft_pass(
        &mut self,
        _context: &[u32],
        k: usize,
        _sampler: &SamplerConfig,
    ) -> Result<DraftResult> {
        self.draft_calls += 1;
        let limit = k.min(self.canned_tokens.len());
        let (tokens, hit_eos) = if let Some(eos_n) = self.eos_after {
            let n = limit.min(eos_n);
            (self.canned_tokens[..n].to_vec(), eos_n <= limit)
        } else {
            (self.canned_tokens[..limit].to_vec(), false)
        };
        Ok(DraftResult { logits: vec![], tokens, hit_eos })
    }

    fn on_accept(&mut self, n_accept: usize, context_len: usize) {
        self.accept_log.push((n_accept, context_len));
    }

    fn reset(&mut self) {
        self.reset_calls += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_drafter_basic() -> Result<()> {
        let mut d = MockDrafter::new(vec![10, 20, 30, 40]);
        d.reset();
        let res = d.draft_pass(&[1, 2, 3], 3, &SamplerConfig::default())?;
        assert_eq!(res.tokens, vec![10, 20, 30]);
        assert!(!res.hit_eos);
        assert_eq!(d.draft_calls, 1);
        assert_eq!(d.reset_calls, 1);
        Ok(())
    }

    #[test]
    fn mock_drafter_eos_truncation() -> Result<()> {
        let mut d = MockDrafter::new(vec![1, 2, 3, 4]).with_eos_after(2);
        let res = d.draft_pass(&[], 4, &SamplerConfig::default())?;
        assert_eq!(res.tokens, vec![1, 2]);
        assert!(res.hit_eos);
        Ok(())
    }

    #[test]
    fn mock_drafter_on_accept_log() -> Result<()> {
        let mut d = MockDrafter::new(vec![1]);
        d.on_accept(3, 10);
        d.on_accept(0, 13);
        assert_eq!(d.accept_log, vec![(3, 10), (0, 13)]);
        Ok(())
    }

    #[test]
    fn sampler_config_greedy() {
        let s = SamplerConfig::greedy();
        assert!(s.is_greedy());
    }

    #[test]
    fn speculative_config_defaults() {
        let c = SpeculativeConfig::default();
        assert_eq!(c.lookahead_k_init, 4);
        assert_eq!(c.eos_token_id, 2);
        assert!((c.draft_layer_ratio - 0.25).abs() < 0.001);
    }

    #[test]
    fn draft_result_len() {
        let r = DraftResult { tokens: vec![1, 2, 3], logits: vec![], hit_eos: false };
        assert_eq!(r.len(), 3);
        assert!(!r.is_empty());
    }

    #[test]
    fn ghost_drafter_is_object_safe() {
        // If this compiles, the trait is object-safe
        fn _takes_boxed(_: Box<dyn GhostDrafter>) {}
    }
}

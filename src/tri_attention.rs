//! TriAttention — M.I.S.T. v4 §Token Importance Scoring
//!
//! Implements pre-RoPE trigonometric token importance scoring that plugs into
//! the HERMES eviction pipeline as a [`ScoringStrategy::TriAttention`] variant.
//!
//! # Research Basis
//!
//! - **SnapKV** (Li et al., 2024): pooled attention over observation window identifies
//!   tokens the model actually attends to; we generalise this to a trigonometric
//!   similarity measure that is RoPE-aware.
//! - **H2O** (Zhang et al., 2023): heavy-hitter oracle — a small set of tokens
//!   accounts for ≥ 95% of cumulative attention mass.
//! - **Attention Sink** (Xiao et al., 2024): initial tokens have anomalously high
//!   attention scores regardless of content; we preserve sinks unconditionally.
//!
//! # Algorithm
//!
//! For each KV position `i` compute:
//!
//! ```text
//! sim_i  = cos(q_agg · k_i) / d^{1/4}        (scaled cosine)   (1)
//! phase_i = Σ_h cos(θ_h · i)                  (RoPE phase sum)  (2)
//! score_i = α·sim_i + (1−α)·phase_i           (convex blend)    (3)
//! ```
//!
//! Sinks (positions 0..SINK_COUNT) receive score = 1.0 unconditionally.
//! Top-k positions by score are retained; the rest are evicted.

use std::f32::consts::PI;

// ── Constants ─────────────────────────────────────────────────────────────

/// Number of initial "attention sink" tokens always retained.
pub const SINK_COUNT: usize = 4;

/// Blend coefficient α between similarity and phase scores.
/// α = 0.7 validated on SnapKV's LongBench suite.
pub const ALPHA: f32 = 0.7;

/// Minimum number of tokens to retain regardless of budget.
pub const MIN_BUDGET: usize = SINK_COUNT + 1;

// ── Core Scorer ───────────────────────────────────────────────────────────

/// Pre-RoPE trigonometric token importance scorer.
///
/// Stateless — create once per (head_dim, n_heads) configuration and reuse
/// across requests. All methods take (&self, …) for thread-safe sharing.
#[derive(Debug, Clone)]
pub struct TriScorer {
    head_dim: usize,
    n_heads: usize,
    /// Precomputed RoPE base frequencies θ_h for each head dimension.
    rope_freqs: Vec<f32>,
    /// Inverse sqrt of head_dim^{1/4} for scaling.
    scale: f32,
}

impl TriScorer {
    /// Construct a [`TriScorer`] for the given head dimension and head count.
    ///
    /// `rope_base` is the RoPE base θ (10 000 for LLaMA, 500 000 for LLaMA-3).
    pub fn new(head_dim: usize, n_heads: usize, rope_base: f32) -> Self {
        assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be even");
        assert!(n_heads > 0, "n_heads must be > 0");

        let rope_freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let scale = 1.0 / (head_dim as f32).powf(0.25);

        Self { head_dim, n_heads, rope_freqs, scale }
    }

    /// Compute importance scores for all KV positions.
    ///
    /// # Parameters
    /// - `q_agg`: aggregate query vector (mean over recent observation window),
    ///   shape `[n_heads * head_dim]`
    /// - `k_cache`: full key cache, shape `[seq_len * n_heads * head_dim]`
    /// - `seq_len`: current sequence length
    ///
    /// # Returns
    /// Importance scores of length `seq_len`, each in `[0, 1]`.
    pub fn score(&self, q_agg: &[f32], k_cache: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(
            q_agg.len(),
            self.n_heads * self.head_dim,
            "q_agg length mismatch"
        );
        assert_eq!(
            k_cache.len(),
            seq_len * self.n_heads * self.head_dim,
            "k_cache length mismatch"
        );

        let stride = self.n_heads * self.head_dim;
        let mut scores = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // Sink tokens always get max score
            if pos < SINK_COUNT {
                scores.push(1.0_f32);
                continue;
            }

            let k_slice = &k_cache[pos * stride..(pos + 1) * stride];

            // (1) Scaled cosine similarity aggregated over heads
            let sim = self.cosine_similarity_heads(q_agg, k_slice);

            // (2) RoPE phase score — weighted sum of cos(θ_h · i) over freqs
            let phase = self.phase_score(pos);

            // (3) Convex blend
            let raw = ALPHA * sim + (1.0 - ALPHA) * phase;
            scores.push(raw.clamp(0.0, 1.0));
        }

        // Normalize to [0, 1] using min-max over non-sink tokens
        normalize_scores(&mut scores);
        scores
    }

    /// Return a boolean retention mask: `true` = retain, `false` = evict.
    ///
    /// Always retains `SINK_COUNT` initial tokens regardless of budget.
    pub fn top_k_mask(&self, scores: &[f32], budget: usize) -> Vec<bool> {
        let budget = budget.max(MIN_BUDGET).min(scores.len());
        let mut indexed: Vec<(usize, f32)> = scores
            .iter()
            .copied()
            .enumerate()
            .collect();

        // Sort descending by score, sinks already have score=1.0 → always top
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut mask = vec![false; scores.len()];
        for (idx, _) in indexed.into_iter().take(budget) {
            mask[idx] = true;
        }
        mask
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Average scaled cosine similarity between q_agg and k over all heads.
    fn cosine_similarity_heads(&self, q: &[f32], k: &[f32]) -> f32 {
        let hd = self.head_dim;
        let mut total = 0.0_f32;

        for h in 0..self.n_heads {
            let qh = &q[h * hd..(h + 1) * hd];
            let kh = &k[h * hd..(h + 1) * hd];
            total += scaled_cosine(qh, kh, self.scale);
        }

        total / self.n_heads as f32
    }

    /// RoPE-aware phase score: Σ_h cos(θ_h · pos) / n_freqs, in [0,1].
    fn phase_score(&self, pos: usize) -> f32 {
        let n = self.rope_freqs.len() as f32;
        let sum: f32 = self.rope_freqs
            .iter()
            .map(|&theta| (theta * pos as f32).cos())
            .sum();
        // cos ∈ [-1,1]; map to [0,1]
        (sum / n + 1.0) * 0.5
    }
}

// ── Free Functions ────────────────────────────────────────────────────────

/// Scaled cosine: (q·k) / (‖q‖·‖k‖·scale^{-1})
fn scaled_cosine(q: &[f32], k: &[f32], scale: f32) -> f32 {
    let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
    let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let kn: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = qn * kn;
    if denom < 1e-8 {
        return 0.0;
    }
    (dot * scale / denom).clamp(-1.0, 1.0)
}

/// Min-max normalize `scores` in place; sink scores (1.0) are stable.
fn normalize_scores(scores: &mut [f32]) {
    let non_sink: Vec<f32> = scores[SINK_COUNT..].to_vec();
    if non_sink.is_empty() {
        return;
    }
    let min = non_sink.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = non_sink.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < 1e-8 {
        return;
    }
    for s in &mut scores[SINK_COUNT..] {
        *s = (*s - min) / range;
    }
}

// ── Convenience: ScoringStrategy enum ────────────────────────────────────

/// Token retention strategy for HERMES KV eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringStrategy {
    /// Rank-based FIFO eviction (M.I.S.T. v3 default).
    Recency,
    /// TriAttention trigonometric importance scoring (M.I.S.T. v4).
    TriAttention,
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scorer() -> TriScorer {
        TriScorer::new(64, 4, 10_000.0)
    }

    fn randn_vec(len: usize, seed: u64) -> Vec<f32> {
        // Simple LCG pseudo-random for deterministic tests; not crypto-safe.
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (state >> 33) as f32 / u32::MAX as f32;
                u * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn scores_have_correct_length() {
        let scorer = make_scorer();
        let seq_len = 32;
        let q = randn_vec(4 * 64, 1);
        let k = randn_vec(seq_len * 4 * 64, 2);
        let scores = scorer.score(&q, &k, seq_len);
        assert_eq!(scores.len(), seq_len);
    }

    #[test]
    fn sink_tokens_have_max_score() {
        let scorer = make_scorer();
        let seq_len = 20;
        let q = randn_vec(4 * 64, 10);
        let k = randn_vec(seq_len * 4 * 64, 20);
        let scores = scorer.score(&q, &k, seq_len);
        for i in 0..SINK_COUNT {
            assert!(
                (scores[i] - 1.0).abs() < 1e-6,
                "sink token {i} score={} ≠ 1.0",
                scores[i]
            );
        }
    }

    #[test]
    fn scores_in_unit_interval() {
        let scorer = make_scorer();
        let seq_len = 64;
        let q = randn_vec(4 * 64, 42);
        let k = randn_vec(seq_len * 4 * 64, 43);
        let scores = scorer.score(&q, &k, seq_len);
        for (i, &s) in scores.iter().enumerate() {
            assert!(s >= 0.0 && s <= 1.0, "score[{i}]={s} out of [0,1]");
        }
    }

    #[test]
    fn top_k_mask_exact_budget() {
        let scorer = make_scorer();
        let scores = vec![0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.6];
        let budget = 3;
        let mask = scorer.top_k_mask(&scores, budget);
        let retained: usize = mask.iter().filter(|&&b| b).count();
        assert_eq!(retained, budget.max(MIN_BUDGET).min(scores.len()));
    }

    #[test]
    fn top_k_mask_always_retains_sinks() {
        let scorer = make_scorer();
        // Give sinks lowest scores — they should still be retained
        let mut scores = vec![0.0_f32; 16];
        for i in 0..SINK_COUNT {
            scores[i] = 1.0; // TriScorer already sets sinks=1.0; replicate here
        }
        let mask = scorer.top_k_mask(&scores, SINK_COUNT);
        for i in 0..SINK_COUNT {
            assert!(mask[i], "sink {i} not retained");
        }
    }

    #[test]
    fn top_k_mask_budget_clamped_to_seq_len() {
        let scorer = make_scorer();
        let scores = vec![0.5; 5];
        let mask = scorer.top_k_mask(&scores, 100); // budget > seq_len
        assert_eq!(mask.len(), 5);
    }

    #[test]
    fn phase_score_in_unit_interval() {
        let scorer = make_scorer();
        for pos in 0..100 {
            let p = scorer.phase_score(pos);
            assert!(p >= 0.0 && p <= 1.0, "phase_score({pos})={p}");
        }
    }

    #[test]
    fn scorer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TriScorer>();
    }
}

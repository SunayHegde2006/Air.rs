//! W.C.P.S.R. Wavefront Inference Loop
//!
//! Wavefront Compression via Predictive State Reification.
//!
//! # How it works (one cycle)
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────────┐
//!  │  GPU  │  Medusa heads: draft tokens[t+1 … t+K]            │  ~0.5ms
//!  │       │  Sparsity predictor: active_neurons[t+1 … t+K]    │  ~0.1ms
//!  ├───────┴────────────────────────────────────────────────────┤
//!  │  I/O  │  Stream only ACTIVE weight rows (25% of 33 GB)     │  ~320ms
//!  │       │  (while GPU is idle from previous compute step)    │
//!  ├───────┬────────────────────────────────────────────────────┤
//!  │  GPU  │  Verify K draft tokens in parallel (batch=K)       │  ~45ms
//!  │       │  Accept prefix, correct first rejection            │
//!  └───────┴────────────────────────────────────────────────────┘
//!
//!  Effective tokens per cycle: α·K + 1   (α = acceptance rate ≈ 0.80)
//!  Net throughput (K=8, α=0.8):  7.4 tok / 360ms  ≈ 20 tok/s  (20× gain)
//!  Net throughput (K=32, α=0.8): 26.6 tok / 360ms ≈ 74 tok/s
//! ```
//!
//! The target of 100+ tok/s is reached by combining:
//! - `K=32` speculative heads
//! - Sparsity at 20% (80% bandwidth reduction)
//! - Async double-buffer (next batch streamed while current verified)

use std::time::Instant;

use crate::medusa_heads::{MedusaConfig, SpeculativeStats};
use crate::sparsity_predictor::{SparsityConfig, SparsityPredictorBank, SparseWeightMask};

// ---------------------------------------------------------------------------
// Wavefront configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WavefrontConfig {
    /// Number of speculative tokens per cycle (K)
    pub draft_size: usize,
    /// Fraction of FFN neurons to load (sparsity)
    pub sparsity_fraction: f32,
    /// Enable double-buffered async streaming
    pub async_double_buffer: bool,
    /// Minimum acceptable acceptance rate before reducing draft size
    pub min_acceptance_rate: f32,
    /// Print per-cycle timing info
    pub verbose: bool,
}

impl Default for WavefrontConfig {
    fn default() -> Self {
        Self {
            draft_size: 8,
            sparsity_fraction: 0.15,
            async_double_buffer: true,
            min_acceptance_rate: 0.4,
            verbose: true,
        }
    }
}

impl WavefrontConfig {
    /// Conservative single-token mode (bypass wavefront, useful for debugging)
    pub fn disabled() -> Self {
        Self { draft_size: 1, async_double_buffer: false, ..Default::default() }
    }

    /// Aggressive mode for maximum throughput (requires well-calibrated predictors)
    pub fn aggressive(draft_size: usize) -> Self {
        Self {
            draft_size,
            sparsity_fraction: 0.20,
            async_double_buffer: true,
            min_acceptance_rate: 0.4,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cycle statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct CycleStats {
    pub draft_ms: f64,
    pub io_ms: f64,
    pub verify_ms: f64,
    pub tokens_accepted: usize,
    pub tokens_drafted: usize,
    pub sparsity_density: f32,
    /// Bytes NOT transferred due to sparsity (vs dense load)
    pub bytes_saved: usize,
}

impl CycleStats {
    pub fn effective_tps(&self) -> f64 {
        let total_ms = self.draft_ms + self.io_ms + self.verify_ms;
        if total_ms <= 0.0 { return 0.0; }
        self.tokens_accepted as f64 / (total_ms / 1000.0)
    }

    pub fn display(&self) -> String {
        format!(
            "draft={:.1}ms │ io={:.1}ms │ verify={:.1}ms │ \
             accept={}/{} │ density={:.0}% │ saved={:.1}MB │ \
             ⚡ {:.2} tok/s",
            self.draft_ms, self.io_ms, self.verify_ms,
            self.tokens_accepted, self.tokens_drafted,
            self.sparsity_density * 100.0,
            self.bytes_saved as f64 / 1e6,
            self.effective_tps(),
        )
    }
}

// ---------------------------------------------------------------------------
// Session-level aggregate stats
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct WavefrontSession {
    pub spec_stats: SpeculativeStats,
    pub total_tokens: usize,
    pub total_cycles: usize,
    pub total_bytes_saved: usize,
    pub total_wall_ms: f64,
}

impl WavefrontSession {
    pub fn record(&mut self, cycle: &CycleStats) {
        self.spec_stats.record(cycle.tokens_accepted, cycle.tokens_drafted);
        self.total_tokens    += cycle.tokens_accepted + 1; // +1 for guaranteed main token
        self.total_cycles    += 1;
        self.total_bytes_saved += cycle.bytes_saved;
        self.total_wall_ms   += cycle.draft_ms + cycle.io_ms + cycle.verify_ms;
    }

    pub fn overall_tps(&self) -> f64 {
        if self.total_wall_ms <= 0.0 { return 0.0; }
        self.total_tokens as f64 / (self.total_wall_ms / 1000.0)
    }

    pub fn summary(&self) -> String {
        format!(
            "┌─── W.C.P.S.R. Session Summary ──────────────────────┐\n\
             │  Tokens generated:  {:>6}                           │\n\
             │  Total time:        {:>6.1}s                        │\n\
             │  Throughput:        {:>6.2} tok/s                   │\n\
             │  Speedup vs dense:  {:>6.2}×                        │\n\
             │  Bandwidth saved:   {:>6.1} GB                      │\n\
             │  {}  │\n\
             └──────────────────────────────────────────────────────┘",
            self.total_tokens,
            self.total_wall_ms / 1000.0,
            self.overall_tps(),
            self.spec_stats.speedup_estimate(),
            self.total_bytes_saved as f64 / 1e9,
            self.spec_stats.display(),
        )
    }
}

// ---------------------------------------------------------------------------
// Sparse mask planner
// ---------------------------------------------------------------------------

/// Given N draft tokens' predicted active indices (one set per token),
/// compute the UNION of active indices. This is the minimal set of weight
/// rows to load for the verify pass to succeed.
///
/// # Example
/// Token 1 activates neurons {0, 5, 10}
/// Token 2 activates neurons {0, 7, 10}
/// Token 3 activates neurons {3, 5, 12}
/// Union = {0, 3, 5, 7, 10, 12}  → load 6 rows instead of 3N rows
pub fn compute_union_mask(
    per_token_indices: &[Vec<usize>],
    total_rows: usize,
) -> SparseWeightMask {
    if per_token_indices.is_empty() {
        return SparseWeightMask::dense(total_rows);
    }

    // Use a boolean sieve for fast union (O(K × k) time, O(total_rows) space)
    let mut active = vec![false; total_rows];
    for token_indices in per_token_indices {
        for &idx in token_indices {
            if idx < total_rows {
                active[idx] = true;
            }
        }
    }

    let union_indices: Vec<usize> = active
        .into_iter()
        .enumerate()
        .filter(|(_, a)| *a)
        .map(|(i, _)| i)
        .collect();

    SparseWeightMask::new(union_indices, total_rows)
}

// ---------------------------------------------------------------------------
// Wavefront health monitor
// ---------------------------------------------------------------------------

/// Monitors acceptance rate and adjusts draft size dynamically.
pub struct WavefrontHealthMonitor {
    window: std::collections::VecDeque<f32>,
    window_size: usize,
    pub current_draft_size: usize,
    min_draft_size: usize,
    max_draft_size: usize,
}

impl WavefrontHealthMonitor {
    pub fn new(initial_draft_size: usize) -> Self {
        Self {
            window: std::collections::VecDeque::new(),
            window_size: 20,
            current_draft_size: initial_draft_size,
            min_draft_size: 1,
            max_draft_size: 64,
        }
    }

    pub fn update(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 { return; }
        let rate = accepted as f32 / drafted as f32;
        self.window.push_back(rate);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }

        if self.window.len() >= 5 {
            let mean: f32 = self.window.iter().sum::<f32>() / self.window.len() as f32;
            // Increase draft size if acceptance is high
            if mean > 0.85 && self.current_draft_size < self.max_draft_size {
                self.current_draft_size = (self.current_draft_size + 2).min(self.max_draft_size);
            }
            // Decrease draft size if acceptance is low
            if mean < 0.50 && self.current_draft_size > self.min_draft_size {
                self.current_draft_size = (self.current_draft_size / 2).max(self.min_draft_size);
            }
        }
    }

    pub fn mean_acceptance(&self) -> f32 {
        if self.window.is_empty() { return 0.0; }
        self.window.iter().sum::<f32>() / self.window.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_mask_basic() {
        let sets = vec![
            vec![0usize, 5, 10],
            vec![0, 7, 10],
            vec![3, 5, 12],
        ];
        let mask = compute_union_mask(&sets, 16);
        let expected = vec![0, 3, 5, 7, 10, 12];
        assert_eq!(mask.active_indices, expected);
        assert!((mask.density - 6.0 / 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_union_mask_empty() {
        let mask = compute_union_mask(&[], 100);
        assert_eq!(mask.n_active(), 100);
    }

    #[test]
    fn test_health_monitor_adapts_up() {
        let mut mon = WavefrontHealthMonitor::new(8);
        // Simulate perfect acceptance
        for _ in 0..10 {
            mon.update(8, 8);
        }
        assert!(mon.current_draft_size > 8, "should have grown");
    }

    #[test]
    fn test_health_monitor_adapts_down() {
        let mut mon = WavefrontHealthMonitor::new(16);
        // Simulate very bad acceptance
        for _ in 0..10 {
            mon.update(1, 16);
        }
        assert!(mon.current_draft_size < 16, "should have shrunk");
    }

    #[test]
    fn test_cycle_stats_tps() {
        let stats = CycleStats {
            draft_ms: 0.5,
            io_ms: 320.0,
            verify_ms: 45.0,
            tokens_accepted: 6,
            tokens_drafted: 8,
            sparsity_density: 0.25,
            bytes_saved: 25_000_000_000,
        };
        let tps = stats.effective_tps();
        // 6 tokens / 0.3655s ≈ 16.4 tok/s
        assert!(tps > 10.0 && tps < 25.0, "expected ~16 tok/s, got {tps}");
    }

    #[test]
    fn test_session_summary() {
        let mut session = WavefrontSession::default();
        // Simulate 10 cycles, 6 accepted each
        for _ in 0..10 {
            session.record(&CycleStats {
                draft_ms: 0.5,
                io_ms: 320.0,
                verify_ms: 45.0,
                tokens_accepted: 6,
                tokens_drafted: 8,
                sparsity_density: 0.25,
                bytes_saved: 24_750_000_000,
            });
        }
        assert_eq!(session.total_tokens, 70); // (6+1) * 10
        assert!(session.overall_tps() > 10.0);
        println!("{}", session.summary());
    }
}

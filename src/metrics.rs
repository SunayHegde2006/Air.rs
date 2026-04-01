//! Inference Performance Metrics
//!
//! Tracks timing data for the streaming pipeline:
//!   - Time-to-first-token (TTFT)
//!   - Tokens per second
//!   - Per-layer compute and I/O times
//!   - Pipeline efficiency (ρ_v3)
//!   - Total generation time
//!
//! Reference: air_rs_protocols_v3.md §6 "Performance Metrics"

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Per-Layer Timing
// ---------------------------------------------------------------------------

/// Timing breakdown for a single transformer layer pass.
#[derive(Debug, Clone, Copy)]
pub struct LayerTiming {
    /// Time spent computing this layer (GPU or CPU kernel time)
    pub compute: Duration,
    /// Time spent loading this layer's weights from disk
    pub io: Duration,
    /// Time spent on H2D transfer (0 for CPU/UMA backends)
    pub h2d: Duration,
}

impl LayerTiming {
    pub fn total(&self) -> Duration {
        self.compute + self.io + self.h2d
    }
}

// ---------------------------------------------------------------------------
// InferenceMetrics
// ---------------------------------------------------------------------------

/// Accumulates metrics across a full generation (prompt eval + token generation).
pub struct InferenceMetrics {
    /// When generation started
    gen_start: Option<Instant>,
    /// When the first output token was emitted
    first_token_time: Option<Instant>,
    /// When the most recent token was emitted
    last_token_time: Option<Instant>,
    /// Total tokens generated (not including prompt)
    pub tokens_generated: usize,
    /// Prompt tokens evaluated
    pub prompt_tokens: usize,
    /// Per-layer timings (accumulated over all tokens)
    layer_timings: Vec<LayerTiming>,
    /// Running sums for aggregate statistics
    total_compute: Duration,
    total_io: Duration,
    total_h2d: Duration,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            gen_start: None,
            first_token_time: None,
            last_token_time: None,
            tokens_generated: 0,
            prompt_tokens: 0,
            layer_timings: Vec::new(),
            total_compute: Duration::ZERO,
            total_io: Duration::ZERO,
            total_h2d: Duration::ZERO,
        }
    }

    /// Mark the start of generation.
    pub fn start(&mut self) {
        self.gen_start = Some(Instant::now());
    }

    /// Mark that the first output token has been emitted.
    pub fn mark_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
    }

    /// Record that a token was generated.
    pub fn record_token(&mut self) {
        self.tokens_generated += 1;
        self.last_token_time = Some(Instant::now());
    }

    /// Record timing for a single layer pass.
    pub fn record_layer(&mut self, timing: LayerTiming) {
        self.total_compute += timing.compute;
        self.total_io += timing.io;
        self.total_h2d += timing.h2d;
        self.layer_timings.push(timing);
    }

    /// Time-to-first-token: time from generation start to first output token.
    pub fn ttft(&self) -> Option<Duration> {
        match (self.gen_start, self.first_token_time) {
            (Some(start), Some(first)) => Some(first.duration_since(start)),
            _ => None,
        }
    }

    /// Total generation time from start to the last token.
    pub fn total_time(&self) -> Option<Duration> {
        match (self.gen_start, self.last_token_time) {
            (Some(start), Some(last)) => Some(last.duration_since(start)),
            _ => None,
        }
    }

    /// Total elapsed seconds (0.0 if not started).
    pub fn elapsed_secs(&self) -> f64 {
        self.total_time()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Time-to-first-token in milliseconds (0.0 if not recorded).
    pub fn ttft_ms(&self) -> f64 {
        self.ttft()
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    /// Tokens per second (generation phase only, excluding TTFT).
    pub fn tokens_per_second(&self) -> Option<f64> {
        if self.tokens_generated <= 1 {
            return None;
        }

        match (self.first_token_time, self.last_token_time) {
            (Some(first), Some(last)) => {
                let gen_duration = last.duration_since(first);
                if gen_duration.as_secs_f64() > 0.0 {
                    // -1 because the first token was already emitted at first_token_time
                    Some((self.tokens_generated - 1) as f64 / gen_duration.as_secs_f64())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Pipeline efficiency ρ: ratio of useful compute to total time.
    /// ρ = T_compute / max(T_compute, T_io + T_h2d)
    pub fn pipeline_efficiency(&self) -> f64 {
        let io_total = self.total_io + self.total_h2d;
        if self.total_compute.is_zero() && io_total.is_zero() {
            return 1.0;
        }
        let max_time = self.total_compute.max(io_total);
        self.total_compute.as_secs_f64() / max_time.as_secs_f64()
    }

    /// Average layer compute time.
    pub fn avg_layer_compute(&self) -> Duration {
        if self.layer_timings.is_empty() {
            return Duration::ZERO;
        }
        self.total_compute / self.layer_timings.len() as u32
    }

    /// Average layer IO time.
    pub fn avg_layer_io(&self) -> Duration {
        if self.layer_timings.is_empty() {
            return Duration::ZERO;
        }
        self.total_io / self.layer_timings.len() as u32
    }

    /// Calculate tokens/sec from explicit duration and count (for testing).
    pub fn calc_tokens_per_sec(tokens: usize, duration: Duration) -> f64 {
        if duration.as_secs_f64() > 0.0 {
            tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate pipeline efficiency from explicit timings (for testing).
    pub fn calc_rho(compute: Duration, io: Duration) -> f64 {
        if compute.is_zero() && io.is_zero() {
            return 1.0;
        }
        let max_time = compute.max(io);
        compute.as_secs_f64() / max_time.as_secs_f64()
    }

    /// Format a human-readable metrics summary.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push("┌─── Inference Metrics ───┐".to_string());

        if let Some(ttft) = self.ttft() {
            lines.push(format!("│ TTFT:           {:>7.1}ms │", ttft.as_secs_f64() * 1000.0));
        }

        if let Some(tps) = self.tokens_per_second() {
            lines.push(format!("│ Tokens/sec:     {:>7.1}   │", tps));
        }

        lines.push(format!("│ Tokens:         {:>7}   │", self.tokens_generated));

        if let Some(total) = self.total_time() {
            lines.push(format!("│ Total time:     {:>7.1}s  │", total.as_secs_f64()));
        }

        let rho = self.pipeline_efficiency();
        lines.push(format!("│ Pipeline ρ:     {:>7.1}%  │", rho * 100.0));

        if !self.layer_timings.is_empty() {
            let avg_compute = self.avg_layer_compute();
            let avg_io = self.avg_layer_io();
            lines.push(format!("│ Avg compute/L:  {:>7.2}ms │", avg_compute.as_secs_f64() * 1000.0));
            lines.push(format!("│ Avg IO/L:       {:>7.2}ms │", avg_io.as_secs_f64() * 1000.0));
        }

        lines.push("└─────────────────────────┘".to_string());

        lines.join("\n")
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_timing_total() {
        let timing = LayerTiming {
            compute: Duration::from_millis(10),
            io: Duration::from_millis(5),
            h2d: Duration::from_millis(2),
        };
        assert_eq!(timing.total(), Duration::from_millis(17));
    }

    #[test]
    fn test_calc_tokens_per_sec() {
        let tps = InferenceMetrics::calc_tokens_per_sec(100, Duration::from_secs(2));
        assert!((tps - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_calc_tokens_per_sec_zero_duration() {
        let tps = InferenceMetrics::calc_tokens_per_sec(100, Duration::ZERO);
        assert_eq!(tps, 0.0);
    }

    #[test]
    fn test_calc_rho_equal_compute_io() {
        let rho = InferenceMetrics::calc_rho(
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calc_rho_io_bound() {
        // compute=5ms, IO=20ms → ρ = 5/20 = 0.25
        let rho = InferenceMetrics::calc_rho(
            Duration::from_millis(5),
            Duration::from_millis(20),
        );
        assert!((rho - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_calc_rho_compute_bound() {
        // compute=20ms, IO=5ms → ρ = 20/20 = 1.0
        let rho = InferenceMetrics::calc_rho(
            Duration::from_millis(20),
            Duration::from_millis(5),
        );
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_record_layers() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_layer(LayerTiming {
            compute: Duration::from_millis(10),
            io: Duration::from_millis(5),
            h2d: Duration::ZERO,
        });
        metrics.record_layer(LayerTiming {
            compute: Duration::from_millis(20),
            io: Duration::from_millis(15),
            h2d: Duration::ZERO,
        });

        assert_eq!(metrics.avg_layer_compute(), Duration::from_millis(15));
        assert_eq!(metrics.avg_layer_io(), Duration::from_millis(10));
    }

    #[test]
    fn test_metrics_pipeline_efficiency() {
        let mut metrics = InferenceMetrics::new();
        // Total compute = 30ms, Total IO = 20ms → ρ = 30/30 = 1.0
        metrics.record_layer(LayerTiming {
            compute: Duration::from_millis(30),
            io: Duration::from_millis(20),
            h2d: Duration::ZERO,
        });
        let rho = metrics.pipeline_efficiency();
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_summary_no_crash() {
        let metrics = InferenceMetrics::new();
        let summary = metrics.summary();
        assert!(summary.contains("Inference Metrics"));
    }
}

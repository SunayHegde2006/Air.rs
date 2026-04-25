//! Residency score computation — R(t,τ) from STRIX Protocol §3.2–3.6.
//!
//! The residency score determines how urgently a tensor should be in VRAM.
//! It is a weighted sum of four components, clamped to [0.0, 1.0]:
//!
//! ```text
//! R(t,τ) = w_u·U(t,τ) + w_p·P(t,τ) + w_s·S(t) + w_c·C(t)
//! ```

/// Weights for the four residency score components.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ScoreWeights {
    /// Urgency weight (how close is the tensor to being needed?).
    pub urgency: f32,
    /// Predictive weight (how recently was it accessed?).
    pub predictive: f32,
    /// Sticky weight (thrash prevention — penalise repeated evictions).
    pub sticky: f32,
    /// Cost weight (how expensive is it to reload?).
    pub cost: f32,
}

impl Default for ScoreWeights {
    /// Default weights from STRIX Protocol §20.3.
    fn default() -> Self {
        Self {
            urgency: 0.45,
            predictive: 0.30,
            sticky: 0.15,
            cost: 0.10,
        }
    }
}

impl ScoreWeights {
    /// Sum of all weights (should equal 1.0 for proper normalisation).
    pub fn sum(&self) -> f32 {
        self.urgency + self.predictive + self.sticky + self.cost
    }
}

// ── Component Functions ──────────────────────────────────────────────────

/// Urgency component U(t,τ) — sigmoid function of distance to next use.
///
/// Returns ≈1.0 when `distance ≤ β` and decays smoothly to ≈0 beyond.
///
/// Formula (STRIX Protocol §3.3):
///   `U(t,τ) = 1.0 / (1.0 + exp(α × (distance - β)))`
/// where `α = 0.8` (steepness) and `β = W` (prefetch window size).
pub fn urgency(distance_layers: usize, prefetch_window: usize) -> f32 {
    if prefetch_window == 0 {
        return if distance_layers == 0 { 1.0 } else { 0.0 };
    }
    let alpha: f32 = 0.8;
    let beta: f32 = prefetch_window as f32;
    let d = distance_layers as f32;
    1.0 / (1.0 + (alpha * (d - beta)).exp())
}

/// Predictive component P(t,τ) — exponential recency decay.
///
/// Tensors accessed more recently get higher scores.
/// `decay` controls how quickly the score drops (higher = faster decay).
pub fn predictive(last_access_step: u64, current_step: u64, decay: f32) -> f32 {
    if current_step <= last_access_step {
        return 1.0;
    }
    let delta = (current_step - last_access_step) as f32;
    (-decay * delta).exp()
}

/// Sticky component S(t) — logarithmic thrash prevention.
///
/// Tensors that have been evicted many times get a higher score to prevent
/// further thrashing (evict-reload-evict cycles).
pub fn sticky(eviction_count: u32) -> f32 {
    if eviction_count == 0 {
        return 0.0;
    }
    let s = (1.0 + eviction_count as f32).ln() / (1.0 + 20.0_f32).ln();
    s.min(1.0)
}

/// Cost component C(t) — normalised reload cost.
///
/// Larger tensors are more expensive to reload, so they should be
/// retained longer. Normalised against `max_tensor_bytes`.
pub fn cost(size_bytes: usize, max_tensor_bytes: usize) -> f32 {
    if max_tensor_bytes == 0 {
        return 0.0;
    }
    let ratio = size_bytes as f32 / max_tensor_bytes as f32;
    ratio.min(1.0)
}

/// Compute the full residency score R(t,τ), clamped to [0.0, 1.0].
pub fn residency_score(
    weights: &ScoreWeights,
    urgency_val: f32,
    predictive_val: f32,
    sticky_val: f32,
    cost_val: f32,
) -> f32 {
    let raw = weights.urgency * urgency_val
        + weights.predictive * predictive_val
        + weights.sticky * sticky_val
        + weights.cost * cost_val;
    raw.clamp(0.0, 1.0)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urgency_at_distance_zero() {
        let u = urgency(0, 3);
        assert!(u > 0.9, "urgency at d=0 should be ~1.0, got {u}");
    }

    #[test]
    fn urgency_at_window_center() {
        // At d = β = W, sigmoid should be ~0.5 (center of sigmoid)
        let u = urgency(3, 3);
        assert!(
            (u - 0.5).abs() < 0.01,
            "urgency at d=W should be ~0.5, got {u}"
        );
    }

    #[test]
    fn urgency_decays_with_distance() {
        let window = 3;
        // With α=0.8, decay is gentler; need larger distance for <0.1
        let u_far = urgency(window + 6, window);
        assert!(u_far < 0.1, "urgency well beyond W should be <0.1, got {u_far}");

        // Monotonically decreasing
        let u0 = urgency(0, window);
        let u1 = urgency(1, window);
        let u2 = urgency(2, window);
        assert!(u0 > u1);
        assert!(u1 > u2);
    }

    #[test]
    fn urgency_zero_window() {
        assert_eq!(urgency(0, 0), 1.0);
        assert_eq!(urgency(1, 0), 0.0);
    }

    #[test]
    fn predictive_recency_decay() {
        let p0 = predictive(10, 10, 0.1);
        assert!((p0 - 1.0).abs() < 1e-6, "same step should be 1.0");

        let p5 = predictive(5, 10, 0.1);
        let p20 = predictive(0, 20, 0.1);
        assert!(p5 > p20, "more recent access should score higher");
    }

    #[test]
    fn sticky_increases_with_evictions() {
        let s0 = sticky(0);
        let s1 = sticky(1);
        let s5 = sticky(5);
        let s20 = sticky(20);
        assert_eq!(s0, 0.0);
        assert!(s1 > s0);
        assert!(s5 > s1);
        assert!(s20 > s5);
        // s20 should be ~1.0 (at the cap)
        assert!((s20 - 1.0).abs() < 0.01);
    }

    #[test]
    fn cost_normalised_to_unit() {
        assert_eq!(cost(0, 1000), 0.0);
        assert!((cost(500, 1000) - 0.5).abs() < 1e-6);
        assert!((cost(1000, 1000) - 1.0).abs() < 1e-6);
        // Larger than max clamped to 1.0
        assert!((cost(2000, 1000) - 1.0).abs() < 1e-6);
        // Zero max
        assert_eq!(cost(100, 0), 0.0);
    }

    #[test]
    fn residency_score_in_range() {
        let w = ScoreWeights::default();
        let score = residency_score(&w, 1.0, 1.0, 1.0, 1.0);
        assert!(score >= 0.0 && score <= 1.0, "score must be in [0,1], got {score}");
    }

    #[test]
    fn residency_score_weights_sum() {
        let w = ScoreWeights::default();
        let sum = w.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "default weights should sum to 1.0, got {sum}"
        );
    }
}

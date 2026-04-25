//! VRAM Pressure Manager (STRIX Protocol §10.1–10.3).
//!
//! Monitors VRAM utilisation and classifies pressure into five levels.
//! Each level maps to concrete actions: from normal prefetch (Green)
//! through emergency eviction (Critical).
//!
//! The manager is hardware/OS agnostic — it operates on (used, total)
//! byte counts that come from the HAL layer.

use std::fmt;

// ── Pressure Level ──────────────────────────────────────────────────────

/// VRAM pressure classification (§10.1).
///
/// Thresholds are based on *used/total* ratio:
///
/// | Level    | Utilisation | Meaning                              |
/// |----------|-------------|--------------------------------------|
/// | Green    | < 60%       | Normal operation, full prefetch       |
/// | Yellow   | 60–75%      | Reduce prefetch window                |
/// | Orange   | 75–85%      | Evict Class C, shrink KV cache        |
/// | Red      | 85–95%      | Evict Class B, suspend prefetch       |
/// | Critical | ≥ 95%       | Emergency: evict everything possible  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    Green,
    Yellow,
    Orange,
    Red,
    Critical,
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Green => write!(f, "Green (<60%)"),
            Self::Yellow => write!(f, "Yellow (60-75%)"),
            Self::Orange => write!(f, "Orange (75-85%)"),
            Self::Red => write!(f, "Red (85-95%)"),
            Self::Critical => write!(f, "Critical (≥95%)"),
        }
    }
}

// ── Pressure Action ─────────────────────────────────────────────────────

/// Recommended action for the current pressure level (§10.2).
#[derive(Debug, Clone, PartialEq)]
pub enum PressureAction {
    /// Normal operation — no changes needed.
    Continue,
    /// Reduce the prefetch window to `new_window` layers.
    ReducePrefetch { new_window: usize },
    /// Evict tensors of the given class (C first, then B).
    EvictClass { min_class: u8 },
    /// Suspend all prefetch operations entirely.
    SuspendPrefetch,
    /// Trigger KV cache quantisation (e.g. FP16 → INT8).
    CompressKvCache,
    /// Emergency: evict all non-pinned tensors.
    EmergencyEvict,
}

// ── VramPressureManager ─────────────────────────────────────────────────

/// Evaluates and manages VRAM pressure (§10).
///
/// Created once at startup with the GPU's total VRAM and safety margin.
/// Call `evaluate()` each scheduler tick to get the current level and
/// `recommended_actions()` for what to do about it.
#[derive(Debug, Clone)]
pub struct VramPressureManager {
    /// Total VRAM in bytes.
    total_vram: usize,
    /// Safety margin subtracted from total (bytes).
    safety_margin: usize,
    /// Effective budget = total - safety.
    budget: usize,
    /// Default prefetch window (from config).
    default_prefetch_window: usize,
}

impl VramPressureManager {
    /// Create a new pressure manager.
    ///
    /// - `total_vram`: GPU VRAM in bytes
    /// - `safety_margin`: bytes reserved for OS/driver overhead
    /// - `default_prefetch_window`: normal prefetch window size (layers)
    pub fn new(total_vram: usize, safety_margin: usize, default_prefetch_window: usize) -> Self {
        let budget = total_vram.saturating_sub(safety_margin);
        Self {
            total_vram,
            safety_margin,
            budget,
            default_prefetch_window,
        }
    }

    /// Total VRAM in bytes (as reported by the HAL at startup).
    pub fn total_vram(&self) -> usize {
        self.total_vram
    }

    /// Safety margin in bytes (reserved for OS/driver overhead).
    pub fn safety_margin(&self) -> usize {
        self.safety_margin
    }

    /// Effective VRAM budget (total minus safety margin).
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Evaluate current pressure from raw byte counts.
    ///
    /// `used` should come from `GpuHal::vram_used()`.
    pub fn evaluate(&self, used: usize) -> PressureLevel {
        if self.budget == 0 {
            return PressureLevel::Critical;
        }
        let ratio = used as f64 / self.budget as f64;
        if ratio >= 0.95 {
            PressureLevel::Critical
        } else if ratio >= 0.85 {
            PressureLevel::Red
        } else if ratio >= 0.75 {
            PressureLevel::Orange
        } else if ratio >= 0.60 {
            PressureLevel::Yellow
        } else {
            PressureLevel::Green
        }
    }

    /// Utilisation ratio (0.0–1.0+) for the current usage.
    pub fn utilisation(&self, used: usize) -> f64 {
        if self.budget == 0 {
            return 1.0;
        }
        used as f64 / self.budget as f64
    }

    /// Recommended actions for a given pressure level (§10.2).
    pub fn recommended_actions(&self, level: PressureLevel) -> Vec<PressureAction> {
        match level {
            PressureLevel::Green => vec![PressureAction::Continue],

            PressureLevel::Yellow => vec![PressureAction::ReducePrefetch {
                new_window: (self.default_prefetch_window / 2).max(1),
            }],

            PressureLevel::Orange => vec![
                PressureAction::EvictClass { min_class: 2 }, // Class C
                PressureAction::CompressKvCache,
            ],

            PressureLevel::Red => vec![
                PressureAction::EvictClass { min_class: 1 }, // Class B
                PressureAction::SuspendPrefetch,
                PressureAction::CompressKvCache,
            ],

            PressureLevel::Critical => vec![PressureAction::EmergencyEvict],
        }
    }
}

// ── KV Cache Budget (§3.5) ──────────────────────────────────────────────

/// Compute the KV cache VRAM cost for a given model configuration.
///
/// Formula: `2 × seq_len × n_heads × head_dim × n_layers × dtype_bytes`
///
/// The factor of 2 accounts for both K and V caches.
///
/// # Example (LLaMA 70B, Q8_0 KV, 4K context)
/// ```text
/// let bytes = strix::vram_pressure::kv_cache_budget(
///     4096, 64, 128, 80, 1  // seq, heads, head_dim, layers, Q8_0
/// );
/// // ~ 5.37 GB
/// ```
pub fn kv_cache_budget(
    context_len: usize,
    n_heads: usize,
    head_dim: usize,
    n_layers: usize,
    dtype_bytes: usize,
) -> usize {
    2 * context_len * n_heads * head_dim * n_layers * dtype_bytes
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> VramPressureManager {
        // 8 GB total, 512 MB safety → 7.5 GB budget
        VramPressureManager::new(
            8 * 1024 * 1024 * 1024,
            512 * 1024 * 1024,
            3, // default prefetch window
        )
    }

    #[test]
    fn budget_calculation() {
        let mgr = make_manager();
        let expected = 8 * 1024 * 1024 * 1024 - 512 * 1024 * 1024;
        assert_eq!(mgr.budget(), expected);
    }

    #[test]
    fn pressure_green() {
        let mgr = make_manager();
        // 50% usage → Green
        let used = mgr.budget() / 2;
        assert_eq!(mgr.evaluate(used), PressureLevel::Green);
    }

    #[test]
    fn pressure_yellow() {
        let mgr = make_manager();
        // 65% usage → Yellow
        let used = (mgr.budget() as f64 * 0.65) as usize;
        assert_eq!(mgr.evaluate(used), PressureLevel::Yellow);
    }

    #[test]
    fn pressure_orange() {
        let mgr = make_manager();
        // 80% usage → Orange
        let used = (mgr.budget() as f64 * 0.80) as usize;
        assert_eq!(mgr.evaluate(used), PressureLevel::Orange);
    }

    #[test]
    fn pressure_red() {
        let mgr = make_manager();
        // 90% usage → Red
        let used = (mgr.budget() as f64 * 0.90) as usize;
        assert_eq!(mgr.evaluate(used), PressureLevel::Red);
    }

    #[test]
    fn pressure_critical() {
        let mgr = make_manager();
        // 96% usage → Critical
        let used = (mgr.budget() as f64 * 0.96) as usize;
        assert_eq!(mgr.evaluate(used), PressureLevel::Critical);
    }

    #[test]
    fn pressure_at_exact_boundaries() {
        let mgr = make_manager();
        let b = mgr.budget();
        // Exactly 60% → Yellow (≥60%)
        let used_60 = (b as f64 * 0.60) as usize;
        assert_eq!(mgr.evaluate(used_60), PressureLevel::Yellow);
        // Exactly 75% → Orange
        let used_75 = (b as f64 * 0.75) as usize;
        assert_eq!(mgr.evaluate(used_75), PressureLevel::Orange);
        // Exactly 85% → Red
        let used_85 = (b as f64 * 0.85) as usize;
        assert_eq!(mgr.evaluate(used_85), PressureLevel::Red);
        // Exactly 95% → Critical
        let used_95 = (b as f64 * 0.95) as usize;
        assert_eq!(mgr.evaluate(used_95), PressureLevel::Critical);
    }

    #[test]
    fn pressure_zero_budget() {
        let mgr = VramPressureManager::new(100, 200, 3); // safety > total
        assert_eq!(mgr.budget(), 0);
        assert_eq!(mgr.evaluate(0), PressureLevel::Critical);
    }

    #[test]
    fn recommended_actions_green() {
        let mgr = make_manager();
        let actions = mgr.recommended_actions(PressureLevel::Green);
        assert_eq!(actions, vec![PressureAction::Continue]);
    }

    #[test]
    fn recommended_actions_yellow_halves_prefetch() {
        let mgr = make_manager();
        let actions = mgr.recommended_actions(PressureLevel::Yellow);
        assert!(actions.contains(&PressureAction::ReducePrefetch { new_window: 1 }));
    }

    #[test]
    fn recommended_actions_orange_evicts_class_c() {
        let mgr = make_manager();
        let actions = mgr.recommended_actions(PressureLevel::Orange);
        assert!(actions.contains(&PressureAction::EvictClass { min_class: 2 }));
        assert!(actions.contains(&PressureAction::CompressKvCache));
    }

    #[test]
    fn recommended_actions_red_suspends_prefetch() {
        let mgr = make_manager();
        let actions = mgr.recommended_actions(PressureLevel::Red);
        assert!(actions.contains(&PressureAction::SuspendPrefetch));
    }

    #[test]
    fn recommended_actions_critical_emergency() {
        let mgr = make_manager();
        let actions = mgr.recommended_actions(PressureLevel::Critical);
        assert!(actions.contains(&PressureAction::EmergencyEvict));
    }

    #[test]
    fn utilisation_ratio() {
        let mgr = make_manager();
        let u = mgr.utilisation(mgr.budget() / 2);
        assert!((u - 0.5).abs() < 0.001);
    }

    // --- KV Cache Budget ---

    #[test]
    fn kv_cache_budget_llama_7b() {
        // LLaMA 7B: 32 layers, 32 heads, 128 head_dim, 4K context, FP16 (2 bytes)
        let bytes = kv_cache_budget(4096, 32, 128, 32, 2);
        // Expected: 2 × 4096 × 32 × 128 × 32 × 2 = 2,147,483,648 (2 GB)
        assert_eq!(bytes, 2_147_483_648);
    }

    #[test]
    fn kv_cache_budget_llama_70b_q8() {
        // LLaMA 70B: 80 layers, 64 heads (GQA), 128 head_dim, 4K ctx, Q8 (1 byte)
        let bytes = kv_cache_budget(4096, 64, 128, 80, 1);
        // 2 × 4096 × 64 × 128 × 80 × 1 = 5,368,709,120 (~5.37 GB)
        assert_eq!(bytes, 5_368_709_120);
    }

    #[test]
    fn kv_cache_budget_zero_context() {
        assert_eq!(kv_cache_budget(0, 32, 128, 32, 2), 0);
    }

    #[test]
    fn pressure_level_ordering() {
        assert!(PressureLevel::Green < PressureLevel::Yellow);
        assert!(PressureLevel::Yellow < PressureLevel::Orange);
        assert!(PressureLevel::Orange < PressureLevel::Red);
        assert!(PressureLevel::Red < PressureLevel::Critical);
    }
}

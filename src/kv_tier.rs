//! Tiered KV Cache — Pinned / Active / Warm / Cold
//!
//! Extends the base KvCacheManager with a 4-tier eviction policy:
//!
//! | Tier   | Where          | Latency  | Use case                        |
//! |--------|----------------|----------|---------------------------------|
//! | Pinned | VRAM (locked)  | 0 ns     | System prompt, tool results     |
//! | Active | VRAM           | ~0.1 µs  | Recent context (sliding window) |
//! | Warm   | System RAM     | ~5 µs    | Older context, may be recalled  |
//! | Cold   | Disk (mmap)    | ~50 µs   | Very old context, rarely used   |
//!
//! The tier manager tracks recency-of-access per token position and
//! promotes/demotes entries to keep VRAM usage within budget.
//!
//! Reference: air_rs_protocols_v3.md §5 "Tiered KV Cache"

use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Tier Classification
// ---------------------------------------------------------------------------

/// Which storage tier a KV cache entry is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvTier {
    /// Pinned in VRAM — never evicted (system prompt, tool context)
    Pinned,
    /// Active in VRAM — recent tokens within the sliding window
    Active,
    /// Warm in system RAM — evicted from VRAM but still fast to reload
    Warm,
    /// Cold on disk — evicted from RAM, loaded via mmap on demand
    Cold,
}

impl std::fmt::Display for KvTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvTier::Pinned => write!(f, "Pinned (VRAM)"),
            KvTier::Active => write!(f, "Active (VRAM)"),
            KvTier::Warm   => write!(f, "Warm (RAM)"),
            KvTier::Cold   => write!(f, "Cold (Disk)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Token Entry — tracks per-position metadata
// ---------------------------------------------------------------------------

/// Metadata for a single token position in the KV cache.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    /// Absolute position in the sequence
    pub position: usize,
    /// Current storage tier
    pub tier: KvTier,
    /// How many times this position has been attended to
    pub access_count: u64,
    /// Last time this position was accessed
    pub last_access: Instant,
    /// Whether this position is pinned (cannot be demoted)
    pub is_pinned: bool,
}

impl TokenEntry {
    pub fn new(position: usize, tier: KvTier) -> Self {
        Self {
            position,
            tier,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: tier == KvTier::Pinned,
        }
    }

    /// Record that this position was just accessed.
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Tier Budget
// ---------------------------------------------------------------------------

/// Memory budgets for each tier (in number of token positions).
#[derive(Debug, Clone, Copy)]
pub struct TierBudget {
    /// Max positions in Pinned tier (limited by VRAM reserved for pinning)
    pub pinned_max: usize,
    /// Max positions in Active tier (sliding window size)
    pub active_max: usize,
    /// Max positions in Warm tier (system RAM budget)
    pub warm_max: usize,
    // Cold tier has no limit — bounded by disk space
}

impl Default for TierBudget {
    fn default() -> Self {
        Self {
            pinned_max: 512,    // ~512 tokens of system prompt
            active_max: 2048,   // sliding window of 2K recent tokens
            warm_max: 8192,     // 8K tokens in RAM
        }
    }
}

impl TierBudget {
    /// Budget for a given VRAM size in bytes and bytes-per-token-per-layer.
    pub fn from_vram(vram_bytes: u64, bytes_per_token: u64) -> Self {
        if bytes_per_token == 0 {
            return Self::default();
        }
        let total_vram_tokens = (vram_bytes / bytes_per_token) as usize;

        // Reserve 20% for pinned, 80% for active
        let pinned = (total_vram_tokens * 20) / 100;
        let active = total_vram_tokens - pinned;

        Self {
            pinned_max: pinned.max(64),
            active_max: active.max(256),
            warm_max: active * 4, // 4x the VRAM budget in RAM
        }
    }
}

// ---------------------------------------------------------------------------
// Tier Manager
// ---------------------------------------------------------------------------

/// Manages tier assignments and demotion/promotion for KV cache entries.
///
/// This works alongside the existing `KvCacheManager` — it doesn't hold
/// the actual tensor data, just the tier metadata and eviction logic.
pub struct TierManager {
    /// Per-position tier metadata (ordered by position)
    entries: VecDeque<TokenEntry>,
    /// Budget configuration
    budget: TierBudget,
    /// Total demotions performed (Active→Warm)
    pub demotions_to_warm: u64,
    /// Total demotions performed (Warm→Cold)
    pub demotions_to_cold: u64,
    /// Total promotions performed (Cold/Warm→Active)
    pub promotions: u64,
}

impl TierManager {
    pub fn new(budget: TierBudget) -> Self {
        Self {
            entries: VecDeque::new(),
            budget,
            demotions_to_warm: 0,
            demotions_to_cold: 0,
            promotions: 0,
        }
    }

    /// Add a new token position to the cache (initially Active).
    pub fn add_token(&mut self, position: usize) {
        let entry = TokenEntry::new(position, KvTier::Active);
        self.entries.push_back(entry);
        self.enforce_budgets();
    }

    /// Pin a range of positions (e.g., system prompt tokens).
    /// Pinned entries are never demoted.
    pub fn pin_range(&mut self, start: usize, end: usize) {
        let mut pinned_count = self.count_tier(KvTier::Pinned);

        for entry in self.entries.iter_mut() {
            if entry.position >= start && entry.position < end {
                if pinned_count < self.budget.pinned_max {
                    entry.tier = KvTier::Pinned;
                    entry.is_pinned = true;
                    pinned_count += 1;
                }
            }
        }
    }

    /// Record that a set of positions were just attended to.
    pub fn touch_positions(&mut self, positions: &[usize]) {
        for entry in self.entries.iter_mut() {
            if positions.contains(&entry.position) {
                entry.touch();
            }
        }
    }

    /// Get the current tier for a given position.
    pub fn get_tier(&self, position: usize) -> Option<KvTier> {
        self.entries.iter()
            .find(|e| e.position == position)
            .map(|e| e.tier)
    }

    /// Enforce budgets by demoting oldest Active→Warm and Warm→Cold.
    pub fn enforce_budgets(&mut self) {
        // Demote Active → Warm (oldest first, skip pinned)
        let active_count = self.count_tier(KvTier::Active);
        if active_count > self.budget.active_max {
            let excess = active_count - self.budget.active_max;
            let mut demoted = 0;

            for entry in self.entries.iter_mut() {
                if demoted >= excess {
                    break;
                }
                if entry.tier == KvTier::Active && !entry.is_pinned {
                    entry.tier = KvTier::Warm;
                    self.demotions_to_warm += 1;
                    demoted += 1;
                }
            }
        }

        // Demote Warm → Cold (oldest first)
        let warm_count = self.count_tier(KvTier::Warm);
        if warm_count > self.budget.warm_max {
            let excess = warm_count - self.budget.warm_max;
            let mut demoted = 0;

            for entry in self.entries.iter_mut() {
                if demoted >= excess {
                    break;
                }
                if entry.tier == KvTier::Warm {
                    entry.tier = KvTier::Cold;
                    self.demotions_to_cold += 1;
                    demoted += 1;
                }
            }
        }
    }

    /// Promote a position from Warm/Cold back to Active (cache hit).
    /// Moves the entry to the back of the deque (most recent) so it
    /// isn't immediately demoted by enforce_budgets.
    pub fn promote(&mut self, position: usize) -> bool {
        // Find the index of the entry to promote
        let idx = self.entries.iter().position(|e| {
            e.position == position
                && (e.tier == KvTier::Warm || e.tier == KvTier::Cold)
        });

        if let Some(idx) = idx {
            // Remove from current position in the deque
            let mut entry = self.entries.remove(idx).unwrap();
            entry.tier = KvTier::Active;
            entry.touch();
            // Push to back = most recent, won't be first to demote
            self.entries.push_back(entry);
            self.promotions += 1;
            self.enforce_budgets();
            true
        } else {
            false
        }
    }

    /// Count entries in a given tier.
    pub fn count_tier(&self, tier: KvTier) -> usize {
        self.entries.iter().filter(|e| e.tier == tier).count()
    }

    /// Total number of tracked positions.
    pub fn total_positions(&self) -> usize {
        self.entries.len()
    }

    /// Get tier distribution as (pinned, active, warm, cold).
    pub fn distribution(&self) -> (usize, usize, usize, usize) {
        (
            self.count_tier(KvTier::Pinned),
            self.count_tier(KvTier::Active),
            self.count_tier(KvTier::Warm),
            self.count_tier(KvTier::Cold),
        )
    }

    /// Reset all tier state (for a new generation).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.demotions_to_warm = 0;
        self.demotions_to_cold = 0;
        self.promotions = 0;
    }

    /// Format a summary of the tier distribution.
    pub fn summary(&self) -> String {
        let (p, a, w, c) = self.distribution();
        format!(
            "KV Tiers: Pinned={} Active={} Warm={} Cold={} (total={})",
            p, a, w, c, self.total_positions()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_budget() -> TierBudget {
        TierBudget {
            pinned_max: 4,
            active_max: 8,
            warm_max: 8,
        }
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", KvTier::Pinned), "Pinned (VRAM)");
        assert_eq!(format!("{}", KvTier::Cold), "Cold (Disk)");
    }

    #[test]
    fn test_add_tokens_within_budget() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..5 {
            mgr.add_token(i);
        }
        assert_eq!(mgr.count_tier(KvTier::Active), 5);
        assert_eq!(mgr.count_tier(KvTier::Warm), 0);
    }

    #[test]
    fn test_active_to_warm_demotion() {
        let mut mgr = TierManager::new(small_budget());
        // Add 12 tokens — budget is active_max=8, so 4 get demoted
        for i in 0..12 {
            mgr.add_token(i);
        }
        assert_eq!(mgr.count_tier(KvTier::Active), 8);
        assert_eq!(mgr.count_tier(KvTier::Warm), 4);
        assert_eq!(mgr.demotions_to_warm, 4);
    }

    #[test]
    fn test_warm_to_cold_demotion() {
        let mut mgr = TierManager::new(small_budget());
        // Add 20 tokens — active_max=8, warm_max=8, so 4 go to Cold
        for i in 0..20 {
            mgr.add_token(i);
        }
        assert_eq!(mgr.count_tier(KvTier::Active), 8);
        assert_eq!(mgr.count_tier(KvTier::Warm), 8);
        assert_eq!(mgr.count_tier(KvTier::Cold), 4);
        assert!(mgr.demotions_to_cold > 0);
    }

    #[test]
    fn test_pinning() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..10 {
            mgr.add_token(i);
        }

        // Pin first 4 positions
        mgr.pin_range(0, 4);
        assert_eq!(mgr.count_tier(KvTier::Pinned), 4);

        // Pinned entries should not be demoted when budget enforced
        for i in 10..20 {
            mgr.add_token(i);
        }
        assert_eq!(mgr.count_tier(KvTier::Pinned), 4);
    }

    #[test]
    fn test_pinning_respects_budget() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..10 {
            mgr.add_token(i);
        }

        // Try to pin 8 but budget is pinned_max=4
        mgr.pin_range(0, 8);
        assert_eq!(mgr.count_tier(KvTier::Pinned), 4);
    }

    #[test]
    fn test_promotion() {
        let mut mgr = TierManager::new(small_budget());
        // Fill up and force some to Warm
        for i in 0..12 {
            mgr.add_token(i);
        }

        // Position 0 should be in Warm (it was the oldest)
        assert_eq!(mgr.get_tier(0), Some(KvTier::Warm));

        // Promote it back
        assert!(mgr.promote(0));
        assert_eq!(mgr.get_tier(0), Some(KvTier::Active));
        assert_eq!(mgr.promotions, 1);
    }

    #[test]
    fn test_promotion_nonexistent_fails() {
        let mut mgr = TierManager::new(small_budget());
        mgr.add_token(0);
        // Position 99 doesn't exist
        assert!(!mgr.promote(99));
    }

    #[test]
    fn test_touch_positions() {
        let mut mgr = TierManager::new(small_budget());
        mgr.add_token(0);
        mgr.add_token(1);

        mgr.touch_positions(&[0, 1]);

        let entry = mgr.entries.iter().find(|e| e.position == 0).unwrap();
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_clear() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..20 {
            mgr.add_token(i);
        }
        mgr.clear();
        assert_eq!(mgr.total_positions(), 0);
        assert_eq!(mgr.demotions_to_warm, 0);
    }

    #[test]
    fn test_distribution() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..20 {
            mgr.add_token(i);
        }
        let (p, a, w, c) = mgr.distribution();
        assert_eq!(p + a + w + c, 20);
    }

    #[test]
    fn test_summary() {
        let mut mgr = TierManager::new(small_budget());
        for i in 0..5 {
            mgr.add_token(i);
        }
        let summary = mgr.summary();
        assert!(summary.contains("Active=5"));
        assert!(summary.contains("total=5"));
    }

    #[test]
    fn test_budget_from_vram() {
        // 1 GB VRAM, 256 bytes per token per layer
        let budget = TierBudget::from_vram(1_073_741_824, 256);
        assert!(budget.pinned_max >= 64);
        assert!(budget.active_max >= 256);
        assert!(budget.warm_max > budget.active_max);
    }

    #[test]
    fn test_budget_from_vram_zero_bytes() {
        // Edge case: 0 bytes per token
        let budget = TierBudget::from_vram(1_073_741_824, 0);
        assert_eq!(budget.pinned_max, TierBudget::default().pinned_max);
    }

    #[test]
    fn test_default_budget() {
        let budget = TierBudget::default();
        assert_eq!(budget.pinned_max, 512);
        assert_eq!(budget.active_max, 2048);
        assert_eq!(budget.warm_max, 8192);
    }
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 4: HERMES Hierarchical KV Eviction
// ============================================================================
//
// Based on: "HERMES: KV Cache as Hierarchical Memory" (2025)
//
// I(t) = α·recency + β·access_density + γ·position_weight
//   recency(t)       = exp(-λ · age)
//   access_density   = log(1 + count) / log(10)
//   position_weight  = sigmoid(σ · (prefix_len - pos))

/// Weights for the HERMES importance score.
#[derive(Debug, Clone)]
pub struct HermesWeights {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub sigma: f64,
    pub prefix_len: usize,
}

impl Default for HermesWeights {
    fn default() -> Self {
        Self {
            alpha: 0.40,
            beta: 0.35,
            gamma: 0.25,
            lambda: 0.05,
            sigma: 0.1,
            prefix_len: 128,
        }
    }
}

impl HermesWeights {
    /// Compute HERMES importance score for one TokenEntry.
    pub fn score(&self, entry: &TokenEntry, current_step: u64) -> f64 {
        let elapsed = current_step.saturating_sub(entry.access_count);
        let recency = (-self.lambda * elapsed as f64).exp();
        let density = ((1.0 + entry.access_count as f64).ln()
            / (10.0f64).ln())
            .min(1.0);
        let pos_diff = self.prefix_len as f64 - entry.position as f64;
        let pos_weight = 1.0 / (1.0 + (-self.sigma * pos_diff).exp());
        self.alpha * recency + self.beta * density + self.gamma * pos_weight
    }
}

/// HERMES-augmented tier manager — evicts lowest-importance tokens first.
pub struct HermesTierManager {
    pub inner: TierManager,
    pub weights: HermesWeights,
    pub step: u64,
    pub importance_evictions: u64,
}

impl HermesTierManager {
    pub fn new(budget: TierBudget, weights: HermesWeights) -> Self {
        Self {
            inner: TierManager::new(budget),
            weights,
            step: 0,
            importance_evictions: 0,
        }
    }

    /// Add token and enforce budget via importance-score eviction.
    pub fn add_token(&mut self, position: usize) {
        self.step += 1;
        self.inner.entries.push_back(TokenEntry::new(position, KvTier::Active));
        self.enforce_budgets_hermes();
    }

    pub fn pin_range(&mut self, start: usize, end: usize) {
        self.inner.pin_range(start, end);
    }
    pub fn touch_positions(&mut self, positions: &[usize]) {
        self.inner.touch_positions(positions);
    }
    pub fn get_tier(&self, position: usize) -> Option<KvTier> {
        self.inner.get_tier(position)
    }
    pub fn promote(&mut self, position: usize) -> bool {
        self.inner.promote(position)
    }
    pub fn distribution(&self) -> (usize, usize, usize, usize) {
        self.inner.distribution()
    }

    pub fn summary(&self) -> String {
        let (p, a, w, c) = self.distribution();
        format!(
            "HERMES KV: Pinned={} Active={} Warm={} Cold={} step={} evictions={}",
            p, a, w, c, self.step, self.importance_evictions
        )
    }

    /// Importance-scored budget enforcement (replaces base LRU policy).
    pub fn enforce_budgets_hermes(&mut self) {
        let budget = self.inner.budget;
        let step = self.step;
        let weights = self.weights.clone();

        // Active → Warm: evict lowest-scored Active (non-pinned) entries.
        let active_count = self.inner.count_tier(KvTier::Active);
        if active_count > budget.active_max {
            let excess = active_count - budget.active_max;
            let mut scored: Vec<(usize, f64)> = self.inner.entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.tier == KvTier::Active && !e.is_pinned)
                .map(|(idx, e)| (idx, weights.score(e, step)))
                .collect();
            // Ascending → lowest importance first.
            scored.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for &(idx, _) in scored.iter().take(excess) {
                self.inner.entries[idx].tier = KvTier::Warm;
                self.inner.demotions_to_warm += 1;
                self.importance_evictions += 1;
            }
        }

        // Warm → Cold: oldest first (Warm is already cooled, LRU is fine).
        let warm_count = self.inner.count_tier(KvTier::Warm);
        if warm_count > budget.warm_max {
            let excess = warm_count - budget.warm_max;
            let mut demoted = 0;
            for entry in self.inner.entries.iter_mut() {
                if demoted >= excess {
                    break;
                }
                if entry.tier == KvTier::Warm {
                    entry.tier = KvTier::Cold;
                    self.inner.demotions_to_cold += 1;
                    demoted += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod hermes_tests {
    use super::*;

    fn small_budget() -> TierBudget {
        TierBudget { pinned_max: 2, active_max: 4, warm_max: 4 }
    }

    #[test]
    fn hermes_score_recency_decreases_with_age() {
        let w = HermesWeights { alpha: 1.0, beta: 0.0, gamma: 0.0, ..HermesWeights::default() };
        let mut e_recent = TokenEntry::new(0, KvTier::Active);
        e_recent.access_count = 90;
        let mut e_old = TokenEntry::new(1, KvTier::Active);
        e_old.access_count = 10;
        assert!(w.score(&e_recent, 100) > w.score(&e_old, 100));
    }

    #[test]
    fn hermes_score_access_density() {
        let w = HermesWeights { alpha: 0.0, beta: 1.0, gamma: 0.0, ..HermesWeights::default() };
        let mut low = TokenEntry::new(0, KvTier::Active);
        low.access_count = 1;
        let mut high = TokenEntry::new(1, KvTier::Active);
        high.access_count = 100;
        assert!(w.score(&high, 200) > w.score(&low, 200));
    }

    #[test]
    fn hermes_score_prefix_bonus() {
        let w = HermesWeights {
            alpha: 0.0, beta: 0.0, gamma: 1.0, prefix_len: 64, sigma: 0.1,
            ..HermesWeights::default()
        };
        let prefix = TokenEntry::new(10, KvTier::Active);
        let suffix = TokenEntry::new(200, KvTier::Active);
        assert!(w.score(&prefix, 0) > w.score(&suffix, 0));
    }

    #[test]
    fn hermes_evicts_on_overflow() {
        let mut mgr = HermesTierManager::new(small_budget(), HermesWeights::default());
        for i in 0..7 {
            mgr.add_token(i);
        }
        assert_eq!(mgr.inner.count_tier(KvTier::Active), 4);
        assert!(mgr.importance_evictions > 0);
    }

    #[test]
    fn hermes_pinned_not_evicted() {
        let mut mgr = HermesTierManager::new(small_budget(), HermesWeights::default());
        for i in 0..4 { mgr.add_token(i); }
        mgr.pin_range(0, 2);
        for i in 4..12 { mgr.add_token(i); }
        assert_eq!(mgr.inner.count_tier(KvTier::Pinned), 2);
    }

    #[test]
    fn hermes_high_access_survives_eviction() {
        let budget = TierBudget { pinned_max: 0, active_max: 2, warm_max: 10 };
        let weights = HermesWeights {
            alpha: 0.0, beta: 1.0, gamma: 0.0,
            ..HermesWeights::default()
        };
        let mut mgr = HermesTierManager::new(budget, weights);
        mgr.add_token(0);
        mgr.add_token(1);
        // Many touches → high access density for position 0
        for _ in 0..20 {
            mgr.touch_positions(&[0]);
        }
        mgr.add_token(2);
        mgr.add_token(3);
        // Token 0 should still be Active due to high access density.
        assert_eq!(mgr.get_tier(0), Some(KvTier::Active));
    }

    #[test]
    fn hermes_weights_sum_to_one() {
        let w = HermesWeights::default();
        assert!((w.alpha + w.beta + w.gamma - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hermes_summary_format() {
        let mut mgr = HermesTierManager::new(small_budget(), HermesWeights::default());
        for i in 0..5 { mgr.add_token(i); }
        let s = mgr.summary();
        assert!(s.contains("HERMES KV") && s.contains("Active="));
    }
}

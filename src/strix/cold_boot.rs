//! Cold Boot Sequence — staged model loading for STRIX cold start.
//!
//! When the model first loads (no tensors in VRAM), the cold boot sequence
//! determines the optimal loading order based on tensor classification:
//!
//! ```text
//! Phase 1: Class A (embeddings, norms, lm_head) → Pin in VRAM
//! Phase 2: Class B layers [0, prefetch_window) → Load to Hot
//! Phase 3: Remaining Class B → Stage to Warm (RAM)
//! Phase 4: Class C → Leave Cold (on disk)
//! Phase 5: Class D → Skip entirely (Archival)
//! ```
//!
//! This ensures the inference pipeline can start producing tokens as soon
//! as Phase 1 + Phase 2 complete, while remaining layers stream in
//! progressively.

use super::registry::TensorRegistry;
use super::types::{ResidencyState, TensorClass, TensorId};
use std::time::Duration;

// ── ColdBootStep ─────────────────────────────────────────────────────────

/// A single step in the cold boot loading sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct ColdBootStep {
    /// Tensor to load.
    pub tensor_id: TensorId,
    /// Target residency state after loading.
    pub target_state: ResidencyState,
    /// Loading phase (1 = highest priority).
    pub phase: u8,
    /// Size in bytes to transfer.
    pub size_bytes: usize,
}

// ── ColdBootPlan ─────────────────────────────────────────────────────────

/// Complete cold boot loading plan with per-phase byte estimates.
#[derive(Debug, Clone)]
pub struct ColdBootPlan {
    /// Ordered list of loading steps.
    pub steps: Vec<ColdBootStep>,
    /// Total bytes in Phase 1 (Class A → Pinned).
    pub phase1_bytes: usize,
    /// Total bytes in Phase 2 (Class B prefetch window → Hot).
    pub phase2_bytes: usize,
    /// Total bytes in Phase 3 (Remaining Class B → Warm/RAM).
    pub phase3_bytes: usize,
    /// Total bytes in Phase 4 (Class C → stay Cold).
    pub phase4_bytes: usize,
    /// Number of tensors skipped (Class D).
    pub skipped_count: usize,
}

impl ColdBootPlan {
    /// Total bytes that must be transferred during cold boot.
    pub fn total_transfer_bytes(&self) -> usize {
        self.phase1_bytes + self.phase2_bytes + self.phase3_bytes
    }

    /// Bytes needed before inference can start (Phase 1 + Phase 2).
    pub fn critical_path_bytes(&self) -> usize {
        self.phase1_bytes + self.phase2_bytes
    }

    /// Total number of steps in the plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

// ── ColdBootSequence ─────────────────────────────────────────────────────

/// Orchestrates the initial model loading sequence.
pub struct ColdBootSequence {
    /// Number of layers to eager-load into VRAM during cold boot.
    prefetch_window: usize,
}

impl ColdBootSequence {
    /// Create a new cold boot sequence with the given prefetch window.
    pub fn new(prefetch_window: usize) -> Self {
        Self { prefetch_window }
    }

    /// Analyze the tensor registry and produce a prioritized loading plan.
    pub fn plan(&self, registry: &TensorRegistry) -> ColdBootPlan {
        let mut steps = Vec::new();
        let mut phase1_bytes = 0usize;
        let mut phase2_bytes = 0usize;
        let mut phase3_bytes = 0usize;
        let mut phase4_bytes = 0usize;

        // ── Phase 1: Class A → Pin ──────────────────────────────────
        for id in registry.by_class(TensorClass::A) {
            if let Some(meta) = registry.get(id) {
                steps.push(ColdBootStep {
                    tensor_id: id,
                    target_state: ResidencyState::Pinned,
                    phase: 1,
                    size_bytes: meta.size_bytes,
                });
                phase1_bytes += meta.size_bytes;
            }
        }

        // ── Phase 2: Class B within prefetch window → Hot ───────────
        let mut class_b_ids: Vec<(TensorId, Option<usize>, usize)> = registry
            .by_class(TensorClass::B)
            .into_iter()
            .filter_map(|id| {
                let meta = registry.get(id)?;
                Some((id, meta.layer_id, meta.size_bytes))
            })
            .collect();
        // Sort by layer ID (None → end)
        class_b_ids.sort_by_key(|(_, layer, _)| layer.unwrap_or(usize::MAX));

        for (id, layer_id, size_bytes) in &class_b_ids {
            let within_window = layer_id
                .map(|l| l < self.prefetch_window)
                .unwrap_or(false);

            if within_window {
                steps.push(ColdBootStep {
                    tensor_id: *id,
                    target_state: ResidencyState::Hot,
                    phase: 2,
                    size_bytes: *size_bytes,
                });
                phase2_bytes += size_bytes;
            } else {
                // Phase 3: remaining Class B → Warm (RAM staging)
                steps.push(ColdBootStep {
                    tensor_id: *id,
                    target_state: ResidencyState::Warm,
                    phase: 3,
                    size_bytes: *size_bytes,
                });
                phase3_bytes += size_bytes;
            }
        }

        // ── Phase 4: Class C → Cold (leave on disk) ─────────────────
        for id in registry.by_class(TensorClass::C) {
            if let Some(meta) = registry.get(id) {
                steps.push(ColdBootStep {
                    tensor_id: id,
                    target_state: ResidencyState::Cold,
                    phase: 4,
                    size_bytes: meta.size_bytes,
                });
                phase4_bytes += meta.size_bytes;
            }
        }

        // ── Phase 5: Class D → Skip ─────────────────────────────────
        let skipped_count = registry.by_class(TensorClass::D).len();

        ColdBootPlan {
            steps,
            phase1_bytes,
            phase2_bytes,
            phase3_bytes,
            phase4_bytes,
            skipped_count,
        }
    }

    /// Estimate cold boot time from a plan and storage throughput.
    ///
    /// `throughput_mb_s`: sustained read throughput in MB/s (e.g. 3500 for NVMe).
    pub fn estimate_time(plan: &ColdBootPlan, throughput_mb_s: f64) -> Duration {
        if throughput_mb_s <= 0.0 {
            return Duration::MAX;
        }
        let total_bytes = plan.total_transfer_bytes() as f64;
        let total_mb = total_bytes / (1024.0 * 1024.0);
        let seconds = total_mb / throughput_mb_s;
        Duration::from_secs_f64(seconds)
    }

    /// Estimate time to first token (Phase 1 + Phase 2 only).
    pub fn estimate_ttft(plan: &ColdBootPlan, throughput_mb_s: f64) -> Duration {
        if throughput_mb_s <= 0.0 {
            return Duration::MAX;
        }
        let critical_bytes = plan.critical_path_bytes() as f64;
        let critical_mb = critical_bytes / (1024.0 * 1024.0);
        let seconds = critical_mb / throughput_mb_s;
        Duration::from_secs_f64(seconds)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strix::types::DType;

    fn setup_registry() -> TensorRegistry {
        let mut reg = TensorRegistry::new();
        // Class A: embedding
        reg.register("embd".into(), vec![32000, 4096], DType::F16, 256_000_000, TensorClass::A, None);
        // Class A: output norm
        reg.register("norm".into(), vec![4096], DType::F32, 16_384, TensorClass::A, None);
        // Class B: 4 layers
        for l in 0..4 {
            reg.register(
                format!("blk.{l}.q"),
                vec![4096, 4096],
                DType::Q4_K,
                8_000_000,
                TensorClass::B,
                Some(l),
            );
        }
        // Class C: KV cache
        reg.register("kv.0".into(), vec![2048, 128], DType::F16, 500_000, TensorClass::C, Some(0));
        // Class D: archival
        reg.register("archive".into(), vec![100], DType::F32, 400, TensorClass::D, None);
        reg
    }

    #[test]
    fn plan_ordering_respects_class_priority() {
        let cbs = ColdBootSequence::new(2);
        let reg = setup_registry();
        let plan = cbs.plan(&reg);

        // Phase 1 steps should come before Phase 2, etc.
        let phases: Vec<u8> = plan.steps.iter().map(|s| s.phase).collect();
        // Verify phases are non-decreasing
        for window in phases.windows(2) {
            assert!(window[0] <= window[1], "Phases should be ordered: {:?}", phases);
        }
    }

    #[test]
    fn class_a_always_phase_1() {
        let cbs = ColdBootSequence::new(2);
        let reg = setup_registry();
        let plan = cbs.plan(&reg);

        // All Class A tensors should be in Phase 1
        let phase1_ids: Vec<TensorId> = plan.steps
            .iter()
            .filter(|s| s.phase == 1)
            .map(|s| s.tensor_id)
            .collect();
        assert_eq!(phase1_ids.len(), 2); // embd + norm
        for id in &phase1_ids {
            let meta = reg.get(*id).unwrap();
            assert_eq!(meta.class, TensorClass::A);
        }
    }

    #[test]
    fn class_d_skipped() {
        let cbs = ColdBootSequence::new(2);
        let reg = setup_registry();
        let plan = cbs.plan(&reg);

        assert_eq!(plan.skipped_count, 1);
        // Class D should NOT appear in the steps
        let has_class_d = plan.steps.iter().any(|s| {
            reg.get(s.tensor_id)
                .map(|m| m.class == TensorClass::D)
                .unwrap_or(false)
        });
        assert!(!has_class_d, "Class D tensors should be skipped");
    }

    #[test]
    fn byte_estimates_correct() {
        let cbs = ColdBootSequence::new(2);
        let reg = setup_registry();
        let plan = cbs.plan(&reg);

        // Phase 1: 256_000_000 + 16_384 = 256_016_384
        assert_eq!(plan.phase1_bytes, 256_016_384);
        // Phase 2: layers 0 and 1 (within window 2) = 2 × 8_000_000
        assert_eq!(plan.phase2_bytes, 16_000_000);
        // Phase 3: layers 2 and 3 = 2 × 8_000_000
        assert_eq!(plan.phase3_bytes, 16_000_000);
        // Phase 4: KV cache = 500_000
        assert_eq!(plan.phase4_bytes, 500_000);
    }

    #[test]
    fn empty_model_plan() {
        let cbs = ColdBootSequence::new(2);
        let reg = TensorRegistry::new();
        let plan = cbs.plan(&reg);

        assert_eq!(plan.step_count(), 0);
        assert_eq!(plan.total_transfer_bytes(), 0);
        assert_eq!(plan.skipped_count, 0);
    }

    #[test]
    fn time_estimate() {
        let cbs = ColdBootSequence::new(2);
        let reg = setup_registry();
        let plan = cbs.plan(&reg);

        // 3500 MB/s NVMe
        let total_time = ColdBootSequence::estimate_time(&plan, 3500.0);
        assert!(total_time.as_secs_f64() > 0.0);

        let ttft = ColdBootSequence::estimate_ttft(&plan, 3500.0);
        assert!(ttft <= total_time, "TTFT should be ≤ total time");
    }
}

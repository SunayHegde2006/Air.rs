//! Tensor metadata — per-tensor record held by the STRIX registry.
//!
//! `TensorMeta` tracks everything STRIX needs about a single tensor:
//! its identity, shape, classification, current residency state, and
//! access statistics used by the residency scoring function.

use super::types::{DType, ResidencyState, TensorClass, TensorId};

/// Per-tensor metadata record (STRIX Protocol §17).
///
/// The registry holds one `TensorMeta` for every tensor in the model.
/// The scheduler reads these to compute residency scores; the memory
/// manager writes to them when tensors move between tiers.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Unique identifier within this STRIX session.
    pub id: TensorId,
    /// Canonical tensor name (e.g. `"model.layers.12.self_attn.q_proj.weight"`).
    pub name: String,
    /// Tensor dimensions (e.g. `[4096, 4096]` for a weight matrix).
    pub shape: Vec<usize>,
    /// Element data type (or quantisation format).
    pub dtype: DType,
    /// Total size in bytes on the storage tier.
    pub size_bytes: usize,
    /// Priority classification (A/B/C/D).
    pub class: TensorClass,
    /// Current memory tier location.
    pub residency: ResidencyState,
    /// Layer index this tensor belongs to (None for non-layer tensors like embeddings).
    pub layer_id: Option<usize>,
    /// Number of times this tensor has been evicted from VRAM.
    pub eviction_count: u32,
    /// Token step at which this tensor was last accessed.
    pub last_access_step: u64,
    /// Number of active `ResidencyGuard`s preventing eviction.
    pub guard_count: u32,
}

impl TensorMeta {
    /// Create a new `TensorMeta` in the `Cold` state with zero access history.
    pub fn new(
        id: TensorId,
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        size_bytes: usize,
        class: TensorClass,
        layer_id: Option<usize>,
    ) -> Self {
        // Class A tensors start as Pinned once loaded; all others start Cold.
        let residency = ResidencyState::Cold;

        Self {
            id,
            name,
            shape,
            dtype,
            size_bytes,
            class,
            residency,
            layer_id,
            eviction_count: 0,
            last_access_step: 0,
            guard_count: 0,
        }
    }

    /// Returns `true` if this tensor is currently protected by at least one guard.
    pub fn is_guarded(&self) -> bool {
        self.guard_count > 0
    }

    /// Returns `true` if this tensor can be evicted.
    ///
    /// A tensor is evictable if:
    /// - It is `Hot` (not `Pinned`)
    /// - It has no active guards
    pub fn is_evictable(&self) -> bool {
        self.residency == ResidencyState::Hot && self.guard_count == 0
    }

    /// Record an access at the given token step.
    pub fn touch(&mut self, step: u64) {
        self.last_access_step = step;
    }

    /// Record an eviction event.
    pub fn record_eviction(&mut self) {
        self.eviction_count = self.eviction_count.saturating_add(1);
        self.residency = ResidencyState::Warm;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> TensorMeta {
        TensorMeta::new(
            TensorId(0),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![4096, 4096],
            DType::Q4_K,
            8_388_608,
            TensorClass::B,
            Some(0),
        )
    }

    #[test]
    fn tensor_meta_construction() {
        let m = sample_meta();
        assert_eq!(m.id, TensorId(0));
        assert_eq!(m.class, TensorClass::B);
        assert_eq!(m.residency, ResidencyState::Cold);
        assert_eq!(m.eviction_count, 0);
        assert_eq!(m.last_access_step, 0);
        assert_eq!(m.guard_count, 0);
        assert_eq!(m.layer_id, Some(0));
    }

    #[test]
    fn tensor_meta_evictable() {
        let mut m = sample_meta();
        // Cold tensor is not evictable
        assert!(!m.is_evictable());

        // Hot tensor with no guards is evictable
        m.residency = ResidencyState::Hot;
        assert!(m.is_evictable());

        // Hot tensor with a guard is NOT evictable
        m.guard_count = 1;
        assert!(!m.is_evictable());

        // Pinned tensor is NOT evictable
        m.guard_count = 0;
        m.residency = ResidencyState::Pinned;
        assert!(!m.is_evictable());
    }

    #[test]
    fn tensor_meta_touch_and_eviction() {
        let mut m = sample_meta();
        m.touch(42);
        assert_eq!(m.last_access_step, 42);

        m.residency = ResidencyState::Hot;
        m.record_eviction();
        assert_eq!(m.eviction_count, 1);
        assert_eq!(m.residency, ResidencyState::Warm);
    }
}

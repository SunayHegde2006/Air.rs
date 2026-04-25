//! Tensor Registry — central tracking of all tensors in a STRIX session.
//!
//! The `TensorRegistry` is the single source of truth for tensor metadata.
//! It owns every `TensorMeta` record and provides typed query methods for
//! filtering by class, layer, and residency state.
//!
//! The `ResidencyGuard` provides RAII-based eviction protection: while a
//! guard exists for a tensor, that tensor cannot be evicted from VRAM.

use super::meta::TensorMeta;
use super::types::{DType, ResidencyState, TensorClass, TensorId};
use std::collections::HashMap;

// ── TensorRegistry ───────────────────────────────────────────────────────

/// Central registry holding all tensor metadata records.
///
/// Tensors are registered once during model loading and tracked for the
/// entire session. The registry is **not** thread-safe — callers must
/// synchronise externally if shared across threads.
pub struct TensorRegistry {
    tensors: HashMap<TensorId, TensorMeta>,
    next_id: u32,
}

impl TensorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a new tensor and return its assigned `TensorId`.
    ///
    /// The tensor starts in `ResidencyState::Cold` with zero access history.
    pub fn register(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        size_bytes: usize,
        class: TensorClass,
        layer_id: Option<usize>,
    ) -> TensorId {
        let id = TensorId(self.next_id);
        self.next_id += 1;

        let meta = TensorMeta::new(id, name, shape, dtype, size_bytes, class, layer_id);
        self.tensors.insert(id, meta);
        id
    }

    /// Look up a tensor by ID.
    pub fn get(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.get(&id)
    }

    /// Look up a tensor by ID (mutable).
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut TensorMeta> {
        self.tensors.get_mut(&id)
    }

    /// Total number of registered tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    // ── Query Methods ────────────────────────────────────────────────

    /// All tensor IDs belonging to a given class.
    pub fn by_class(&self, class: TensorClass) -> Vec<TensorId> {
        self.tensors
            .values()
            .filter(|m| m.class == class)
            .map(|m| m.id)
            .collect()
    }

    /// All tensor IDs belonging to a given layer.
    pub fn by_layer(&self, layer_id: usize) -> Vec<TensorId> {
        self.tensors
            .values()
            .filter(|m| m.layer_id == Some(layer_id))
            .map(|m| m.id)
            .collect()
    }

    /// All tensor IDs in a given residency state.
    pub fn by_residency(&self, state: ResidencyState) -> Vec<TensorId> {
        self.tensors
            .values()
            .filter(|m| m.residency == state)
            .map(|m| m.id)
            .collect()
    }

    /// All tensors that are currently evictable (Hot + unguarded).
    pub fn evictable(&self) -> Vec<TensorId> {
        self.tensors
            .values()
            .filter(|m| m.is_evictable())
            .map(|m| m.id)
            .collect()
    }

    /// Iterate over all tensor metadata records.
    pub fn iter(&self) -> impl Iterator<Item = &TensorMeta> {
        self.tensors.values()
    }

    /// Compute aggregate statistics about the registry.
    pub fn stats(&self) -> RegistryStats {
        let mut stats = RegistryStats::default();
        for meta in self.tensors.values() {
            stats.total_count += 1;
            stats.total_bytes += meta.size_bytes;
            match meta.residency {
                ResidencyState::Hot | ResidencyState::Pinned => {
                    stats.vram_bytes += meta.size_bytes;
                    stats.vram_count += 1;
                }
                ResidencyState::Warm | ResidencyState::Loading | ResidencyState::Staging => {
                    stats.ram_bytes += meta.size_bytes;
                    stats.ram_count += 1;
                }
                ResidencyState::Cold | ResidencyState::Archival => {
                    stats.storage_bytes += meta.size_bytes;
                    stats.storage_count += 1;
                }
            }
        }
        stats
    }
}

impl Default for TensorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── RegistryStats ────────────────────────────────────────────────────────

/// Aggregate statistics snapshot from `TensorRegistry::stats()`.
#[derive(Debug, Default, Clone)]
pub struct RegistryStats {
    pub total_count: usize,
    pub total_bytes: usize,
    pub vram_count: usize,
    pub vram_bytes: usize,
    pub ram_count: usize,
    pub ram_bytes: usize,
    pub storage_count: usize,
    pub storage_bytes: usize,
}

// ── ResidencyGuard ───────────────────────────────────────────────────────

/// RAII guard that prevents a tensor from being evicted while held.
///
/// Increments `TensorMeta::guard_count` on creation, decrements on drop.
/// While any guard exists for a tensor, `is_evictable()` returns `false`.
///
/// # Safety Contract
///
/// The guard borrows the registry mutably to increment the count, but the
/// guard itself only stores the tensor ID. Dropping the guard requires
/// a mutable reference to the registry again (via `release()`).
///
/// This is an explicit-release pattern rather than true RAII because we
/// cannot hold a `&mut TensorRegistry` inside the guard while the caller
/// also needs access to the registry.
pub struct ResidencyGuard {
    tensor_id: TensorId,
}

impl ResidencyGuard {
    /// Acquire a guard for the given tensor.
    ///
    /// Increments `guard_count` on the tensor's metadata.
    pub fn acquire(registry: &mut TensorRegistry, tensor_id: TensorId) -> Option<Self> {
        let meta = registry.get_mut(tensor_id)?;
        meta.guard_count += 1;
        Some(Self { tensor_id })
    }

    /// The tensor ID this guard protects.
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }

    /// Release the guard, decrementing the tensor's guard count.
    ///
    /// Must be called explicitly (not Drop-based) because we need `&mut TensorRegistry`.
    pub fn release(self, registry: &mut TensorRegistry) {
        if let Some(meta) = registry.get_mut(self.tensor_id) {
            meta.guard_count = meta.guard_count.saturating_sub(1);
        }
        // self is consumed, preventing double-release
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry_with_tensors() -> TensorRegistry {
        let mut reg = TensorRegistry::new();
        // Class A embedding (no layer)
        reg.register(
            "token_embd.weight".into(),
            vec![32000, 4096],
            DType::F16,
            32000 * 4096 * 2,
            TensorClass::A,
            None,
        );
        // Class B layer 0 weights
        reg.register(
            "blk.0.attn_q.weight".into(),
            vec![4096, 4096],
            DType::Q4_K,
            8_388_608,
            TensorClass::B,
            Some(0),
        );
        // Class B layer 1 weights
        reg.register(
            "blk.1.attn_q.weight".into(),
            vec![4096, 4096],
            DType::Q4_K,
            8_388_608,
            TensorClass::B,
            Some(1),
        );
        // Class C (KV cache placeholder)
        reg.register(
            "kv.layer.0".into(),
            vec![2048, 128],
            DType::F16,
            2048 * 128 * 2,
            TensorClass::C,
            Some(0),
        );
        reg
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = TensorRegistry::new();
        let id = reg.register(
            "test.weight".into(),
            vec![10, 10],
            DType::F32,
            400,
            TensorClass::B,
            Some(0),
        );
        assert_eq!(reg.len(), 1);
        let meta = reg.get(id).unwrap();
        assert_eq!(meta.name, "test.weight");
        assert_eq!(meta.size_bytes, 400);
        assert_eq!(meta.residency, ResidencyState::Cold);
    }

    #[test]
    fn sequential_ids() {
        let mut reg = TensorRegistry::new();
        let id0 = reg.register("a".into(), vec![1], DType::F32, 4, TensorClass::B, None);
        let id1 = reg.register("b".into(), vec![1], DType::F32, 4, TensorClass::B, None);
        let id2 = reg.register("c".into(), vec![1], DType::F32, 4, TensorClass::B, None);
        assert_eq!(id0, TensorId(0));
        assert_eq!(id1, TensorId(1));
        assert_eq!(id2, TensorId(2));
    }

    #[test]
    fn filter_by_class() {
        let reg = make_registry_with_tensors();
        let class_a = reg.by_class(TensorClass::A);
        assert_eq!(class_a.len(), 1);
        let class_b = reg.by_class(TensorClass::B);
        assert_eq!(class_b.len(), 2);
        let class_c = reg.by_class(TensorClass::C);
        assert_eq!(class_c.len(), 1);
        let class_d = reg.by_class(TensorClass::D);
        assert!(class_d.is_empty());
    }

    #[test]
    fn filter_by_layer() {
        let reg = make_registry_with_tensors();
        let layer0 = reg.by_layer(0);
        assert_eq!(layer0.len(), 2); // B + C on layer 0
        let layer1 = reg.by_layer(1);
        assert_eq!(layer1.len(), 1); // B on layer 1
        let layer99 = reg.by_layer(99);
        assert!(layer99.is_empty());
    }

    #[test]
    fn filter_by_residency() {
        let reg = make_registry_with_tensors();
        // All tensors start Cold
        let cold = reg.by_residency(ResidencyState::Cold);
        assert_eq!(cold.len(), 4);
        let hot = reg.by_residency(ResidencyState::Hot);
        assert!(hot.is_empty());
    }

    #[test]
    fn evictable_logic() {
        let mut reg = make_registry_with_tensors();
        // All Cold — none evictable
        assert!(reg.evictable().is_empty());

        // Make tensor 1 Hot — it becomes evictable
        reg.get_mut(TensorId(1)).unwrap().residency = ResidencyState::Hot;
        assert_eq!(reg.evictable().len(), 1);

        // Add a guard — no longer evictable
        reg.get_mut(TensorId(1)).unwrap().guard_count = 1;
        assert!(reg.evictable().is_empty());
    }

    #[test]
    fn residency_guard_acquire_release() {
        let mut reg = make_registry_with_tensors();
        let id = TensorId(0);

        // Acquire guard
        let guard = ResidencyGuard::acquire(&mut reg, id).unwrap();
        assert_eq!(reg.get(id).unwrap().guard_count, 1);

        // Acquire second guard
        let guard2 = ResidencyGuard::acquire(&mut reg, id).unwrap();
        assert_eq!(reg.get(id).unwrap().guard_count, 2);

        // Release first guard
        guard.release(&mut reg);
        assert_eq!(reg.get(id).unwrap().guard_count, 1);

        // Release second guard
        guard2.release(&mut reg);
        assert_eq!(reg.get(id).unwrap().guard_count, 0);
    }

    #[test]
    fn stats_accuracy() {
        let mut reg = make_registry_with_tensors();
        // All Cold → storage
        let stats = reg.stats();
        assert_eq!(stats.total_count, 4);
        assert_eq!(stats.storage_count, 4);
        assert_eq!(stats.vram_count, 0);
        assert_eq!(stats.ram_count, 0);

        // Move one to Hot → VRAM
        reg.get_mut(TensorId(1)).unwrap().residency = ResidencyState::Hot;
        let stats = reg.stats();
        assert_eq!(stats.vram_count, 1);
        assert_eq!(stats.storage_count, 3);
        assert_eq!(stats.vram_bytes, 8_388_608);
    }
}

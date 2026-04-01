//! STRIX Bridge — high-level orchestrator for tensor memory management.
//!
//! `StrixBridge` wires together the registry, arena, scheduler, and I/O engine
//! into a single façade with simple `register` / `load` / `evict` / `tick`
//! semantics. The bridge is the primary integration point for
//! `WeightStreamer` and `InferenceGenerator`.
//!
//! STRIX Protocol §14 — orchestration layer.

use super::arena::{Allocation, VramArena};
use super::config::StrixConfig;
use super::io_engine::{IoEngine, IoPriority, IoRequest};
use super::registry::{RegistryStats, TensorRegistry};
use super::scheduler::{ResidencyScheduler, SchedulerAction};
use super::types::{DType, ResidencyState, TensorClass, TensorId};
use std::collections::HashMap;

// ── BridgeError ──────────────────────────────────────────────────────────

/// Errors returned by bridge operations.
#[derive(Debug)]
pub enum BridgeError {
    /// Tensor not found in the registry.
    UnknownTensor(TensorId),
    /// No arena space available for loading.
    OutOfVram { needed: usize, available: usize },
    /// Tensor is already in the requested state.
    AlreadyInState { id: TensorId, state: ResidencyState },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::UnknownTensor(id) => write!(f, "unknown tensor {:?}", id),
            BridgeError::OutOfVram { needed, available } => {
                write!(f, "need {} bytes, only {} available", needed, available)
            }
            BridgeError::AlreadyInState { id, state } => {
                write!(f, "tensor {:?} already in state {:?}", id, state)
            }
        }
    }
}

impl std::error::Error for BridgeError {}

// ── BridgeStats ──────────────────────────────────────────────────────────

/// Aggregate statistics from the bridge.
#[derive(Debug, Default, Clone)]
pub struct BridgeStats {
    /// Registry stats (tensor counts per tier).
    pub registry: RegistryStats,
    /// Arena utilisation ratio (0.0–1.0).
    pub arena_utilization: f64,
    /// Arena bytes used.
    pub arena_used: usize,
    /// Arena bytes available.
    pub arena_available: usize,
    /// Number of I/O requests pending.
    pub io_pending: usize,
    /// Cumulative loads executed.
    pub total_loads: u64,
    /// Cumulative evictions executed.
    pub total_evictions: u64,
}

// ── StrixBridge ──────────────────────────────────────────────────────────

/// High-level orchestrator that connects all STRIX modules.
///
/// The bridge owns the registry, arena, scheduler, and I/O engine.
/// External code interacts exclusively through this façade.
pub struct StrixBridge {
    registry: TensorRegistry,
    arena: VramArena,
    scheduler: ResidencyScheduler,
    io_engine: IoEngine,
    /// Arena allocations keyed by TensorId.
    allocations: HashMap<TensorId, Allocation>,
    /// Cumulative counters.
    total_loads: u64,
    total_evictions: u64,
}

impl StrixBridge {
    /// Build a bridge from configuration and VRAM budget.
    ///
    /// `vram_total`: total VRAM in bytes (from `GpuHal::info()`).
    pub fn new(config: &StrixConfig, vram_total: usize) -> Self {
        let safety_margin = config.vram_safety_margin_mb * 1024 * 1024;
        Self {
            registry: TensorRegistry::new(),
            arena: VramArena::new(vram_total, safety_margin),
            scheduler: ResidencyScheduler::new(config),
            io_engine: IoEngine::new(4), // 4 concurrent I/O ops
            allocations: HashMap::new(),
            total_loads: 0,
            total_evictions: 0,
        }
    }

    // ── Registration ─────────────────────────────────────────────────

    /// Register a new tensor. Returns the assigned `TensorId`.
    pub fn register_tensor(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        size_bytes: usize,
        class: TensorClass,
        layer_id: Option<usize>,
    ) -> TensorId {
        self.registry
            .register(name, shape, dtype, size_bytes, class, layer_id)
    }

    // ── Load / Evict ─────────────────────────────────────────────────

    /// Mark a tensor as loaded into VRAM.
    ///
    /// Allocates space in the arena and transitions the tensor to `Hot`.
    /// If the tensor is Class A, it transitions to `Pinned` instead.
    pub fn load_tensor(&mut self, id: TensorId) -> Result<Allocation, BridgeError> {
        let meta = self
            .registry
            .get(id)
            .ok_or(BridgeError::UnknownTensor(id))?;

        if meta.residency == ResidencyState::Hot || meta.residency == ResidencyState::Pinned {
            return Err(BridgeError::AlreadyInState {
                id,
                state: meta.residency,
            });
        }

        let size = meta.size_bytes;
        let is_class_a = meta.class == TensorClass::A;

        // Try to allocate in the arena.
        let alloc = self
            .arena
            .allocate(size, 256) // 256-byte alignment for GPU
            .ok_or(BridgeError::OutOfVram {
                needed: size,
                available: self.arena.available(),
            })?;

        // Transition state.
        let meta = self.registry.get_mut(id).unwrap();
        meta.residency = if is_class_a {
            ResidencyState::Pinned
        } else {
            ResidencyState::Hot
        };

        self.allocations.insert(id, alloc);
        self.total_loads += 1;
        Ok(alloc)
    }

    /// Evict a tensor from VRAM.
    ///
    /// Frees the arena allocation and transitions the tensor to `Warm`.
    pub fn evict_tensor(&mut self, id: TensorId) -> Result<(), BridgeError> {
        let meta = self
            .registry
            .get(id)
            .ok_or(BridgeError::UnknownTensor(id))?;

        if meta.residency != ResidencyState::Hot {
            return Err(BridgeError::AlreadyInState {
                id,
                state: meta.residency,
            });
        }

        // Free the arena allocation.
        if let Some(alloc) = self.allocations.remove(&id) {
            self.arena.free(alloc);
        }

        // Transition to Warm and record eviction.
        let meta = self.registry.get_mut(id).unwrap();
        meta.residency = ResidencyState::Warm;
        meta.eviction_count += 1;
        self.total_evictions += 1;
        Ok(())
    }

    // ── Tick (scheduling) ────────────────────────────────────────────

    /// Run one scheduling pass and execute the resulting actions.
    ///
    /// Returns the actions that were executed.
    pub fn tick(
        &mut self,
        current_layer: usize,
        current_step: u64,
    ) -> Vec<SchedulerAction> {
        let actions = self
            .scheduler
            .tick(&self.registry, &self.arena, current_layer, current_step);

        // Execute each action.
        for action in &actions {
            match action {
                SchedulerAction::Evict(id) => {
                    let _ = self.evict_tensor(*id);
                }
                SchedulerAction::Load(id) => {
                    let _ = self.load_tensor(*id);
                }
                SchedulerAction::Pin(id) => {
                    // Pin = load as Class A.
                    let _ = self.load_tensor(*id);
                }
                SchedulerAction::Noop => {}
            }
        }

        actions
    }

    // ── Prefetch ─────────────────────────────────────────────────────

    /// Submit prefetch I/O requests for tensors in upcoming layers.
    ///
    /// `cursor_layer`: the layer about to execute.
    /// `window`: how many layers ahead to prefetch.
    pub fn prefetch_window(&mut self, cursor_layer: usize, window: usize) {
        for offset in 1..=window {
            let target_layer = cursor_layer + offset;
            for id in self.registry.by_layer(target_layer) {
                if let Some(meta) = self.registry.get(id) {
                    if meta.residency == ResidencyState::Cold
                        || meta.residency == ResidencyState::Warm
                    {
                        let priority = IoPriority::from_class(meta.class, true);
                        self.io_engine.submit(IoRequest {
                            tensor_id: id,
                            file_offset: 0, // Real offset would come from GGUF index
                            size: meta.size_bytes,
                            priority,
                        });
                    }
                }
            }
        }
    }

    // ── Access ────────────────────────────────────────────────────────

    /// Immutable access to the tensor registry.
    pub fn registry(&self) -> &TensorRegistry {
        &self.registry
    }

    /// Mutable access to the tensor registry.
    pub fn registry_mut(&mut self) -> &mut TensorRegistry {
        &mut self.registry
    }

    /// Immutable access to the arena.
    pub fn arena(&self) -> &VramArena {
        &self.arena
    }

    /// Immutable access to the I/O engine.
    pub fn io_engine(&self) -> &IoEngine {
        &self.io_engine
    }

    // ── Stats ────────────────────────────────────────────────────────

    /// Aggregate stats combining registry, arena, and I/O engine.
    pub fn stats(&self) -> BridgeStats {
        BridgeStats {
            registry: self.registry.stats(),
            arena_utilization: self.arena.utilization(),
            arena_used: self.arena.used(),
            arena_available: self.arena.available(),
            io_pending: self.io_engine.pending_count(),
            total_loads: self.total_loads,
            total_evictions: self.total_evictions,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> StrixConfig {
        StrixConfig {
            vram_safety_margin_mb: 0,
            prefetch_window_layers: 2,
            eviction_headroom_fraction: 0.0,
            ..StrixConfig::default()
        }
    }

    #[test]
    fn register_and_load() {
        let mut bridge = StrixBridge::new(&small_config(), 10_000);
        let id = bridge.register_tensor(
            "test.weight".into(),
            vec![100, 100],
            DType::Q4_K,
            1000,
            TensorClass::B,
            Some(0),
        );
        assert_eq!(bridge.registry().len(), 1);

        let alloc = bridge.load_tensor(id).unwrap();
        assert_eq!(alloc.size, 1000);
        assert_eq!(
            bridge.registry().get(id).unwrap().residency,
            ResidencyState::Hot,
        );
        assert_eq!(bridge.arena().used(), 1000);
    }

    #[test]
    fn load_and_evict_round_trip() {
        let mut bridge = StrixBridge::new(&small_config(), 10_000);
        let id = bridge.register_tensor(
            "w".into(),
            vec![50],
            DType::F16,
            500,
            TensorClass::B,
            Some(0),
        );

        bridge.load_tensor(id).unwrap();
        assert_eq!(bridge.arena().used(), 500);

        bridge.evict_tensor(id).unwrap();
        assert_eq!(bridge.arena().used(), 0);
        assert_eq!(
            bridge.registry().get(id).unwrap().residency,
            ResidencyState::Warm,
        );
        assert_eq!(bridge.registry().get(id).unwrap().eviction_count, 1);
    }

    #[test]
    fn class_a_gets_pinned() {
        let mut bridge = StrixBridge::new(&small_config(), 10_000);
        let id = bridge.register_tensor(
            "embd".into(),
            vec![1000],
            DType::F16,
            2000,
            TensorClass::A,
            None,
        );

        bridge.load_tensor(id).unwrap();
        assert_eq!(
            bridge.registry().get(id).unwrap().residency,
            ResidencyState::Pinned,
        );
    }

    #[test]
    fn oom_on_load() {
        let mut bridge = StrixBridge::new(&small_config(), 500);
        let id = bridge.register_tensor(
            "big".into(),
            vec![10000],
            DType::F32,
            1000,
            TensorClass::B,
            Some(0),
        );

        let result = bridge.load_tensor(id);
        assert!(result.is_err());
    }

    #[test]
    fn tick_drives_pin_and_load() {
        let mut bridge = StrixBridge::new(&small_config(), 100_000);
        // Class A embedding
        bridge.register_tensor(
            "embd".into(),
            vec![1000],
            DType::F16,
            2000,
            TensorClass::A,
            None,
        );
        // Class B layer 0
        bridge.register_tensor(
            "blk.0.q".into(),
            vec![1000],
            DType::Q4_K,
            1000,
            TensorClass::B,
            Some(0),
        );

        let actions = bridge.tick(0, 0);
        // Should have Pin (for class A) and Load (for layer 0 class B)
        assert!(!actions.is_empty());
        // After tick, the class A tensor should be Pinned
        assert_eq!(
            bridge.registry().get(TensorId(0)).unwrap().residency,
            ResidencyState::Pinned,
        );
    }

    #[test]
    fn prefetch_submits_io_requests() {
        let mut bridge = StrixBridge::new(&small_config(), 100_000);
        for l in 0..5 {
            bridge.register_tensor(
                format!("blk.{l}.q"),
                vec![100],
                DType::Q4_K,
                500,
                TensorClass::B,
                Some(l),
            );
        }

        bridge.prefetch_window(0, 2);
        // Should queue requests for layers 1 and 2
        assert!(bridge.io_engine().pending_count() > 0);
    }

    #[test]
    fn stats_aggregation() {
        let mut bridge = StrixBridge::new(&small_config(), 10_000);
        bridge.register_tensor(
            "w".into(),
            vec![100],
            DType::F16,
            1000,
            TensorClass::B,
            Some(0),
        );
        bridge.load_tensor(TensorId(0)).unwrap();

        let stats = bridge.stats();
        assert_eq!(stats.registry.total_count, 1);
        assert_eq!(stats.registry.vram_count, 1);
        assert_eq!(stats.arena_used, 1000);
        assert_eq!(stats.total_loads, 1);
        assert_eq!(stats.total_evictions, 0);
    }
}

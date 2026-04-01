//! Streamer Adapter — thin façade bridging `WeightStreamer` with STRIX.
//!
//! `StrixStreamerAdapter` translates `WeightStreamer`-style `load_layer` /
//! `release_layer` calls into STRIX `StrixBridge` operations. This is the
//! glue that lets the existing inference pipeline use STRIX-managed memory
//! without changing its public API.
//!
//! STRIX Protocol §15 — integration layer.

use super::bridge::{BridgeError, StrixBridge};
use super::types::{DType, TensorClass, TensorId};

// ── LayerMapping ─────────────────────────────────────────────────────────

/// Tracks which `TensorId`s belong to a layer.
#[derive(Debug, Clone)]
pub struct LayerMapping {
    /// All tensor IDs that belong to this layer.
    pub tensor_ids: Vec<TensorId>,
    /// Total size of the layer in bytes.
    pub total_bytes: usize,
}

// ── StrixStreamerAdapter ─────────────────────────────────────────────────

/// Adapter between `WeightStreamer` and `StrixBridge`.
///
/// The streamer adapter knows how to:
/// 1. Bulk-register an entire model's tensors into the bridge
/// 2. Load/release layers as atomic units
/// 3. Drive prefetching for upcoming layers
pub struct StrixStreamerAdapter {
    /// The underlying STRIX bridge.
    bridge: StrixBridge,
    /// Layer → tensor-ID mapping (built during registration).
    layer_map: Vec<LayerMapping>,
    /// Total number of layers registered.
    num_layers: usize,
    /// Current inference cursor (layer being executed).
    cursor: usize,
    /// Current generation step.
    step: u64,
}

impl StrixStreamerAdapter {
    /// Create a new adapter wrapping an existing bridge.
    pub fn new(bridge: StrixBridge) -> Self {
        Self {
            bridge,
            layer_map: Vec::new(),
            num_layers: 0,
            cursor: 0,
            step: 0,
        }
    }

    // ── Bulk Registration ────────────────────────────────────────────

    /// Register a model's embedding tensors (Class A).
    ///
    /// Returns the assigned `TensorId`s.
    pub fn register_embeddings(
        &mut self,
        tensors: Vec<(String, Vec<usize>, DType, usize)>,
    ) -> Vec<TensorId> {
        tensors
            .into_iter()
            .map(|(name, shape, dtype, size)| {
                self.bridge
                    .register_tensor(name, shape, dtype, size, TensorClass::A, None)
            })
            .collect()
    }

    /// Register a layer's weight tensors (Class B).
    ///
    /// All tensors are assigned to the given `layer_id`.
    /// Returns the `LayerMapping` for this layer.
    pub fn register_layer(
        &mut self,
        layer_id: usize,
        tensors: Vec<(String, Vec<usize>, DType, usize)>,
    ) -> LayerMapping {
        let mut total_bytes = 0;
        let tensor_ids: Vec<TensorId> = tensors
            .into_iter()
            .map(|(name, shape, dtype, size)| {
                total_bytes += size;
                self.bridge.register_tensor(
                    name,
                    shape,
                    dtype,
                    size,
                    TensorClass::B,
                    Some(layer_id),
                )
            })
            .collect();

        let mapping = LayerMapping {
            tensor_ids,
            total_bytes,
        };

        // Ensure the layer_map is large enough.
        while self.layer_map.len() <= layer_id {
            self.layer_map.push(LayerMapping {
                tensor_ids: Vec::new(),
                total_bytes: 0,
            });
        }
        self.layer_map[layer_id] = mapping.clone();
        self.num_layers = self.num_layers.max(layer_id + 1);

        mapping
    }

    // ── Layer Operations ─────────────────────────────────────────────

    /// Load all tensors for a layer into VRAM.
    ///
    /// Returns `Ok(())` on success, or the first error encountered.
    pub fn load_layer(&mut self, layer_id: usize) -> Result<(), BridgeError> {
        if layer_id >= self.layer_map.len() {
            return Ok(()); // No tensors for this layer.
        }

        let ids: Vec<TensorId> = self.layer_map[layer_id].tensor_ids.clone();
        for id in ids {
            // Skip tensors already loaded.
            if let Some(meta) = self.bridge.registry().get(id) {
                use super::types::ResidencyState;
                if meta.residency == ResidencyState::Hot
                    || meta.residency == ResidencyState::Pinned
                {
                    continue;
                }
            }
            self.bridge.load_tensor(id)?;
        }
        Ok(())
    }

    /// Release (evict) all tensors for a layer from VRAM.
    ///
    /// Returns `Ok(())` on success, or the first error encountered.
    pub fn release_layer(&mut self, layer_id: usize) -> Result<(), BridgeError> {
        if layer_id >= self.layer_map.len() {
            return Ok(());
        }

        let ids: Vec<TensorId> = self.layer_map[layer_id].tensor_ids.clone();
        for id in ids {
            if let Some(meta) = self.bridge.registry().get(id) {
                use super::types::ResidencyState;
                if meta.residency == ResidencyState::Hot {
                    self.bridge.evict_tensor(id)?;
                }
            }
        }
        Ok(())
    }

    // ── Inference Cursor ─────────────────────────────────────────────

    /// Advance the inference cursor and run a scheduler tick.
    ///
    /// Call this once per layer execution. It drives prefetching and
    /// eviction decisions automatically.
    pub fn advance_cursor(&mut self) {
        self.cursor += 1;
        if self.cursor >= self.num_layers {
            // Wrap around for next generation pass.
            self.cursor = 0;
            self.step += 1;
        }
        self.bridge.tick(self.cursor, self.step);
    }

    /// Get the current layer cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Get the current generation step.
    pub fn step(&self) -> u64 {
        self.step
    }

    // ── Access ────────────────────────────────────────────────────────

    /// Access the underlying bridge.
    pub fn bridge(&self) -> &StrixBridge {
        &self.bridge
    }

    /// Mutable access to the underlying bridge.
    pub fn bridge_mut(&mut self) -> &mut StrixBridge {
        &mut self.bridge
    }

    /// Number of registered layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::StrixConfig;

    fn test_config() -> StrixConfig {
        StrixConfig {
            vram_safety_margin_mb: 0,
            prefetch_window_layers: 2,
            eviction_headroom_fraction: 0.0,
            ..StrixConfig::default()
        }
    }

    fn setup_adapter() -> StrixStreamerAdapter {
        let bridge = StrixBridge::new(&test_config(), 100_000);
        let mut adapter = StrixStreamerAdapter::new(bridge);

        // Register embeddings.
        adapter.register_embeddings(vec![
            ("token_embd".into(), vec![32000, 4096], DType::F16, 2000),
        ]);

        // Register 4 layers with one weight each.
        for l in 0..4 {
            adapter.register_layer(l, vec![
                (format!("blk.{l}.q"), vec![4096, 4096], DType::Q4_K, 1000),
            ]);
        }

        adapter
    }

    #[test]
    fn register_and_load_layer() {
        let mut adapter = setup_adapter();
        assert_eq!(adapter.num_layers(), 4);

        adapter.load_layer(0).unwrap();
        let stats = adapter.bridge().stats();
        assert_eq!(stats.registry.vram_count, 1); // layer 0 loaded
    }

    #[test]
    fn load_and_release_round_trip() {
        let mut adapter = setup_adapter();
        adapter.load_layer(1).unwrap();

        let used_before = adapter.bridge().arena().used();
        assert!(used_before > 0);

        adapter.release_layer(1).unwrap();
        assert_eq!(adapter.bridge().arena().used(), 0);
    }

    #[test]
    fn advance_cursor_wraps() {
        let mut adapter = setup_adapter();
        assert_eq!(adapter.cursor(), 0);
        assert_eq!(adapter.step(), 0);

        // Advance through all layers.
        for _ in 0..4 {
            adapter.advance_cursor();
        }
        // After 4 advances on 4 layers, should wrap to 0 and increment step.
        assert_eq!(adapter.cursor(), 0);
        assert_eq!(adapter.step(), 1);
    }

    #[test]
    fn double_load_skips_already_loaded() {
        let mut adapter = setup_adapter();
        adapter.load_layer(2).unwrap();
        // Double load should succeed without error (skips already-loaded tensors).
        adapter.load_layer(2).unwrap();
    }
}

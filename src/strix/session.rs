//! STRIX Phase 4 — Air.rs integration session API.
//!
//! Provides the entry-point for the inference loop to interact with STRIX:
//!
//! - `StrixSession::open(model_info, config, vram)` — parse model, register
//!   tensors, run cold boot
//! - `notify_layer_start(layer)` / `notify_layer_end(layer)` — cursor
//!   advancement with prefetch triggering
//! - `acquire_tensor(name)` / `release_tensor(id)` — tensor access with
//!   guard counting that prevents eviction while in use
//!
//! STRIX Protocol §20 — Air.rs integration API.

use super::bridge::{BridgeStats, StrixBridge};
use super::compat::{
    classify_tensor, normalize_tensor_name, parse_model_file,
    CompatError, GgufTensorInfo, ModelArchitecture, UnifiedModel,
};
use super::config::StrixConfig;
use super::types::{TensorClass, TensorId};
use std::collections::HashMap;

// ── SessionError ─────────────────────────────────────────────────────────

/// Errors from session operations.
#[derive(Debug)]
pub enum SessionError {
    /// Tensor not found by name.
    TensorNotFound(String),
    /// Tensor is not currently GPU-resident.
    NotResident(String),
    /// Compatibility layer error (GGUF parsing).
    Compat(CompatError),
    /// Bridge error (VRAM allocation, etc.).
    Bridge(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::TensorNotFound(name) => {
                write!(f, "tensor not found: '{}'", name)
            }
            SessionError::NotResident(name) => {
                write!(f, "tensor '{}' is not GPU-resident", name)
            }
            SessionError::Compat(e) => write!(f, "compat error: {}", e),
            SessionError::Bridge(msg) => write!(f, "bridge error: {}", msg),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<CompatError> for SessionError {
    fn from(e: CompatError) -> Self {
        SessionError::Compat(e)
    }
}

// ── SessionState ─────────────────────────────────────────────────────────

/// Tracks the current state of a STRIX session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session created but not yet loaded.
    Created,
    /// Cold boot in progress (loading Class A tensors).
    Booting,
    /// Ready for inference.
    Ready,
    /// Session closed.
    Closed,
}

// ── StrixSession ─────────────────────────────────────────────────────────

/// High-level session for the inference loop to interact with STRIX.
///
/// The session owns the bridge and maintains a name → TensorId lookup
/// for the tensor access API.
pub struct StrixSession {
    /// The STRIX bridge orchestrating memory management.
    bridge: StrixBridge,
    /// Name → TensorId mapping for O(1) lookup by name.
    name_map: HashMap<String, TensorId>,
    /// Detected model architecture.
    architecture: ModelArchitecture,
    /// Current execution cursor (layer index).
    cursor_layer: usize,
    /// Total number of layers in the model.
    total_layers: usize,
    /// Token step counter (incremented each layer start).
    step: u64,
    /// Session state.
    state: SessionState,
    /// Number of layers ahead to prefetch.
    prefetch_window: usize,
}

impl StrixSession {
    /// Open a new STRIX session from pre-parsed tensor info.
    ///
    /// This is the primary entry-point for the inference loop. It:
    /// 1. Creates a bridge with the given VRAM budget
    /// 2. Registers all tensors (with auto-classification)
    /// 3. Sets up the name → id lookup
    ///
    /// After `open`, call `cold_boot()` to load Class A tensors.
    pub fn open(
        tensors: &[GgufTensorInfo],
        architecture: ModelArchitecture,
        config: &StrixConfig,
        vram_bytes: usize,
    ) -> Result<Self, SessionError> {
        let mut bridge = StrixBridge::new(config, vram_bytes);
        let mut name_map = HashMap::with_capacity(tensors.len());
        let mut max_layer: usize = 0;

        for tensor_info in tensors {
            let norm = normalize_tensor_name(&tensor_info.name);
            let class = classify_tensor(&norm);
            let layer_id = norm.layer;

            if let Some(l) = layer_id {
                max_layer = max_layer.max(l);
            }

            let id = bridge.register_tensor(
                tensor_info.name.clone(),
                tensor_info.shape.clone(),
                tensor_info.dtype,
                tensor_info.size_bytes,
                class,
                layer_id,
            );

            name_map.insert(tensor_info.name.clone(), id);
        }

        Ok(Self {
            bridge,
            name_map,
            architecture,
            cursor_layer: 0,
            total_layers: if max_layer > 0 { max_layer + 1 } else { 0 },
            step: 0,
            state: SessionState::Created,
            prefetch_window: config.prefetch_window_layers,
        })
    }

    /// Open a session from a `UnifiedModel` (any format: GGUF, SafeTensors, PyTorch, ONNX).
    ///
    /// This bridges the multi-format model compatibility layer into the
    /// STRIX session lifecycle. The `UnifiedTensorInfo` structs carry the
    /// same data as `GgufTensorInfo` (name, shape, dtype, size_bytes).
    pub fn open_unified(
        model: &UnifiedModel,
        config: &StrixConfig,
        vram_bytes: usize,
    ) -> Result<Self, SessionError> {
        // Convert UnifiedTensorInfo → GgufTensorInfo for the existing pipeline
        let tensors: Vec<GgufTensorInfo> = model.tensors.iter().map(|t| {
            GgufTensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                offset: t.data_offset,
                size_bytes: t.size_bytes,
            }
        }).collect();

        Self::open(&tensors, model.architecture, config, vram_bytes)
    }

    /// Open a session directly from a model file path (auto-detects format).
    ///
    /// Combines `parse_model_file()` + `open_unified()` into a single call.
    /// Supported formats: `.gguf`, `.safetensors`, `.bin`/`.pt`/`.pth`, `.onnx`.
    pub fn open_from_file(
        path: &std::path::Path,
        config: &StrixConfig,
        vram_bytes: usize,
    ) -> Result<Self, SessionError> {
        let model = parse_model_file(path)?;
        Self::open_unified(&model, config, vram_bytes)
    }

    /// Run cold boot — load all Class A (pinned) tensors into VRAM.
    ///
    /// Should be called once after `open()` before inference starts.
    pub fn cold_boot(&mut self) -> Result<(), SessionError> {
        self.state = SessionState::Booting;

        // Collect Class A tensor IDs
        let class_a_ids: Vec<TensorId> = self
            .name_map
            .values()
            .filter(|id| {
                self.bridge
                    .registry()
                    .get(**id)
                    .is_some_and(|m| m.class == TensorClass::A)
            })
            .copied()
            .collect();

        for id in class_a_ids {
            self.bridge
                .load_tensor(id)
                .map_err(|e| SessionError::Bridge(format!("{}", e)))?;
        }

        self.state = SessionState::Ready;
        Ok(())
    }

    /// Notify that inference is starting a new layer.
    ///
    /// - Advances the cursor
    /// - Increments the step counter
    /// - Triggers prefetch for upcoming layers
    /// - Runs a scheduler tick
    pub fn notify_layer_start(&mut self, layer_idx: usize) {
        self.cursor_layer = layer_idx;
        self.step += 1;

        // Prefetch upcoming layers
        self.bridge
            .prefetch_window(layer_idx, self.prefetch_window);

        // Run one scheduling cycle
        self.bridge.tick(layer_idx, self.step);
    }

    /// Notify that a layer has completed.
    ///
    /// Currently a no-op — the scheduler handles eviction decisions
    /// during `tick()`. This hook exists for future layer-end analytics.
    pub fn notify_layer_end(&mut self, _layer_idx: usize) {
        // Reserved for future per-layer analytics / eviction hints.
    }

    /// Look up a tensor by name, returning its `TensorId`.
    pub fn tensor_id(&self, name: &str) -> Option<TensorId> {
        self.name_map.get(name).copied()
    }

    /// Acquire access to a tensor by name.
    ///
    /// Increments the tensor's `guard_count`, preventing the scheduler
    /// from evicting it. The caller **must** call `release_tensor(id)`
    /// when done (typically via a scope guard or manual cleanup).
    ///
    /// Returns the `TensorId` for subsequent access.
    pub fn acquire_tensor(&mut self, name: &str) -> Result<TensorId, SessionError> {
        let id = self
            .name_map
            .get(name)
            .copied()
            .ok_or_else(|| SessionError::TensorNotFound(name.to_string()))?;

        // Ensure the tensor is GPU-resident
        {
            let meta = self
                .bridge
                .registry()
                .get(id)
                .ok_or_else(|| SessionError::TensorNotFound(name.to_string()))?;

            if !meta.residency.is_gpu_resident() {
                return Err(SessionError::NotResident(name.to_string()));
            }
        }

        // Increment guard count and update access step
        let meta = self.bridge.registry_mut().get_mut(id).unwrap();
        meta.guard_count += 1;
        meta.last_access_step = self.step;

        Ok(id)
    }

    /// Release a previously acquired tensor.
    ///
    /// Decrements the tensor's `guard_count`. Once the count reaches 0,
    /// the scheduler is free to evict the tensor.
    pub fn release_tensor(&mut self, id: TensorId) {
        if let Some(meta) = self.bridge.registry_mut().get_mut(id) {
            meta.guard_count = meta.guard_count.saturating_sub(1);
        }
    }

    /// Query the current guard count for a tensor.
    pub fn guard_count(&self, id: TensorId) -> u32 {
        self.bridge
            .registry()
            .get(id)
            .map_or(0, |m| m.guard_count)
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Current layer cursor position.
    pub fn cursor_layer(&self) -> usize {
        self.cursor_layer
    }

    /// Total number of layers in the model.
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Token step counter.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Detected model architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Number of registered tensors.
    pub fn tensor_count(&self) -> usize {
        self.name_map.len()
    }

    /// Aggregate bridge statistics.
    pub fn stats(&self) -> BridgeStats {
        self.bridge.stats()
    }

    /// Immutable access to the bridge (for advanced inspection).
    pub fn bridge(&self) -> &StrixBridge {
        &self.bridge
    }

    /// Close the session.
    pub fn close(&mut self) {
        self.state = SessionState::Closed;
    }
}

// ── SessionGuard (RAII convenience wrapper) ──────────────────────────────

/// RAII guard that automatically releases a tensor when dropped.
///
/// Created via `SessionGuard::new(&mut session, name)`. On drop, it
/// calls `session.release_tensor(id)` to decrement the guard count.
///
/// # Lifetime
///
/// The guard borrows the session mutably. If you need to hold multiple
/// guards simultaneously, use the explicit `acquire_tensor` / `release_tensor`
/// API instead.
pub struct SessionGuard<'a> {
    session: &'a mut StrixSession,
    tensor_id: TensorId,
}

impl<'a> SessionGuard<'a> {
    /// Acquire a tensor guard by name.
    pub fn new(session: &'a mut StrixSession, name: &str) -> Result<Self, SessionError> {
        let id = session.acquire_tensor(name)?;
        Ok(Self {
            session,
            tensor_id: id,
        })
    }

    /// The tensor ID this guard protects.
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }
}

impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        self.session.release_tensor(self.tensor_id);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strix::types::DType;

    fn test_config() -> StrixConfig {
        StrixConfig {
            vram_safety_margin_mb: 0,
            prefetch_window_layers: 2,
            eviction_headroom_fraction: 0.0,
            ..StrixConfig::default()
        }
    }

    fn sample_tensors() -> Vec<GgufTensorInfo> {
        vec![
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                shape: vec![4096, 32000],
                dtype: DType::F16,
                offset: 0,
                size_bytes: 2000,
            },
            GgufTensorInfo {
                name: "output_norm.weight".to_string(),
                shape: vec![4096],
                dtype: DType::F32,
                offset: 2000,
                size_bytes: 500,
            },
            GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![4096, 4096],
                dtype: DType::Q4_K,
                offset: 2500,
                size_bytes: 1000,
            },
            GgufTensorInfo {
                name: "blk.0.ffn_down.weight".to_string(),
                shape: vec![11008, 4096],
                dtype: DType::Q4_K,
                offset: 3500,
                size_bytes: 1500,
            },
            GgufTensorInfo {
                name: "blk.1.attn_q.weight".to_string(),
                shape: vec![4096, 4096],
                dtype: DType::Q4_K,
                offset: 5000,
                size_bytes: 1000,
            },
        ]
    }

    #[test]
    fn session_open_registers_tensors() {
        let tensors = sample_tensors();
        let session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();

        assert_eq!(session.tensor_count(), 5);
        assert_eq!(session.total_layers(), 2); // layers 0 and 1
        assert_eq!(session.state(), SessionState::Created);
        assert_eq!(session.architecture(), ModelArchitecture::Llama);
    }

    #[test]
    fn session_cold_boot_loads_class_a() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();

        session.cold_boot().unwrap();
        assert_eq!(session.state(), SessionState::Ready);

        // Class A tensors (token_embd) should be loaded; output_norm is Class C
        let stats = session.stats();
        assert!(stats.arena_used > 0);
        assert!(stats.total_loads >= 1); // at least the token_embd Class A tensor
    }

    #[test]
    fn session_layer_notifications() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();
        session.cold_boot().unwrap();

        session.notify_layer_start(0);
        assert_eq!(session.cursor_layer(), 0);
        assert_eq!(session.step(), 1);

        session.notify_layer_end(0);

        session.notify_layer_start(1);
        assert_eq!(session.cursor_layer(), 1);
        assert_eq!(session.step(), 2);
    }

    #[test]
    fn session_acquire_release_guard_count() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();
        session.cold_boot().unwrap();

        // token_embd is Class A, loaded during cold boot
        let id = session.acquire_tensor("token_embd.weight").unwrap();
        assert_eq!(id, TensorId(0)); // first registered tensor
        assert_eq!(session.guard_count(id), 1);

        // Acquire again — guard count should be 2
        let _ = session.acquire_tensor("token_embd.weight").unwrap();
        assert_eq!(session.guard_count(id), 2);

        // Release once — guard count back to 1
        session.release_tensor(id);
        assert_eq!(session.guard_count(id), 1);

        // Release again — back to 0
        session.release_tensor(id);
        assert_eq!(session.guard_count(id), 0);
    }

    #[test]
    fn session_raii_guard_drops_correctly() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();
        session.cold_boot().unwrap();

        let id;
        {
            let guard = SessionGuard::new(&mut session, "token_embd.weight").unwrap();
            id = guard.tensor_id();
            // Guard is alive — count should be 1
            // (Can't check here because guard borrows session mutably)
        }
        // Guard dropped — count should be 0
        assert_eq!(session.guard_count(id), 0);
    }

    #[test]
    fn session_acquire_not_found() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();

        let result = session.acquire_tensor("nonexistent.weight");
        assert!(result.is_err());
    }

    #[test]
    fn session_acquire_not_resident() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();
        // Don't cold boot — nothing is loaded

        // blk.0.attn_q is Class B, not yet loaded
        let result = session.acquire_tensor("blk.0.attn_q.weight");
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_id_lookup() {
        let tensors = sample_tensors();
        let session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();

        assert!(session.tensor_id("token_embd.weight").is_some());
        assert!(session.tensor_id("blk.0.attn_q.weight").is_some());
        assert!(session.tensor_id("nonexistent").is_none());
    }

    #[test]
    fn session_close() {
        let tensors = sample_tensors();
        let mut session = StrixSession::open(
            &tensors,
            ModelArchitecture::Llama,
            &test_config(),
            100_000,
        )
        .unwrap();

        session.close();
        assert_eq!(session.state(), SessionState::Closed);
    }
}

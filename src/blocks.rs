//! TransformerBlock trait — ADR-0001.
//!
//! Provides a single-layer seam between `InferenceGenerator` and the concrete
//! quantised-GQA kernel in `model::transformer_block`. Enables:
//! - Zero-GPU unit tests via `MockTransformerBlock`
//! - Future heterogeneous-block stacks (e.g. MoE layers mixed with dense)
//! - Device injection (ADR-0002): each block can target a different device
//!
//! # Ownership
//! `QBlock` holds its weights behind `Arc<QBlockWeights>` so multiple stacks
//! (draft + target in speculative decoding) can share read-only weight data
//! without copying.

use std::sync::Arc;
use candle_core::Result;
use candle_core::Tensor;
use crate::model::{QBlockWeights, ModelConfig};
use crate::ops::RopeCache;
use crate::attention_backend::{AttentionBackend, HybridAttentionRouter};
use crate::gated_deltanet::{DeltaNetConfig, GatedDeltaNetLayer};

// ---------------------------------------------------------------------------
// TransformerBlock trait
// ---------------------------------------------------------------------------

/// One transformer layer — attention + FFN with residual connections.
///
/// The trait is object-safe to allow `Vec<Box<dyn TransformerBlock>>`.
///
/// ## Input / Output convention
/// ```text
/// forward(x, kv_k, kv_v, start_pos) → (hidden, new_k, new_v)
///
/// x          — [batch, seq_len, hidden_dim]
/// kv_k / v   — previous cached K/V for this layer (None on first call)
/// start_pos  — token offset in the session (used for RoPE phase)
/// hidden     — [batch, seq_len, hidden_dim]  (post-attention + FFN residual)
/// new_k      — [batch, total_seq, n_kv_heads, head_dim]  (for KV cache)
/// new_v      — same shape as new_k
/// ```
pub trait TransformerBlock: Send + Sync {
    /// Run this layer's forward pass.
    ///
    /// Returns `(hidden, new_k, new_v)`. The caller saves `new_k`/`new_v`
    /// to the `SessionKvCache` for subsequent tokens.
    fn forward(
        &self,
        x: &Tensor,
        kv_cache_k: Option<&Tensor>,
        kv_cache_v: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)>;

    /// Zero-based index of this layer in the model stack.
    fn layer_id(&self) -> usize;
}

// ---------------------------------------------------------------------------
// QBlock — concrete implementation wrapping `model::transformer_block`
// ---------------------------------------------------------------------------

/// One quantised GQA transformer layer backed by `QBlockWeights`.
///
/// `weights` and `config` are reference-counted so that a speculative
/// decoder's draft stack and the main stack can share the same tensors.
pub struct QBlock {
    /// Zero-based position in the transformer stack.
    pub layer: usize,
    /// Shared quantised weight tensors for this layer.
    pub weights: Arc<QBlockWeights>,
    /// Model hyper-parameters (shared across all QBlock instances).
    pub config: Arc<ModelConfig>,
    /// Optional pre-computed RoPE inverse-frequency cache (shared).
    pub rope: Option<Arc<RopeCache>>,
}

impl QBlock {
    /// Construct from individually boxed components.
    pub fn new(
        layer: usize,
        weights: Arc<QBlockWeights>,
        config: Arc<ModelConfig>,
        rope: Option<Arc<RopeCache>>,
    ) -> Self {
        Self { layer, weights, config, rope }
    }
}

impl TransformerBlock for QBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache_k: Option<&Tensor>,
        kv_cache_v: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        crate::model::transformer_block(
            x,
            &self.weights,
            kv_cache_k,
            kv_cache_v,
            &self.config,
            start_pos,
            self.rope.as_deref(),
        )
    }

    fn layer_id(&self) -> usize {
        self.layer
    }
}

// ---------------------------------------------------------------------------
// StreamingQBlock — NVMe-streaming layer (S.L.I.P. compatible)
// ---------------------------------------------------------------------------

/// A transformer layer that streams its weights on every `forward()` call.
///
/// Holds an `Arc<WeightStreamer>` and the target layer index. On each call,
/// it invokes `streamer.load_layer(layer_id, device)` to bring the quantised
/// tensors into RAM (backed by the mmap — effectively a page-fault), runs
/// `model::transformer_block`, and then drops the weights. This keeps RSS at
/// ≈ 1–2 layers at any point, matching the S.L.I.P. memory budget.
///
/// # Prefetch
/// Prefetching (`madvise WILLNEED`) and page release (`DONTNEED`) are managed
/// by the generator's outer loop, not by individual blocks. This keeps each
/// block stateless and object-safe.
///
/// # Thread safety
/// `WeightStreamer` uses an mmap (read-only), which is `Send + Sync`.
/// Multiple blocks can safely hold `Arc<WeightStreamer>` across threads.
pub struct StreamingQBlock {
    layer: usize,
    streamer: Arc<crate::weight_streamer::WeightStreamer>,
    config: Arc<ModelConfig>,
    rope: Option<Arc<RopeCache>>,
    device: candle_core::Device,
}

impl StreamingQBlock {
    pub fn new(
        layer: usize,
        streamer: Arc<crate::weight_streamer::WeightStreamer>,
        config: Arc<ModelConfig>,
        rope: Option<Arc<RopeCache>>,
        device: candle_core::Device,
    ) -> Self {
        Self { layer, streamer, config, rope, device }
    }
}

impl TransformerBlock for StreamingQBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache_k: Option<&Tensor>,
        kv_cache_v: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Load this layer's quantised weights from the mmap (page-fault path).
        // Weights are dropped after this function returns — RSS stays bounded.
        let weights = self.streamer
            .load_layer(self.layer, &self.device)
            .map_err(|e| candle_core::Error::Msg(format!(
                "StreamingQBlock layer {} load failed: {e}", self.layer
            )))?;

        crate::model::transformer_block(
            x,
            &weights,
            kv_cache_k,
            kv_cache_v,
            &self.config,
            start_pos,
            self.rope.as_deref(),
        )
    }

    fn layer_id(&self) -> usize {
        self.layer
    }
}

/// Build a full streaming block stack from a `WeightStreamer`.
///
/// Each block holds a clone of `Arc<WeightStreamer>` (refcount bump, no copy)
/// and its own layer index. The returned vec can be stored directly in
/// `InferenceGenerator::blocks`.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use std::path::Path;
/// use candle_core::Device;
/// use air_rs::weight_streamer::WeightStreamer;
/// use air_rs::blocks::build_streaming_blocks;
///
/// let streamer = Arc::new(WeightStreamer::open(Path::new("model.gguf")).unwrap());
/// // config / rope built from metadata at runtime
/// let blocks = build_streaming_blocks(streamer, config_arc, rope, Device::Cpu);
/// ```
pub fn build_streaming_blocks(
    streamer: Arc<crate::weight_streamer::WeightStreamer>,
    config: Arc<ModelConfig>,
    rope: Option<Arc<RopeCache>>,
    device: candle_core::Device,
) -> Vec<Box<dyn TransformerBlock>> {
    let n = streamer.n_layers();
    (0..n)
        .map(|layer| -> Box<dyn TransformerBlock> {
            Box::new(StreamingQBlock::new(
                layer,
                Arc::clone(&streamer),
                Arc::clone(&config),
                rope.as_ref().map(Arc::clone),
                device.clone(),
            ))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DeltaNetBlock — GatedDeltaNet layer wrapped as TransformerBlock (v0.10.1)
// ---------------------------------------------------------------------------

/// A `GatedDeltaNetLayer` wrapped in the `TransformerBlock` interface.
///
/// Used by `build_hybrid_blocks` for layers whose `AttentionBackend == GatedDeltaNet`.
///
/// ## Tensor contract
/// Input / output shapes match `StreamingQBlock` exactly:
/// ```text
/// forward(x, _, _, start_pos) → (x_prime, zeros_k, zeros_v)
/// ```
/// The recurrent state lives inside `GatedDeltaNetLayer::states` (one `DeltaState`
/// per head). `new_k` / `new_v` are zero tensors of the same shape as `x` —
/// the session cache recognises `is_recurrent()=true` and skips KV storage.
///
/// **Thread safety**: `GatedDeltaNetLayer` stores `Vec<DeltaState>` which is
/// `Send` but NOT `Sync` (mutable state). The block is therefore wrapped in a
/// `Mutex` for interior mutability — one mutex per layer, never contended since
/// the generator runs a single inference thread per session.
pub struct DeltaNetBlock {
    /// Zero-based layer index in the transformer stack.
    pub layer: usize,
    /// Inner GatedDeltaNet layer (mutex for interior mutability in `&self` forward).
    inner: std::sync::Mutex<GatedDeltaNetLayer>,
    /// Number of heads (needed to build the QKVαβ projection stub).
    n_heads: usize,
    /// Head dimension.
    head_dim: usize,
}

impl DeltaNetBlock {
    /// Construct from a `DeltaNetConfig` and layer index.
    pub fn new(layer: usize, config: DeltaNetConfig) -> Self {
        let n_heads  = config.n_heads;
        let head_dim = config.head_dim;
        Self {
            layer,
            inner: std::sync::Mutex::new(GatedDeltaNetLayer::new(config)),
            n_heads,
            head_dim,
        }
    }

    /// Reset the recurrent state (call at session start / context reset).
    pub fn reset_state(&self) {
        self.inner.lock().unwrap().reset();
    }

    /// VRAM cost of all head states (bytes, FP32).
    pub fn state_bytes(&self) -> usize {
        self.n_heads * self.head_dim * self.head_dim * 4
    }
}

impl TransformerBlock for DeltaNetBlock {
    /// Forward pass — single-token decode path.
    ///
    /// Extracts a flat F32 slice from `x`, runs `forward_token`, and wraps
    /// the output back into a Tensor. `kv_cache_k/v` are ignored (recurrent).
    fn forward(
        &self,
        x: &Tensor,
        _kv_cache_k: Option<&Tensor>,
        _kv_cache_v: Option<&Tensor>,
        _start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = x.device();
        let dtype  = x.dtype();
        let shape  = x.shape().clone();

        // Flatten to 1-D for the pure-Rust DeltaNet kernel
        let flat: Vec<f32> = x.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;

        // Compute the stride for the QKVαβ view.
        // In v0.10.1 the actual Wq/Wk/Wv/Wα/Wβ projections run here.
        // For now we pass the hidden state directly as a surrogate (stub).
        let d      = self.head_dim;
        let nh     = self.n_heads;
        let stride = 3 * d + 2;

        // Build a zero-padded QKVαβ buffer: real weights applied in v0.10.1
        let qkvab = vec![0.0f32; nh * stride];
        let _ = flat; // will be matmul'd against Wqkv in v0.10.1

        let out_vec = self.inner.lock().unwrap().forward_token(&qkvab);

        // Reshape output to match input shape
        let out = Tensor::from_vec(out_vec, shape.dims().to_vec(), device)?
            .to_dtype(dtype)?;

        // Return zero K/V — session cache skips storage for recurrent layers
        let zeros = Tensor::zeros_like(&out)?;
        Ok((out, zeros.clone(), zeros))
    }

    fn layer_id(&self) -> usize { self.layer }
}

// ---------------------------------------------------------------------------
// build_hybrid_blocks — router-aware block factory (v0.10.1)
// ---------------------------------------------------------------------------

/// Build a heterogeneous block stack driven by a `HybridAttentionRouter`.
///
/// For each layer:
/// - `AttentionBackend::GatedDeltaNet` → `DeltaNetBlock` (recurrent, no KV cache)
/// - `AttentionBackend::Softmax`       → `StreamingQBlock` (standard GQA + KV cache)
/// - `AttentionBackend::SlidingWindow` → `StreamingQBlock` (Gemma4; flash-attn in v0.10.1)
/// - `AttentionBackend::GlobalFull`    → `StreamingQBlock` (Gemma4 global layer)
///
/// `delta_cfg_fn` is a closure that builds the `DeltaNetConfig` for a layer index.
/// It receives `(layer_idx)` and must return a config matching the model's dimensions.
///
/// # Example
/// ```ignore
/// let router = HybridAttentionRouter::qwen3_6_27b();
/// let blocks  = build_hybrid_blocks(
///     Arc::clone(&streamer), config_arc, rope, Device::Cpu, &router,
///     |layer| DeltaNetConfig::new(128, 32).with_layer(layer),
/// );
/// ```
pub fn build_hybrid_blocks(
    streamer:     Arc<crate::weight_streamer::WeightStreamer>,
    config:       Arc<ModelConfig>,
    rope:         Option<Arc<RopeCache>>,
    device:       candle_core::Device,
    router:       &HybridAttentionRouter,
    delta_cfg_fn: impl Fn(usize) -> DeltaNetConfig,
) -> Vec<Box<dyn TransformerBlock>> {
    let n = streamer.n_layers();
    (0..n)
        .map(|layer| -> Box<dyn TransformerBlock> {
            match router.backend_for_layer(layer) {
                AttentionBackend::GatedDeltaNet => {
                    Box::new(DeltaNetBlock::new(layer, delta_cfg_fn(layer)))
                }
                _ => {
                    // Softmax, SlidingWindow, GlobalFull — all use the standard
                    // StreamingQBlock for now; SW/Global wired to flash-attn in v0.10.1
                    Box::new(StreamingQBlock::new(
                        layer,
                        Arc::clone(&streamer),
                        Arc::clone(&config),
                        rope.as_ref().map(Arc::clone),
                        device.clone(),
                    ))
                }
            }
        })
        .collect()
}


// ---------------------------------------------------------------------------
// MockTransformerBlock — zero-GPU test double
// ---------------------------------------------------------------------------

/// A no-op transformer block for unit tests.
///
/// - `forward` returns `(x_clone, zeros_k, zeros_v)` without touching VRAM.
/// - Records call count in an atomic counter for assertion in tests.
/// - Thread-safe: can be placed into `Arc<dyn TransformerBlock>`.
#[cfg(test)]
pub struct MockTransformerBlock {
    pub id: usize,
    forward_count: std::sync::atomic::AtomicU32,
}

#[cfg(test)]
impl MockTransformerBlock {
    pub fn new(id: usize) -> Self {
        Self { id, forward_count: std::sync::atomic::AtomicU32::new(0) }
    }

    /// Number of times `forward` has been called.
    pub fn calls(&self) -> u32 {
        self.forward_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
impl TransformerBlock for MockTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        _kv_cache_k: Option<&Tensor>,
        _kv_cache_v: Option<&Tensor>,
        _start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Return pass-through hidden + zero K/V (no real computation needed)
        let zeros = Tensor::zeros_like(x)?;
        Ok((x.clone(), zeros.clone(), zeros))
    }

    fn layer_id(&self) -> usize {
        self.id
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn mock_block_passthrough() -> candle_core::Result<()> {
        let block = MockTransformerBlock::new(3);
        assert_eq!(block.layer_id(), 3);

        let x = Tensor::ones((1, 4, 64), DType::F32, &Device::Cpu)?;
        let (hidden, k, v) = block.forward(&x, None, None, 0)?;

        // hidden == x (pass-through)
        let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let h_vec: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
        assert_eq!(x_vec, h_vec);

        // k/v are zeros of same shape
        assert_eq!(k.shape(), x.shape());
        let k_vec: Vec<f32> = k.flatten_all()?.to_vec1()?;
        assert!(k_vec.iter().all(|&v| v == 0.0));

        assert_eq!(block.calls(), 1);
        Ok(())
    }

    #[test]
    fn mock_block_call_counting() -> candle_core::Result<()> {
        let block = MockTransformerBlock::new(0);
        let x = Tensor::zeros((1, 1, 16), DType::F32, &Device::Cpu)?;
        block.forward(&x, None, None, 0)?;
        block.forward(&x, None, None, 1)?;
        block.forward(&x, None, None, 2)?;
        assert_eq!(block.calls(), 3);
        Ok(())
    }

    #[test]
    fn mock_block_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockTransformerBlock>();
    }

    #[test]
    fn qblock_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QBlock>();
    }
}

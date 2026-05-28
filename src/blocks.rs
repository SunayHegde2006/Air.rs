pub use crate::layer_pipeline::{LayerUnit, LayerExecutionContext};
use std::sync::Arc;
use crate::model::{QBlockWeights, ModelConfig};
use crate::ops::RopeCache;
use crate::kv_cache::LayerCache;
use crate::attention_backend::{HybridAttentionRouter, AttentionBackend};
use crate::gated_deltanet::{DeltaNetConfig, DeltaState, GatedDeltaNetLayer};
use candle_core::{Tensor, Device, DType, Result};

// ---------------------------------------------------------------------------
// QBlock — concrete implementation wrapping `model::transformer_block`
// ---------------------------------------------------------------------------

/// One quantised GQA transformer layer backed by `QBlockWeights`.
///
/// `weights` and `config` are reference-counted so that a speculative
/// decoder's draft stack and the main stack can share the same tensors.
#[derive(Clone)]
pub struct QBlock {
    /// Zero-based position in the transformer stack.
    pub layer: usize,
    /// Shared quantised weight tensors for this layer.
    pub weights: Arc<QBlockWeights>,
    /// Model hyper-parameters (shared across all QBlock instances).
    pub config: Arc<ModelConfig>,
    /// Optional pre-computed RoPE inverse-frequency cache (shared).
    pub rope: Option<Arc<RopeCache>>,
    /// Optional dual RoPE cache (Gemma 4).
    pub dual_rope: Option<Arc<crate::dual_rope::DualRopeCache>>,
}

impl QBlock {
    pub fn layer_id(&self) -> usize { self.layer }

    /// Construct from individually boxed components.
    pub fn new(
        layer: usize,
        weights: Arc<QBlockWeights>,
        config: Arc<ModelConfig>,
        rope: Option<Arc<RopeCache>>,
        dual_rope: Option<Arc<crate::dual_rope::DualRopeCache>>,
    ) -> Self {
        Self { layer, weights, config, rope, dual_rope }
    }
}

impl LayerUnit for QBlock {
    fn execute(
        &self,
        ctx: &LayerExecutionContext,
    ) -> Result<(Tensor, LayerCache)> {
        let (k_in, v_in, mut delta_in) = match ctx.state {
            Some(LayerCache::Attention { k, v }) => (Some(k), Some(v), None),
            Some(LayerCache::Recurrent(s)) => (None, None, Some(s.clone())),
            _ => (None, None, None),
        };

        let (out, nk, nv) = crate::model::transformer_block(
            self.layer,
            ctx.x,
            ctx.weights.unwrap_or(&self.weights),
            k_in,
            v_in,
            delta_in.as_mut(),
            ctx.config,
            ctx.pos,
            ctx.rope_cache.or(self.rope.as_deref()),
            ctx.dual_cache.or(self.dual_rope.as_deref()),
            ctx.mask,
            ctx.tp,
        )?;

        let next_cache = if let Some(s) = delta_in {
            LayerCache::Recurrent(s)
        } else {
            LayerCache::Attention { k: nk, v: nv }
        };

        Ok((out, next_cache))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
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
#[derive(Clone)]
pub struct StreamingQBlock {
    layer: usize,
    streamer: Arc<crate::weight_streamer::WeightStreamer>,
    config: Arc<ModelConfig>,
    rope: Option<Arc<RopeCache>>,
    dual_rope: Option<Arc<crate::dual_rope::DualRopeCache>>,
    device: candle_core::Device,
}

impl StreamingQBlock {
    pub fn new(
        layer: usize,
        streamer: Arc<crate::weight_streamer::WeightStreamer>,
        config: Arc<ModelConfig>,
        rope: Option<Arc<RopeCache>>,
        dual_rope: Option<Arc<crate::dual_rope::DualRopeCache>>,
        device: candle_core::Device,
    ) -> Self {
        Self { layer, streamer, config, rope, dual_rope, device }
    }
}

impl LayerUnit for StreamingQBlock {
    fn execute(
        &self,
        ctx: &LayerExecutionContext,
    ) -> Result<(Tensor, LayerCache)> {
        let weights = self.streamer.load_layer(self.layer, &self.device, ctx.tp)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let (k_in, v_in, mut delta_in) = match ctx.state {
            Some(LayerCache::Attention { k, v }) => (Some(k), Some(v), None),
            Some(LayerCache::Recurrent(s)) => (None, None, Some(s.clone())),
            _ => (None, None, None),
        };

        let (out, nk, nv) = crate::model::transformer_block(
            self.layer,
            ctx.x,
            ctx.weights.unwrap_or(&weights),
            k_in,
            v_in,
            delta_in.as_mut(),
            ctx.config,
            ctx.pos,
            ctx.rope_cache.or(self.rope.as_deref()),
            ctx.dual_cache.or(self.dual_rope.as_deref()),
            ctx.mask,
            ctx.tp,
        )?;

        let next_cache = if let Some(s) = delta_in {
            LayerCache::Recurrent(s)
        } else {
            LayerCache::Attention { k: nk, v: nv }
        };

        Ok((out, next_cache))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
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
    dual_rope: Option<Arc<crate::dual_rope::DualRopeCache>>,
    device: candle_core::Device,
) -> Vec<Box<dyn LayerUnit>> {
    let n = streamer.n_layers();
    (0..n)
        .map(|layer| -> Box<dyn LayerUnit> {
            Box::new(StreamingQBlock::new(
                layer,
                Arc::clone(&streamer),
                Arc::clone(&config),
                rope.as_ref().map(Arc::clone),
                dual_rope.as_ref().map(Arc::clone),
                device.clone(),
            ))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DeltaNetBlock — GatedDeltaNet layer wrapped as TransformerBlock
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
#[derive(Clone)]
pub struct DeltaNetBlock {
    /// Zero-based layer index in the transformer stack.
    pub layer: usize,
    /// Inner GatedDeltaNet layer — wrapped in Arc<Mutex> so DeltaNetBlock is Clone.
    inner: std::sync::Arc<std::sync::Mutex<GatedDeltaNetLayer>>,
    /// Weight streamer for lazy loading.
    streamer: std::sync::Arc<crate::weight_streamer::WeightStreamer>,
    /// Model configuration.
    config: std::sync::Arc<crate::model::ModelConfig>,
    /// Number of heads.
    n_heads: usize,
    /// Head dimension.
    head_dim: usize,
}

impl DeltaNetBlock {
    /// Construct from a `DeltaNetConfig` and layer index.
    pub fn new(
        layer: usize,
        config: DeltaNetConfig,
        streamer: std::sync::Arc<crate::weight_streamer::WeightStreamer>,
        model_cfg: std::sync::Arc<crate::model::ModelConfig>,
    ) -> Self {
        let n_heads  = config.n_heads;
        let head_dim = config.head_dim;
        Self {
            layer,
            inner: std::sync::Arc::new(std::sync::Mutex::new(GatedDeltaNetLayer::new(config))),
            streamer,
            config: model_cfg,
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

impl LayerUnit for DeltaNetBlock {
    fn execute(
        &self,
        ctx: &LayerExecutionContext,
    ) -> Result<(Tensor, LayerCache)> {
        let mut delta_state = match ctx.state {
            Some(LayerCache::Recurrent(s)) => s.clone(),
            _ => {
                crate::gated_deltanet::DeltaState::zeros_on_device(
                    self.n_heads, self.head_dim, self.head_dim, &ctx.x.device()
                )?
            }
        };

        // Load real weights from the streamer.
        let weights_streamed = self.streamer.load_layer(self.layer, &ctx.x.device(), ctx.tp)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let (out, _, _) = crate::model::transformer_block(
            self.layer,
            ctx.x,
            &weights_streamed,
            None,
            None,
            Some(&mut delta_state),
            ctx.config,
            ctx.pos,
            None,
            None,
            None,
            ctx.tp,
        )?;

        Ok((out, LayerCache::Recurrent(delta_state)))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// build_hybrid_blocks — router-aware block factory
// ---------------------------------------------------------------------------

/// Build a heterogeneous block stack driven by a `HybridAttentionRouter`.
///
/// For each layer:
/// - `AttentionBackend::GatedDeltaNet` → `DeltaNetBlock` (recurrent, no KV cache)
/// - `AttentionBackend::Softmax`       → `StreamingQBlock` (standard GQA + KV cache)
/// - `AttentionBackend::SlidingWindow` → `StreamingQBlock` (Gemma4; flash-attn integrated)
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
    dual_rope:    Option<Arc<crate::dual_rope::DualRopeCache>>,
    device:       candle_core::Device,
    router:       &HybridAttentionRouter,
    delta_cfg_fn: impl Fn(usize) -> DeltaNetConfig,
) -> Vec<Box<dyn LayerUnit>> {
    let n = streamer.n_layers();
    (0..n)
        .map(|layer| -> Box<dyn LayerUnit> {
            match router.backend_for_layer(layer) {
                AttentionBackend::GatedDeltaNet => {
                    Box::new(DeltaNetBlock::new(
                        layer, 
                        delta_cfg_fn(layer),
                        std::sync::Arc::clone(&streamer),
                        std::sync::Arc::clone(&config),
                    ))
                }
                _ => {
                    Box::new(StreamingQBlock::new(
                        layer,
                        Arc::clone(&streamer),
                        Arc::clone(&config),
                        rope.as_ref().map(Arc::clone),
                        dual_rope.as_ref().map(Arc::clone),
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
#[derive(Clone)]
pub struct MockTransformerBlock {
    pub id: usize,
    forward_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

#[cfg(test)]
impl MockTransformerBlock {
    pub fn new(id: usize) -> Self {
        Self { id, forward_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)) }
    }

    /// Number of times `execute` has been called.
    pub fn calls(&self) -> u32 {
        self.forward_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn layer_id(&self) -> usize {
        self.id
    }
}

#[cfg(test)]
impl LayerUnit for MockTransformerBlock {
    fn execute(
        &self,
        ctx: &LayerExecutionContext,
    ) -> Result<(Tensor, LayerCache)> {
        self.forward_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((ctx.x.clone(), ctx.state.cloned().unwrap_or(LayerCache::Empty)))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
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

        let cfg = Arc::new(ModelConfig::default());
        let weights = QBlockWeights::default_for_test();
        let x = Tensor::ones((1, 4, 64), DType::F32, &Device::Cpu)?;
        let ctx = LayerExecutionContext {
            x: &x,
            weights: Some(&weights),
            state: None,
            pos: 0,
            config: &cfg,
            rope_cache: None,
            dual_cache: None,
            mask: None,
            tp: None,
        };
        let (hidden, cache_out) = block.execute(&ctx)?;

        // hidden == x (pass-through)
        let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let h_vec: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
        assert_eq!(x_vec, h_vec);

        // cache out is Empty (mock doesn't write KV)
        assert!(matches!(cache_out, crate::kv_cache::LayerCache::Empty));

        assert_eq!(block.calls(), 1);
        Ok(())
    }

    #[test]
    fn mock_block_call_counting() -> candle_core::Result<()> {
        let block = MockTransformerBlock::new(0);
        let cfg = Arc::new(ModelConfig::default());
        let weights = QBlockWeights::default_for_test();
        let x = Tensor::zeros((1, 1, 16), DType::F32, &Device::Cpu)?;
        let mut ctx = LayerExecutionContext {
            x: &x,
            weights: Some(&weights),
            state: None,
            pos: 0,
            config: &cfg,
            rope_cache: None,
            dual_cache: None,
            mask: None,
            tp: None,
        };
        block.execute(&ctx)?;
        ctx.pos = 1;
        block.execute(&ctx)?;
        ctx.pos = 2;
        block.execute(&ctx)?;
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

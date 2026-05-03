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

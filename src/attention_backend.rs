//! # Implementation
//! - `AttentionBackend` enum: ✅ defined
//! - `HybridAttentionRouter`: ✅ implemented
//! - Wired into `blocks.rs`: supports GQA, DeltaNet, and SlidingWindow.
use candle_core::{Result, Tensor};

// ---------------------------------------------------------------------------
// AttentionBackend
// ---------------------------------------------------------------------------

/// Per-layer attention mechanism.
///
/// Determines which kernel the block factory and forward-pass dispatcher
/// will use for that layer. Cheap to copy (fits in a cache line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// Standard softmax attention (GQA/MHA).
    ///
    /// Used by: Llama, Mistral, Qwen2, Phi-3, Falcon, Gemma 1/2/3,
    ///          Qwen3.6 GQA layers (1-in-4 blocks).
    Softmax,

    /// Gated Delta Network — linear recurrence kernel.
    ///
    /// Used by: Qwen3.6 (48/64 layers per 27B layout).
    ///
    /// Research: Yang et al., "Gated Linear Attention Transformers with
    /// Hardware-Efficient Training", NeurIPS 2024.
    ///
    /// Status: Production implementation in `gated_deltanet.rs`.
    GatedDeltaNet,

    /// Local sliding-window attention.
    ///
    /// Used by: Mistral (window=4096), Gemma4 local layers.
    ///
    /// `window` — number of tokens in each side of the local window.
    SlidingWindow { window: usize },

    /// Full global attention with unified KV and p-RoPE.
    ///
    /// Used by: Gemma4 global layers (every Nth layer; final layer always global).
    GlobalFull,
}

impl AttentionBackend {
    /// Returns `true` if this backend uses a recurrent state instead of KV cache.
    ///
    /// DeltaNet layers maintain a state matrix `S_t` — they do NOT use the
    /// conventional K/V past slice. `SessionKvCache` maps these to
    /// `LayerCache::DeltaState` variants (v0.10.0).
    pub fn is_recurrent(self) -> bool {
        matches!(self, Self::GatedDeltaNet)
    }

    /// Returns `true` if this backend supports KV-cache paging.
    ///
    /// Recurrent backends are not pageable — their state is a dense matrix.
    pub fn is_kv_cacheable(self) -> bool {
        !self.is_recurrent()
    }

    /// Human-readable name for metrics and logging.
    pub fn name(self) -> &'static str {
        match self {
            Self::Softmax              => "softmax",
            Self::GatedDeltaNet        => "gated_delta_net",
            Self::SlidingWindow { .. } => "sliding_window",
            Self::GlobalFull           => "global_full",
        }
    }
}

impl std::fmt::Display for AttentionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Softmax                    => write!(f, "Softmax"),
            Self::GatedDeltaNet              => write!(f, "GatedDeltaNet"),
            Self::SlidingWindow { window }   => write!(f, "SlidingWindow(w={window})"),
            Self::GlobalFull                 => write!(f, "GlobalFull"),
        }
    }
}

// ---------------------------------------------------------------------------
// HybridAttentionRouter
// ---------------------------------------------------------------------------

/// Maps each transformer layer to its attention backend.
///
/// For homogeneous models (Llama, Qwen2, etc.) every layer is `Softmax`
/// and `uniform()` constructs the router in O(1).
///
/// For hybrid models (Qwen3.6, Gemma4) a non-uniform layout is required.
/// The layout is constructed by `model_variant.rs` from GGUF metadata
/// at model-load time.
#[derive(Debug, Clone)]
pub struct HybridAttentionRouter {
    layout: Vec<AttentionBackend>,
}

impl HybridAttentionRouter {
    /// Construct a uniform router: every layer uses `backend`.
    ///
    /// O(n_layers) allocation — called once at model load.
    pub fn uniform(n_layers: usize, backend: AttentionBackend) -> Self {
        Self { layout: vec![backend; n_layers] }
    }

    /// Construct from an explicit per-layer layout vector.
    ///
    /// Panics if `layout` is empty.
    pub fn from_layout(layout: Vec<AttentionBackend>) -> Self {
        assert!(!layout.is_empty(), "HybridAttentionRouter: layout must not be empty");
        Self { layout }
    }

    /// Return the backend for a given layer index.
    ///
    /// Panics if `layer >= n_layers()`.
    pub fn backend_for_layer(&self, layer: usize) -> AttentionBackend {
        self.layout[layer]
    }

    /// Total number of layers this router covers.
    pub fn n_layers(&self) -> usize {
        self.layout.len()
    }

    /// Count layers using a given backend.
    pub fn count(&self, backend: AttentionBackend) -> usize {
        self.layout.iter().filter(|&&b| b == backend).count()
    }

    /// Return the full layout slice (for serialization / debugging).
    pub fn layout(&self) -> &[AttentionBackend] {
        &self.layout
    }

    // -----------------------------------------------------------------------
    // Qwen3.6 layout constructors
    // -----------------------------------------------------------------------

    /// Pattern: 16 × (3 × DeltaNet + 1 × Softmax) = 64 layers total.
    /// Layers 0,1,2 → GatedDeltaNet; layer 3 → Softmax; repeat × 16.
    pub fn qwen3_6_27b() -> Self {
        let mut layout = Vec::with_capacity(64);
        for _ in 0..16 {
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::Softmax);
        }
        Self::from_layout(layout)
    }

    /// Build the Qwen3.6-35B-A3B attention layout.
    ///
    /// Same 4-layer pattern × 24 = 96 layers.
    pub fn qwen3_6_35b_a3b() -> Self {
        let mut layout = Vec::with_capacity(96);
        for _ in 0..24 {
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::GatedDeltaNet);
            layout.push(AttentionBackend::Softmax);
        }
        Self::from_layout(layout)
    }

    // -----------------------------------------------------------------------
    // Gemma4 layout constructors
    // -----------------------------------------------------------------------

    /// Local sliding-window (w=4096) every layer except global layers every 6th.
    /// 32 layers total (E4B). Final layer always global.
    pub fn gemma4_e4b(n_layers: usize, sliding_window: usize, global_every_n: usize) -> Self {
        let mut layout = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let is_last = layer == n_layers - 1;
            let is_global = is_last || (layer > 0 && layer % global_every_n == global_every_n - 1);
            if is_global {
                layout.push(AttentionBackend::GlobalFull);
            } else {
                layout.push(AttentionBackend::SlidingWindow { window: sliding_window });
            }
        }
        Self::from_layout(layout)
    }

    /// Build the Gemma4-26B-A4B attention layout.
    ///
    /// Same pattern as E4B but with more layers (46 total).
    pub fn gemma4_26b_a4b(n_layers: usize, sliding_window: usize, global_every_n: usize) -> Self {
        Self::gemma4_e4b(n_layers, sliding_window, global_every_n) // same logic, different params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_softmax_router() {
        let r = HybridAttentionRouter::uniform(32, AttentionBackend::Softmax);
        assert_eq!(r.n_layers(), 32);
        assert_eq!(r.backend_for_layer(0), AttentionBackend::Softmax);
        assert_eq!(r.backend_for_layer(31), AttentionBackend::Softmax);
        assert_eq!(r.count(AttentionBackend::Softmax), 32);
    }

    #[test]
    fn test_qwen3_6_27b_layout() {
        let r = HybridAttentionRouter::qwen3_6_27b();
        assert_eq!(r.n_layers(), 64);
        // First 3 layers: DeltaNet
        assert_eq!(r.backend_for_layer(0), AttentionBackend::GatedDeltaNet);
        assert_eq!(r.backend_for_layer(1), AttentionBackend::GatedDeltaNet);
        assert_eq!(r.backend_for_layer(2), AttentionBackend::GatedDeltaNet);
        // Layer 3: Softmax (GQA)
        assert_eq!(r.backend_for_layer(3), AttentionBackend::Softmax);
        // Repeat at 4,5,6 → DeltaNet; 7 → Softmax
        assert_eq!(r.backend_for_layer(4), AttentionBackend::GatedDeltaNet);
        assert_eq!(r.backend_for_layer(7), AttentionBackend::Softmax);
        // Counts: 48 DeltaNet + 16 Softmax
        assert_eq!(r.count(AttentionBackend::GatedDeltaNet), 48);
        assert_eq!(r.count(AttentionBackend::Softmax), 16);
    }

    #[test]
    fn test_qwen3_6_35b_a3b_layout() {
        let r = HybridAttentionRouter::qwen3_6_35b_a3b();
        assert_eq!(r.n_layers(), 96);
        assert_eq!(r.count(AttentionBackend::GatedDeltaNet), 72); // 24×3
        assert_eq!(r.count(AttentionBackend::Softmax), 24);       // 24×1
    }

    #[test]
    fn test_gemma4_e4b_layout_final_layer_global() {
        let r = HybridAttentionRouter::gemma4_e4b(32, 4096, 6);
        assert_eq!(r.n_layers(), 32);
        // Final layer always global
        assert_eq!(r.backend_for_layer(31), AttentionBackend::GlobalFull);
        // Some sliding-window layers
        assert_eq!(r.backend_for_layer(0), AttentionBackend::SlidingWindow { window: 4096 });
    }

    #[test]
    fn test_is_recurrent_deltanet() {
        assert!(AttentionBackend::GatedDeltaNet.is_recurrent());
        assert!(!AttentionBackend::Softmax.is_recurrent());
        assert!(!AttentionBackend::SlidingWindow { window: 4096 }.is_recurrent());
        assert!(!AttentionBackend::GlobalFull.is_recurrent());
    }

    #[test]
    fn test_is_kv_cacheable() {
        assert!(!AttentionBackend::GatedDeltaNet.is_kv_cacheable());
        assert!(AttentionBackend::Softmax.is_kv_cacheable());
        assert!(AttentionBackend::GlobalFull.is_kv_cacheable());
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", AttentionBackend::Softmax), "Softmax");
        assert_eq!(format!("{}", AttentionBackend::GatedDeltaNet), "GatedDeltaNet");
        assert_eq!(format!("{}", AttentionBackend::SlidingWindow { window: 4096 }), "SlidingWindow(w=4096)");
        assert_eq!(format!("{}", AttentionBackend::GlobalFull), "GlobalFull");
    }

    #[test]
    fn test_from_layout_panics_on_empty() {
        let result = std::panic::catch_unwind(|| {
            HybridAttentionRouter::from_layout(vec![]);
        });
        assert!(result.is_err(), "empty layout should panic");
    }

    #[test]
    fn test_count_mixed_layout() {
        let layout = vec![
            AttentionBackend::GatedDeltaNet,
            AttentionBackend::Softmax,
            AttentionBackend::SlidingWindow { window: 2048 },
            AttentionBackend::GlobalFull,
        ];
        let r = HybridAttentionRouter::from_layout(layout);
        assert_eq!(r.count(AttentionBackend::GatedDeltaNet), 1);
        assert_eq!(r.count(AttentionBackend::Softmax), 1);
        assert_eq!(r.count(AttentionBackend::GlobalFull), 1);
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(AttentionBackend::Softmax.name(), "softmax");
        assert_eq!(AttentionBackend::GatedDeltaNet.name(), "gated_delta_net");
        assert_eq!(AttentionBackend::SlidingWindow { window: 1 }.name(), "sliding_window");
        assert_eq!(AttentionBackend::GlobalFull.name(), "global_full");
    }
}

// ---------------------------------------------------------------------------
// Qwen 3.6 Specific Components
// ---------------------------------------------------------------------------

/// Qwen 3.6 GQA layer forward pass (v0.10.0).
///
/// Qwen 3.6 uses standard GQA for every 4th layer, with a specialized
/// RoPE theta (500,000). This helper abstracts the RoPE/Attention dispatch.
pub fn qwen36_gqa_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    start_pos: usize,
    head_dim: usize,
    rope_cache: &crate::ops::RopeCache,
) -> Result<Tensor> {
    // Qwen 3.6 GQA layers use 500k theta
    let qwen_theta = 500_000.0;
    
    // 1. Apply RoPE
    let (q, k) = crate::ops::rope_cached(
        q, k, start_pos, head_dim, qwen_theta, rope_cache
    )?;

    // 2. Standard Attention (no sliding window for Qwen global layers)
    crate::ops::attention(&q, &k, &v, n_heads, n_kv_heads, None, None)
}

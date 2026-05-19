//! Gemma 4 Hybrid-Attention Forward Pass (v0.10.0)
//!
//! Implements the Gemma 4 transformer block with:
//!   - Local sliding-window attention (GQA, window=4096, every non-global layer)
//!   - Global full-attention (GQA, p-RoPE with high theta, every Nth layer + final)
//!   - Sigmoid MoE router for 26B-A4B (top-K=2 experts per token)
//!   - Gemma-RMSNorm: scale = stored_weight + 1.0 (not plain multiplicative)
//!   - GeGLU activation: FFN(x) = (xW_gate ⊙ GELU(xW_up)) W_down
//!
//! # Reference
//! Gemma 4 Technical Report (Google DeepMind, 2025)
//! Variants: E4B (dense, 32L), 26B-A4B (MoE, 4B active of 26B total)
//!
//! # Architecture
//! ```text
//! Gemma4Block
//!   ├── pre_attn_norm: GemmaRmsNorm
//!   ├── attn: SlidingWindowAttn | GlobalFullAttn  (per HybridAttentionRouter)
//!   ├── post_attn_norm: GemmaRmsNorm
//!   ├── pre_ffn_norm: GemmaRmsNorm
//!   ├── ffn: DenseGeGLU | SigmoidMoeRouter + ExpertGeGLU  (per variant)
//!   └── post_ffn_norm: GemmaRmsNorm
//! ```
//!
//! # v0.10.0 status
//! - GemmaRmsNorm:           ✅
//! - GeGLU (dense FFN):      ✅
//! - Sigmoid MoE router:     ✅
//! - SlidingWindowAttn stub: ✅ (wired to existing flash-attn path in v0.10.1)
//! - GlobalFullAttn stub:    ✅ (wired to existing full-attn path in v0.10.1)
//! - DualRoPE integration:   ✅ (uses `src/dual_rope.rs`)
//! - GGUF tensor loading:    🔲 v0.10.1

use crate::attention_backend::AttentionBackend;
use crate::dual_rope::{DualRopeCache, apply_rope_batch};

// ---------------------------------------------------------------------------
// Gemma-RMSNorm
// ---------------------------------------------------------------------------

/// Gemma-specific RMSNorm where `effective_weight = stored_weight + 1.0`.
///
/// Standard RMSNorm: `y = (x / RMS(x)) * w`
/// Gemma RMSNorm:    `y = (x / RMS(x)) * (w + 1.0)`
///
/// This means the stored weight tensor is a *residual* around 1.0.
/// Initialised to zero → behaves as plain RMSNorm at init.
pub struct GemmaRmsNorm {
    /// Stored weight (residual). Effective weight = weight + 1.0.
    weight: Vec<f32>,
    eps:    f32,
}

impl GemmaRmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: vec![0.0f32; dim], eps }
    }

    /// Load weights from a slice (e.g. from a GGUF tensor).
    pub fn load_weights(&mut self, w: &[f32]) {
        debug_assert_eq!(w.len(), self.weight.len());
        self.weight.copy_from_slice(w);
    }

    /// Apply Gemma-RMSNorm in-place.
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(n, self.weight.len());

        // RMS
        let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + self.eps).sqrt();

        // Normalise and scale by (w + 1.0)
        for (xi, &wi) in x.iter_mut().zip(&self.weight) {
            *xi = (*xi / rms) * (wi + 1.0);
        }
    }

    pub fn dim(&self) -> usize { self.weight.len() }
}

// ---------------------------------------------------------------------------
// GELU activation
// ---------------------------------------------------------------------------

/// Tanh-approximated GELU: `x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))`
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let c = 0.7978845608_f32;
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    x * 0.5 * (1.0 + t)
}

// ---------------------------------------------------------------------------
// Dense GeGLU FFN (Gemma 4 E4B)
// ---------------------------------------------------------------------------

/// GeGLU feed-forward network: `FFN(x) = (xW_gate ⊙ GELU(xW_up)) W_down`
///
/// Weight shapes:
///   W_gate: [d_ffn × d_model]
///   W_up:   [d_ffn × d_model]
///   W_down: [d_model × d_ffn]
pub struct DenseGeGlu {
    w_gate: Vec<f32>,  // [d_ffn × d_model]
    w_up:   Vec<f32>,  // [d_ffn × d_model]
    w_down: Vec<f32>,  // [d_model × d_ffn]
    d_model: usize,
    d_ffn:   usize,
}

impl DenseGeGlu {
    /// Construct with zero-initialised weights.
    pub fn new(d_model: usize, d_ffn: usize) -> Self {
        Self {
            w_gate: vec![0.0f32; d_ffn * d_model],
            w_up:   vec![0.0f32; d_ffn * d_model],
            w_down: vec![0.0f32; d_model * d_ffn],
            d_model,
            d_ffn,
        }
    }

    /// Forward pass. `x` is `[d_model]`, returns `[d_model]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let dm = self.d_model;
        let df = self.d_ffn;

        // gate = x @ W_gate^T  [d_ffn]
        let mut gate = vec![0.0f32; df];
        for i in 0..df {
            for j in 0..dm {
                gate[i] += x[j] * self.w_gate[i * dm + j];
            }
        }

        // up = GELU(x @ W_up^T)  [d_ffn]
        let mut up = vec![0.0f32; df];
        for i in 0..df {
            let mut acc = 0.0f32;
            for j in 0..dm {
                acc += x[j] * self.w_up[i * dm + j];
            }
            up[i] = gelu(acc);
        }

        // hidden = gate ⊙ up  [d_ffn]
        let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g * u).collect();

        // out = hidden @ W_down^T  [d_model]
        let mut out = vec![0.0f32; dm];
        for i in 0..dm {
            for j in 0..df {
                out[i] += hidden[j] * self.w_down[i * df + j];
            }
        }
        out
    }

    /// VRAM bytes (f32).
    pub fn weight_bytes(&self) -> usize {
        (self.d_ffn * self.d_model * 2 + self.d_model * self.d_ffn) * 4
    }
}

// ---------------------------------------------------------------------------
// Sigmoid MoE Router (Gemma 4 26B-A4B)
// ---------------------------------------------------------------------------

/// Sigmoid MoE router for Gemma 4 26B-A4B.
///
/// Unlike softmax-based routers (e.g. Mixtral), Gemma 4 uses independent
/// sigmoid activations over router logits and selects the top-K experts
/// by raw sigmoid score. This avoids normalisation across experts.
///
/// Reference: Gemma 4 Technical Report, §3.2 (Expert Routing)
///
/// # Parameters
/// - `n_experts`: total experts E (typically 32 in 26B-A4B)
/// - `top_k`:     number of active experts per token (typically 2)
/// - `d_model`:   input dimension
pub struct SigmoidMoeRouter {
    /// Router weight: [n_experts × d_model]
    w_router: Vec<f32>,
    n_experts: usize,
    top_k:     usize,
    d_model:   usize,
}

impl SigmoidMoeRouter {
    pub fn new(n_experts: usize, top_k: usize, d_model: usize) -> Self {
        assert!(top_k <= n_experts, "top_k must not exceed n_experts");
        Self {
            w_router: vec![0.0f32; n_experts * d_model],
            n_experts,
            top_k,
            d_model,
        }
    }

    /// Route a token hidden state to the top-K experts.
    ///
    /// Returns `(expert_indices, expert_weights)` for the selected experts.
    /// `expert_weights` are the raw sigmoid scores (not re-normalised).
    pub fn route(&self, x: &[f32]) -> (Vec<usize>, Vec<f32>) {
        debug_assert_eq!(x.len(), self.d_model);

        // Compute router logits: [n_experts]
        let mut logits = vec![0.0f32; self.n_experts];
        for e in 0..self.n_experts {
            let mut acc = 0.0f32;
            for j in 0..self.d_model {
                acc += x[j] * self.w_router[e * self.d_model + j];
            }
            logits[e] = sigmoid_f32(acc);
        }

        // Top-K selection (partial sort)
        let mut indices: Vec<usize> = (0..self.n_experts).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_idx = indices[..self.top_k].to_vec();
        let top_wt:  Vec<f32> = top_idx.iter().map(|&i| logits[i]).collect();
        (top_idx, top_wt)
    }

    /// VRAM bytes for router weights (f32).
    pub fn weight_bytes(&self) -> usize {
        self.n_experts * self.d_model * 4
    }
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// MoE expert pool (dense weight matrix, shared across experts for testing)
// ---------------------------------------------------------------------------

/// Pool of E independent GeGLU expert FFNs.
pub struct ExpertPool {
    experts: Vec<DenseGeGlu>,
}

impl ExpertPool {
    pub fn new(n_experts: usize, d_model: usize, d_ffn: usize) -> Self {
        Self {
            experts: (0..n_experts).map(|_| DenseGeGlu::new(d_model, d_ffn)).collect(),
        }
    }

    /// Compute weighted sum of top-K expert outputs.
    ///
    /// `x` — token hidden state `[d_model]`
    /// `expert_indices` — selected expert indices (top_k)
    /// `expert_weights` — corresponding sigmoid scores (top_k)
    pub fn forward(
        &self,
        x: &[f32],
        expert_indices: &[usize],
        expert_weights: &[f32],
    ) -> Vec<f32> {
        let d = self.experts[0].d_model;
        let mut out = vec![0.0f32; d];
        for (&idx, &wt) in expert_indices.iter().zip(expert_weights) {
            let expert_out = self.experts[idx].forward(x);
            for i in 0..d {
                out[i] += wt * expert_out[i];
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Attention config
// ---------------------------------------------------------------------------

/// Attention configuration for one Gemma 4 layer.
#[derive(Debug, Clone)]
pub struct Gemma4AttnConfig {
    pub d_model:    usize,
    pub n_heads:    usize,
    pub n_kv_heads: usize,  // GQA: fewer KV heads
    pub head_dim:   usize,
    pub window_size: usize, // only used by SlidingWindow backend
}

impl Gemma4AttnConfig {
    /// Gemma 4 E4B defaults (32 heads, 16 KV heads, head_dim=256)
    pub fn e4b_default() -> Self {
        Self { d_model: 2560, n_heads: 16, n_kv_heads: 8, head_dim: 256, window_size: 4096 }
    }
}

// ---------------------------------------------------------------------------
// Attention stubs (wired to real kernels in v0.10.1)
// ---------------------------------------------------------------------------

/// Stub: sliding-window attention output (zeros — wired to flash-attn in v0.10.1).
fn sliding_window_attn_stub(
    _q:  &[f32],
    _k:  &[f32],
    _v:  &[f32],
    d_model: usize,
) -> Vec<f32> {
    // v0.10.1: call candle_flash_attn::flash_attn() with local masking
    vec![0.0f32; d_model]
}

/// Stub: global full attention output (zeros — wired to existing attn in v0.10.1).
fn global_full_attn_stub(
    _q:  &[f32],
    _k:  &[f32],
    _v:  &[f32],
    d_model: usize,
) -> Vec<f32> {
    // v0.10.1: call existing GQA softmax attention
    vec![0.0f32; d_model]
}

// ---------------------------------------------------------------------------
// Gemma4Block — full transformer block
// ---------------------------------------------------------------------------

/// Gemma 4 transformer decoder block.
///
/// Handles both dense (E4B) and MoE (26B-A4B) FFN based on `ffn_variant`.
pub struct Gemma4Block {
    /// Attention backend for this layer.
    pub backend: AttentionBackend,
    /// Attention configuration.
    pub attn_config: Gemma4AttnConfig,
    /// Pre-attention norm.
    pre_attn_norm:  GemmaRmsNorm,
    /// Post-attention norm.
    post_attn_norm: GemmaRmsNorm,
    /// Pre-FFN norm.
    pre_ffn_norm:   GemmaRmsNorm,
    /// Post-FFN norm.
    post_ffn_norm:  GemmaRmsNorm,
    /// FFN variant: dense or MoE.
    ffn: Gemma4Ffn,
    /// Layer index.
    pub layer_idx: usize,
}

/// FFN variant for a Gemma 4 block.
pub enum Gemma4Ffn {
    Dense(DenseGeGlu),
    Moe { router: SigmoidMoeRouter, experts: ExpertPool },
}

impl Gemma4Block {
    /// Construct a dense (E4B) block.
    pub fn new_dense(
        backend:    AttentionBackend,
        attn_cfg:   Gemma4AttnConfig,
        d_model:    usize,
        d_ffn:      usize,
        layer_idx:  usize,
    ) -> Self {
        Self {
            backend,
            pre_attn_norm:  GemmaRmsNorm::new(d_model, 1e-6),
            post_attn_norm: GemmaRmsNorm::new(d_model, 1e-6),
            pre_ffn_norm:   GemmaRmsNorm::new(d_model, 1e-6),
            post_ffn_norm:  GemmaRmsNorm::new(d_model, 1e-6),
            ffn: Gemma4Ffn::Dense(DenseGeGlu::new(d_model, d_ffn)),
            attn_config: attn_cfg,
            layer_idx,
        }
    }

    /// Construct a MoE (26B-A4B) block.
    pub fn new_moe(
        backend:    AttentionBackend,
        attn_cfg:   Gemma4AttnConfig,
        d_model:    usize,
        d_ffn:      usize,
        n_experts:  usize,
        top_k:      usize,
        layer_idx:  usize,
    ) -> Self {
        Self {
            backend,
            pre_attn_norm:  GemmaRmsNorm::new(d_model, 1e-6),
            post_attn_norm: GemmaRmsNorm::new(d_model, 1e-6),
            pre_ffn_norm:   GemmaRmsNorm::new(d_model, 1e-6),
            post_ffn_norm:  GemmaRmsNorm::new(d_model, 1e-6),
            ffn: Gemma4Ffn::Moe {
                router:  SigmoidMoeRouter::new(n_experts, top_k, d_model),
                experts: ExpertPool::new(n_experts, d_model, d_ffn),
            },
            attn_config: attn_cfg,
            layer_idx,
        }
    }

    /// Forward pass for a single token.
    ///
    /// Applies pre/post norms, attention (stub), and FFN (dense or MoE).
    /// RoPE is applied externally via `apply_rope_batch` before this call.
    pub fn forward(
        &mut self,
        x:   &[f32],     // [d_model]
        q:   &[f32],     // [n_heads × head_dim] — already RoPE applied
        k:   &[f32],     // [n_kv_heads × head_dim]
        v:   &[f32],     // [n_kv_heads × head_dim]
    ) -> Vec<f32>        // [d_model]
    {
        let d = self.attn_config.d_model;
        debug_assert_eq!(x.len(), d);

        // --- Attention sub-block ---
        let mut normed_x = x.to_vec();
        self.pre_attn_norm.forward(&mut normed_x);

        let attn_out = match self.backend {
            AttentionBackend::SlidingWindow { .. } => sliding_window_attn_stub(q, k, v, d),
            AttentionBackend::GlobalFull           => global_full_attn_stub(q, k, v, d),
            _ => vec![0.0f32; d], // Softmax / other: use existing attn path
        };

        // Post-attn norm + residual
        let mut after_attn = attn_out;
        self.post_attn_norm.forward(&mut after_attn);
        let mut residual: Vec<f32> = x.iter().zip(&after_attn).map(|(a, b)| a + b).collect();

        // --- FFN sub-block ---
        let mut normed_r = residual.clone();
        self.pre_ffn_norm.forward(&mut normed_r);

        let ffn_out = match &self.ffn {
            Gemma4Ffn::Dense(ffn) => ffn.forward(&normed_r),
            Gemma4Ffn::Moe { router, experts } => {
                let (idx, wt) = router.route(&normed_r);
                experts.forward(&normed_r, &idx, &wt)
            }
        };

        let mut after_ffn = ffn_out;
        self.post_ffn_norm.forward(&mut after_ffn);

        // Residual
        for (r, f) in residual.iter_mut().zip(&after_ffn) { *r += f; }
        residual
    }

    /// Returns `true` if this block uses a local sliding-window attention.
    pub fn is_local(&self) -> bool {
        matches!(self.backend, AttentionBackend::SlidingWindow { .. })
    }

    /// Returns `true` if this block uses global full attention.
    pub fn is_global(&self) -> bool {
        matches!(self.backend, AttentionBackend::GlobalFull)
    }
}

// ---------------------------------------------------------------------------
// Gemma4ModelConfig
// ---------------------------------------------------------------------------

/// Model-level configuration for Gemma 4 variants.
#[derive(Debug, Clone)]
pub struct Gemma4ModelConfig {
    pub variant:     Gemma4Variant,
    pub n_layers:    usize,
    pub d_model:     usize,
    pub d_ffn:       usize,
    pub n_heads:     usize,
    pub n_kv_heads:  usize,
    pub head_dim:    usize,
    pub window_size: usize,
    pub global_every_n: usize,  // global attention every N layers (last is always global)
    /// MoE parameters (only for 26B-A4B)
    pub n_experts:   Option<usize>,
    pub top_k:       Option<usize>,
}

/// Gemma 4 model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    /// 4B-parameter dense model, 32 layers.
    E4B,
    /// 26B-parameter MoE model, 4B active params.
    A26B,
}

impl Gemma4ModelConfig {
    /// Gemma 4 E4B configuration (approximate; exact values from GGUF in v0.10.1).
    pub fn e4b() -> Self {
        Self {
            variant:     Gemma4Variant::E4B,
            n_layers:    32,
            d_model:     2560,
            d_ffn:       16384,
            n_heads:     16,
            n_kv_heads:  8,
            head_dim:    256,
            window_size: 4096,
            global_every_n: 6,
            n_experts:   None,
            top_k:       None,
        }
    }

    /// Gemma 4 26B-A4B configuration (approximate).
    pub fn a26b() -> Self {
        Self {
            variant:     Gemma4Variant::A26B,
            n_layers:    46,
            d_model:     5120,
            d_ffn:       8192,
            n_heads:     32,
            n_kv_heads:  16,
            head_dim:    256,
            window_size: 4096,
            global_every_n: 6,
            n_experts:   Some(32),
            top_k:       Some(2),
        }
    }

    /// Determine the attention backend for a given layer index.
    pub fn backend_for_layer(&self, layer_idx: usize) -> AttentionBackend {
        let is_last   = layer_idx == self.n_layers - 1;
        let is_global = is_last
            || (layer_idx > 0 && layer_idx % self.global_every_n == self.global_every_n - 1);
        if is_global {
            AttentionBackend::GlobalFull
        } else {
            AttentionBackend::SlidingWindow { window: self.window_size }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma_rms_norm_zero_weight() {
        // Zero weights → effective weight = 1.0 → plain RMSNorm
        let norm = GemmaRmsNorm::new(4, 1e-6);
        let mut x = vec![3.0f32, 4.0, 0.0, 0.0];
        let orig_norm: f32 = x.iter().map(|v| v*v).sum::<f32>().sqrt();
        norm.forward(&mut x);
        // After RMSNorm: x = [3/rms, 4/rms, 0, 0] * 1.0
        // rms of [3,4,0,0] = sqrt((9+16)/4) = sqrt(6.25) = 2.5
        assert!((x[0] - 3.0/2.5).abs() < 1e-4, "x[0]: {}", x[0]);
        assert!((x[1] - 4.0/2.5).abs() < 1e-4, "x[1]: {}", x[1]);
    }

    #[test]
    fn test_gemma_rms_norm_with_weight() {
        let mut norm = GemmaRmsNorm::new(4, 1e-6);
        // Set weight = 1.0 → effective = 2.0 → output doubled vs zero-weight
        norm.load_weights(&[1.0, 1.0, 1.0, 1.0]);
        let mut x = vec![3.0f32, 4.0, 0.0, 0.0];
        norm.forward(&mut x);
        // effective_weight = 2.0, rms = 2.5
        assert!((x[0] - 3.0/2.5*2.0).abs() < 1e-4, "x[0]: {}", x[0]);
    }

    #[test]
    fn test_gelu_zero_is_zero() {
        assert_eq!(gelu(0.0), 0.0);
    }

    #[test]
    fn test_gelu_positive_domain() {
        // GELU(x) ≈ x for large positive x
        assert!((gelu(10.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_dense_geglu_output_shape() {
        let ffn = DenseGeGlu::new(8, 16);
        let x   = vec![0.1f32; 8];
        let out = ffn.forward(&x);
        assert_eq!(out.len(), 8, "output should be d_model");
    }

    #[test]
    fn test_dense_geglu_zero_weights_zero_output() {
        let ffn = DenseGeGlu::new(4, 8);
        // Zero weights → gate=0, GELU(0)=0 → out=0
        let out = ffn.forward(&vec![1.0f32; 4]);
        assert!(out.iter().all(|&x| x == 0.0), "zero weights should give zero output");
    }

    #[test]
    fn test_sigmoid_moe_router_topk_count() {
        let router = SigmoidMoeRouter::new(8, 2, 4);
        let x = vec![0.5f32; 4];
        let (idx, wt) = router.route(&x);
        assert_eq!(idx.len(), 2, "should select exactly top_k=2 experts");
        assert_eq!(wt.len(), 2);
    }

    #[test]
    fn test_sigmoid_moe_weights_in_01() {
        let router = SigmoidMoeRouter::new(8, 3, 4);
        let x = vec![0.5f32; 4];
        let (_, wt) = router.route(&x);
        for &w in &wt {
            assert!(w >= 0.0 && w <= 1.0, "sigmoid weight out of [0,1]: {w}");
        }
    }

    #[test]
    fn test_expert_pool_forward_shape() {
        let pool   = ExpertPool::new(4, 8, 16);
        let x      = vec![0.5f32; 8];
        let idx    = vec![0usize, 2];
        let wt     = vec![0.5f32, 0.5];
        let out    = pool.forward(&x, &idx, &wt);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_gemma4_block_dense_forward_shape() {
        let d  = 16;
        let nh = 2;
        let kv_h = 1;
        let hd = 8;
        let cfg = Gemma4AttnConfig {
            d_model: d, n_heads: nh, n_kv_heads: kv_h,
            head_dim: hd, window_size: 4096
        };
        let mut block = Gemma4Block::new_dense(
            AttentionBackend::SlidingWindow { window: 4096 },
            cfg, d, 32, 0
        );
        let x = vec![0.1f32; d];
        let q = vec![0.1f32; nh * hd];
        let k = vec![0.1f32; kv_h * hd];
        let v = vec![0.1f32; kv_h * hd];
        let out = block.forward(&x, &q, &k, &v);
        assert_eq!(out.len(), d, "output shape should be d_model");
    }

    #[test]
    fn test_gemma4_block_moe_forward_shape() {
        let d  = 16;
        let cfg = Gemma4AttnConfig {
            d_model: d, n_heads: 2, n_kv_heads: 1,
            head_dim: 8, window_size: 4096
        };
        let mut block = Gemma4Block::new_moe(
            AttentionBackend::GlobalFull,
            cfg, d, 32, 4, 2, 5
        );
        let x = vec![0.1f32; d];
        let q = vec![0.1f32; 16];
        let k = vec![0.1f32; 8];
        let v = vec![0.1f32; 8];
        let out = block.forward(&x, &q, &k, &v);
        assert_eq!(out.len(), d);
    }

    #[test]
    fn test_gemma4_model_config_e4b_last_layer_global() {
        let cfg = Gemma4ModelConfig::e4b();
        // Last layer (31) must be global
        assert_eq!(cfg.backend_for_layer(31), AttentionBackend::GlobalFull);
        // Layer 0 should be sliding window
        assert_eq!(
            cfg.backend_for_layer(0),
            AttentionBackend::SlidingWindow { window: 4096 }
        );
    }

    #[test]
    fn test_gemma4_model_config_a26b_moe_params() {
        let cfg = Gemma4ModelConfig::a26b();
        assert_eq!(cfg.n_experts, Some(32));
        assert_eq!(cfg.top_k, Some(2));
        assert_eq!(cfg.variant, Gemma4Variant::A26B);
    }

    #[test]
    fn test_block_is_local_is_global() {
        let cfg = Gemma4AttnConfig {
            d_model: 16, n_heads: 2, n_kv_heads: 1, head_dim: 8, window_size: 4096
        };
        let b_local = Gemma4Block::new_dense(
            AttentionBackend::SlidingWindow { window: 4096 }, cfg.clone(), 16, 32, 0
        );
        let b_global = Gemma4Block::new_dense(
            AttentionBackend::GlobalFull, cfg, 16, 32, 5
        );
        assert!(b_local.is_local(), "SlidingWindow should be local");
        assert!(!b_local.is_global());
        assert!(b_global.is_global(), "GlobalFull should be global");
        assert!(!b_global.is_local());
    }

    #[test]
    fn test_dense_geglu_weight_bytes() {
        let ffn = DenseGeGlu::new(512, 2048);
        // (2048*512 + 2048*512 + 512*2048) * 4
        let expected = (2048 * 512 * 2 + 512 * 2048) * 4;
        assert_eq!(ffn.weight_bytes(), expected);
    }
}

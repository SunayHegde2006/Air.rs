//! Transformer mathematical operations for LLaMA-family models.
//!
//! All operations use Candle's tensor API which dispatches to cuBLAS/cuDNN
//! on CUDA devices automatically. No hand-written CUDA kernels needed —
//! Candle handles the GPU dispatch.

use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// RMSNorm — Root Mean Square Layer Normalization
// ---------------------------------------------------------------------------
// Used instead of LayerNorm in LLaMA. Cheaper (no mean subtraction, no bias).
//
//   RMSNorm(x) = x * weight / sqrt(mean(x²) + eps)
//
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    // Upcast to f32 for numerical stability during norm computation
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    // Cast back and apply learned scale
    x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)
}

// ---------------------------------------------------------------------------
// C5 — LayerNorm (Falcon, Phi-2, GPT-2, StableLM, BLOOM, OLMo, GLM)
// ---------------------------------------------------------------------------
// Standard Layer Normalization with learned affine parameters:
//
//   LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
//
// Differences from RMSNorm:
//   - Computes mean (not just RMS), so it centers the distribution
//   - Has an additive bias term in addition to the scale weight
//   - More expensive (~2× compute) but required for many non-LLaMA architectures
//
/// Layer normalization as used in Falcon, Phi-2, GPT-2, BLOOM, OLMo, StableLM.
///
/// # Arguments
/// * `x`      — input tensor `[..., hidden_dim]`
/// * `weight` — learned scale (γ) `[hidden_dim]`
/// * `bias`   — learned shift (β) `[hidden_dim]`; pass `None` for no-bias variant
/// * `eps`    — numerical stability epsilon (typically `1e-5`)
pub fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    let x_dtype = x.dtype();
    // Upcast to f32 for numerical stability
    let x_f32 = x.to_dtype(DType::F32)?;
    // mean(x) over last dim, kept for broadcasting
    let mean = x_f32.mean_keepdim(D::Minus1)?;
    // variance = mean((x - mean)²)
    let x_centered = x_f32.broadcast_sub(&mean)?;
    let variance = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
    // Normalize
    let x_normed = x_centered.broadcast_div(&(variance + eps)?.sqrt()?)?;
    // Apply learned affine transform: x_normed * weight + bias
    let scaled = x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)?;
    match bias {
        Some(b) => scaled.broadcast_add(b),
        None => Ok(scaled),
    }
}

// ---------------------------------------------------------------------------
// C5b — Gemma RMSNorm variant (weight = stored_weight + 1.0)
// ---------------------------------------------------------------------------
// Gemma stores (weight - 1) in GGUF, so the effective scale is w + 1.0.
// This is also called "RMSNorm with unit offset" in the Gemma paper.
//
/// Gemma-specific RMSNorm: effective weight = stored_weight + 1.0.
///
/// Applied in all Gemma 2/3/4 and CodeGemma transformer layers.
/// Unlike standard RMSNorm, the bias-free scale is shifted by 1.
pub fn rms_norm_gemma(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let x_normed = x_f32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    // Effective weight = stored_weight + 1.0
    let effective_weight = (weight + 1.0_f64)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(&effective_weight)
}

// ---------------------------------------------------------------------------
// I3 — GELU activation (GPT-2, BLOOM, some Falcon)
// ---------------------------------------------------------------------------
// Gaussian Error Linear Unit — smoother alternative to ReLU/SiLU.
//
//   GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
//
// GPT-2 uses the tanh approximation (above). The exact form uses erf().
// We implement the tanh approximation for consistency with llama.cpp.
//
/// GELU activation (tanh approximation) as used in GPT-2 and BLOOM.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    // Constants: sqrt(2/π) ≈ 0.7978845608, coefficient = 0.044715
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    const COEFF: f64 = 0.044715;
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    // inner = x + 0.044715 * x³
    let x_cubed = x_f32.powf(3.0)?;
    let inner = (x_f32.clone() + (x_cubed * COEFF)?)?;
    // tanh(sqrt(2/π) * inner)
    let tanh_arg = (inner * SQRT_2_OVER_PI)?;
    let tanh_val = tanh_arg.tanh()?;
    // 0.5 * x * (1 + tanh_val)
    let one = Tensor::ones_like(&tanh_val)?;
    let out = (x_f32 * (one + tanh_val)?)?.affine(0.5, 0.0)?;
    out.to_dtype(x_dtype)
}

// ---------------------------------------------------------------------------
// I4 — GeGLU activation (Gemma 2/3/4, CodeGemma)
// ---------------------------------------------------------------------------
// Gated activation where the gate uses GELU instead of SiLU:
//
//   GeGLU(gate, up) = GELU(gate) ⊙ up
//
// Contrast with SwiGLU (LLaMA): SiLU(gate) ⊙ up
//
/// GeGLU FFN forward: computes gate×up with GELU gating.
///
/// # Arguments
/// * `x`      — normed input `[batch, seq, hidden_dim]`
/// * `w_gate` — gate projection (quantized)
/// * `w_up`   — up projection (quantized)
/// * `w_down` — down projection (quantized)
pub fn geglu_ffn(
    x: &Tensor,
    w_gate: &candle_core::quantized::QMatMul,
    w_up: &candle_core::quantized::QMatMul,
    w_down: &candle_core::quantized::QMatMul,
) -> Result<Tensor> {
    use candle_core::Module;
    let gate = w_gate.forward(x)?;
    let up = w_up.forward(x)?;
    // GeGLU: GELU(gate) * up
    let gated = (gelu(&gate)? * up)?;
    w_down.forward(&gated)
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE) — with frequency caching
// ---------------------------------------------------------------------------
// Encodes absolute position information into Q and K vectors by rotating
// pairs of dimensions. This is what lets the model know token order.
//
// For each pair (x_i, x_{i+1}), rotate by angle θ_i * position:
//   x_i'     = x_i * cos(θ) - x_{i+1} * sin(θ)
//   x_{i+1}' = x_i * sin(θ) + x_{i+1} * cos(θ)
//
// P0-1 OPTIMIZATION: inv_freq is pre-computed and cached per (head_dim, rope_theta)
// key, avoiding repeated powf() computations across layers and tokens.

/// Cache for pre-computed RoPE inverse frequency bands.
///
/// Key: (head_dim, rope_theta encoded as u64 bits) → inv_freq tensor on device.
/// Thread-safe via Mutex for concurrent access from multiple generators.
pub struct RopeCache {
    inv_freq_cache: Mutex<HashMap<(usize, u64), Tensor>>,
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeCache {
    pub fn new() -> Self {
        Self {
            inv_freq_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or compute the inverse frequency tensor for the given head_dim and rope_theta.
    fn get_inv_freq(&self, head_dim: usize, rope_theta: f64, device: &Device) -> Result<Tensor> {
        let key = (head_dim, rope_theta.to_bits());
        let mut cache = self.inv_freq_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let tensor = Tensor::new(inv_freq, device)?;
        cache.insert(key, tensor.clone());
        Ok(tensor)
    }
}

/// Apply RoPE using a shared cache for inverse frequencies.
pub fn rope_cached(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    head_dim: usize,
    rope_theta: f64,
    cache: &RopeCache,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let dtype = q.dtype();
    let seq_len = q.dim(1)?;

    let inv_freq = cache.get_inv_freq(head_dim, rope_theta, device)?;

    let positions: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
    let angles = positions.matmul(&inv_freq.unsqueeze(0)?)?;

    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;

    let q_rotated = apply_rotary_emb(q, &cos, &sin)?;
    let k_rotated = apply_rotary_emb(k, &cos, &sin)?;

    Ok((q_rotated, k_rotated))
}

/// Uncached rope — retained for backward compatibility and tests.
pub fn rope(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    head_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let dtype = q.dtype();
    let seq_len = q.dim(1)?;

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq, device)?;

    let positions: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
    let angles = positions.matmul(&inv_freq.unsqueeze(0)?)?;

    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;

    let q_rotated = apply_rotary_emb(q, &cos, &sin)?;
    let k_rotated = apply_rotary_emb(k, &cos, &sin)?;

    Ok((q_rotated, k_rotated))
}

/// Apply rotary embedding to a tensor of shape [batch, seq_len, n_heads, head_dim]
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    // Split last dimension into two halves
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    // Broadcast cos/sin to match x shape: [seq_len, half_dim] → [1, seq_len, 1, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
    let rot_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rot_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    Tensor::cat(&[&rot_x1, &rot_x2], D::Minus1)
}

// ---------------------------------------------------------------------------
// Grouped Query Attention (GQA)
// ---------------------------------------------------------------------------
// GQA: Multiple Q heads share fewer K/V heads (e.g., 32 Q heads, 8 KV heads)
// to reduce memory. K/V heads are repeated to match Q head count, then
// standard multi-head attention is applied.
//
// Uses Flash Attention when available (--features flash-attn) for O(N) memory
// and ~2× throughput. Falls back to standard matmul attention otherwise.

/// Unified attention dispatcher — picks Flash or Standard path automatically.
///
/// Input shapes: [batch, seq, heads, dim]
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    {
        // Flash Attention only works on CUDA with F16/BF16 tensors.
        // Fall back to standard path on CPU or if dtype isn't supported.
        let on_cuda = !q.device().is_cpu();
        let dtype_ok = matches!(q.dtype(), DType::F16 | DType::BF16);
        if on_cuda && dtype_ok {
            return flash_grouped_query_attention(q, k, v, n_heads, n_kv_heads);
        }
    }
    grouped_query_attention(q, k, v, n_heads, n_kv_heads)
}

// ---------------------------------------------------------------------------
// Flash Attention Path (CUDA only, behind feature flag)
// ---------------------------------------------------------------------------

/// Flash Attention GQA — O(N) memory, fused softmax+masking in a single kernel.
///
/// Requires CUDA device and F16/BF16 inputs.
/// Input shapes: [batch, seq, heads, dim]
#[cfg(feature = "flash-attn")]
pub fn flash_grouped_query_attention(
    q: &Tensor,     // [batch, seq_q, n_heads, head_dim]
    k: &Tensor,     // [batch, seq_kv, n_kv_heads, head_dim]
    v: &Tensor,     // [batch, seq_kv, n_kv_heads, head_dim]
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Repeat K/V heads to match Q if GQA
    let repeat_factor = n_heads / n_kv_heads;
    let (k, v) = if repeat_factor > 1 {
        // Flash Attention expects [batch, seq, heads, dim]
        // We need to expand in the heads dimension
        let (batch, seq_kv, _n_kv, hd) = k.dims4()?;
        let k = k
            .unsqueeze(3)?  // [batch, seq_kv, n_kv_heads, 1, head_dim]
            .expand(&[batch, seq_kv, n_kv_heads, repeat_factor, hd])?
            .contiguous()?
            .reshape(&[batch, seq_kv, n_heads, hd])?;
        let v = v
            .unsqueeze(3)?
            .expand(&[batch, seq_kv, n_kv_heads, repeat_factor, hd])?
            .contiguous()?
            .reshape(&[batch, seq_kv, n_heads, hd])?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    // Ensure contiguity for Flash Attention
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    // candle-flash-attn: FlashAttn with causal masking built-in
    let flash = candle_flash_attn::FlashAttn {
        softmax_scale,
        alibi_slopes: None,
        window_size_left: None,
        window_size_right: Some(0), // causal: can only attend to current + past
        softcap: None,
    };

    // Apply the fused flash attention kernel
    q.apply_op3_no_bwd(&k, &v, &flash)
}

// ---------------------------------------------------------------------------
// Standard Attention Path (CPU + GPU fallback)
// ---------------------------------------------------------------------------

pub fn grouped_query_attention(
    q: &Tensor,     // [batch, seq_q, n_heads, head_dim]
    k: &Tensor,     // [batch, total_seq, n_kv_heads, head_dim]
    v: &Tensor,     // [batch, total_seq, n_kv_heads, head_dim]
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Transpose to [batch, heads, seq, dim] for matmul
    // .contiguous() is required because transpose() creates non-contiguous
    // views and Candle's matmul only works on contiguous tensors.
    let q = q.transpose(1, 2)?.contiguous()?; // [batch, n_heads, seq_q, head_dim]
    let k = k.transpose(1, 2)?.contiguous()?; // [batch, n_kv_heads, seq_kv, head_dim]
    let v = v.transpose(1, 2)?.contiguous()?; // [batch, n_kv_heads, seq_kv, head_dim]

    // Repeat K/V heads to match Q head count if GQA (n_heads > n_kv_heads)
    let repeat_factor = n_heads / n_kv_heads;
    let (k, v) = if repeat_factor > 1 {
        let k = repeat_kv(&k, repeat_factor)?;
        let v = repeat_kv(&v, repeat_factor)?;
        (k, v)
    } else {
        (k, v)
    };

    // Q·Kᵀ scaled dot product: [batch, n_heads, seq_q, head_dim] × [batch, n_heads, head_dim, seq_kv]
    // P0-2: k is already contiguous from line 199, transpose of a contiguous
    // tensor yields a view that matmul can handle without explicit contiguous().
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let attn_weights = (q.matmul(&k_t)? * scale)?;

    // Causal mask: prevent attending to future positions
    let seq_q = attn_weights.dim(D::Minus2)?;
    let seq_kv = attn_weights.dim(D::Minus1)?;
    let attn_weights = apply_causal_mask(&attn_weights, seq_q, seq_kv)?;

    // Softmax over key dimension
    let attn_weights = softmax(&attn_weights, D::Minus1)?;

    // Weighted sum of values: [batch, n_heads, seq_q, head_dim]
    // P0-2: softmax output is already contiguous (constructed from exp/div),
    // and v is contiguous from line 200. No extra .contiguous() needed.
    let output = attn_weights.matmul(&v)?;

    // Transpose back: [batch, seq_q, n_heads, head_dim]
    output.transpose(1, 2)?.contiguous()
}

/// Repeat KV heads to match the number of Q heads in GQA.
/// [batch, n_kv_heads, seq, dim] → [batch, n_heads, seq, dim]
fn repeat_kv(x: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    // Expand: [batch, n_kv_heads, 1, seq, dim] → [batch, n_kv_heads, repeat, seq, dim]
    let x = x
        .unsqueeze(2)?
        .expand(&[batch, n_kv_heads, repeat, seq_len, head_dim])?
        .contiguous()?
        .reshape(&[batch, n_kv_heads * repeat, seq_len, head_dim])?;
    Ok(x)
}

// ---------------------------------------------------------------------------
// SiLU-Gated Feed-Forward Network (SwiGLU)
// ---------------------------------------------------------------------------
// LLaMA uses SwiGLU instead of standard ReLU FFN:
//   FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
//
// SiLU(x) = x * sigmoid(x)
//
pub fn silu_ffn(
    x: &Tensor,
    w_gate: &candle_core::quantized::QMatMul,  // [intermediate_dim, hidden_dim] quantized
    w_up: &candle_core::quantized::QMatMul,    // [intermediate_dim, hidden_dim] quantized
    w_down: &candle_core::quantized::QMatMul,  // [hidden_dim, intermediate_dim] quantized
) -> Result<Tensor> {
    use candle_core::Module;

    // gate_proj(x) and up_proj(x): quantized linear projections
    let gate = w_gate.forward(x)?;
    let up = w_up.forward(x)?;

    // SiLU activation on gate, then element-wise multiply with up
    let activated = silu(&gate)?.mul(&up)?;

    // down_proj: project back to hidden_dim (quantized)
    w_down.forward(&activated)
}

/// SiLU (Sigmoid Linear Unit): x * σ(x) = x / (1 + exp(-x))
///
/// P1-2: Optimized 3-op form computing x / (1 + exp(-x)) directly.
/// Avoids intermediate `ones_like` and `broadcast_div(ones, ...)` allocations.
/// On CUDA, Candle dispatches each op to cuDNN kernels automatically.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // x / (1 + exp(-x))  — numerically equivalent to x * sigmoid(x)
    let denom = (x.neg()?.exp()? + 1.0)?;
    x.broadcast_div(&denom)
}

// ---------------------------------------------------------------------------
// Phi-3 Partial RoPE — rotate only the first fraction of head_dim
// ---------------------------------------------------------------------------
// Phi-3 uses partial rotary embeddings: only the first `partial_factor * head_dim`
// dimensions of Q and K are rotated. The remaining dimensions pass through unchanged.
//
// GGUF key: `phi3.rope.partial_factor` (typically 0.5 for Phi-3-mini)
//
/// RoPE applied to only the first `partial_factor × head_dim` dimensions.
///
/// Remaining dims are concatenated unchanged. Reuses the existing [`RopeCache`].
///
/// # Arguments
/// * `q`, `k`        — `[batch, seq, heads, head_dim]`
/// * `start_pos`     — position offset for generation
/// * `head_dim`      — full head dimension
/// * `rope_theta`    — base frequency
/// * `partial_factor`— fraction to rotate (0.0–1.0, e.g. 0.5)
/// * `cache`         — shared Rope inv-freq cache
pub fn rope_partial_cached(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    head_dim: usize,
    rope_theta: f64,
    partial_factor: f64,
    cache: &RopeCache,
) -> Result<(Tensor, Tensor)> {
    let rotated_dim = ((head_dim as f64 * partial_factor) as usize).max(2) & !1; // even
    let pass_dim = head_dim - rotated_dim;

    if pass_dim == 0 {
        // No partial — full standard RoPE
        return rope_cached(q, k, start_pos, head_dim, rope_theta, cache);
    }

    // Split last dim into rotated + pass-through
    let q_rot = q.narrow(3, 0, rotated_dim)?;
    let q_pass = q.narrow(3, rotated_dim, pass_dim)?;
    let k_rot = k.narrow(3, 0, rotated_dim)?;
    let k_pass = k.narrow(3, rotated_dim, pass_dim)?;

    // Apply RoPE only to the rotated portion
    let (q_rot, k_rot) = rope_cached(&q_rot, &k_rot, start_pos, rotated_dim, rope_theta, cache)?;

    // Concatenate back
    let q_out = Tensor::cat(&[&q_rot, &q_pass], 3)?;
    let k_out = Tensor::cat(&[&k_rot, &k_pass], 3)?;
    Ok((q_out, k_out))
}

// ---------------------------------------------------------------------------
// Sliding-Window GQA — Mistral / Phi-3 attention with window constraint
// ---------------------------------------------------------------------------
// Combines grouped-query attention with the sliding window causal mask.
// After computing raw attention scores, positions outside the window get -inf.
//
/// Grouped Query Attention with sliding window causal mask (Mistral, Phi-3).
///
/// # Arguments
/// * `q`           — `[batch, seq_q, n_heads, head_dim]`
/// * `k`, `v`      — `[batch, seq_kv, n_kv_heads, head_dim]`
/// * `n_heads`     — number of query heads
/// * `n_kv_heads`  — number of KV heads (GQA repeat factor = n_heads / n_kv_heads)
/// * `window_size` — tokens per attention window
/// * `start_pos`   — absolute position of first query token
pub fn sliding_window_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    window_size: usize,
    start_pos: usize,
) -> Result<Tensor> {
    let (batch, seq_q, _n_heads_q, head_dim) = q.dims4()?;
    let (_batch, seq_kv, _n_kv_heads, _head_dim_kv) = k.dims4()?;

    let repeat = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Transpose for bmm: [batch, heads, seq, head_dim]
    let q_t = q.transpose(1, 2)?;
    let k_t = k.transpose(1, 2)?;
    let v_t = v.transpose(1, 2)?;

    // GQA: repeat KV heads to match Q heads
    let k_exp = if repeat > 1 {
        k_t.repeat((1, repeat, 1, 1))?
    } else {
        k_t
    };
    let v_exp = if repeat > 1 {
        v_t.repeat((1, repeat, 1, 1))?
    } else {
        v_t
    };

    // Scaled dot-product: [batch, n_heads, seq_q, seq_kv]
    let k_exp_t = k_exp.transpose(2, 3)?;
    let scores = (q_t.matmul(&k_exp_t)? * scale)?;

    // Apply sliding-window causal mask
    let scores = sliding_window_attention(&scores, seq_q, seq_kv, window_size, start_pos)?;

    // Softmax + weighted sum
    let weights = softmax(&scores, D::Minus1)?;
    let out = weights.matmul(&v_exp)?; // [batch, n_heads, seq_q, head_dim]

    // Transpose back: [batch, seq_q, n_heads, head_dim]
    out.transpose(1, 2)
}

// ---------------------------------------------------------------------------
// Softmax — Numerically Stable
// ---------------------------------------------------------------------------
pub fn softmax(x: &Tensor, dim: D) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

// ---------------------------------------------------------------------------
// Causal Mask — Fused On-Device
// ---------------------------------------------------------------------------
// Creates upper-triangular mask of -inf to prevent attending to future tokens.
//
//   [0,    -inf, -inf]
//   [0,    0,    -inf]
//   [0,    0,    0   ]
//
pub fn apply_causal_mask(scores: &Tensor, seq_q: usize, seq_kv: usize) -> Result<Tensor> {
    if seq_q == 1 {
        // Single token generation — no masking needed (it can attend to all past)
        return Ok(scores.clone());
    }
    let device = scores.device();
    let dtype = scores.dtype();
    // Build causal mask: mask[i][j] = 0 if j <= i + (seq_kv - seq_q), else -inf
    // This handles the case where seq_kv > seq_q (KV cache already has past tokens)
    let offset = seq_kv - seq_q;
    let mask: Vec<f32> = (0..seq_q)
        .flat_map(|i| {
            (0..seq_kv).map(move |j| {
                if j <= i + offset {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::new(mask, device)?
        .reshape((seq_q, seq_kv))?
        .to_dtype(dtype)?;

    // Broadcast add: scores [batch, heads, seq_q, seq_kv] + mask [seq_q, seq_kv]
    scores.broadcast_add(&mask)
}

/// Phase-aware causal mask — §7 Integration Contract.
///
/// Dispatches to the correct masking strategy based on the ARB batch phase:
///
/// | Phase   | seq_q | Behaviour |
/// |---------|-------|-----------|
/// | Decode  | 1     | **No mask** — single token attends to all cached KV |
/// | Prefill | > 1   | **Causal** — upper-triangular -∞ mask |
/// | Mixed   | any   | Falls back to `apply_causal_mask` (general path) |
///
/// # Arguments
/// * `scores` — attention weight tensor `[batch, heads, seq_q, seq_kv]`
/// * `is_prefill` — true when the batch is in full prefill mode (`all_prefill`)
/// * `seq_q` — query sequence length (number of new tokens)
/// * `seq_kv` — key/value sequence length (cached + new)
pub fn apply_causal_mask_phased(
    scores: &Tensor,
    is_prefill: bool,
    seq_q: usize,
    seq_kv: usize,
) -> Result<Tensor> {
    if !is_prefill || seq_q == 1 {
        // Decode phase: single query token — attends to all past KV freely.
        // No masking needed (and mask would be a 1×seq_kv row of all zeros).
        return Ok(scores.clone());
    }
    // Prefill phase with multiple query tokens — apply causal upper-tri mask.
    apply_causal_mask(scores, seq_q, seq_kv)
}

// ---------------------------------------------------------------------------
// I1 — Sliding Window Attention (Mistral, Gemma 3, StarCoder2)
// ---------------------------------------------------------------------------
// Restricts each token to attend only to the W most-recent tokens, where W
// is the window size. This bounds KV-cache memory to O(W) instead of O(seq_len)
// and reduces attention compute from O(seq²) to O(seq·W).
//
// Mistral 7B: window = 4096 (alternating local/global in Mistral 3+)
// Gemma 3:    interleaved — every 6th layer is global, rest are local W=512
// StarCoder2: window = 16384
//
// Implementation: we add a "left edge" causal bias on top of the standard
// causal mask — positions further left than `window_size` get -inf.
//
/// Apply sliding window + causal mask to attention scores.
///
/// Combined effect:
/// - Causal: position j can only attend to positions ≤ j (no future)
/// - Sliding: position j can only attend to positions ≥ j - window_size + 1
///
/// # Arguments
/// * `scores`      — raw attention scores `[batch, heads, seq_q, seq_kv]`
/// * `seq_q`       — query sequence length
/// * `seq_kv`      — key/value sequence length (seq_q ≤ seq_kv in decode)
/// * `window_size` — maximum number of past tokens to attend to (e.g., 4096)
/// * `start_pos`   — position of the first query token in the full sequence
///
/// # Notes
/// During prefill: `start_pos = 0`, `seq_q = seq_kv = prompt_len`
/// During decode:  `start_pos = kv_cache_len`, `seq_q = 1`
pub fn sliding_window_attention(
    scores: &Tensor,
    seq_q: usize,
    seq_kv: usize,
    window_size: usize,
    start_pos: usize,
) -> Result<Tensor> {
    let device = scores.device();
    let dtype = scores.dtype();
    // Build the mask: 1 = attend, 0 = mask out
    // For query position q_i (absolute = start_pos + i) and key position k_j (absolute = j):
    //   attend if: k_j <= q_i (causal) AND k_j >= q_i - window_size + 1 (sliding)
    let neg_inf = f32::NEG_INFINITY;

    let mut mask_data = vec![0.0f32; seq_q * seq_kv];
    for i in 0..seq_q {
        let q_abs = start_pos + i;
        for j in 0..seq_kv {
            let k_abs = j; // KV positions are absolute (0..seq_kv)
            let causal_ok = k_abs <= q_abs;
            let window_ok = k_abs + window_size > q_abs; // k_abs >= q_abs - window_size + 1
            if !causal_ok || !window_ok {
                mask_data[i * seq_kv + j] = neg_inf;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, 1, seq_q, seq_kv), device)?
        .to_dtype(dtype)?;
    scores.broadcast_add(&mask)
}

// ---------------------------------------------------------------------------
// I2 — YaRN / NTK-aware RoPE Scaling (128K+ context: Llama 3.1, Qwen, Gemma 3)
// ---------------------------------------------------------------------------
// Standard RoPE uses base freq θ_i = θ_base^(-2i/d) which gives poor
// extrapolation beyond the training context length.
//
// Two scaling approaches (both implemented here):
//
// 1. NTK-aware linear scaling (simple):
//    θ_i' = (θ_base * scale)^(-2i/d)
//    where scale = desired_ctx / train_ctx
//    Used by: Code Llama, Qwen 2.5 (basic extend)
//
// 2. YaRN (Yet Another RoPE extensioN — Peng et al. 2023):
//    Applies non-uniform frequency interpolation:
//    - High-frequency dims: unchanged (short-distance dependencies)
//    - Low-frequency dims: scaled by 1/s (global position)
//    - Mid-frequency dims: interpolated (linear ramp)
//    Used by: Llama 3.1, Qwen 2.5 (128K models), Gemma 3
//
// Reference: https://arxiv.org/abs/2309.00071

/// Configuration for YaRN/NTK-aware RoPE scaling.
#[derive(Debug, Clone)]
pub struct RopeScalingConfig {
    /// Scaling type
    pub kind: RopeScalingKind,
    /// Original training context length
    pub original_max_position: usize,
    /// Desired extended context length
    pub extended_max_position: usize,
    /// YaRN: low-frequency wavelen threshold (default: 1.0)
    pub low_freq_factor: f64,
    /// YaRN: high-frequency wavelen threshold (default: 4.0)
    pub high_freq_factor: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingKind {
    /// Basic linear scale — multiply θ_base by scale factor
    Ntk,
    /// YaRN non-uniform scaling (Peng et al. 2023)
    Yarn,
}

impl Default for RopeScalingConfig {
    fn default() -> Self {
        Self {
            kind: RopeScalingKind::Yarn,
            original_max_position: 4096,
            extended_max_position: 131072, // 128K
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
        }
    }
}

impl RopeScalingConfig {
    /// Llama 3.1 / 3.3 YaRN config (train=8192, extend=131072)
    pub fn llama3_1() -> Self {
        Self {
            kind: RopeScalingKind::Yarn,
            original_max_position: 8192,
            extended_max_position: 131072,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
        }
    }

    /// Qwen 2.5 / Qwen3 NTK config (train=32768, extend=131072)
    pub fn qwen_extended() -> Self {
        Self {
            kind: RopeScalingKind::Ntk,
            original_max_position: 32768,
            extended_max_position: 131072,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
        }
    }

    pub fn scale_factor(&self) -> f64 {
        self.extended_max_position as f64 / self.original_max_position as f64
    }
}

/// Compute scaled RoPE inverse frequencies using YaRN or NTK scaling.
///
/// Returns inv_freq tensor `[head_dim/2]` with scaled frequencies.
/// Drop-in replacement for the unscaled inv_freq used in `rope_cached`.
///
/// # Arguments
/// * `head_dim`   — attention head dimension (must be even)
/// * `rope_theta` — base frequency (e.g., 10000.0 for Llama, 1000000.0 for Llama3)
/// * `cfg`        — scaling configuration
/// * `device`     — target device
pub fn rope_scaled_inv_freq(
    head_dim: usize,
    rope_theta: f64,
    cfg: &RopeScalingConfig,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let scale = cfg.scale_factor();

    let inv_freqs: Vec<f64> = (0..half_dim)
        .map(|i| {
            // Base inverse frequency for dimension pair i
            let base_freq = rope_theta.powf(-(2.0 * i as f64) / head_dim as f64);

            match cfg.kind {
                RopeScalingKind::Ntk => {
                    // NTK: uniformly scale the base → effectively extends context
                    // by rotating slower, so positions don't wrap around as fast
                    let scaled_theta = rope_theta * scale.powf(head_dim as f64 / (head_dim as f64 - 2.0));
                    scaled_theta.powf(-(2.0 * i as f64) / head_dim as f64)
                }
                RopeScalingKind::Yarn => {
                    // YaRN: non-uniform interpolation
                    // wavelength of this frequency band
                    let wavelen = 2.0 * std::f64::consts::PI / base_freq;
                    let low_wavelen = cfg.original_max_position as f64 / cfg.low_freq_factor;
                    let high_wavelen = cfg.original_max_position as f64 / cfg.high_freq_factor;

                    if wavelen < high_wavelen {
                        // High-frequency band: keep unchanged (short-range deps)
                        base_freq
                    } else if wavelen > low_wavelen {
                        // Low-frequency band: scale down by 1/s (global position info)
                        base_freq / scale
                    } else {
                        // Mid-frequency band: smooth linear ramp between the two
                        let ramp = (low_wavelen / wavelen - 1.0)
                            / (low_wavelen / high_wavelen - 1.0);
                        let ramp = ramp.clamp(0.0, 1.0);
                        base_freq * (1.0 - ramp) + (base_freq / scale) * ramp
                    }
                }
            }
        })
        .collect();

    let inv_freqs_f32: Vec<f32> = inv_freqs.iter().map(|&v| v as f32).collect();
    Tensor::from_vec(inv_freqs_f32, (half_dim,), device)
}

/// Apply RoPE using pre-scaled inverse frequencies (YaRN/NTK path).
///
/// Drop-in for `rope_cached` when long-context scaling is active.
/// Call `rope_scaled_inv_freq()` once at model load time, cache the result.
pub fn rope_with_inv_freq(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // inv_freq: [half_dim] → build position-dependent rotation
    let seq_len = q.dim(1)?;
    let half_dim = inv_freq.dim(0)?;
    let device = q.device();

    // pos_ids: [seq_len]
    let pos_ids: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos_ids = Tensor::from_vec(pos_ids, (seq_len,), device)?;

    // freqs: [seq_len, half_dim] = outer(pos_ids, inv_freq)
    let pos_col = pos_ids.unsqueeze(1)?;           // [seq, 1]
    let inv_row = inv_freq.unsqueeze(0)?;           // [1, half_dim]
    let freqs = pos_col.broadcast_mul(&inv_row)?;  // [seq, half_dim]

    let cos = freqs.cos()?; // [seq, half_dim]
    let sin = freqs.sin()?;

    let apply_rope = |x: &Tensor| -> Result<Tensor> {
        let (batch, seq, heads, dim) = x.dims4()?;
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;       // [b,s,h, half]
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?; // [b,s,h, half]
        let cos_b = cos.reshape((1, seq, 1, half_dim))?;
        let sin_b = sin.reshape((1, seq, 1, half_dim))?;
        let rot = (x1.broadcast_mul(&cos_b)? - x2.broadcast_mul(&sin_b)?)?;
        let rot2 = (x1.broadcast_mul(&sin_b)? + x2.broadcast_mul(&cos_b)?)?;
        let _ = (batch, heads, dim); // suppress unused warnings
        Tensor::cat(&[&rot, &rot2], D::Minus1)
    };

    Ok((apply_rope(q)?, apply_rope(k)?))
}

// ---------------------------------------------------------------------------
// I6 — ALiBi Positional Encoding (MPT, Baichuan 7B, BLOOM)
// ---------------------------------------------------------------------------
// Attention with Linear Biases (Press et al. 2021, arXiv:2108.12409).
// Instead of adding position embeddings to tokens, ALiBi adds a linear
// penalty to attention scores based on the distance between query and key:
//
//   attn_score(q_i, k_j) -= slope_h * |i - j|
//
// where slope_h is a head-specific geometric slope:
//   slope_h = 2^(-8h/n_heads)   for h in [1..n_heads]
//
// Advantages:
// - No position embedding lookup (works at any context length)
// - Better length extrapolation than sinusoidal / RoPE
// - Causal masking: positions with j > i get -inf (standard causal mask)
//
// Used by: MPT-7B/30B, DBRX, Baichuan 7B, BLOOM-176B, and others

/// Compute ALiBi head slopes.
///
/// Returns a `Vec<f64>` of length `n_heads`, one slope per attention head.
/// Slopes follow a geometric progression: `2^(-8/n_heads)`, `2^(-16/n_heads)`, ...
pub fn alibi_slopes(n_heads: usize) -> Vec<f64> {
    let ratio = 2.0_f64.powf(8.0 / n_heads as f64);
    (1..=n_heads)
        .map(|h| 1.0 / ratio.powi(h as i32))
        .collect()
}

/// Add ALiBi position bias to attention scores.
///
/// Modifies `scores` (Q·Kᵀ / √d) in-place by subtracting the ALiBi penalty.
/// The causal mask is also applied (future positions → -inf).
///
/// # Arguments
/// * `scores`   — raw attention scores `[batch, n_heads, seq_q, seq_kv]`
/// * `seq_q`    — query sequence length
/// * `seq_kv`   — key/value sequence length
/// * `start_pos`— absolute position of first query token (for decode: kv_cache_len)
/// * `slopes`   — per-head slopes from `alibi_slopes(n_heads)` — precompute and cache
pub fn apply_alibi(
    scores: &Tensor,
    seq_q: usize,
    seq_kv: usize,
    start_pos: usize,
    slopes: &[f64],
) -> Result<Tensor> {
    let n_heads = slopes.len();
    let device = scores.device();
    let dtype = scores.dtype();

    // Build ALiBi bias matrix: [n_heads, seq_q, seq_kv]
    // bias[h, i, j] = -slope_h * |q_abs_i - j|  (causal: j > q_abs → -inf)
    let neg_inf = f32::NEG_INFINITY;
    let mut bias_data = vec![0.0f32; n_heads * seq_q * seq_kv];

    for h in 0..n_heads {
        let slope = slopes[h] as f32;
        for i in 0..seq_q {
            let q_abs = (start_pos + i) as isize;
            for j in 0..seq_kv {
                let k_abs = j as isize;
                let val = if k_abs > q_abs {
                    // Future position — causal mask
                    neg_inf
                } else {
                    // ALiBi penalty: linear distance
                    -slope * (q_abs - k_abs) as f32
                };
                bias_data[h * seq_q * seq_kv + i * seq_kv + j] = val;
            }
        }
    }

    let bias = Tensor::from_vec(bias_data, (n_heads, seq_q, seq_kv), device)?
        .unsqueeze(0)?         // [1, n_heads, seq_q, seq_kv]
        .to_dtype(dtype)?;
    scores.broadcast_add(&bias)
}

// ===========================================================================
// P1 — Parallel Attention + FFN (Falcon Family)
// ===========================================================================
// Falcon 7B/40B/180B use a "parallel" transformer block where the attention
// sublayer and FFN sublayer run on the *same* residual, then their outputs are
// summed (as opposed to sequential where FFN takes the attention output).
//
// Standard (sequential):
//   h = x + attn(norm(x))
//   out = h + ffn(norm(h))          ← FFN sees attn output
//
// Parallel (Falcon):
//   normed = norm(x)
//   out = x + attn(normed) + ffn(normed)   ← both see same normed input
//
// Benefits: fewer sequential operations; better GPU utilisation on large models.
// Implementation: caller computes attn_out + ffn_out and passes both here.

/// Combine attention and FFN residual outputs using Falcon parallel layout.
///
/// `residual + attn_out + ffn_out`
///
/// # Arguments
/// * `residual` — original input `x` before the transformer block
/// * `attn_out` — output of the attention sublayer (applied to normed input)
/// * `ffn_out`  — output of the FFN sublayer (also applied to normed input)
///
/// Returns the combined output `[batch, seq, hidden]`.
pub fn parallel_attn_ffn(
    residual: &Tensor,
    attn_out: &Tensor,
    ffn_out: &Tensor,
) -> Result<Tensor> {
    // Two adds: (residual + attn_out) + ffn_out
    // Both attn_out and ffn_out must be the same shape as residual.
    (residual.broadcast_add(attn_out)?.broadcast_add(ffn_out))
}

/// Falcon-style single shared LayerNorm for both attn + FFN paths.
///
/// In Falcon, a single `input_layernorm` is applied to `x`, and the result
/// is fed to *both* the attention and FFN. This fn returns the normed tensor
/// for reuse. Caller should call `layer_norm(x, weight, bias, eps)` once and
/// pass the result to both attn and FFN projections.
///
/// This is a documentation helper — the actual norm is `layer_norm()`.
/// Call `layer_norm()` once, reuse result for both paths.
#[inline(always)]
pub fn falcon_shared_norm<'a>(normed: &'a Tensor) -> &'a Tensor {
    normed // pass-through; serves as documentation/naming
}

// ===========================================================================
// P2 — Partial Rotary Embeddings (Phi-2, StableLM, Neox)
// ===========================================================================
// Some models apply RoPE only to the *first rotary_dim* channels of each head,
// leaving the remaining channels unchanged (no rotation applied):
//
//   q_rot = rope(q[:, :, :, :rotary_dim])    ← rotated
//   q_pass = q[:, :, :, rotary_dim:]         ← kept as-is
//   q_out = cat([q_rot, q_pass], dim=-1)
//
// Phi-2: rotary_dim = 32, head_dim = 80 (40% rotated)
// StableLM: rotary_dim = 32, head_dim = 64 (50% rotated)
// GPT-NeoX: rotary_dim = head_dim (100% — standard)
//
// This is distinct from YaRN (frequency scaling) — it's partial *coverage*.

/// Apply RoPE to only the first `rotary_dim` channels of each attention head.
///
/// Channels `[rotary_dim..]` are left unchanged (passed through unrotated).
///
/// # Arguments
/// * `q`, `k`       — input queries/keys `[batch, seq, n_heads, head_dim]`
/// * `start_pos`    — position offset for KV cache
/// * `rotary_dim`   — number of channels to rotate (must be even, ≤ head_dim)
/// * `rope_theta`   — RoPE base frequency
/// * `device`       — compute device
///
/// # Model-specific values
/// | Model | head_dim | rotary_dim |
/// |---|---|---|
/// | Phi-2 | 80 | 32 |
/// | StableLM 3B | 64 | 32 |
/// | GPT-NeoX 20B | 64 | 64 |
pub fn rope_partial(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    rotary_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dim(D::Minus1)?;
    assert!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "rotary_dim ({rotary_dim}) must be even and ≤ head_dim ({head_dim})"
    );

    let pass_dim = head_dim - rotary_dim;
    let seq_len = q.dim(1)?;
    let device = q.device();
    let dtype = q.dtype();

    // Compute rotation angles for the first rotary_dim channels
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

    let positions: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos = Tensor::from_vec(positions, (seq_len,), device)?.unsqueeze(1)?;
    let angles = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [seq, half]

    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;

    // Broadcast cos/sin to [1, seq, 1, half]
    let cos_b = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin_b = sin.unsqueeze(0)?.unsqueeze(2)?;

    let apply_partial = |x: &Tensor| -> Result<Tensor> {
        // Split into rotated and pass-through parts
        let x_rot = x.narrow(D::Minus1, 0, rotary_dim)?;
        let x_pass = if pass_dim > 0 {
            Some(x.narrow(D::Minus1, rotary_dim, pass_dim)?)
        } else {
            None
        };

        // Apply RoPE to rotated part
        let x1 = x_rot.narrow(D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(D::Minus1, half, half)?;
        let rot1 = (x1.broadcast_mul(&cos_b)? - x2.broadcast_mul(&sin_b)?)?;
        let rot2 = (x1.broadcast_mul(&sin_b)? + x2.broadcast_mul(&cos_b)?)?;
        let rotated = Tensor::cat(&[&rot1, &rot2], D::Minus1)?;

        // Concatenate with pass-through
        if let Some(pass) = x_pass {
            Tensor::cat(&[&rotated, &pass], D::Minus1)
        } else {
            Ok(rotated)
        }
    };

    Ok((apply_partial(q)?, apply_partial(k)?))
}

// ===========================================================================
// P3 — Learned Positional Embeddings (GPT-2 style)
// ===========================================================================
// GPT-2, older BERT-style models, and some Falcon variants use a learned
// position embedding table (a [max_seq_len, hidden_dim] weight matrix) instead
// of sinusoidal or RoPE. Each position index maps to a learnable vector that
// is *added* to the token embeddings before the transformer stack.
//
// The table is typically stored in GGUF as `position_embd.weight`.
// At inference: just index into the table by position and add to token embeds.
//
// GPT-2: max_pos=1024, hidden=768/1024/1280/1600
// GPT-2-XL: max_pos=1024, hidden=1600

/// Add learned positional embeddings to token embeddings.
///
/// # Arguments
/// * `token_embeds` — token embedding tensor `[batch, seq_len, hidden_dim]`
/// * `pos_table`    — learned position embedding table `[max_seq, hidden_dim]`
/// * `start_pos`    — position of the first token (usually 0 for prefill, cache_len for decode)
///
/// # Returns
/// `[batch, seq_len, hidden_dim]` — combined embeddings.
pub fn add_learned_pos_embeds(
    token_embeds: &Tensor,
    pos_table: &Tensor,
    start_pos: usize,
) -> Result<Tensor> {
    let seq_len = token_embeds.dim(1)?;
    let max_pos = pos_table.dim(0)?;
    assert!(
        start_pos + seq_len <= max_pos,
        "position out of learned table range: start_pos({start_pos}) + seq_len({seq_len}) > max_pos({max_pos})"
    );

    // Slice the position table: [seq_len, hidden_dim]
    let pos_slice = pos_table.narrow(0, start_pos, seq_len)?;

    // Broadcast to [1, seq_len, hidden_dim] and add to [batch, seq_len, hidden_dim]
    let pos_broadcast = pos_slice.unsqueeze(0)?;
    token_embeds.broadcast_add(&pos_broadcast)
}

/// Build sinusoidal positional embedding table (GPT-2 fallback if no learned table).
///
/// Some GPT-2 checkpoints lack a learned table and use sinusoidal instead.
/// `[max_seq_len, hidden_dim]` — matches GPT-2 layout.
pub fn sinusoidal_pos_table(
    max_seq_len: usize,
    hidden_dim: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let half = hidden_dim / 2;
    let mut data = vec![0.0f32; max_seq_len * hidden_dim];

    for pos in 0..max_seq_len {
        for i in 0..half {
            let angle = pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / hidden_dim as f32);
            data[pos * hidden_dim + 2 * i] = angle.sin();
            data[pos * hidden_dim + 2 * i + 1] = angle.cos();
        }
    }

    Tensor::from_vec(data, (max_seq_len, hidden_dim), device)
}

// ===========================================================================
// P4 — Blocksparse Attention (Phi-3 Small)
// ===========================================================================
// Phi-3 Small uses a blocksparse attention pattern defined by a 2D block mask.
// Rather than attending to all positions, each query block only attends to a
// sparse subset of key/value blocks, dramatically reducing compute for long seqs.
//
// The pattern is given at model load time as a binary matrix:
//   block_mask[q_block][k_block] = 1 → this (q,k) block attends
//   block_mask[q_block][k_block] = 0 → skip (set to -inf)
//
// Phi-3 Small 8K: block_size=128, 64 blocks for 8K context
// Phi-3 Small 128K: block_size=128, 1024 blocks with sparse cross-block pattern
//
// Implementation: expand the block mask to a full token-level mask and apply
// as additive bias (0=attend, -inf=skip). Production code would use actual
// block-sparse CUDA kernels, but this CPU-compatible fallback is fully correct.

/// Configuration for blocksparse attention.
#[derive(Debug, Clone)]
pub struct BlocksparseConfig {
    /// Block size in tokens (default 128 for Phi-3 Small)
    pub block_size: usize,
    /// Number of blocks per sequence (seq_len / block_size)
    pub n_blocks: usize,
    /// Attention pattern: `pattern[q_block * n_blocks + k_block] = true` → can attend
    pub pattern: Vec<bool>,
}

impl BlocksparseConfig {
    /// Create a blocksparse config.
    ///
    /// # Arguments
    /// * `block_size` — tokens per block (e.g. 128)
    /// * `pattern`    — flat [n_blocks × n_blocks] bool matrix (row = query block)
    pub fn new(block_size: usize, n_blocks: usize, pattern: Vec<bool>) -> Self {
        assert_eq!(
            pattern.len(), n_blocks * n_blocks,
            "pattern len {} != n_blocks² {}", pattern.len(), n_blocks * n_blocks
        );
        Self { block_size, n_blocks, pattern }
    }

    /// Phi-3 Small default: sliding window with stride-1 global blocks.
    /// Every block attends to: itself + the w previous blocks (causal sliding).
    pub fn phi3_small(n_blocks: usize, window_blocks: usize) -> Self {
        let mut pattern = vec![false; n_blocks * n_blocks];
        for q in 0..n_blocks {
            // Always attend to self (causal)
            pattern[q * n_blocks + q] = true;
            // Attend to previous `window_blocks` blocks
            let start = q.saturating_sub(window_blocks);
            for k in start..=q {
                pattern[q * n_blocks + k] = true;
            }
        }
        Self::new(128, n_blocks, pattern)
    }

    /// Check if query block `q_blk` attends to key block `k_blk`.
    pub fn attends(&self, q_blk: usize, k_blk: usize) -> bool {
        self.pattern[q_blk * self.n_blocks + k_blk]
    }
}

/// Apply blocksparse attention mask to raw attention scores.
///
/// Positions in non-attended blocks are set to NEG_INFINITY before softmax.
/// This is the correct-but-dense fallback; production uses sparse CUDA kernels.
///
/// # Arguments
/// * `scores` — raw attention scores `[batch, heads, seq_q, seq_kv]`
/// * `config` — blocksparse pattern configuration
/// * `start_pos` — absolute position of first query token (for decode)
pub fn apply_blocksparse_mask(
    scores: &Tensor,
    config: &BlocksparseConfig,
    start_pos: usize,
) -> Result<Tensor> {
    let seq_q = scores.dim(D::Minus2)?;
    let seq_kv = scores.dim(D::Minus1)?;
    let device = scores.device();
    let dtype = scores.dtype();
    let neg_inf = f32::NEG_INFINITY;

    let mut mask_data = vec![0.0f32; seq_q * seq_kv];

    for qi in 0..seq_q {
        let q_abs = start_pos + qi;
        let q_blk = q_abs / config.block_size;
        for ki in 0..seq_kv {
            let k_blk = ki / config.block_size;
            // Causal: future positions always masked
            let causal_ok = ki <= q_abs;
            // Blocksparse: check if this block pair attends
            let block_ok = q_blk < config.n_blocks
                && k_blk < config.n_blocks
                && config.attends(q_blk.min(config.n_blocks - 1), k_blk.min(config.n_blocks - 1));
            if !causal_ok || !block_ok {
                mask_data[qi * seq_kv + ki] = neg_inf;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, 1, seq_q, seq_kv), device)?
        .to_dtype(dtype)?;
    scores.broadcast_add(&mask)
}

// ===========================================================================
// P9 — Interleaved Rotary Embeddings (GLM-style: GLM-4, GLM-5, GLM-5.1)
// ===========================================================================
// ChatGLM uses "interleaved" rotary embeddings where the head dimension is
// paired differently — instead of [x0,x1,...,xH/2, x_{H/2},...,x_H]:
//
// Standard RoPE pairing: pairs are (i, i + H/2) → cat style
// GLM/CodeGemma pairing: pairs are (2i, 2i+1) → interleaved style
//
//   x_pair = [(x[0],x[1]), (x[2],x[3]), ..., (x[H-2],x[H-1])]
//   rotated[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
//   rotated[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
//
// This is sometimes called "interleaved" or "adjacent-pair" RoPE.
// GLM-4 (9B/130B), GLM-5.1 (9B), CodeGemma all use this variant.

/// Apply interleaved (GLM-style) rotary positional embeddings.
///
/// Pairs adjacent dimensions `(2i, 2i+1)` for rotation instead of
/// splitting the head dimension in half.
///
/// # Arguments
/// * `q`, `k`    — `[batch, seq, n_heads, head_dim]` (head_dim must be even)
/// * `start_pos` — KV-cache offset
/// * `rope_theta`— base frequency (GLM-4 default: 10000.0)
pub fn rope_interleaved(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dim(D::Minus1)?;
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let half = head_dim / 2;
    let seq_len = q.dim(1)?;
    let device = q.device();
    let dtype = q.dtype();

    // Compute frequencies for each pair index [0..half]
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

    let positions: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let pos = Tensor::from_vec(positions, (seq_len,), device)?.unsqueeze(1)?;
    let angles = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [seq, half]
    let cos = angles.cos()?.to_dtype(dtype)?; // [seq, half]
    let sin = angles.sin()?.to_dtype(dtype)?;

    // Broadcast to [1, seq, 1, half]
    let cos_b = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin_b = sin.unsqueeze(0)?.unsqueeze(2)?;

    let apply_interleaved = |x: &Tensor| -> Result<Tensor> {
        // x: [batch, seq, heads, head_dim]
        // Extract even and odd dimensions: [b, s, h, half]
        // Even: x[:,:,:,0::2], Odd: x[:,:,:,1::2]
        // candle narrow trick: stride not directly supported, so we use indexing
        // Alternative: unstack and restack pairs — correct but allocates
        // Fast path: reshape x to [b, s, h, half, 2], then unpack
        let shape = x.shape().dims().to_vec();
        let batch = shape[0];
        let seq = shape[1];
        let heads = shape[2];

        // Reshape to [b, s, h, half, 2] to expose adjacent pairs
        let x_pairs = x.reshape((batch, seq, heads, half, 2))?;

        // Extract even (pair dim=0) and odd (pair dim=1) — narrow on last dim
        let x_even = x_pairs.narrow(D::Minus1, 0, 1)?.reshape((batch, seq, heads, half))?;
        let x_odd  = x_pairs.narrow(D::Minus1, 1, 1)?.reshape((batch, seq, heads, half))?;

        // Apply rotation: (even*cos - odd*sin, even*sin + odd*cos)
        let rot_even = (x_even.broadcast_mul(&cos_b)? - x_odd.broadcast_mul(&sin_b)?)?;
        let rot_odd  = (x_even.broadcast_mul(&sin_b)? + x_odd.broadcast_mul(&cos_b)?)?;

        // Interleave back: [b, s, h, half, 2] then reshape to [b, s, h, head_dim]
        let rot_even_u = rot_even.unsqueeze(D::Minus1)?; // [b, s, h, half, 1]
        let rot_odd_u  = rot_odd.unsqueeze(D::Minus1)?;  // [b, s, h, half, 1]
        let interleaved = Tensor::cat(&[&rot_even_u, &rot_odd_u], D::Minus1)?; // [b,s,h,half,2]
        interleaved.reshape((batch, seq, heads, head_dim))
    };

    Ok((apply_interleaved(q)?, apply_interleaved(k)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], device)?;
        let weight = Tensor::ones((3,), DType::F32, device)?;
        let result = rms_norm(&x, &weight, 1e-5)?;
        // RMS = sqrt(mean(1+4+9)) = sqrt(14/3) ≈ 2.16
        // normed ≈ [0.46, 0.93, 1.39]
        let vals: Vec<f32> = result.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 0.4629).abs() < 0.01);
        assert!((vals[2] - 1.3887).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_silu_activation() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], device)?;
        let result = silu(&x)?;
        let vals: Vec<f32> = result.to_vec1()?;
        // silu(0) = 0, silu(1) ≈ 0.731, silu(-1) ≈ -0.269
        assert!((vals[0]).abs() < 0.001);
        assert!((vals[1] - 0.731).abs() < 0.01);
        assert!((vals[2] + 0.269).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = &Device::Cpu;
        let scores = Tensor::zeros((1, 1, 3, 3), DType::F32, device)?;
        let masked = apply_causal_mask(&scores, 3, 3)?;
        let vals: Vec<f32> = masked.flatten_all()?.to_vec1()?;
        // Position (0,1), (0,2), (1,2) should be -inf
        assert!(vals[1].is_infinite() && vals[1] < 0.0); // [0][1]
        assert!(vals[2].is_infinite() && vals[2] < 0.0); // [0][2]
        assert!(!vals[3].is_infinite());                   // [1][0] = 0
        assert!(vals[5].is_infinite() && vals[5] < 0.0); // [1][2]
        Ok(())
    }
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 1: SageAttention3 FP4 Microscaling
// ============================================================================
//
// Based on: "SageAttention3: Microscaling FP4 Attention" (2025)
//
// FP4 E2M1 format: 1 sign bit | 2 exponent bits | 1 mantissa bit
// Microscaling: per-block-of-16 scale factors stored in f32.
//
// Algorithm:
//   1. Partition Q, K into blocks of BLOCK=16 elements.
//   2. Per block, compute scale = max(|x|) / fp4_max.
//   3. Quantize x → round(x / scale) clamped to FP4 range.
//   4. Dequantize for attention compute (emulated in f32 on CPU).
//   5. Standard softmax attention with dequantized Q·Kᵀ.
//
// On CPU this provides a faithful emulation of FP4 precision for testing
// and model exploration. CUDA path would fuse steps 1-4 into one kernel.

/// FP4 E2M1 representable values (positive, excluding ±0).
/// Full set: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × sign
const FP4_POSITIVE_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Maximum representable FP4 magnitude.
const FP4_MAX: f32 = 6.0;
/// Block size for microscaling (16 elements share one scale factor).
const FP4_BLOCK: usize = 16;

/// Quantize one f32 value to the nearest FP4 E2M1 representable value.
#[inline(always)]
fn quantize_fp4(x: f32) -> f32 {
    let sign = x.signum();
    let abs = x.abs();
    // Find closest FP4 positive value by linear scan (tiny table).
    let mut best = 0.0f32;
    let mut best_dist = f32::MAX;
    for &v in &FP4_POSITIVE_VALUES {
        let d = (abs - v).abs();
        if d < best_dist {
            best_dist = d;
            best = v;
        }
    }
    sign * best
}

/// Quantize a slice to FP4 with microscaling (block size = FP4_BLOCK).
/// Returns (quantized values as f32, per-block scale factors).
fn fp4_quantize_blocked(data: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = data.len();
    let n_blocks = (n + FP4_BLOCK - 1) / FP4_BLOCK;
    let mut quant = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * FP4_BLOCK;
        let end = (start + FP4_BLOCK).min(n);
        let block = &data[start..end];

        // Per-block scale: max absolute value / FP4_MAX
        let block_max = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if block_max > 0.0 { block_max / FP4_MAX } else { 1.0 };
        scales.push(scale);

        for &x in block {
            // Scale-then-quantize
            let scaled = x / scale;
            let q = quantize_fp4(scaled.clamp(-FP4_MAX, FP4_MAX));
            // Dequantize immediately (emulation: store the error-rounded value)
            quant.push(q * scale);
        }
    }
    (quant, scales)
}

/// SageAttention3-style FP4 microscaling attention (CPU emulation).
///
/// Quantizes Q and K to FP4 with per-block microscaling before computing
/// the attention score matrix. Values V remain in full precision.
///
/// # Arguments
/// * `q` — `[batch, seq_q, n_heads, head_dim]`
/// * `k` — `[batch, seq_kv, n_kv_heads, head_dim]`
/// * `v` — `[batch, seq_kv, n_kv_heads, head_dim]`
///
/// # Returns
/// Attention output `[batch, seq_q, n_heads, head_dim]`
pub fn fp4_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    // Materialise tensors as f32 for FP4 quantization.
    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;

    let q_raw: Vec<f32> = q_f32.flatten_all()?.to_vec1()?;
    let k_raw: Vec<f32> = k_f32.flatten_all()?.to_vec1()?;

    // FP4 round-trip for Q and K.
    let (q_quant, _q_scales) = fp4_quantize_blocked(&q_raw);
    let (k_quant, _k_scales) = fp4_quantize_blocked(&k_raw);

    // Reconstruct quantized tensors on the original device.
    let device = q.device();
    let original_dtype = q.dtype();
    let q_qt = Tensor::from_vec(q_quant, q_f32.shape().clone(), device)?
        .to_dtype(original_dtype)?;
    let k_qt = Tensor::from_vec(k_quant, k_f32.shape().clone(), device)?
        .to_dtype(original_dtype)?;

    // Standard GQA with FP4-degraded Q and K.
    grouped_query_attention(&q_qt, &k_qt, v, n_heads, n_kv_heads)
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 2: KIMI Linear Attention
// ============================================================================
//
// Based on: "KIMI LINEAR: An Expressive Efficient Attention Architecture" (2025)
//
// Replaces softmax: Attention(Q,K,V)_i = φ(Q_i) · (Σ_{j≤i} φ(K_j) Vⱼᵀ)
//                                         ─────────────────────────────────
//                                         φ(Q_i) · (Σ_{j≤i} φ(K_j))
//
// Kernel function φ(x) = ELU(x) + 1  (always positive, supports causal sum)
//
// Complexity: O(N · D) vs O(N² · D) — massive memory savings for long context.
//
// Causal implementation via cumulative state:
//   S_i = S_{i-1} + φ(K_i) ⊗ V_i   (state matrix, D×D)
//   z_i = z_{i-1} + φ(K_i)           (normalizer, D)
//   o_i = φ(Q_i) · S_i / (φ(Q_i) · z_i)

/// ELU kernel function φ(x) = max(0, x) + min(0, exp(x) - 1) + 1.
/// Always ≥ 0, enabling linear attention's cumulative sum to be valid.
#[inline(always)]
fn phi_elu(x: f32) -> f32 {
    if x >= 0.0 { x + 1.0 } else { x.exp() }
}

/// KIMI-style causal linear attention (CPU reference implementation).
///
/// Operates on a single head's Q, K, V vectors for one sequence.
/// For production use, call `linear_attention_kimi` which handles batches.
///
/// # Arguments
/// * `q_seq` — `[seq_q, d]` query vectors for one head
/// * `k_seq` — `[seq_kv, d]` key vectors for one head  
/// * `v_seq` — `[seq_kv, d]` value vectors for one head
///
/// # Returns
/// Output `[seq_q, d]`
fn linear_attention_head(
    q_seq: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Cumulative state: S[d_k, d_v] and z[d_k]
    let mut state = vec![0.0f32; head_dim * head_dim]; // S: [d, d]
    let mut normalizer = vec![0.0f32; head_dim];         // z: [d]
    let mut out = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        // φ(K_t): [head_dim]
        let k_t = &k_seq[t * head_dim..(t + 1) * head_dim];
        let v_t = &v_seq[t * head_dim..(t + 1) * head_dim];
        let phi_k: Vec<f32> = k_t.iter().map(|&x| phi_elu(x)).collect();

        // Update state: S += outer(φ(K_t), V_t)
        for i in 0..head_dim {
            for j in 0..head_dim {
                state[i * head_dim + j] += phi_k[i] * v_t[j];
            }
        }
        // Update normalizer: z += φ(K_t)
        for i in 0..head_dim {
            normalizer[i] += phi_k[i];
        }

        // φ(Q_t): [head_dim]
        let q_t = &q_seq[t * head_dim..(t + 1) * head_dim];
        let phi_q: Vec<f32> = q_t.iter().map(|&x| phi_elu(x)).collect();

        // o_t = φ(Q_t) · S  (matmul with state row-by-row)
        // Then divide by φ(Q_t) · z for normalization.
        let denom: f32 = phi_q.iter().zip(&normalizer).map(|(q, z)| q * z).sum::<f32>().max(1e-6);
        for j in 0..head_dim {
            let num: f32 = phi_q.iter().enumerate().map(|(i, &q)| q * state[i * head_dim + j]).sum();
            out[t * head_dim + j] = num / denom;
        }
    }
    out
}

/// KIMI linear attention — sub-quadratic O(N·D) causal attention.
///
/// Input shapes: `[batch, seq, n_heads, head_dim]`.
/// No softmax; uses ELU kernel for positive-definite attention.
///
/// # Key properties
/// - Memory: O(D²) state per head instead of O(N) attention matrix
/// - Naturally handles infinite streaming (state summarises all past)
/// - No attention sinks (no softmax → no sink competition)
pub fn linear_attention_kimi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let (batch, seq_q, _nh, head_dim) = q.dims4()?;
    let seq_kv = k.dim(1)?;
    let repeat = n_heads / n_kv_heads;

    // Work in f32 for the linear attention recurrence.
    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;
    let v_f32 = v.to_dtype(DType::F32)?;

    let q_data: Vec<f32> = q_f32.flatten_all()?.to_vec1()?;
    let k_data: Vec<f32> = k_f32.flatten_all()?.to_vec1()?;
    let v_data: Vec<f32> = v_f32.flatten_all()?.to_vec1()?;

    let mut out_data = vec![0.0f32; batch * seq_q * n_heads * head_dim];

    // Strides for [batch, seq, heads, dim] layout
    let q_batch_stride   = seq_q * n_heads * head_dim;
    let k_batch_stride   = seq_kv * n_kv_heads * head_dim;
    let v_batch_stride   = seq_kv * n_kv_heads * head_dim;
    let out_batch_stride = seq_q * n_heads * head_dim;

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / repeat; // GQA head index

            // Extract this head's Q sequence: [seq_q, head_dim]
            let mut q_head = Vec::with_capacity(seq_q * head_dim);
            for t in 0..seq_q {
                let base = b * q_batch_stride + t * n_heads * head_dim + h * head_dim;
                q_head.extend_from_slice(&q_data[base..base + head_dim]);
            }

            // Extract this KV head's K and V sequences: [seq_kv, head_dim]
            let mut k_head = Vec::with_capacity(seq_kv * head_dim);
            let mut v_head = Vec::with_capacity(seq_kv * head_dim);
            for t in 0..seq_kv {
                let k_base = b * k_batch_stride + t * n_kv_heads * head_dim + kv_h * head_dim;
                let v_base = b * v_batch_stride + t * n_kv_heads * head_dim + kv_h * head_dim;
                k_head.extend_from_slice(&k_data[k_base..k_base + head_dim]);
                v_head.extend_from_slice(&v_data[v_base..v_base + head_dim]);
            }

            let head_out = linear_attention_head(&q_head, &k_head, &v_head, seq_q, head_dim);

            // Write back
            for t in 0..seq_q {
                let dst = b * out_batch_stride + t * n_heads * head_dim + h * head_dim;
                out_data[dst..dst + head_dim].copy_from_slice(&head_out[t * head_dim..(t + 1) * head_dim]);
            }
        }
    }

    let device = q.device();
    let out = Tensor::from_vec(out_data, (batch, seq_q, n_heads, head_dim), device)?
        .to_dtype(q.dtype())?;
    Ok(out)
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 2b: Gated Attention
// ============================================================================
//
// Based on: "Gated Attention for Large Language Models" (2025)
//
// Standard attention has "attention sinks" — certain tokens (BOS, punctuation)
// absorb disproportionate attention probability, causing information loss.
//
// Gated attention adds a sigmoid gate on the attention OUTPUT per-head:
//   gate_i = sigmoid(W_gate · x_i)   (learned per-head gate)
//   out_i  = gate_i ⊙ Attention(Q_i, K, V)
//
// Benefits:
//   - Gate can suppress sink-head outputs (gate ≈ 0 for unimportant heads)
//   - Natural sparsity emerges during training (many gates close to 0)
//   - Eliminates attention-sink problem without modifying QKV computation

/// Apply per-head sigmoid gating to attention output.
///
/// # Arguments
/// * `attn_out`  — attention output `[batch, seq, n_heads, head_dim]`
/// * `x`         — normed input (pre-attention) `[batch, seq, hidden_dim]`
///   where `hidden_dim = n_heads * head_dim`
/// * `w_gate`    — gate weight `[n_heads, head_dim, head_dim]` or `[hidden_dim, hidden_dim]`
///
/// This simplified variant uses a learned scalar gate per-head derived from
/// the mean of the input across head_dim (no extra matmul required).
/// For full production use, replace with `W_gate @ x_per_head`.
///
/// # Gate computation
/// `gate_h = sigmoid(mean(x_h))` where `x_h = x[..head_dim*h .. head_dim*(h+1)]`
pub fn gated_attention_output(
    attn_out: &Tensor,
    x: &Tensor,
    n_heads: usize,
) -> Result<Tensor> {
    let (batch, seq, _n_h, head_dim) = attn_out.dims4()?;
    let hidden_dim = n_heads * head_dim;

    // x shape: [batch, seq, hidden_dim] — flatten heads out.
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;
    let out_f32 = attn_out.to_dtype(DType::F32)?;
    let mut out_data: Vec<f32> = out_f32.flatten_all()?.to_vec1()?;

    // Compute per-head gates and apply.
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..n_heads {
                // Mean of this head's slice of x → scalar gate
                let x_base = b * seq * hidden_dim + t * hidden_dim + h * head_dim;
                let x_head = &x_data[x_base..x_base + head_dim];
                let mean: f32 = x_head.iter().sum::<f32>() / head_dim as f32;
                // sigmoid(mean)
                let gate = 1.0 / (1.0 + (-mean).exp());

                // Apply gate to attention output for this head
                let out_base = b * seq * n_heads * head_dim + t * n_heads * head_dim + h * head_dim;
                for i in 0..head_dim {
                    out_data[out_base + i] *= gate;
                }
            }
        }
    }

    let device = attn_out.device();
    Tensor::from_vec(out_data, (batch, seq, n_heads, head_dim), device)?
        .to_dtype(attn_out.dtype())
}

/// Full gated attention forward: runs GQA then applies per-head sigmoid gating.
///
/// Drop-in for `attention()` when attention-sink suppression is desired.
///
/// # Arguments
/// * `q`, `k`, `v` — attention inputs `[batch, seq, heads, head_dim]`
/// * `x`           — pre-attention normed residual `[batch, seq, hidden_dim]`
/// * `n_heads`, `n_kv_heads` — head counts for GQA
pub fn gated_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    x: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let attn_out = grouped_query_attention(q, k, v, n_heads, n_kv_heads)?;
    gated_attention_output(&attn_out, x, n_heads)
}

// ============================================================================
// Tests — Optimal Compounding Stack (Layers 1-3)
// ============================================================================

#[cfg(test)]
mod ocs_attn_tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    fn rand_tensor(shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        // Deterministic pseudo-random via LCG for reproducibility.
        let mut state: u64 = 0xDEAD_CAFE;
        let data: Vec<f32> = (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map [0, u64::MAX] → [-1, 1]
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        }).collect();
        Tensor::from_vec(data, shape, &Device::Cpu)
    }

    // ── FP4 Tests ────────────────────────────────────────────────────────────

    #[test]
    fn fp4_quantize_zero_is_zero() {
        assert_eq!(quantize_fp4(0.0), 0.0);
    }

    #[test]
    fn fp4_quantize_exact_representable() {
        // Each FP4 value should round-trip exactly.
        for &v in &FP4_POSITIVE_VALUES {
            assert_eq!(quantize_fp4(v), v, "FP4 value {v} should be exact");
            assert_eq!(quantize_fp4(-v), -v, "FP4 value -{v} should be exact");
        }
    }

    #[test]
    fn fp4_quantize_rounds_to_nearest() {
        // 0.75 is equidistant (0.25) from both 0.5 and 1.0.
        // Linear scan returns the lower value on ties → 0.5.
        assert_eq!(quantize_fp4(0.75), 0.5);
        // 0.2 is 0.2 from 0.0 and 0.3 from 0.5 → nearest is 0.0.
        assert_eq!(quantize_fp4(0.2), 0.0);
        // 1.25 is 0.25 from 1.0 and 0.25 from 1.5 → lower wins: 1.0.
        assert_eq!(quantize_fp4(1.25), 1.0);
    }

    #[test]
    fn fp4_blocked_quantize_shape() {
        let data: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
        let (quant, scales) = fp4_quantize_blocked(&data);
        assert_eq!(quant.len(), 48);
        assert_eq!(scales.len(), 3); // 48 / 16 = 3 blocks
    }

    #[test]
    fn fp4_blocked_scale_non_negative() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let (_quant, scales) = fp4_quantize_blocked(&data);
        for s in scales {
            assert!(s > 0.0, "scale must be positive");
        }
    }

    #[test]
    fn fp4_attention_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let k = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let v = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let out = fp4_attention(&q, &k, &v, 2, 2)?;
        assert_eq!(out.dims(), &[1, 4, 2, 8]);
        Ok(())
    }

    #[test]
    fn fp4_attention_finite_outputs() -> Result<()> {
        let dev = &Device::Cpu;
        let q = rand_tensor(&[1, 4, 2, 8])?;
        let k = rand_tensor(&[1, 4, 2, 8])?;
        let v = rand_tensor(&[1, 4, 2, 8])?;
        let out = fp4_attention(&q, &k, &v, 2, 2)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for v in vals { assert!(v.is_finite(), "FP4 attention produced non-finite output"); }
        Ok(())
    }

    // ── Linear Attention (KIMI) Tests ─────────────────────────────────────────

    #[test]
    fn phi_elu_positive_always() {
        // φ must always be ≥ 0
        for x in [-5.0f32, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0] {
            let v = phi_elu(x);
            assert!(v >= 0.0, "phi_elu({x}) = {v} < 0");
        }
    }

    #[test]
    fn phi_elu_positive_input_is_xp1() {
        // For x > 0: φ(x) = x + 1
        assert!((phi_elu(2.0) - 3.0).abs() < 1e-6);
        assert!((phi_elu(0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn linear_attention_head_shape() {
        let seq = 4;
        let dim = 8;
        let q = vec![0.1f32; seq * dim];
        let k = vec![0.1f32; seq * dim];
        let v = vec![0.2f32; seq * dim];
        let out = linear_attention_head(&q, &k, &v, seq, dim);
        assert_eq!(out.len(), seq * dim);
    }

    #[test]
    fn linear_attention_head_all_finite() {
        let seq = 6;
        let dim = 16;
        // Use deterministic values
        let q: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let k: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let v: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.005).collect();
        let out = linear_attention_head(&q, &k, &v, seq, dim);
        for (i, x) in out.iter().enumerate() {
            assert!(x.is_finite(), "linear_attention_head output[{i}] = {x} is non-finite");
        }
    }

    #[test]
    fn linear_attention_kimi_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let k = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let v = Tensor::zeros((1, 4, 2, 8), DType::F32, dev)?;
        let out = linear_attention_kimi(&q, &k, &v, 2, 2)?;
        assert_eq!(out.dims(), &[1, 4, 2, 8]);
        Ok(())
    }

    #[test]
    fn linear_attention_kimi_finite_outputs() -> Result<()> {
        let dev = &Device::Cpu;
        let q = rand_tensor(&[1, 6, 2, 8])?;
        let k = rand_tensor(&[1, 6, 2, 8])?;
        let v = rand_tensor(&[1, 6, 2, 8])?;
        let out = linear_attention_kimi(&q, &k, &v, 2, 2)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for v in vals { assert!(v.is_finite(), "Linear attention produced non-finite output"); }
        Ok(())
    }

    #[test]
    fn linear_attention_kimi_gqa() -> Result<()> {
        // GQA: 4 Q heads, 2 KV heads
        let dev = &Device::Cpu;
        let q = rand_tensor(&[1, 4, 4, 8])?;
        let k = rand_tensor(&[1, 4, 2, 8])?;
        let v = rand_tensor(&[1, 4, 2, 8])?;
        let out = linear_attention_kimi(&q, &k, &v, 4, 2)?;
        assert_eq!(out.dims(), &[1, 4, 4, 8]);
        Ok(())
    }

    // ── Gated Attention Tests ────────────────────────────────────────────────

    #[test]
    fn gated_attention_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let attn_out = Tensor::ones((1, 4, 2, 8), DType::F32, dev)?;
        let x = Tensor::ones((1, 4, 16), DType::F32, dev)?;
        let gated = gated_attention_output(&attn_out, &x, 2)?;
        assert_eq!(gated.dims(), &[1, 4, 2, 8]);
        Ok(())
    }

    #[test]
    fn gated_attention_gate_bounds() -> Result<()> {
        // sigmoid output ∈ (0, 1), so gated output ≤ input
        let dev = &Device::Cpu;
        let attn_out = Tensor::ones((1, 2, 2, 4), DType::F32, dev)?;
        let x = Tensor::ones((1, 2, 8), DType::F32, dev)?;
        let gated = gated_attention_output(&attn_out, &x, 2)?;
        let vals: Vec<f32> = gated.flatten_all()?.to_vec1()?;
        for v in vals {
            // sigmoid(1.0) ≈ 0.731, so output ≈ 0.731 * 1.0
            assert!(v > 0.0 && v <= 1.0, "gate output out of range: {v}");
        }
        Ok(())
    }

    #[test]
    fn gated_attention_negative_x_suppresses() -> Result<()> {
        // Very negative x → gate ≈ 0 → output near 0
        let dev = &Device::Cpu;
        let attn_out = Tensor::ones((1, 1, 1, 4), DType::F32, dev)?;
        let x = Tensor::from_vec(vec![-100.0f32; 4], (1, 1, 4), dev)?;
        let gated = gated_attention_output(&attn_out, &x, 1)?;
        let vals: Vec<f32> = gated.flatten_all()?.to_vec1()?;
        for v in vals { assert!(v.abs() < 1e-3, "gate should suppress: {v}"); }
        Ok(())
    }

    #[test]
    fn gated_attention_full_forward_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let q = rand_tensor(&[1, 4, 2, 8])?;
        let k = rand_tensor(&[1, 4, 2, 8])?;
        let v = rand_tensor(&[1, 4, 2, 8])?;
        let x = rand_tensor(&[1, 4, 16])?;
        let out = gated_attention(&q, &k, &v, &x, 2, 2)?;
        assert_eq!(out.dims(), &[1, 4, 2, 8]);
        Ok(())
    }
}

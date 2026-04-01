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

//! Transformer mathematical operations for LLaMA-family models.
//!
//! All operations use Candle's tensor API which dispatches to cuBLAS/cuDNN
//! on CUDA devices automatically. No hand-written CUDA kernels needed —
//! Candle handles the GPU dispatch.

use candle_core::{DType, Result, Tensor, D};

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
// Rotary Position Embeddings (RoPE)
// ---------------------------------------------------------------------------
// Encodes absolute position information into Q and K vectors by rotating
// pairs of dimensions. This is what lets the model know token order.
//
// For each pair (x_i, x_{i+1}), rotate by angle θ_i * position:
//   x_i'     = x_i * cos(θ) - x_{i+1} * sin(θ)
//   x_{i+1}' = x_i * sin(θ) + x_{i+1} * cos(θ)
//
pub fn rope(
    q: &Tensor,
    k: &Tensor,
    start_pos: usize,
    head_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let dtype = q.dtype();
    let seq_len = q.dim(1)?; // [batch, seq_len, n_heads, head_dim]

    // Compute frequency bands: θ_i = rope_theta^(-2i/d) for i in 0..d/2
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq, device)?; // [half_dim]

    // Position indices
    let positions: Vec<f32> = (start_pos..start_pos + seq_len)
        .map(|p| p as f32)
        .collect();
    let positions = Tensor::new(positions, device)?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: angles[pos][dim] = pos * inv_freq[dim]
    let angles = positions.matmul(&inv_freq.unsqueeze(0)?)?; // [seq_len, half_dim]

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
// Standard multi-head attention but K/V can have fewer heads than Q.
// Each K/V head is shared across (n_heads / n_kv_heads) Q heads.
//
//   Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k + mask) · V
//
pub fn grouped_query_attention(
    q: &Tensor,     // [batch, seq_len, n_heads, head_dim]
    k: &Tensor,     // [batch, total_seq, n_kv_heads, head_dim]
    v: &Tensor,     // [batch, total_seq, n_kv_heads, head_dim]
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Transpose to [batch, heads, seq, dim] for matmul
    let q = q.transpose(1, 2)?; // [batch, n_heads, seq_q, head_dim]
    let k = k.transpose(1, 2)?; // [batch, n_kv_heads, seq_kv, head_dim]
    let v = v.transpose(1, 2)?; // [batch, n_kv_heads, seq_kv, head_dim]

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
    let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

    // Causal mask: prevent attending to future positions
    let seq_q = attn_weights.dim(D::Minus2)?;
    let seq_kv = attn_weights.dim(D::Minus1)?;
    let attn_weights = apply_causal_mask(&attn_weights, seq_q, seq_kv)?;

    // Softmax over key dimension
    let attn_weights = softmax(&attn_weights, D::Minus1)?;

    // Weighted sum of values: [batch, n_heads, seq_q, head_dim]
    let output = attn_weights.matmul(&v)?;

    // Transpose back: [batch, seq_q, n_heads, head_dim]
    output.transpose(1, 2)
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

/// SiLU (Sigmoid Linear Unit): x * σ(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus_exp = (Tensor::ones_like(&exp_neg_x)? + exp_neg_x)?;
    let sigmoid = Tensor::ones_like(&one_plus_exp)?.broadcast_div(&one_plus_exp)?;
    x.mul(&sigmoid)
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
// Causal Mask
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

//! P5 — Multi-head Latent Attention (MLA)
//!
//! MLA is DeepSeek's memory-efficient attention variant used in DeepSeek V2/V3/R1 full.
//! It uses low-rank projections to compress the KV cache while maintaining quality.
//!
//! # Architecture
//!
//! Standard MHA KV cache: `O(seq × n_kv_heads × head_dim)` per layer
//! MLA KV cache: `O(seq × kv_lora_rank)` — dramatically smaller
//!
//! Forward pass:
//! ```text
//! q = W_q × x           OR   q = W_qA × W_qB × x  (low-rank Q)
//! c_kv = W_c_kv × x          // compress to [seq, kv_lora_rank]
//! k = W_uk × c_kv + rope(W_kr × x)   // decompress K + add RoPE on rope-part
//! v = W_uv × c_kv               // decompress V
//! ```
//!
//! # Key Config (DeepSeek V2/V3)
//! - `q_lora_rank`: 1536 (V2/V3) or None (use full W_q)
//! - `kv_lora_rank`: 512 (V2/V3)
//! - `qk_nope_head_dim`: 128 (no-RoPE component)
//! - `qk_rope_head_dim`: 64 (RoPE component)
//! - `v_head_dim`: 128
//! - Total head_dim = nope + rope = 192

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// MLA configuration for one attention layer.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Number of Q heads (full Q heads, not KV)
    pub n_heads: usize,
    /// Rank for Q low-rank projection (None = use full W_q directly)
    pub q_lora_rank: Option<usize>,
    /// KV compression rank — the actual KV cache size per token
    pub kv_lora_rank: usize,
    /// Head dim for the non-RoPE Q/K component
    pub qk_nope_head_dim: usize,
    /// Head dim for the RoPE Q/K component
    pub qk_rope_head_dim: usize,
    /// Value head dimension
    pub v_head_dim: usize,
    /// Attention scale (1/sqrt(qk_total_head_dim))
    pub softmax_scale: f32,
}

impl MlaConfig {
    /// DeepSeek-V2 / DeepSeek-V3 configuration.
    pub fn deepseek_v2(n_heads: usize) -> Self {
        let qk_nope = 128;
        let qk_rope = 64;
        let total_head_dim = qk_nope + qk_rope;
        Self {
            n_heads,
            q_lora_rank: Some(1536),
            kv_lora_rank: 512,
            qk_nope_head_dim: qk_nope,
            qk_rope_head_dim: qk_rope,
            v_head_dim: 128,
            softmax_scale: 1.0 / (total_head_dim as f32).sqrt(),
        }
    }

    /// DeepSeek-R1 uses the same MLA as V2/V3.
    pub fn deepseek_r1(n_heads: usize) -> Self {
        Self::deepseek_v2(n_heads)
    }
}

// ---------------------------------------------------------------------------
// MLA Weights
// ---------------------------------------------------------------------------

/// Weights for one MLA attention layer.
pub struct MlaWeights {
    // --- Q projection ---
    /// Low-rank Q down projection: [hidden, q_lora_rank] (optional)
    pub w_qa: Option<QMatMul>,
    /// Low-rank Q up projection: [q_lora_rank, n_heads*(qk_nope+qk_rope)] (optional)
    pub w_qb: Option<QMatMul>,
    /// Full Q projection (used when q_lora_rank is None): [hidden, n_heads*(qk_nope+qk_rope)]
    pub w_q: Option<QMatMul>,

    // --- KV compression ---
    /// KV down projection: [hidden, kv_lora_rank + qk_rope]
    /// The +qk_rope part is the "decoupled K rope" component
    pub w_kv_a: QMatMul,
    /// Norm before KV up projection
    pub kv_norm_weight: Tensor,
    /// KV up projection: [kv_lora_rank, n_heads*(qk_nope + v_head_dim)]
    pub w_kv_b: QMatMul,

    // --- Output ---
    /// Output projection: [n_heads * v_head_dim, hidden]
    pub w_o: QMatMul,
}

// ---------------------------------------------------------------------------
// MLA KV Cache
// ---------------------------------------------------------------------------

/// The compressed KV cache for MLA — stores `c_kv` and `k_rope` per token.
///
/// This is dramatically smaller than standard KV cache:
/// - Standard: `n_layers × seq × n_kv_heads × head_dim`
/// - MLA:      `n_layers × seq × (kv_lora_rank + qk_rope_head_dim)`
pub struct MlaKvCache {
    /// Compressed KV latent: [seq_cached, kv_lora_rank]
    pub c_kv: Option<Tensor>,
    /// Decoupled RoPE K component: [seq_cached, qk_rope_head_dim]
    pub k_rope: Option<Tensor>,
}

impl MlaKvCache {
    pub fn new() -> Self {
        Self { c_kv: None, k_rope: None }
    }

    /// Append new compressed KV to cache. Returns (c_kv_all, k_rope_all).
    pub fn append(&mut self, new_c_kv: Tensor, new_k_rope: Tensor) -> Result<(Tensor, Tensor)> {
        let c_kv_all = match &self.c_kv {
            None => new_c_kv.clone(),
            Some(old) => Tensor::cat(&[old, &new_c_kv], 0)?,
        };
        let k_rope_all = match &self.k_rope {
            None => new_k_rope.clone(),
            Some(old) => Tensor::cat(&[old, &new_k_rope], 0)?,
        };
        self.c_kv = Some(c_kv_all.clone());
        self.k_rope = Some(k_rope_all.clone());
        Ok((c_kv_all, k_rope_all))
    }

    /// How many tokens are cached.
    pub fn seq_len(&self) -> usize {
        self.c_kv.as_ref().map(|t| t.dim(0).unwrap_or(0)).unwrap_or(0)
    }
}

impl Default for MlaKvCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RMSNorm (inline — avoid circular dep with ops.rs)
// ---------------------------------------------------------------------------
fn rms_norm_mla(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(weight)
}

// ---------------------------------------------------------------------------
// MLA Forward Pass
// ---------------------------------------------------------------------------

/// Run one MLA attention forward pass.
///
/// # Arguments
/// * `x`      — input hidden states `[batch, seq_q, hidden_dim]`
/// * `weights`— MLA weight tensors
/// * `cache`  — mutable KV cache (updated in-place)
/// * `cfg`    — MLA configuration
/// * `start_pos` — current position in sequence (for RoPE)
///
/// # Returns
/// Output tensor `[batch, seq_q, hidden_dim]`.
pub fn mla_forward(
    x: &Tensor,
    weights: &MlaWeights,
    cache: &mut MlaKvCache,
    cfg: &MlaConfig,
    start_pos: usize,
) -> Result<Tensor> {
    let (batch, seq_q, _hidden) = x.dims3()?;
    let device = x.device();
    let dtype = x.dtype();

    // ── Step 1: Compute Q ────────────────────────────────────────────────────
    let q_full = match (&weights.w_qa, &weights.w_qb, &weights.w_q) {
        (Some(w_qa), Some(w_qb), _) => {
            // Low-rank Q: x -> w_qa -> w_qb
            let q_a = w_qa.forward(x)?;
            w_qb.forward(&q_a)?
        }
        (_, _, Some(w_q)) => w_q.forward(x)?,
        _ => candle_core::bail!("MLA: must provide either (w_qa, w_qb) or w_q"),
    };
    // q_full: [batch, seq_q, n_heads * (qk_nope + qk_rope)]
    let qk_total = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    let q_reshaped = q_full.reshape((batch, seq_q, cfg.n_heads, qk_total))?;
    let q_nope = q_reshaped.narrow(D::Minus1, 0, cfg.qk_nope_head_dim)?;
    let q_rope = q_reshaped.narrow(D::Minus1, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim)?;

    // ── Step 2: KV compression ───────────────────────────────────────────────
    // w_kv_a projects to [kv_lora_rank + qk_rope]
    let kv_a_out = weights.w_kv_a.forward(x)?; // [b, seq_q, kv_lora_rank + qk_rope]
    let c_kv_new = kv_a_out.narrow(D::Minus1, 0, cfg.kv_lora_rank)?;
    let k_rope_new = kv_a_out.narrow(D::Minus1, cfg.kv_lora_rank, cfg.qk_rope_head_dim)?;

    // Flatten seq dim for cache append: [seq_q, kv_lora_rank]
    let c_kv_flat = c_kv_new.reshape((batch * seq_q, cfg.kv_lora_rank))?;
    let k_rope_flat = k_rope_new.reshape((batch * seq_q, cfg.qk_rope_head_dim))?;

    let (c_kv_all, k_rope_all) = cache.append(c_kv_flat, k_rope_flat)?;
    let seq_kv = cache.seq_len();

    // ── Step 3: KV decompression ─────────────────────────────────────────────
    // Norm the compressed KV before decompression
    let c_kv_normed = rms_norm_mla(&c_kv_all, &weights.kv_norm_weight, 1e-6)?;

    // Decompress: [seq_kv, n_heads * (qk_nope + v_head_dim)]
    let kv_b_out = weights.w_kv_b.forward(&c_kv_normed)?;
    let kv_total = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let kv_reshaped = kv_b_out.reshape((1, seq_kv, cfg.n_heads, kv_total))?;
    let k_nope = kv_reshaped.narrow(D::Minus1, 0, cfg.qk_nope_head_dim)?;
    let v = kv_reshaped.narrow(D::Minus1, cfg.qk_nope_head_dim, cfg.v_head_dim)?;

    // ── Step 4: Apply RoPE to rope components ────────────────────────────────
    // (simplified: add position-dependent phase — full RoPE from ops.rs in caller)
    // Here we just return the RoPE-ready tensors; caller can apply rope_partial()
    // to q_rope and k_rope_all before the dot product.

    // For the nope component: Q·K = q_nope·k_nope → this part needs no RoPE
    // For the rope component: need separate dot product after RoPE application
    // Simplified fusion: concatenate nope+rope back for standard GQA attention

    let k_rope_all_reshaped = k_rope_all
        .reshape((1, seq_kv, 1, cfg.qk_rope_head_dim))?
        .expand((1, seq_kv, cfg.n_heads, cfg.qk_rope_head_dim))?;
    let k_full = Tensor::cat(&[&k_nope, &k_rope_all_reshaped], D::Minus1)?;

    let q_full_cat = Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?;

    // ── Step 5: Scaled dot-product attention ────────────────────────────────
    // Standard GQA with the reconstructed Q, K, V (1 KV head group = n_heads)
    // q: [b, seq_q, n_heads, qk_total], k: [1, seq_kv, n_heads, qk_total]
    let scale = cfg.softmax_scale as f64;
    let q_t = q_full_cat.transpose(1, 2)?.contiguous()?; // [b, n_heads, seq_q, qk_total]
    let k_t = k_full.transpose(1, 2)?.contiguous()?;
    let v_t = v.transpose(1, 2)?.contiguous()?;           // [1, n_heads, seq_kv, v_head_dim]

    let attn_weights = (q_t.matmul(&k_t.transpose(D::Minus2, D::Minus1)?)? * scale)?;

    // Causal mask
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![0.0f32; seq_q * seq_kv];
    for i in 0..seq_q {
        let q_abs = start_pos + i;
        for j in 0..seq_kv {
            if j > q_abs {
                mask_data[i * seq_kv + j] = neg_inf;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (1, 1, seq_q, seq_kv), device)?
        .to_dtype(dtype)?;
    let attn_weights = attn_weights.broadcast_add(&mask)?;

    // Softmax over last dim: softmax(logits) = exp(x - max) / sum(exp)
    let attn_max = attn_weights.max_keepdim(D::Minus1)?;
    let attn_exp = (attn_weights.broadcast_sub(&attn_max)?.exp())?;
    let attn_sum = attn_exp.sum_keepdim(D::Minus1)?;
    let attn_probs = attn_exp.broadcast_div(&attn_sum)?;
    let context = attn_probs.matmul(&v_t)?; // [b, n_heads, seq_q, v_head_dim]

    // ── Step 6: Output projection ────────────────────────────────────────────
    let context = context.transpose(1, 2)?  // [b, seq_q, n_heads, v_head_dim]
        .contiguous()?
        .reshape((batch, seq_q, cfg.n_heads * cfg.v_head_dim))?;
    weights.w_o.forward(&context)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_config_deepseek_v2() {
        let cfg = MlaConfig::deepseek_v2(128);
        assert_eq!(cfg.n_heads, 128);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.qk_nope_head_dim, 128);
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.v_head_dim, 128);
        assert!(cfg.softmax_scale > 0.0);
    }

    #[test]
    fn test_mla_kv_cache_append() {
        let device = candle_core::Device::Cpu;
        let mut cache = MlaKvCache::new();
        assert_eq!(cache.seq_len(), 0);

        let c_kv1 = Tensor::zeros((4, 512), DType::F32, &device).unwrap();
        let k_rope1 = Tensor::zeros((4, 64), DType::F32, &device).unwrap();
        cache.append(c_kv1, k_rope1).unwrap();
        assert_eq!(cache.seq_len(), 4);

        let c_kv2 = Tensor::zeros((2, 512), DType::F32, &device).unwrap();
        let k_rope2 = Tensor::zeros((2, 64), DType::F32, &device).unwrap();
        cache.append(c_kv2, k_rope2).unwrap();
        assert_eq!(cache.seq_len(), 6);
    }
}

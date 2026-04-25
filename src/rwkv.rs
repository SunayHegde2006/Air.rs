//! P7 — RWKV WKV Mechanism (RWKV-4 through RWKV-6)
//!
//! RWKV (Receptance Weighted Key Value) is an RNN-style model with linear
//! time-and-memory inference, unlike transformer attention which is quadratic.
//!
//! # WKV Recurrence (RWKV-4/5)
//!
//! For each layer and each channel dimension, the WKV recurrence is:
//! ```text
//! wkv_t = (e^(u+k_t) * v_t + a_{t-1}) / (e^(u+k_t) + b_{t-1})
//! a_t   = e^(-w) * a_{t-1} + e^k_t * v_t
//! b_t   = e^(-w) * b_{t-1} + e^k_t
//! ```
//! Output: `o_t = sigmoid(r_t) * wkv_t`
//!
//! # RWKV-6 (Eagle/Finch architecture)
//! Adds LoRA-style dynamic decay (w), dynamic u bonus, multi-head WKV.
//!
//! # Architecture
//! ```text
//! x → Time-mix → WKV → o
//!   ↘ Channel-mix (FFN analog)
//! ```
//!
//! References:
//! - RWKV-4: https://arxiv.org/abs/2305.13048
//! - RWKV-6: https://arxiv.org/abs/2404.05892

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// RWKV model variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RwkvVersion {
    V4,
    V5,
    V6,
}

/// RWKV configuration for one layer.
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    /// Hidden dimension
    pub d_model: usize,
    /// Number of attention heads (RWKV-5/6 only; V4 uses 1 per channel)
    pub n_heads: usize,
    /// Head dimension (d_model / n_heads)
    pub head_dim: usize,
    /// Model version
    pub version: RwkvVersion,
}

impl RwkvConfig {
    /// RWKV-4 config (single-head per channel)
    pub fn rwkv4(d_model: usize) -> Self {
        Self { d_model, n_heads: d_model, head_dim: 1, version: RwkvVersion::V4 }
    }

    /// RWKV-5 / RWKV-6 (Eagle 7B) config
    pub fn rwkv6_7b() -> Self {
        let d_model = 4096;
        let n_heads = 64;
        Self { d_model, n_heads, head_dim: d_model / n_heads, version: RwkvVersion::V6 }
    }
}

// ---------------------------------------------------------------------------
// Time-mix Weights (Attention analog)
// ---------------------------------------------------------------------------

/// Weights for the RWKV time-mix (attention) sublayer.
pub struct RwkvTimeMixWeights {
    // RWKV-4/5: static per-layer, per-channel parameters
    /// Time decay: [d_model] (log negative, i.e., w = -exp(w_log))
    pub time_decay: Tensor,
    /// First-token bonus u: [d_model]
    pub time_first: Tensor,
    /// Time mix lerp for x and x_prev: [d_model]
    pub time_mix_k: Tensor,
    pub time_mix_v: Tensor,
    pub time_mix_r: Tensor,

    // Projections
    pub receptance: QMatMul, // r projection: [d_model, d_model]
    pub key: QMatMul,        // k projection: [d_model, d_model]
    pub value: QMatMul,      // v projection: [d_model, d_model]
    pub output: QMatMul,     // o projection: [d_model, d_model]

    // RWKV-6 additions (None for older versions)
    /// Dynamic decay LoRA A: [d_model, lora_rank]
    pub time_decay_lora_a: Option<QMatMul>,
    /// Dynamic decay LoRA B: [lora_rank, d_model]
    pub time_decay_lora_b: Option<QMatMul>,
}

// ---------------------------------------------------------------------------
// Channel-mix Weights (FFN analog)
// ---------------------------------------------------------------------------

/// Weights for the RWKV channel-mix (FFN) sublayer.
pub struct RwkvChannelMixWeights {
    /// Time mix lerp: [d_model]
    pub time_mix_k: Tensor,
    pub time_mix_r: Tensor,
    /// Key projection: [d_model, d_ffn] (d_ffn ≈ d_model * 4/3.5)
    pub key: QMatMul,
    /// Receptance gate: [d_model, d_model]
    pub receptance: QMatMul,
    /// Value projection: [d_ffn, d_model]
    pub value: QMatMul,
}

// ---------------------------------------------------------------------------
// RWKV Recurrent State
// ---------------------------------------------------------------------------

/// Recurrent state for one RWKV layer.
///
/// Stores the complete state between tokens — O(d_model) not O(seq_len).
#[derive(Clone)]
pub struct RwkvLayerState {
    /// Previous token hidden state for time-mix: [batch, d_model]
    pub x_tm: Option<Tensor>,
    /// WKV numerator state `a`: [batch, d_model]
    pub a: Option<Tensor>,
    /// WKV denominator state `b`: [batch, d_model]
    pub b: Option<Tensor>,
    /// Previous token hidden state for channel-mix: [batch, d_model]
    pub x_cm: Option<Tensor>,
    // RWKV-5/6: multi-head WKV state [batch, n_heads, head_dim, head_dim]
    pub h_wkv: Option<Tensor>,
}

impl RwkvLayerState {
    pub fn new() -> Self {
        Self { x_tm: None, a: None, b: None, x_cm: None, h_wkv: None }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for RwkvLayerState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// RWKV-4 WKV Step
// ---------------------------------------------------------------------------

/// Compute one WKV recurrence step (RWKV-4 style, per-channel).
///
/// # Arguments
/// * `k` — key: [batch, d_model]
/// * `v` — value: [batch, d_model]
/// * `w` — time decay: [d_model] (log negative)
/// * `u` — time first bonus: [d_model]
/// * `state_a` — previous numerator state: [batch, d_model]
/// * `state_b` — previous denominator state: [batch, d_model]
///
/// # Returns
/// `(wkv_out, new_a, new_b)`
pub fn wkv_step(
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    u: &Tensor,
    state_a: &Tensor,
    state_b: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dtype = k.dtype();

    // w is stored as log of negative decay: actual decay = -exp(w)
    // exp(-exp(w)) = decay factor per step
    let w_neg_exp = w.to_dtype(dtype)?.exp()?.neg()?.exp()?; // e^(-exp(w))
    let u = u.to_dtype(dtype)?;

    // Compute e^(u+k) for the first-token bonus
    // This is element-wise, k and u are both [batch, d_model] or [d_model]
    let ek = k.to_dtype(dtype)?.exp()?; // e^k
    let euk = (k.to_dtype(dtype)? + u.unsqueeze(0)?)?.exp()?; // e^(u+k)

    // WKV = (e^(u+k) * v + a) / (e^(u+k) + b)
    let num = euk.broadcast_mul(v)?.broadcast_add(state_a)?;
    let den = (euk.broadcast_add(state_b)? + 1e-6_f64)?; // + eps for stability
    let wkv = (num / den)?;

    // Update state
    let new_a = (w_neg_exp.broadcast_mul(state_a)? + ek.broadcast_mul(v)?)?;
    let new_b = (w_neg_exp.broadcast_mul(state_b)? + ek)?;

    Ok((wkv, new_a, new_b))
}

// ---------------------------------------------------------------------------
// RWKV Time-mix Forward (one token step)
// ---------------------------------------------------------------------------

/// Run one RWKV time-mix step (RWKV-4/5).
///
/// # Arguments
/// * `x`      — input `[batch, d_model]` (single token)
/// * `w`      — time-mix weights
/// * `state`  — mutable layer state (updated in-place)
/// * `cfg`    — model config
///
/// # Returns
/// time-mix output `[batch, d_model]`
pub fn rwkv_time_mix_step(
    x: &Tensor,
    tw: &RwkvTimeMixWeights,
    state: &mut RwkvLayerState,
    cfg: &RwkvConfig,
) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();
    let batch = x.dim(0)?;

    let x_prev = state.x_tm.clone().unwrap_or_else(|| {
        Tensor::zeros_like(x).unwrap()
    });

    // Linear interpolation (time-mix lerp)
    // mix_k: xk = x * mix_k + x_prev * (1 - mix_k)
    let mix_k = tw.time_mix_k.to_dtype(dtype)?.unsqueeze(0)?;
    let mix_v = tw.time_mix_v.to_dtype(dtype)?.unsqueeze(0)?;
    let mix_r = tw.time_mix_r.to_dtype(dtype)?.unsqueeze(0)?;

    let xk = (x.broadcast_mul(&mix_k)? + x_prev.broadcast_mul(&(1.0_f64 - &mix_k)?)?)?;
    let xv = (x.broadcast_mul(&mix_v)? + x_prev.broadcast_mul(&(1.0_f64 - &mix_v)?)?)?;
    let xr = (x.broadcast_mul(&mix_r)? + x_prev.broadcast_mul(&(1.0_f64 - &mix_r)?)?)?;

    // Update previous state
    state.x_tm = Some(x.clone());

    // Projections
    let k = tw.key.forward(&xk.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;
    let v = tw.value.forward(&xv.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;
    let r = tw.receptance.forward(&xr.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;

    // Receptance gate = sigmoid(r)
    let r_gate = (r.neg()?.exp()? + 1.0_f64)?.recip()?;

    // Init states if needed
    let a = state.a.clone().unwrap_or_else(|| {
        Tensor::zeros((batch, cfg.d_model), dtype, device).unwrap()
    });
    let b = state.b.clone().unwrap_or_else(|| {
        Tensor::zeros((batch, cfg.d_model), dtype, device).unwrap()
    });

    // WKV step
    let (wkv, new_a, new_b) = wkv_step(
        &k, &v,
        &tw.time_decay,
        &tw.time_first,
        &a, &b,
    )?;

    state.a = Some(new_a);
    state.b = Some(new_b);

    // Output: sigmoid(r) * wkv → output projection
    let o = tw.output.forward(&(r_gate * wkv)?.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;
    Ok(o)
}

// ---------------------------------------------------------------------------
// RWKV Channel-mix Forward (FFN step)
// ---------------------------------------------------------------------------

/// Run one RWKV channel-mix step (FFN analog).
///
/// # Arguments
/// * `x`     — input `[batch, d_model]`
/// * `cw`    — channel-mix weights
/// * `state` — mutable state (x_cm updated)
/// * `cfg`   — model config
pub fn rwkv_channel_mix_step(
    x: &Tensor,
    cw: &RwkvChannelMixWeights,
    state: &mut RwkvLayerState,
    cfg: &RwkvConfig,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let batch = x.dim(0)?;

    let x_prev = state.x_cm.clone().unwrap_or_else(|| Tensor::zeros_like(x).unwrap());

    let mix_k = cw.time_mix_k.to_dtype(dtype)?.unsqueeze(0)?;
    let mix_r = cw.time_mix_r.to_dtype(dtype)?.unsqueeze(0)?;

    let xk = (x.broadcast_mul(&mix_k)? + x_prev.broadcast_mul(&(1.0_f64 - &mix_k)?)?)?;
    let xr = (x.broadcast_mul(&mix_r)? + x_prev.broadcast_mul(&(1.0_f64 - &mix_r)?)?)?;

    state.x_cm = Some(x.clone());

    let k = cw.key.forward(&xk.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;
    let r = cw.receptance.forward(&xr.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;

    // squared ReLU activation on k (RK² in RWKV)
    let k_act = k.relu()?.sqr()?;

    let v = cw.value.forward(&k_act.unsqueeze(1)?)?.reshape((batch, cfg.d_model))?;

    // Gating: sigmoid(r) * v
    let r_gate = (r.neg()?.exp()? + 1.0_f64)?.recip()?;
    (r_gate * v)
}

// ---------------------------------------------------------------------------
// Full RWKV Block (time-mix + residual + channel-mix + residual)
// ---------------------------------------------------------------------------

/// Combined weight container for one RWKV layer.
pub struct RwkvLayerWeights {
    /// LayerNorm for time-mix input
    pub ln1_weight: Tensor,
    pub ln1_bias: Tensor,
    /// LayerNorm for channel-mix input
    pub ln2_weight: Tensor,
    pub ln2_bias: Tensor,
    pub time_mix: RwkvTimeMixWeights,
    pub channel_mix: RwkvChannelMixWeights,
}

/// Layer norm helper (no candle_nn dep)
fn layer_norm_rwkv(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let var = (x - &mean)?.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = ((x - &mean)? / (var + eps)?.sqrt()?)?;
    normed.broadcast_mul(w)?.broadcast_add(b)
}

/// Run a complete RWKV block (time-mix + channel-mix) for one token.
///
/// # Arguments
/// * `x`     — input `[batch, d_model]` (one token)
/// * `lw`    — layer weights
/// * `state` — mutable layer state
/// * `cfg`   — config
///
/// # Returns
/// output `[batch, d_model]`
pub fn rwkv_block_step(
    x: &Tensor,
    lw: &RwkvLayerWeights,
    state: &mut RwkvLayerState,
    cfg: &RwkvConfig,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let w = lw.ln1_weight.to_dtype(dtype)?;
    let b = lw.ln1_bias.to_dtype(dtype)?;
    let w2 = lw.ln2_weight.to_dtype(dtype)?;
    let b2 = lw.ln2_bias.to_dtype(dtype)?;

    // Time-mix
    let x_ln1 = layer_norm_rwkv(x, &w, &b, 1e-5)?;
    let tm_out = rwkv_time_mix_step(&x_ln1, &lw.time_mix, state, cfg)?;
    let x_after_tm = (x + tm_out)?;

    // Channel-mix
    let x_ln2 = layer_norm_rwkv(&x_after_tm, &w2, &b2, 1e-5)?;
    let cm_out = rwkv_channel_mix_step(&x_ln2, &lw.channel_mix, state, cfg)?;
    (x_after_tm + cm_out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwkv_config() {
        let cfg = RwkvConfig::rwkv6_7b();
        assert_eq!(cfg.d_model, 4096);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.version, RwkvVersion::V6);
    }

    #[test]
    fn test_rwkv_state_reset() {
        let mut state = RwkvLayerState::new();
        assert!(state.a.is_none());
        state.reset();
        assert!(state.a.is_none());
    }

    #[test]
    fn test_wkv_step_basic() {
        use candle_core::Device;
        let dev = &Device::Cpu;
        let d = 4usize;
        let batch = 1usize;

        let k = Tensor::zeros((batch, d), DType::F32, dev).unwrap();
        let v = Tensor::ones((batch, d), DType::F32, dev).unwrap();
        let w = Tensor::full(-1.0f32, (d,), dev).unwrap(); // w = -1 → decay = e^(-e^1)
        let u = Tensor::zeros((d,), DType::F32, dev).unwrap();
        let a = Tensor::zeros((batch, d), DType::F32, dev).unwrap();
        let b = Tensor::zeros((batch, d), DType::F32, dev).unwrap();

        let (wkv, new_a, new_b) = wkv_step(&k, &v, &w, &u, &a, &b).unwrap();
        // With zero state and e^(u+k)=e^0=1, v=1: wkv = 1 / (1 + eps) ≈ 1.0
        let wkv_vals: Vec<f32> = wkv.flatten_all().unwrap().to_vec1().unwrap();
        assert!((wkv_vals[0] - 1.0).abs() < 0.01, "wkv first step should be ~1.0");
        assert!(new_a.dims()[0] == batch);
        assert!(new_b.dims()[0] == batch);
    }
}

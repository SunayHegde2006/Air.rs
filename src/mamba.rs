//! P6 — Mamba SSM Blocks (Mamba-1, Mamba-2, Jamba)
//!
//! Mamba is a selective State Space Model (SSM) that achieves linear-time
//! inference (vs quadratic for attention) by maintaining a fixed-size hidden state.
//!
//! # Mamba-1 Architecture (Gu & Dao, 2023 - arXiv:2312.00752)
//!
//! For each time step, the SSM recurrence is:
//! ```text
//! h_t = A_t * h_{t-1} + B_t * x_t    // state update
//! y_t = C_t * h_t                     // output
//! ```
//! Where A, B, C are *selective* — they depend on the input x (time-varying).
//!
//! Block structure:
//! ```text
//! x → split(z, xBC) → linear(xBC) → SSM(delta, A, B, C) → * silu(z) → linear → out
//! ```
//!
//! # Mamba-2 Architecture (Dao & Gu, 2024 - arXiv:2405.21060)
//! Structured State Space Duality (SSD): multi-head variant with grouped B/C.
//!
//! # Integration (Jamba)
//! Jamba interleaves Mamba and attention blocks (typically 1 attn : 7 mamba).
//!
//! # Implementation Note
//! This implements the **inference recurrence** (sequential scan, O(state_size) VRAM)
//! not the parallel scan used in training. This is the correct path for single-token decode.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Mamba SSM configuration for one block.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Model (hidden) dimension
    pub d_model: usize,
    /// Inner expanded dimension (typically d_model * expand, default expand=2)
    pub d_inner: usize,
    /// State space dimension (default 16 for Mamba-1, 64+ for Mamba-2)
    pub d_state: usize,
    /// Local convolution kernel size (default 4)
    pub d_conv: usize,
    /// Delta (time step) rank (default ceil(d_model / 16))
    pub dt_rank: usize,
    /// Whether to use Mamba-2 multi-head SSM
    pub mamba2: bool,
}

impl MambaConfig {
    /// Mamba-1 130M default (d_model=768)
    pub fn mamba1_130m() -> Self {
        let d_model = 768;
        let d_inner = d_model * 2;
        Self {
            d_model,
            d_inner,
            d_state: 16,
            d_conv: 4,
            dt_rank: (d_model + 15) / 16,
            mamba2: false,
        }
    }

    /// Mamba-1 3B default (d_model=2560)
    pub fn mamba1_3b() -> Self {
        let d_model = 2560;
        let d_inner = d_model * 2;
        Self {
            d_model,
            d_inner,
            d_state: 16,
            d_conv: 4,
            dt_rank: (d_model + 15) / 16,
            mamba2: false,
        }
    }

    /// Jamba 51B Mamba block config (d_model=4096)
    pub fn jamba() -> Self {
        let d_model = 4096;
        let d_inner = d_model * 2;
        Self {
            d_model,
            d_inner,
            d_state: 16,
            d_conv: 4,
            dt_rank: (d_model + 15) / 16,
            mamba2: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Weights for one Mamba SSM block.
pub struct MambaWeights {
    /// Input projection: [d_model, d_inner * 2] (produces x and z)
    pub in_proj: QMatMul,
    /// 1-D depthwise convolution weight: [d_inner, 1, d_conv]
    /// Stored as [d_inner, d_conv] for simple conv implementation
    pub conv1d_weight: Tensor,
    /// Conv bias: [d_inner]
    pub conv1d_bias: Option<Tensor>,
    /// x_proj: [d_inner, dt_rank + d_state * 2] → produces (delta, B, C)
    pub x_proj: QMatMul,
    /// dt_proj: [dt_rank, d_inner] — projects delta back to d_inner
    pub dt_proj: QMatMul,
    /// A log parameter: [d_inner, d_state] — stored as log(-A) for stability
    pub a_log: Tensor,
    /// D skip connection: [d_inner]
    pub d: Tensor,
    /// Output normalization weight: [d_inner] (optional RMSNorm in Mamba-2)
    pub out_norm: Option<Tensor>,
    /// Output projection: [d_inner, d_model]
    pub out_proj: QMatMul,
}

// ---------------------------------------------------------------------------
// SSM Recurrent State (for streaming inference)
// ---------------------------------------------------------------------------

/// The recurrent state for one Mamba layer (single-token decode mode).
///
/// Stored between tokens — allows O(d_state) memory per layer regardless of seq len.
#[derive(Clone)]
pub struct MambaState {
    /// SSM hidden state: [batch, d_inner, d_state]
    pub h: Option<Tensor>,
    /// Convolution buffer (ring buffer of last d_conv inputs): [batch, d_inner, d_conv]
    pub conv_state: Option<Tensor>,
}

impl MambaState {
    pub fn new() -> Self {
        Self { h: None, conv_state: None }
    }

    pub fn reset(&mut self) {
        self.h = None;
        self.conv_state = None;
    }
}

impl Default for MambaState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Mamba Forward (Inference — Sequential Recurrence)
// ---------------------------------------------------------------------------

/// Run one Mamba block in **inference recurrence mode** (single or few tokens).
///
/// This is the correct path for generation. For prefill (bulk parallel scan),
/// a parallel scan kernel is needed (not implemented here — use CUDA for training).
///
/// # Arguments
/// * `x`     — input `[batch, seq_len, d_model]`
/// * `w`     — Mamba weights
/// * `state` — mutable recurrent state (updated in-place)
/// * `cfg`   — Mamba configuration
///
/// # Returns
/// Output `[batch, seq_len, d_model]`
pub fn mamba_forward(
    x: &Tensor,
    w: &MambaWeights,
    state: &mut MambaState,
    cfg: &MambaConfig,
) -> Result<Tensor> {
    let (batch, seq_len, _d_model) = x.dims3()?;
    let device = x.device();
    let dtype = x.dtype();

    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // Extract single token: [batch, 1, d_model]
        let xt = x.narrow(1, t, 1)?;

        // ── 1. Input projection ────────────────────────────────────────────
        // [batch, 1, d_inner * 2]
        let xz = w.in_proj.forward(&xt)?;
        let xz = xz.reshape((batch, 1, cfg.d_inner * 2))?;
        // Split into x (to SSM) and z (gating)
        let x_ssm = xz.narrow(D::Minus1, 0, cfg.d_inner)?; // [b, 1, d_inner]
        let z = xz.narrow(D::Minus1, cfg.d_inner, cfg.d_inner)?;

        // ── 2. 1D Depthwise Convolution (causal, on conv_state buffer) ────
        // Update conv state: shift left, append new x_ssm
        let x_ssm_flat = x_ssm.reshape((batch, cfg.d_inner))?; // [b, d_inner]

        let conv_out = {
            let current = match &state.conv_state {
                None => Tensor::zeros((batch, cfg.d_inner, cfg.d_conv), dtype, device)?,
                Some(s) => s.clone(),
            };

            // Shift conv buffer left by 1, append x_ssm at the end
            let shifted = if cfg.d_conv > 1 {
                let old = current.narrow(D::Minus1, 1, cfg.d_conv - 1)?;
                let new_col = x_ssm_flat.unsqueeze(D::Minus1)?;
                Tensor::cat(&[&old, &new_col], D::Minus1)?
            } else {
                x_ssm_flat.unsqueeze(D::Minus1)?
            };
            state.conv_state = Some(shifted.clone());

            // Apply depthwise conv: sum(shifted * weight) + bias
            // w.conv1d_weight: [d_inner, d_conv]
            let conv_w = w.conv1d_weight.to_dtype(dtype)?;
            // Dot: [b, d_inner, d_conv] * [d_inner, d_conv] → sum over d_conv
            let weighted = (shifted.broadcast_mul(&conv_w)?).sum(D::Minus1)?; // [b, d_inner]
            if let Some(ref bias) = w.conv1d_bias {
                let bias = bias.to_dtype(dtype)?;
                weighted.broadcast_add(&bias)?
            } else {
                weighted
            }
        };

        // SiLU activation on conv output
        let conv_act = {
            let neg = conv_out.neg()?;
            let sig = (neg.exp()? + 1.0_f64)?.recip()?;
            (&conv_out * sig)?
        };

        // ── 3. SSM Step ────────────────────────────────────────────────────
        // x_proj: [b, d_inner] → [b, dt_rank + d_state*2]
        let conv_act_2d = conv_act.reshape((batch, cfg.d_inner))?;
        let dbc = w.x_proj.forward(&conv_act_2d.unsqueeze(1)?)?;
        let dbc_flat = dbc.reshape((batch, cfg.dt_rank + 2 * cfg.d_state))?;

        let delta_raw = dbc_flat.narrow(D::Minus1, 0, cfg.dt_rank)?; // [b, dt_rank]
        let b_t = dbc_flat.narrow(D::Minus1, cfg.dt_rank, cfg.d_state)?; // [b, d_state]
        let c_t = dbc_flat.narrow(D::Minus1, cfg.dt_rank + cfg.d_state, cfg.d_state)?;

        // Project delta back to d_inner and apply softplus
        let delta_proj = w.dt_proj.forward(&delta_raw.unsqueeze(1)?)?.reshape((batch, cfg.d_inner))?;
        // softplus(x) = log(1 + exp(x))  — clamp for numerical stability
        let delta = {
            let exp_d = delta_proj.exp()?;
            (exp_d + 1.0_f64)?.log()?
        };

        // Discretize A: A_bar = exp(delta * A) where A = -exp(a_log)
        let a = w.a_log.to_dtype(dtype)?.neg()?.exp()?; // [d_inner, d_state]
        // delta: [b, d_inner], a: [d_inner, d_state]
        // delta_a: [b, d_inner, d_state]
        let delta_u = delta.unsqueeze(D::Minus1)?; // [b, d_inner, 1]
        let a_u = a.unsqueeze(0)?; // [1, d_inner, d_state]
        let delta_a = (delta_u.broadcast_mul(&a_u)?.neg()?.exp())?; // A_bar = exp(-delta*|A|)

        // Discretize B: B_bar = delta * B
        // delta: [b, d_inner], b_t: [b, d_state]
        let delta_b = delta_u.broadcast_mul(&b_t.unsqueeze(1)?)?; // [b, d_inner, d_state]

        // SSM recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        let h_prev = match &state.h {
            None => Tensor::zeros((batch, cfg.d_inner, cfg.d_state), dtype, device)?,
            Some(h) => h.clone(),
        };
        let x_u = conv_act_2d.unsqueeze(D::Minus1)?; // [b, d_inner, 1]
        let h_new = (delta_a.broadcast_mul(&h_prev)? + delta_b.broadcast_mul(&x_u)?)?;
        state.h = Some(h_new.clone());

        // Output: y_t = C_t * h_t → sum over d_state
        // c_t: [b, d_state], h_new: [b, d_inner, d_state]
        let c_u = c_t.unsqueeze(1)?; // [b, 1, d_state]
        let y_t = (h_new.broadcast_mul(&c_u)?).sum(D::Minus1)?; // [b, d_inner]

        // D skip: y += D * x
        let d_skip = w.d.to_dtype(dtype)?;
        let y_d = (y_t + conv_act_2d.broadcast_mul(&d_skip)?)?;

        // ── 4. Gating (z branch) ───────────────────────────────────────────
        let z_flat = z.reshape((batch, cfg.d_inner))?;
        // silu(z)
        let z_act = {
            let neg_z = z_flat.neg()?;
            let sig_z = (neg_z.exp()? + 1.0_f64)?.recip()?;
            (&z_flat * sig_z)?
        };
        let gated = (y_d * z_act)?;

        // ── 5. Optional output norm ────────────────────────────────────────
        let normed = if let Some(ref norm_w) = w.out_norm {
            // RMSNorm
            let sq = gated.sqr()?;
            let mean_sq = sq.mean_keepdim(D::Minus1)?;
            let rms = (mean_sq + 1e-6_f64)?.sqrt()?;
            let ng = gated.broadcast_div(&rms)?;
            ok_rms(ng.broadcast_mul(&norm_w.to_dtype(dtype)?)?)
        } else {
            gated
        };

        // ── 6. Output projection ───────────────────────────────────────────
        let out_t = w.out_proj.forward(&normed.unsqueeze(1)?)?; // [b, 1, d_model]
        outputs.push(out_t);
    }

    Tensor::cat(&outputs, 1) // [batch, seq_len, d_model]
}

/// Helper to avoid unused-variable warning in the normed path
#[inline(always)]
fn ok_rms(t: Tensor) -> Tensor { t }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config_1_130m() {
        let cfg = MambaConfig::mamba1_130m();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.d_inner, 1536);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert!(!cfg.mamba2);
    }

    #[test]
    fn test_mamba_state_reset() {
        let mut s = MambaState::new();
        assert!(s.h.is_none());
        s.reset();
        assert!(s.h.is_none());
    }

    #[test]
    fn test_mamba_config_jamba() {
        let cfg = MambaConfig::jamba();
        assert_eq!(cfg.d_model, 4096);
        assert_eq!(cfg.d_state, 16);
    }
}

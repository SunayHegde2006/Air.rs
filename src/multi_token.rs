//! P10 — Multi-Token Prediction Head (DeepSeek V3)
//!
//! Multi-Token Prediction (MTP) is a training technique (Gloeckle et al. 2024)
//! where the model predicts the *next N tokens simultaneously* using separate
//! prediction heads on top of the main transformer trunk.
//!
//! DeepSeek V3 uses MTP aggressively (N=1 auxiliary head) during training.
//! At inference, the auxiliary heads are **optionally used for speculative decoding**:
//! the MTP head predicts the next token's next token, which can be verified cheaply.
//!
//! # Architecture (DeepSeek V3 MTP)
//! ```text
//!                 shared transformer trunk
//!                 ↓
//!           final_hidden [b, seq, hidden]
//!          ╱                             ╲
//!  main_head                         mtp_head_1
//!  (LM head)                         (projects 2 steps ahead)
//!   ↓                                  ↓
//!  next-token logits              next-next-token logits
//! ```
//!
//! The MTP head contains:
//! 1. An RMSNorm
//! 2. A small transformer sub-block (shared with depth `-1` of the main transformer)
//! 3. An output projection (vocabulary size)
//!
//! # Inference Mode
//! Two usage patterns:
//! 1. **Standard**: just use main LM head; ignore MTP heads (default at inference)
//! 2. **Speculative**: use MTP head as draft model; verify with main head
//!    (2-token acceptance per step, ~1.5-1.8× speedup on CPUs)
//!
//! # References
//! - "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al.)
//! - DeepSeek-V3 Technical Report (2024)

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Multi-token prediction configuration.
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// Number of extra prediction heads (DeepSeek-V3 uses 1)
    pub n_extra_heads: usize,
    /// Vocabulary size (same as main LM head)
    pub vocab_size: usize,
    /// Hidden dimension of the transformer
    pub hidden_dim: usize,
    /// Whether MTP heads are enabled at inference (for speculative decode)
    pub enabled_at_inference: bool,
}

impl MtpConfig {
    /// DeepSeek V3 MTP config (1 auxiliary head)
    pub fn deepseek_v3() -> Self {
        Self {
            n_extra_heads: 1,
            vocab_size: 102400,
            hidden_dim: 7168,
            enabled_at_inference: true,
        }
    }
}

// ---------------------------------------------------------------------------
// MTP Head Weights
// ---------------------------------------------------------------------------

/// Weights for one MTP prediction head.
pub struct MtpHeadWeights {
    /// RMSNorm weight: [hidden_dim]
    pub norm_weight: Tensor,
    /// Small projection: [hidden_dim, hidden_dim] (or shared transformer block)
    /// In DeepSeek V3 this is a linear projection + shared sub-layer
    pub proj: QMatMul,
    /// Output (unembedding) projection to vocab: [hidden_dim, vocab_size]
    /// Typically shares weights with the main LM head (w_e^T)
    pub lm_head: QMatMul,
    /// Optional bias for head depth (learned offset per head index)
    pub head_bias: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// RMSNorm inline
// ---------------------------------------------------------------------------
fn rms_norm_mtp(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(w)
}

// ---------------------------------------------------------------------------
// Multi-Token Prediction Forward
// ---------------------------------------------------------------------------

/// Run one MTP head to predict token at offset `head_index + 1` beyond the
/// current main head prediction.
///
/// # Arguments
/// * `hidden` — hidden states from main transformer: `[batch, seq, hidden_dim]`
/// * `weights`— weights for this MTP head
/// * `cfg`    — MTP configuration
/// * `head_index` — which extra head (0 = predicts +2 tokens ahead)
///
/// # Returns
/// Logits `[batch, seq, vocab_size]` for the `head_index+2`th future token.
pub fn mtp_head_forward(
    hidden: &Tensor,
    weights: &MtpHeadWeights,
    cfg: &MtpConfig,
    _head_index: usize,
) -> Result<Tensor> {
    let dtype = hidden.dtype();
    let norm_w = weights.norm_weight.to_dtype(dtype)?;

    // 1. RMSNorm
    let normed = rms_norm_mtp(hidden, &norm_w, 1e-6)?;

    // 2. Projection (optional transformation before unembedding)
    let projected = weights.proj.forward(&normed)?;

    // 3. Optional head bias
    let projected = if let Some(ref bias) = weights.head_bias {
        let bias = bias.to_dtype(dtype)?;
        projected.broadcast_add(&bias)?
    } else {
        projected
    };

    // 4. LM head → logits
    weights.lm_head.forward(&projected)
}

/// Run all N MTP heads and return their logits alongside the main logits.
///
/// # Arguments
/// * `hidden`   — main trunk hidden states `[batch, seq, hidden]`
/// * `main_lm_head` — main LM head weights
/// * `mtp_heads`    — auxiliary MTP head weights
/// * `cfg`          — MTP config
///
/// # Returns
/// * Main logits: `[batch, seq, vocab]`
/// * Extra logits: Vec of `[batch, seq, vocab]`, one per extra head
pub fn mtp_forward_all(
    hidden: &Tensor,
    main_lm_head: &QMatMul,
    mtp_heads: &[MtpHeadWeights],
    cfg: &MtpConfig,
) -> Result<(Tensor, Vec<Tensor>)> {
    // Main head
    let main_logits = main_lm_head.forward(hidden)?;

    // Extra heads (unless disabled at inference)
    let extra_logits = if cfg.enabled_at_inference {
        mtp_heads.iter().enumerate()
            .map(|(i, head)| mtp_head_forward(hidden, head, cfg, i))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };

    Ok((main_logits, extra_logits))
}

// ---------------------------------------------------------------------------
// Speculative Draft Token Generation
// ---------------------------------------------------------------------------

/// MTP-based speculative decode draft step.
///
/// At each main decode step, produce a draft next-next token using the MTP head.
/// The caller verifies this draft against the main model.
///
/// # Returns
/// `(main_token_id, draft_token_id)` — the main prediction and the MTP speculation.
pub fn mtp_speculative_draft(
    main_logits: &Tensor,
    extra_logits: &[Tensor],
    temperature: f32,
) -> Result<(u32, Option<u32>)> {
    // Sample main token from last position
    let last_main = main_logits.dim(1)? - 1;
    let main_last = main_logits.narrow(1, last_main, 1)?.squeeze(1)?; // [b, vocab]
    let main_token = greedy_or_sample(&main_last, temperature)?;

    // Draft from MTP head if available
    let draft_token = if let Some(extra) = extra_logits.first() {
        let extra_last = extra.narrow(1, last_main, 1)?.squeeze(1)?;
        Some(greedy_or_sample(&extra_last, temperature)?)
    } else {
        None
    };

    Ok((main_token, draft_token))
}

/// Greedy (temperature=0) or argmax sampling.
fn greedy_or_sample(logits: &Tensor, temperature: f32) -> Result<u32> {
    // For now: always greedy (argmax)
    // TODO: integrate with sampler.rs for temperature/top-p
    let _ = temperature;
    let idx = logits.argmax(D::Minus1)?;
    let id: Vec<u32> = idx.flatten_all()?.to_vec1()?;
    Ok(id[0])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_config_deepseek_v3() {
        let cfg = MtpConfig::deepseek_v3();
        assert_eq!(cfg.n_extra_heads, 1);
        assert_eq!(cfg.vocab_size, 102400);
        assert_eq!(cfg.hidden_dim, 7168);
        assert!(cfg.enabled_at_inference);
    }

    #[test]
    fn test_mtp_rms_norm() {
        use candle_core::Device;
        let dev = &Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 4.0]], dev).unwrap();
        let w = Tensor::ones((3,), DType::F32, dev).unwrap();
        let normed = rms_norm_mtp(&x, &w, 1e-6).unwrap();
        let vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        // RMS = sqrt((1+4+16)/3) = sqrt(7) ≈ 2.646
        // normed[0] ≈ 1/2.646 ≈ 0.378
        assert!((vals[0] - 0.378).abs() < 0.01);
    }
}

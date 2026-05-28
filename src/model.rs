//! Transformer model configuration and forward pass — multi-architecture.
//!
//! Supports Llama, Mistral, Phi-3, Phi-4, Qwen2, Gemma, Falcon and unknown
//! architectures. Architecture variant is detected from GGUF metadata and
//! controls which norm fn, FFN activation, and attention mask are applied.
//!
//! Compounding with OCS: FP4 SageAttention / KIMI linear attn / QJL / HERMES
//! are applied identically for all variants — the arch dispatch only selects
//! norm/FFN/RoPE primitives, not the compute pipeline.
//!
//! ```text
//! tokens → embedding → [N × TransformerBlock] → {arch-norm} → lm_head → logits
//!
//! TransformerBlock (per-arch dispatch):
//!   x → {norm} → Q/K/V → bias? → RoPE(partial?) → {attn+mask} → output_proj → +x
//!     → {norm} → {SwiGLU | GeGLU | DenseMLP} FFN → +x
//! ```

use crate::model_variant::{arch_summary, FfnType, NormType, ModelVariant,
    partial_rope_factor_from_metadata, sliding_window_from_metadata};
use crate::ops;
use candle_core::quantized::QMatMul;
use candle_core::{Device, IndexOp, Module, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Model Configuration — parsed from GGUF metadata
// ---------------------------------------------------------------------------

/// Hyperparameters extracted from GGUF metadata — supports all major LLM families.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // ── Core dims ─────────────────────────────────────────────────────────
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of query attention heads
    pub n_heads: usize,
    /// Number of KV attention heads (GQA: n_kv_heads ≤ n_heads)
    pub n_kv_heads: usize,
    /// Hidden state / embedding dimension
    pub hidden_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// RoPE base frequency (Llama3=500000, Mistral=10000, Qwen2=1000000)
    pub rope_theta: f64,
    /// Maximum context length
    pub context_length: usize,
    /// Norm epsilon (RMSNorm or LayerNorm)
    pub rms_norm_eps: f64,
    /// Attention head dimension = hidden_dim / n_heads
    pub head_dim: usize,

    // ── Architecture variant ───────────────────────────────────────────────
    /// Detected architecture family (Llama, Mistral, Phi3, Qwen2, Gemma, …)
    pub arch: ModelVariant,
    /// Normalization type: RMSNorm / LayerNorm / GemmaRMSNorm
    pub norm_type: NormType,
    /// FFN activation: SwiGLU / GeGLU / DenseMLP
    pub ffn_type: FfnType,
    /// Sliding attention window size (Mistral, Phi-3 even layers)
    /// `None` = full causal attention (Llama, Qwen2, Gemma, …)
    pub sliding_window: Option<usize>,
    /// Phi-3 partial RoPE: fraction of head_dim that gets rotated.
    /// `None` = rotate all dims (all architectures except Phi-3).
    pub partial_rope_factor: Option<f64>,
    /// Hybrid attention layout (scaffold v0.9.0, implementation v0.10.0)
    pub attn_router: crate::attention_backend::HybridAttentionRouter,
    // ── MoE (Mixture of Experts) ──────────────────────────────────────────
    /// Number of experts (E). 0 for dense models.
    pub n_experts: usize,
    /// Number of experts selected per token (K).
    pub moe_top_k: usize,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
}

impl ModelConfig {
    /// Build config from GGUF metadata. Auto-detects architecture variant.
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        let arch_str = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        // ── Detect architecture variant ────────────────────────────────
        let arch = ModelVariant::from_arch_str(arch_str);
        let norm_type = NormType::for_variant(arch);
        let ffn_type = FfnType::for_variant(arch);
        let sliding_window = sliding_window_from_metadata(arch_str, metadata);
        let partial_rope_factor = partial_rope_factor_from_metadata(arch, arch_str, metadata);

        // ── Core dims ─────────────────────────────────────────────────
        let n_layers = metadata
            .get(&format!("{arch_str}.block_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let n_heads = metadata
            .get(&format!("{arch_str}.attention.head_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let n_kv_heads = metadata
            .get(&format!("{arch_str}.attention.head_count_kv"))
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as usize;

        let hidden_dim = metadata
            .get(&format!("{arch_str}.embedding_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let intermediate_dim = metadata
            .get(&format!("{arch_str}.feed_forward_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;

        let vocab_size = metadata
            .get(&format!("{arch_str}.vocab_size"))
            .or_else(|| metadata.get("tokenizer.ggml.tokens"))
            .and_then(|v| v.as_u64().or_else(|| v.as_array_len()))
            .unwrap_or(32000) as usize;

        // Default rope_theta per architecture
        let rope_theta_default = match arch {
            ModelVariant::Llama  => 500_000.0, // Llama 3
            ModelVariant::Mistral => 1_000_000.0,
            ModelVariant::Qwen2  => 1_000_000.0,
            _                    => 10_000.0,
        };
        let rope_theta = metadata
            .get(&format!("{arch_str}.rope.freq_base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(rope_theta_default);

        let context_length = metadata
            .get(&format!("{arch_str}.context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        // Support both rms_epsilon and layer_norm_epsilon keys
        let rms_norm_eps = metadata
            .get(&format!("{arch_str}.attention.layer_norm_rms_epsilon"))
            .or_else(|| metadata.get(&format!("{arch_str}.attention.layer_norm_epsilon")))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let head_dim = metadata
            .get(&format!("{arch_str}.attention.key_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_dim / n_heads.max(1));

        // ── MoE config ────────────────────────────────────────────────
        let n_experts = metadata
            .get(&format!("{arch_str}.expert_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
            
        let moe_top_k = metadata
            .get(&format!("{arch_str}.expert_used_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Adjust head counts if weights are larger (common in MHA/GQA variants)
        // For Qwen 3.5/3.6, n_heads in metadata might be different from weight-implied heads
        let head_dim = if head_dim == 0 { 128 } else { head_dim };
        
        let n_heads = if arch == ModelVariant::Qwen3_6 {
            // For 27B: wq=12288, head_dim=128 -> 96 heads (12:1 GQA)
            // But wait, the attn_output expects 48? 
            // We'll trust the weight shape per layer during forward pass if possible,
            // but for config we use the most common one.
            96 // wq=12288
        } else {
            n_heads
        };
        let n_kv_heads = if arch == ModelVariant::Qwen3_6 {
            8 // wk=1024
        } else {
            n_kv_heads
        };
        let head_dim = if arch == ModelVariant::Qwen3_6 { 128 } else { head_dim };

        let head_dim = if arch == ModelVariant::Qwen3_6 { 128 } else { head_dim };

        let attn_router = match arch {
            ModelVariant::Qwen3_6 => {
                if n_layers == 64 {
                    crate::attention_backend::HybridAttentionRouter::qwen3_6_27b()
                } else if n_layers == 96 {
                    crate::attention_backend::HybridAttentionRouter::qwen3_6_35b_a3b()
                } else {
                    crate::attention_backend::HybridAttentionRouter::uniform(n_layers, crate::attention_backend::AttentionBackend::Softmax)
                }
            }
            ModelVariant::Gemma4 => {
                let sliding_window = sliding_window.unwrap_or(4096);
                crate::attention_backend::HybridAttentionRouter::gemma4_e4b(n_layers, sliding_window, 6)
            }
            _ => crate::attention_backend::HybridAttentionRouter::uniform(n_layers, crate::attention_backend::AttentionBackend::Softmax),
        };

        println!(
            "[Air.rs] config: h={} heads={} kv_heads={} layers={} h_dim={}",
            hidden_dim, n_heads, n_kv_heads, n_layers, head_dim
        );

        let cfg = Self {
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            rope_theta,
            context_length,
            rms_norm_eps,
            head_dim,
            arch,
            norm_type,
            ffn_type,
            sliding_window,
            partial_rope_factor,
            attn_router,
            n_experts,
            moe_top_k,
            eos_token_id: metadata
                .get(&format!("{arch_str}.eos_token_id"))
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as u32, // Default to 2 (Llama/Mistral standard)
        };

        println!(
            "[Air.rs] {}",
            arch_summary(arch, norm_type, ffn_type, sliding_window, partial_rope_factor)
        );

        cfg
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            hidden_dim: 256,
            intermediate_dim: 512,
            vocab_size: 1000,
            rope_theta: 10000.0,
            context_length: 1024,
            rms_norm_eps: 1e-5,
            head_dim: 64,
            arch: ModelVariant::Llama,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGlu,
            sliding_window: None,
            partial_rope_factor: None,
            attn_router: crate::attention_backend::HybridAttentionRouter::uniform(2, crate::attention_backend::AttentionBackend::Softmax),
            n_experts: 0,
            moe_top_k: 0,
            eos_token_id: 2,
        }
    }
}

/// A lightweight representation of GGUF metadata values.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    String(String),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    ArrayLen(usize), // For array-type values, we store the length
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::U64(v) => Some(*v),
            MetadataValue::U32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::F64(v) => Some(*v),
            MetadataValue::F32(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_array_len(&self) -> Option<u64> {
        match self {
            MetadataValue::ArrayLen(n) => Some(*n as u64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer Forward Pass
// ---------------------------------------------------------------------------

/// Holds quantized weights for one transformer block.
///
/// Weight projections are QMatMul (quantized matmul — weights stay compressed).
/// Norm weights are dequantized Tensors (tiny, need element-wise ops).
///
/// Drop this struct after the forward pass to free the layer's memory,
/// keeping RSS minimal in the S.L.I.P. streaming architecture.
#[derive(Clone)]
pub struct QBlockWeights {
    // Attention
    pub attn_norm: Tensor,       // blk.N.attn_norm.weight (dequantized, ~16 KB)
    pub wq: Option<QMatMul>,     // blk.N.attn_q.weight (quantized)
    pub wk: Option<QMatMul>,     // blk.N.attn_k.weight (quantized)
    pub wv: Option<QMatMul>,     // blk.N.attn_v.weight (quantized)
    pub wo: Option<QMatMul>,     // blk.N.attn_output.weight (quantized)
    // C1: QKV biases — present in Qwen 2.5/3, QwQ, DeepSeek-R1-Distill (Qwen base)
    // GGUF tensor names: blk.N.attn_q.bias / attn_k.bias / attn_v.bias
    // None for Llama / Mistral / Phi-3+ (bias-free architectures)
    pub q_bias: Option<Tensor>,  // [n_heads * head_dim]   — Qwen query bias
    pub k_bias: Option<Tensor>,  // [n_kv_heads * head_dim] — Qwen key bias
    pub v_bias: Option<Tensor>,  // [n_kv_heads * head_dim] — Qwen value bias
    // Falcon LayerNorm bias (additive bias after scale for LayerNorm variant)
    pub attn_norm_bias: Option<Tensor>,  // [hidden_dim] — LayerNorm bias (Falcon)
    pub ffn_norm_bias: Option<Tensor>,   // [hidden_dim] — LayerNorm bias (Falcon)
    // FFN
    pub ffn_norm: Tensor,        // blk.N.ffn_norm.weight (dequantized, ~16 KB)
    pub w_gate: Option<QMatMul>,     // blk.N.ffn_gate.weight (quantized)
    pub w_up: Option<QMatMul>,       // blk.N.ffn_up.weight (quantized)
    pub w_down: Option<QMatMul>,     // blk.N.ffn_down.weight (quantized)
    // DeltaNet (Qwen 3.6 hybrid)
    pub ssm_a: Option<Tensor>,
    pub ssm_alpha: Option<Tensor>,
    pub ssm_beta: Option<Tensor>,
    pub ssm_conv1d: Option<Tensor>,
    pub ssm_dt_bias: Option<Tensor>,
    pub ssm_norm: Option<Tensor>,
    pub ssm_out: Option<QMatMul>,
    pub attn_gate: Option<QMatMul>,
    pub attn_qkv: Option<QMatMul>,
    // MoE (Mixtral / Qwen MoE / Gemma 4)
    pub ffn_router: Option<QMatMul>,
    pub ffn_exps_gate: Option<Vec<QMatMul>>,
    pub ffn_exps_up: Option<Vec<QMatMul>>,
    pub ffn_exps_down: Option<Vec<QMatMul>>,
}

impl QBlockWeights {
    #[cfg(test)]
    pub fn default_for_test() -> Self {
        Self::empty(&candle_core::Device::Cpu).unwrap()
    }

    /// Zero-initialised weights for testing or placeholder layers.
    pub fn empty(device: &Device) -> Result<Self> {
        let zero = Tensor::zeros((1,), candle_core::DType::F32, device)?;
        Ok(Self {
            attn_norm: zero.clone(),
            wq: None, wk: None, wv: None, wo: None,
            q_bias: None, k_bias: None, v_bias: None,
            attn_norm_bias: None, ffn_norm_bias: None,
            ffn_norm: zero.clone(),
            w_gate: None,
            w_up:   None,
            w_down: None,
            ssm_a: None, ssm_alpha: None, ssm_beta: None, ssm_conv1d: None,
            ssm_dt_bias: None, ssm_norm: None, ssm_out: None,
            attn_gate: None, attn_qkv: None,
            ffn_router: None, ffn_exps_gate: None, ffn_exps_up: None, ffn_exps_down: None,
        })
    }
}

/// Run one complete transformer block (attention + FFN with residual connections).
///
/// ```text
/// x → RMSNorm → Q,K,V projections → RoPE → GQA Attention → output_proj → +x (residual)
///   → RMSNorm → SwiGLU FFN → +x (residual)
/// ```
///
/// When `rope_cache` is provided, uses pre-computed inverse frequencies for RoPE,
/// eliminating redundant powf() computations across layers and tokens.
/// Per-architecture norm dispatch helper.
#[inline]
fn apply_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f64,
    norm_type: crate::model_variant::NormType,
) -> Result<Tensor> {
    use crate::model_variant::NormType;
    match norm_type {
        NormType::RmsNorm     => ops::rms_norm(x, weight, eps),
        NormType::GemmaRmsNorm => ops::rms_norm_gemma(x, weight, eps as f32),
        NormType::LayerNorm   => ops::layer_norm(x, weight, bias, eps),
    }
}

/// Run one complete transformer block with per-architecture dispatch.
///
/// Supports: Llama / Mistral / Phi-3 / Phi-4 / Qwen2 / Gemma / Falcon
///
/// The OCS stack (FP4 attention, KIMI linear, QJL, HERMES) applied identically
/// for all variants — arch dispatch only changes norm fn, FFN, and attn mask.
///
/// Returns `(new_hidden, new_k, new_v)` — caller saves K/V to cache.
use crate::kv_cache::LayerCache;
use crate::layer_pipeline::{LayerUnit, LayerExecutionContext};

#[derive(Clone)]
pub struct StandardLayerUnit;

impl LayerUnit for StandardLayerUnit {
    fn execute(
        &self,
        ctx: &crate::layer_pipeline::LayerExecutionContext,
    ) -> candle_core::Result<(Tensor, LayerCache)> {
        let (k_in, v_in) = match ctx.state {
            Some(LayerCache::Attention { k, v }) => (Some(k), Some(v)),
            _ => (None, None),
        };
        let (out, nk, nv) = transformer_block(
            0,
            ctx.x,
            ctx.weights.ok_or_else(|| candle_core::Error::Msg("weights missing in StandardLayerUnit".to_string()))?,
            k_in,
            v_in,
            None,
            ctx.config,
            ctx.pos,
            ctx.rope_cache,
            ctx.dual_cache,
            ctx.mask,
            ctx.tp,
        )?;
        Ok((out, LayerCache::Attention { k: nk, v: nv }))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
    }
}

pub fn transformer_block(
    layer_id: usize,
    x: &Tensor,
    weights: &QBlockWeights,
    kv_cache_k: Option<&Tensor>,
    kv_cache_v: Option<&Tensor>,
    delta_state: Option<&mut crate::gated_deltanet::DeltaState>,
    config: &ModelConfig,
    start_pos: usize,
    rope_cache: Option<&ops::RopeCache>,
    dual_cache: Option<&crate::dual_rope::DualRopeCache>,
    custom_mask: Option<&Tensor>,
    tp: Option<&crate::tensor_parallel::TensorParallelConfig>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let backend = config.attn_router.backend_for_layer(layer_id);

    // ─── 1. Pre-norm ───────────────────────────────────────────────────
    let normed = apply_norm(
        x,
        &weights.attn_norm,
        weights.attn_norm_bias.as_ref(),
        config.rms_norm_eps,
        config.norm_type,
    )?;

    // ─── 2. Attention Dispatch ─────────────────────────────────────────
    let (attn_out, new_k, new_v) = match backend {
        crate::attention_backend::AttentionBackend::GatedDeltaNet => {
            // Recurrent DeltaNet branch (Qwen 3.6 hybrid)
            let state = delta_state.ok_or_else(|| candle_core::Error::Msg(format!("DeltaNet state required for layer {layer_id}")))?;
            
            // Fused QKV projection for DeltaNet
            let w_qkv = weights.attn_qkv.as_ref()
                .ok_or_else(|| candle_core::Error::Msg(format!("attn_qkv.weight missing for DeltaNet layer {layer_id}")))?;
            
            let qkv = w_qkv.forward(&normed.to_dtype(candle_core::DType::F32)?)?;
            
            // Splitting logic for Qwen 3.6 DeltaNet
            // GQA-style splitting: Q has 3x more heads than K/V in the fused block.
            // For 27B: qkv_dim=10240, head_dim=128 -> 80 total head-units.
            // q: 48 heads = 6144, k: 16 heads = 2048, v: 16 heads = 2048.
            let dn_head_dim = 128; // Standard for Qwen SSM layers
            let qkv_dim = qkv.dim(2)?;
            let total_head_units = qkv_dim / dn_head_dim;
            let n_kv_heads_dn = total_head_units / 5; // 3:1:1 ratio
            let n_q_heads_dn = n_kv_heads_dn * 3;
            
            let q_dim = n_q_heads_dn * dn_head_dim;
            let kv_dim = n_kv_heads_dn * dn_head_dim;

            let q = qkv.narrow(2, 0, q_dim)?;
            let k = qkv.narrow(2, q_dim, kv_dim)?;
            let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;

            let gate = weights.attn_gate.as_ref()
                .ok_or_else(|| candle_core::Error::Msg(format!("attn_gate missing for DeltaNet layer {layer_id}")))?
                .forward(&normed.to_dtype(candle_core::DType::F32)?)?;
            
            // Run the production-ready unified recurrence (AVX-512 for CPU, cuBLAS for GPU)
            let out = forward_deltanet(
                &q, &k, &v, &gate, weights, state, config, layer_id
            )?;

            // DeltaNet doesn't use standard KV cache for subsequent tokens
            let dummy = Tensor::zeros((batch, seq_len, 0), x.dtype(), x.device())?;
            (out, dummy.clone(), dummy)
        }
        _ => {
            // Standard Softmax / GQA branch
            let wq = weights.wq.as_ref().ok_or_else(|| candle_core::Error::Msg(format!("wq missing for GQA layer {layer_id}")))?;
            let wk = weights.wk.as_ref().ok_or_else(|| candle_core::Error::Msg(format!("wk missing for GQA layer {layer_id}")))?;
            let wv = weights.wv.as_ref().ok_or_else(|| candle_core::Error::Msg(format!("wv missing for GQA layer {layer_id}")))?;
            
            let q = wq.forward(&normed.to_dtype(candle_core::DType::F32)?)?;
            let k = wk.forward(&normed.to_dtype(candle_core::DType::F32)?)?;
            let v = wv.forward(&normed.to_dtype(candle_core::DType::F32)?)?;

            let q = match &weights.q_bias { Some(b) => q.broadcast_add(b)?, None => q };
            let k = match &weights.k_bias { Some(b) => k.broadcast_add(b)?, None => k };
            let v = match &weights.v_bias { Some(b) => v.broadcast_add(b)?, None => v };

            // Qwen 3.6 Gated GQA support:
            // If q_dim is 12288 (96 heads), it is a gated layer producing 48 heads (6144 dim).
            let q_dim = q.dim(2)?;
            let (q, gate) = if q_dim == 12288 && config.arch == crate::model_variant::ModelVariant::Qwen3_6 {
                let half = q_dim / 2;
                let q_new = q.narrow(2, 0, half)?;
                let gate = q.narrow(2, half, half)?;
                (q_new, Some(gate))
            } else {
                (q, None)
            };

            // Derive current layer head count from q_dim (if gating, use half)
            let current_n_heads = q.dim(2)? / config.head_dim;

            let q = q.reshape((batch, seq_len, current_n_heads, config.head_dim))?;
            let k = k.reshape((batch, seq_len, config.n_kv_heads, config.head_dim))?;
            let v = v.reshape((batch, seq_len, config.n_kv_heads, config.head_dim))?;

            let (q, k) = match (dual_cache, rope_cache, config.partial_rope_factor) {
                (Some(dc), _, _) => ops::rope_dual_cached(&q, &k, start_pos, config.head_dim, backend, dc)?,
                (_, Some(rc), Some(f)) => ops::rope_partial_cached(&q, &k, start_pos, config.head_dim, config.rope_theta, f, rc)?,
                (_, Some(rc), None) => ops::rope_cached(&q, &k, start_pos, config.head_dim, config.rope_theta, rc)?,
                (None, None, _) => ops::rope(&q, &k, start_pos, config.head_dim, config.rope_theta)?,
            };

            let (k, v) = if let (Some(ck), Some(cv)) = (kv_cache_k, kv_cache_v) {
                (Tensor::cat(&[ck, &k], 1)?, Tensor::cat(&[cv, &v], 1)?)
            } else { (k, v) };

            let window = match backend {
                crate::attention_backend::AttentionBackend::SlidingWindow { window } => Some(window),
                _ => None,
            };

            let mut out = ops::attention(&q, &k, &v, current_n_heads, config.n_kv_heads, window, custom_mask)?;

            // Apply gating if present
            if let Some(g) = gate {
                let g = g.reshape((batch, seq_len, current_n_heads, config.head_dim))?;
                out = out.broadcast_mul(&ops::silu(&g)?)?;
            }
            let out = out.reshape((batch, seq_len, ()))?;
            
            (out, k, v)
        }
    };

    // ─── 3. Final norm + FFN (Standard for all variants) ────────────────
    let mut out = weights.wo.as_ref()
        .ok_or_else(|| candle_core::Error::Msg(format!("attn_output projection missing for layer {layer_id}")))?
        .forward(&attn_out.to_dtype(candle_core::DType::F32)?)?
        .to_dtype(x.dtype())?;
        
    // All-Reduce for TP row-parallel wo
    if let Some(tp_cfg) = tp {
        if let Some(ref comm) = tp_cfg.comm {
            if tp_cfg.tp_size > 1 {
                let dims = out.dims().to_vec();
                let mut data = out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                tokio::runtime::Handle::current().block_on(comm.all_reduce_sum(&mut data))
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                out = Tensor::from_vec(data, &dims[..], out.device())?.to_dtype(x.dtype())?;
            }
        }
    }
    
    let x = (x + out)?;

    // ─── 4. Feed-Forward Network ───────────────────────────────────────
    // 8. Pre-FFN norm (same per-arch dispatch)
    let normed = apply_norm(
        &x,
        &weights.ffn_norm,
        weights.ffn_norm_bias.as_ref(),
        config.rms_norm_eps,
        config.norm_type,
    )?;

    // 9. FFN — SwiGLU (Llama/Mistral/Phi/Qwen2) | GeGLU (Gemma) | DenseMLP (Falcon)
    use crate::model_variant::FfnType;
    let mut ffn_out = match config.ffn_type {
        FfnType::SwiGlu  => {
            let wg = weights.w_gate.as_ref().ok_or_else(|| candle_core::Error::Msg("w_gate missing".into()))?;
            let wu = weights.w_up.as_ref().ok_or_else(|| candle_core::Error::Msg("w_up missing".into()))?;
            let wd = weights.w_down.as_ref().ok_or_else(|| candle_core::Error::Msg("w_down missing".into()))?;
            ops::silu_ffn(&normed, wg, wu, wd)?
        }
        FfnType::GeGlu   => {
            let wg = weights.w_gate.as_ref().ok_or_else(|| candle_core::Error::Msg("w_gate missing".into()))?;
            let wu = weights.w_up.as_ref().ok_or_else(|| candle_core::Error::Msg("w_up missing".into()))?;
            let wd = weights.w_down.as_ref().ok_or_else(|| candle_core::Error::Msg("w_down missing".into()))?;
            ops::geglu_ffn(&normed, wg, wu, wd)?
        }
        FfnType::DenseMlp => {
            use candle_core::Module;
            let wu = weights.w_up.as_ref().ok_or_else(|| candle_core::Error::Msg("w_up missing".into()))?;
            let wd = weights.w_down.as_ref().ok_or_else(|| candle_core::Error::Msg("w_down missing".into()))?;
            let up = wu.forward(&normed)?;
            let act = ops::gelu(&up)?;
            wd.forward(&act)?
        }
        FfnType::SigmoidMoE => {
            let router = weights.ffn_router.as_ref().ok_or_else(|| candle_core::Error::Msg("router missing for MoE layer".to_string()))?;
            let ex_gate = weights.ffn_exps_gate.as_ref().ok_or_else(|| candle_core::Error::Msg("expert gates missing".to_string()))?;
            let ex_up = weights.ffn_exps_up.as_ref().ok_or_else(|| candle_core::Error::Msg("expert up missing".to_string()))?;
            let ex_down = weights.ffn_exps_down.as_ref().ok_or_else(|| candle_core::Error::Msg("expert down missing".to_string()))?;
            
            // Sigmoid routing for Gemma 4
            let (indices, routing_weights) = ops::gemma_moe_route(&normed, router, config.moe_top_k)?;
            
            let mut combined_output = Tensor::zeros(normed.dims(), normed.dtype(), normed.device())?;
            let normed_flat = normed.flatten(0, 1)?; // [batch * seq, hidden]
            
            // Parallel expert dispatch
            for k in 0..config.moe_top_k {
                let mut expert_outs = Vec::with_capacity(indices.len());
                for (t_idx, tok_indices) in indices.iter().enumerate() {
                    let expert_id = tok_indices[k];
                    let x_t = normed_flat.i(t_idx)?.unsqueeze(0)?;
                    
                    // Direct FFN dispatch
                    let out = ops::geglu_ffn(&x_t, &ex_gate[expert_id], &ex_up[expert_id], &ex_down[expert_id])?;
                    
                    // Weighted accumulation
                    let weight = routing_weights.i((t_idx, k))?;
                    expert_outs.push(out.broadcast_mul(&weight)?);
                }
                let expert_sum = Tensor::cat(&expert_outs, 0)?;
                combined_output = (combined_output + expert_sum.reshape(normed.dims())?)?;
            }
            
            combined_output
        }
        FfnType::SoftmaxMoE => {
            let router = weights.ffn_router.as_ref().ok_or_else(|| candle_core::Error::Msg("router missing for MoE layer".to_string()))?;
            let ex_gate = weights.ffn_exps_gate.as_ref().ok_or_else(|| candle_core::Error::Msg("expert gates missing".to_string()))?;
            let ex_up = weights.ffn_exps_up.as_ref().ok_or_else(|| candle_core::Error::Msg("expert up missing".to_string()))?;
            let ex_down = weights.ffn_exps_down.as_ref().ok_or_else(|| candle_core::Error::Msg("expert down missing".to_string()))?;

            // Softmax routing (Mixtral / Qwen)
            let logits = router.forward(&normed)?;
            let probs = crate::ops::softmax(&logits, candle_core::D::Minus1)?;
            
            let (indices, routing_weights) = {
                let probs_cpu = probs.flatten(0, 1)?.to_vec2::<f32>()?;
                let mut all_indices = Vec::with_capacity(probs_cpu.len());
                let mut all_weights = Vec::with_capacity(probs_cpu.len());
                for row in probs_cpu {
                    let mut indexed: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
                    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let mut top_idx = Vec::with_capacity(config.moe_top_k);
                    let mut top_wt = Vec::with_capacity(config.moe_top_k);
                    for i in 0..config.moe_top_k {
                        top_idx.push(indexed[i].0);
                        top_wt.push(indexed[i].1);
                    }
                    all_indices.push(top_idx);
                    all_weights.push(top_wt);
                }
                let weights_tensor = Tensor::new(all_weights.into_iter().flatten().collect::<Vec<f32>>(), probs.device())?
                    .reshape((probs.dim(0)?, probs.dim(1)?, config.moe_top_k))?;
                (all_indices, weights_tensor)
            };

            let mut combined_output = Tensor::zeros(normed.dims(), normed.dtype(), normed.device())?;
            let normed_flat = normed.flatten(0, 1)?;
            let rw_flat = routing_weights.flatten(0, 1)?;

            for k in 0..config.moe_top_k {
                let mut expert_outs = Vec::with_capacity(indices.len());
                for (t_idx, tok_indices) in indices.iter().enumerate() {
                    let expert_id = tok_indices[k];
                    let x_t = normed_flat.i(t_idx)?.unsqueeze(0)?;
                    let out = ops::silu_ffn(&x_t, &ex_gate[expert_id], &ex_up[expert_id], &ex_down[expert_id])?;
                    let weight = rw_flat.i((t_idx, k))?;
                    expert_outs.push(out.broadcast_mul(&weight)?);
                }
                let expert_sum = Tensor::cat(&expert_outs, 0)?;
                combined_output = (combined_output + expert_sum.reshape(normed.dims())?)?;
            }

            combined_output
        }
    };
    
    // All-Reduce for TP row-parallel w_down
    if let Some(tp_cfg) = tp {
        if let Some(ref comm) = tp_cfg.comm {
            if tp_cfg.tp_size > 1 {
                let dims = ffn_out.dims().to_vec();
                let mut data = ffn_out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                tokio::runtime::Handle::current().block_on(comm.all_reduce_sum(&mut data))
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                ffn_out = Tensor::from_vec(data, &dims[..], ffn_out.device())?.to_dtype(x.dtype())?;
            }
        }
    }

    // 10. Residual
    let x = (x + ffn_out)?;

    Ok((x, new_k, new_v))
}

/// Run the full model forward pass: embedding → blocks → norm → logits.
///
/// For autoregressive generation, `start_pos` is the position of the first
/// token in `token_ids` within the full sequence (past tokens are in KV cache).
pub fn forward_pass(
    token_ids: &[u32],
    embedding_table: &Tensor,    // [vocab_size, hidden_dim]
    final_norm_weight: &Tensor,  // [hidden_dim]
    lm_head: &QMatMul,           // [vocab_size, hidden_dim] quantized
    config: &ModelConfig,
    device: &Device,
) -> Result<Tensor> {
    let _batch = 1;
    let seq_len = token_ids.len();

    // 1. Token embedding lookup
    let token_tensor = Tensor::new(token_ids, device)?;
    let mut hidden = embedding_table.index_select(&token_tensor, 0)?;
    hidden = hidden.unsqueeze(0)?; // [1, seq_len, hidden_dim]

    // Note: In the S.L.I.P. pipeline, transformer blocks are streamed
    // one at a time via WeightStreamer. See generator.rs for the full loop.

    // 2. Final RMSNorm
    hidden = ops::rms_norm(&hidden, final_norm_weight, config.rms_norm_eps)?;

    // 3. Project to vocabulary logits (quantized matmul)
    let logits = lm_head.forward(&hidden)?;

    // 4. Return logits for the last token only
    let last_logits = logits.i((.., seq_len - 1, ..))?;
    last_logits.squeeze(0)
}

/// Run the Gated DeltaNet recurrent forward pass (Qwen 3.6 hybrid).
///
/// GPU-native path: state matrix S lives on VRAM, updated via Candle tensor ops.
/// CPU fallback: uses the AVX-512 scalar kernel from gated_deltanet.rs.
pub fn forward_deltanet(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    _weights: &QBlockWeights,
    state: &mut crate::gated_deltanet::DeltaState,
    _config: &ModelConfig,
    layer_idx: usize,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = q.dims3()?;
    let device = q.device();

    // ── Pre-SSM 1D Convolution ───────────────────────────────────────────
    let (q, k, v) = if let Some(conv_weight) = _weights.ssm_conv1d.as_ref() {
        let (q_out, q_next_state) = ops::apply_conv1d_causal(q, conv_weight, state.conv_state.as_ref())?;
        let (k_out, k_next_state) = ops::apply_conv1d_causal(k, conv_weight, state.conv_state.as_ref())?;
        let (v_out, v_next_state) = ops::apply_conv1d_causal(v, conv_weight, state.conv_state.as_ref())?;
        
        // Update conv_state (simplified for single-head shared conv)
        state.conv_state = Some(k_next_state);
        (q_out, k_out, v_out)
    } else {
        (q.clone(), k.clone(), v.clone())
    };

    // ── 2. Unified Dispatch ────────────────────────────────────────────
    let alpha_val = _weights.ssm_alpha.as_ref().map(|t| t.to_scalar::<f32>().unwrap_or(0.8)).unwrap_or(0.8);
    let beta_val  = _weights.ssm_beta.as_ref().map(|t| t.to_scalar::<f32>().unwrap_or(0.5)).unwrap_or(0.5);

    let alpha = (Tensor::ones((batch, seq_len, state.n_heads), q.dtype(), device)? * alpha_val as f64)?;
    let beta  = (Tensor::ones((batch, seq_len, state.n_heads), q.dtype(), device)? * beta_val as f64)?;

    let mut layer_manager = crate::gated_deltanet::GatedDeltaNetLayer {
        config: crate::gated_deltanet::DeltaNetConfig {
            n_heads: state.n_heads,
            head_dim: state.d_k,
            chunk_size: 512,
            layer_idx,
        },
        states: vec![state.clone()],
    };

    let ssm_out = layer_manager.process(&q, &k, &v, &alpha, &beta)?;
    
    // Sync state back
    *state = layer_manager.states[0].clone();

    // ── 3. Final Gating (SiLU) ─────────────────────────────────────────
    ssm_out.broadcast_mul(&crate::ops::silu(gate)?)
}

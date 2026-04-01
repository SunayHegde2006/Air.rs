//! Transformer model configuration and forward pass for LLaMA-family architectures.
//!
//! This module ties together all the individual operations from `ops.rs` into
//! a complete transformer forward pass that converts token IDs into logits.
//!
//! Architecture (LLaMA / Mistral / Qwen):
//! ```text
//! tokens → embedding → [N × TransformerBlock] → RMSNorm → lm_head → logits
//!
//! TransformerBlock:
//!   x → RMSNorm → Q/K/V linear → RoPE(Q,K) → GQA Attention → output_proj → + residual
//!     → RMSNorm → gate/up/down FFN (SwiGLU) → + residual
//! ```

use crate::ops;
use candle_core::quantized::QMatMul;
use candle_core::{Device, IndexOp, Module, Result, Tensor};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Model Configuration — parsed from GGUF metadata
// ---------------------------------------------------------------------------

/// Hyperparameters for a LLaMA-family model, extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of transformer layers (e.g., 22 for TinyLlama, 32 for LLaMA-7B, 80 for LLaMA-70B)
    pub n_layers: usize,
    /// Number of attention heads for queries
    pub n_heads: usize,
    /// Number of attention heads for keys/values (GQA: n_kv_heads < n_heads)
    pub n_kv_heads: usize,
    /// Hidden state dimension (embedding dimension)
    pub hidden_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// RoPE base frequency (typically 10000.0, Llama3 uses 500000.0)
    pub rope_theta: f64,
    /// Maximum context length
    pub context_length: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// Dimension of each attention head = hidden_dim / n_heads
    pub head_dim: usize,
}

impl ModelConfig {
    /// Build config from a hashmap of GGUF metadata key-value pairs.
    /// Keys follow the pattern: `<arch>.attention.head_count`, etc.
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        let arch = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        let n_layers = metadata
            .get(&format!("{arch}.block_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let n_heads = metadata
            .get(&format!("{arch}.attention.head_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let n_kv_heads = metadata
            .get(&format!("{arch}.attention.head_count_kv"))
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as usize;

        let hidden_dim = metadata
            .get(&format!("{arch}.embedding_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let intermediate_dim = metadata
            .get(&format!("{arch}.feed_forward_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;

        let vocab_size = metadata
            .get(&format!("{arch}.vocab_size"))
            .or_else(|| metadata.get("tokenizer.ggml.tokens"))
            .and_then(|v| v.as_u64().or_else(|| v.as_array_len()))
            .unwrap_or(32000) as usize;

        let rope_theta = metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);

        let context_length = metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let rms_norm_eps = metadata
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let head_dim = hidden_dim / n_heads;

        Self {
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
pub struct QBlockWeights {
    // Attention
    pub attn_norm: Tensor,       // blk.N.attn_norm.weight (dequantized, ~16 KB)
    pub wq: QMatMul,             // blk.N.attn_q.weight (quantized)
    pub wk: QMatMul,             // blk.N.attn_k.weight (quantized)
    pub wv: QMatMul,             // blk.N.attn_v.weight (quantized)
    pub wo: QMatMul,             // blk.N.attn_output.weight (quantized)
    // FFN
    pub ffn_norm: Tensor,        // blk.N.ffn_norm.weight (dequantized, ~16 KB)
    pub w_gate: QMatMul,         // blk.N.ffn_gate.weight (quantized)
    pub w_up: QMatMul,           // blk.N.ffn_up.weight (quantized)
    pub w_down: QMatMul,         // blk.N.ffn_down.weight (quantized)
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
pub fn transformer_block(
    x: &Tensor,
    weights: &QBlockWeights,
    kv_cache_k: Option<&Tensor>,
    kv_cache_v: Option<&Tensor>,
    config: &ModelConfig,
    start_pos: usize,
    rope_cache: Option<&ops::RopeCache>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, seq_len, _hidden) = x.dims3()?;

    // ─── Self-Attention ───────────────────────────────────────────────
    // 1. Pre-attention RMSNorm
    let normed = ops::rms_norm(x, &weights.attn_norm, config.rms_norm_eps)?;

    // 2. Q/K/V projections via quantized matmul (weights stay compressed)
    let q = weights.wq.forward(&normed)?; // [batch, seq, n_heads * head_dim]
    let k = weights.wk.forward(&normed)?; // [batch, seq, n_kv_heads * head_dim]
    let v = weights.wv.forward(&normed)?; // [batch, seq, n_kv_heads * head_dim]

    // 3. Reshape to multi-head: [batch, seq, heads, head_dim]
    let q = q.reshape((batch, seq_len, config.n_heads, config.head_dim))?;
    let k = k.reshape((batch, seq_len, config.n_kv_heads, config.head_dim))?;
    let v = v.reshape((batch, seq_len, config.n_kv_heads, config.head_dim))?;

    // 4. Apply Rotary Position Embeddings to Q and K
    let (q, k) = if let Some(rc) = rope_cache {
        ops::rope_cached(&q, &k, start_pos, config.head_dim, config.rope_theta, rc)?
    } else {
        ops::rope(&q, &k, start_pos, config.head_dim, config.rope_theta)?
    };

    // 5. Concatenate with KV cache (past keys/values for autoregressive generation)
    let (k, v) = if let (Some(cache_k), Some(cache_v)) = (kv_cache_k, kv_cache_v) {
        let k = Tensor::cat(&[cache_k, &k], 1)?;
        let v = Tensor::cat(&[cache_v, &v], 1)?;
        (k, v)
    } else {
        (k, v)
    };

    // 6. Grouped Query Attention
    let attn_out = ops::grouped_query_attention(
        &q,
        &k,
        &v,
        config.n_heads,
        config.n_kv_heads,
    )?;

    // 7. Reshape back and output projection (quantized matmul)
    let attn_out = attn_out.reshape((batch, seq_len, config.n_heads * config.head_dim))?;
    let attn_out = weights.wo.forward(&attn_out)?;

    // 8. Residual connection
    let x = (x + attn_out)?;

    // ─── Feed-Forward Network (SwiGLU) ────────────────────────────────
    // 9. Pre-FFN RMSNorm
    let normed = ops::rms_norm(&x, &weights.ffn_norm, config.rms_norm_eps)?;

    // 10. SiLU-gated FFN (all projections use quantized matmul)
    let ffn_out = ops::silu_ffn(&normed, &weights.w_gate, &weights.w_up, &weights.w_down)?;

    // 11. Residual connection
    let x = (x + ffn_out)?;

    // Return updated hidden state and new K/V for cache
    Ok((x, k, v))
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
    Ok(last_logits.squeeze(0)?)
}

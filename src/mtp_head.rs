//! Native Multi-Token Prediction (MTP) / Medusa Heads Implementation
//!
//! Provides the mathematical pipeline for Predicting multiple tokens ahead in a 
//! single GPU pass. These "draft tokens" are then verified by the main model.
//!
//! Based on:
//! - "Better & Faster LLMs via Multi-token Prediction" (Gloeckle et al., 2024)
//! - Qwen3.6 NEXTN Auxiliary Head Architecture.

use candle_core::{DType, Device, Result, Tensor};
use crate::weight_streamer::WeightStreamer;

/// Single residual block used in MTP/Medusa heads.
/// Architecture: x → SiLU(x @ W1.T) * (x @ W2.T) + x (Swish-Gate Linear Unit)
#[derive(Clone)]
pub struct MtpBlock {
    pub w1: Tensor, // [dim, dim]
    pub w2: Tensor, // [dim, dim]
}

impl MtpBlock {
    pub fn new(w1: Tensor, w2: Tensor) -> Self {
        Self { w1, w2 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.w1.t()?)?.silu()?;
        let proj = x.matmul(&self.w2.t()?)?;
        h.mul(&proj)?.add(x)
    }
}

/// A container for multiple MTP heads, each predicting one position ahead.
pub struct MtpHeadGroup {
    pub blocks: Vec<MtpBlock>,
    pub n_draft_tokens: usize,
    pub hidden_dim: usize,
}

impl MtpHeadGroup {
    /// Load MTP blocks from a WeightStreamer.
    /// Looks for patterns like `mtp_head.{i}.weight0` and `mtp_head.{i}.weight1`.
    pub fn load(
        n_draft_tokens: usize, 
        hidden_dim: usize, 
        streamer: &WeightStreamer, 
        device: &Device
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_draft_tokens);
        
        for i in 0..n_draft_tokens {
            let w1_name = format!("mtp_head.{}.weight0", i);
            let w2_name = format!("mtp_head.{}.weight1", i);
            
            // Try loading. If not present, we skip or could bootstrap.
            // For production, we expect weights.
            let w1 = streamer.load_tensor(&w1_name, device)?;
            let w2 = streamer.load_tensor(&w2_name, device)?;
            
            blocks.push(MtpBlock::new(w1, w2));
        }
        
        Ok(Self { blocks, n_draft_tokens, hidden_dim })
    }

    /// Predict draft tokens for future positions.
    /// 
    /// # Arguments
    /// * `final_hidden` — hidden state from the last layer of the main model [1, hidden_dim]
    /// * `lm_head`     — the main model's output weight [vocab_size, hidden_dim]
    /// 
    /// Returns a tensor of draft logits of shape [n_draft_tokens, vocab_size].
    pub fn predict_logits(&self, final_hidden: &Tensor, lm_head: &Tensor) -> Result<Tensor> {
        let mut head_outputs = Vec::with_capacity(self.n_draft_tokens);
        
        for block in &self.blocks {
            // Each head applies its transform to the BASE hidden state
            let h = block.forward(final_hidden)?;
            head_outputs.push(h);
        }
        
        // Concatenate outputs: [n_draft_tokens, hidden_dim]
        let batch_h = Tensor::cat(&head_outputs, 0)?;
        
        // Project all heads to vocab simultaneously: [n_draft_tokens, vocab_size]
        batch_h.matmul(&lm_head.t()?)
    }
}

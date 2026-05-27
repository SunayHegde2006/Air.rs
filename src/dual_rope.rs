//! Dual RoPE (p-RoPE) — Per-Layer Positional Encoding for Gemma 4 (v0.10.1)
//!
//! Orchestrates high-performance, tensor-resident rotary embeddings:
//! - Pre-computes cos/sin tables on-device for local/global theta.
//! - Eliminates CPU-bound loops and redundant GPU tensor allocations.
//! - Dispatches based on AttentionBackend for hybrid layers.

use crate::attention_backend::AttentionBackend;
use candle_core::{Device, Result, Tensor, D};
use std::sync::Arc;

/// Pre-computed inverse frequency band and cos/sin tensors on-device.
#[derive(Clone)]
pub struct RopeFreqTable {
    pub theta: f64,
    pub head_dim: usize,
    /// Pre-computed cos tensor [max_seq, half_dim] on device.
    pub cos: Option<Tensor>,
    /// Pre-computed sin tensor [max_seq, half_dim] on device.
    pub sin: Option<Tensor>,
}

impl RopeFreqTable {
    /// Create a table config without initializing tensors.
    pub fn new_config(theta: f64, head_dim: usize) -> Self {
        Self { theta, head_dim, cos: None, sin: None }
    }

    /// Actually pre-compute and load tensors to the device.
    pub fn init_on_device(&mut self, max_seq: usize, device: &Device) -> Result<()> {
        let half = self.head_dim / 2;
        
        // 1. Build inv_freq on CPU
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (self.theta as f32).powf(2.0 * i as f32 / self.head_dim as f32))
            .collect();
        let inv_freq_t = Tensor::new(inv_freq, device)?;
        
        // 2. Build position indices on GPU if possible
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions_t = Tensor::new(positions, device)?.unsqueeze(1)?;
        
        // 3. Compute angles: [max_seq, 1] * [1, half] -> [max_seq, half]
        let angles = positions_t.matmul(&inv_freq_t.unsqueeze(0)?)?;
        
        // 4. Cache cos/sin
        self.cos = Some(angles.cos()?);
        self.sin = Some(angles.sin()?);
        
        Ok(())
    }

    /// Slice the cached cos/sin tensors for a specific sequence range.
    pub fn get_slice(&self, start_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.as_ref().ok_or_else(|| candle_core::Error::Msg("RoPE table not initialized".into()))?;
        let sin = self.sin.as_ref().ok_or_else(|| candle_core::Error::Msg("RoPE table not initialized".into()))?;
        
        Ok((
            cos.narrow(0, start_pos, seq_len)?,
            sin.narrow(0, start_pos, seq_len)?
        ))
    }
}

/// Unified Dual RoPE frequency cache for high-performance dispatch.
#[derive(Clone)]
pub struct DualRopeCache {
    pub local:  RopeFreqTable,
    pub global: RopeFreqTable,
    pub head_dim: usize,
}

impl DualRopeCache {
    /// Construct from GGUF metadata strings (no device required yet).
    pub fn from_metadata(
        metadata: &std::collections::HashMap<String, String>,
        head_dim: usize,
    ) -> Self {
        let local_theta = metadata
            .get("gemma4.attention.local_rope_theta")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(10_000.0);
        let global_theta = metadata
            .get("gemma4.attention.global_rope_theta")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1_000_000.0);

        Self {
            local:  RopeFreqTable::new_config(local_theta, head_dim),
            global: RopeFreqTable::new_config(global_theta, head_dim),
            head_dim,
        }
    }

    /// Initialize GPU tensors for the target device and sequence length.
    pub fn init_on_device(&mut self, max_seq: usize, device: &Device) -> Result<()> {
        self.local.init_on_device(max_seq, device)?;
        self.global.init_on_device(max_seq, device)?;
        Ok(())
    }

    /// Optimized entry point for rotary embedding application.
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
        backend: AttentionBackend,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let table = match backend {
            AttentionBackend::SlidingWindow { .. } => &self.local,
            _ => &self.global,
        };
        
        let (cos, sin) = table.get_slice(start_pos, seq_len)?;
        
        // Dispatch to the centralized optimized implementation in ops.rs
        crate::ops::apply_rotary_emb_with_tables(q, k, &cos, &sin)
    }
}

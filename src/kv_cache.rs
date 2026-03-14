//! Interleaved Streaming KV-Cache Manager.
//!
//! During autoregressive generation, each transformer block produces Key and
//! Value vectors that must be retained for future tokens to attend to.
//!
//! The challenge: For large context windows (128k+), the KV cache for all
//! layers doesn't fit in VRAM alongside the model weights.
//!
//! Solution: Store KV cache in System RAM (virtually unlimited), and stream
//! only the current layer's KV slice into VRAM when that layer computes.
//! After computation, save the updated KV back to RAM and free the VRAM.

use candle_core::{Device, Result, Tensor};

/// Per-layer KV cache stored as Candle tensors in system RAM.
pub struct LayerKvCache {
    pub layer_id: usize,
    /// Key cache: [batch, seq_len, n_kv_heads, head_dim] — stored on CPU
    pub k_cache: Option<Tensor>,
    /// Value cache: [batch, seq_len, n_kv_heads, head_dim] — stored on CPU
    pub v_cache: Option<Tensor>,
}

pub struct KvCacheManager {
    layers: Vec<LayerKvCache>,
    device: Device,  // The GPU device to transfer to/from
}

impl KvCacheManager {
    pub fn new(device: Device, num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for id in 0..num_layers {
            layers.push(LayerKvCache {
                layer_id: id,
                k_cache: None,
                v_cache: None,
            });
        }

        Self { device, layers }
    }

    /// Load a specific layer's KV-cache from CPU RAM to GPU VRAM.
    /// Returns None if no cache exists yet (first token / prefill).
    pub fn load_to_device(&self, layer_id: usize) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let layer = &self.layers[layer_id];

        let k = layer
            .k_cache
            .as_ref()
            .map(|t| t.to_device(&self.device))
            .transpose()?;
        let v = layer
            .v_cache
            .as_ref()
            .map(|t| t.to_device(&self.device))
            .transpose()?;

        Ok((k, v))
    }

    /// After computing attention, save the updated K/V tensors back to CPU RAM.
    /// The tensors on GPU will be freed when they go out of scope.
    ///
    /// `new_k` and `new_v` contain ALL keys/values (past + new), already
    /// concatenated by the attention function.
    pub fn save_from_device(
        &mut self,
        layer_id: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<()> {
        let layer = &mut self.layers[layer_id];

        // Move to CPU for storage
        layer.k_cache = Some(new_k.to_device(&Device::Cpu)?);
        layer.v_cache = Some(new_v.to_device(&Device::Cpu)?);

        Ok(())
    }

    /// Get the current sequence length stored in the cache for a given layer.
    pub fn seq_len(&self, layer_id: usize) -> usize {
        self.layers[layer_id]
            .k_cache
            .as_ref()
            .map(|t| t.dim(1).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Clear all cached KV data (e.g., starting a new conversation).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.k_cache = None;
            layer.v_cache = None;
        }
    }

    /// Total memory used by KV cache across all layers (in bytes).
    pub fn memory_usage(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                let k_size = l
                    .k_cache
                    .as_ref()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .unwrap_or(0);
                let v_size = l
                    .v_cache
                    .as_ref()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .unwrap_or(0);
                k_size + v_size
            })
            .sum()
    }
}

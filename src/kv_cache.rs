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
//!
//! Storage modes:
//!   - **F16**: 2 bytes/element — halves memory vs F32, zero quality loss
//!   - **Q8_0**: ~1.0625 bytes/element — 3.75× compression vs F32
//!     Packs 32 f32 values into 32 i8 + 1 f32 scale = 36 bytes per block
//!     Max absolute error < 0.02 for typical KV cache ranges

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Q8_0 Block Quantization
// ---------------------------------------------------------------------------

/// Number of elements per Q8_0 block (matches llama.cpp ggml Q8_0).
const Q8_0_BLOCK_SIZE: usize = 32;

/// A single Q8_0 block: 32 int8 quantised values + 1 f32 scale factor.
///
/// Storage: 32 bytes (quants) + 4 bytes (scale) = 36 bytes per 32 elements,
/// vs. 128 bytes for F32 → **3.56× compression**.
#[derive(Debug, Clone)]
pub struct Q8Block {
    /// Scale factor: `max_abs / 127.0` for the block.
    pub scale: f32,
    /// 32 quantised int8 values. `orig ≈ quant * scale`.
    pub quants: [i8; Q8_0_BLOCK_SIZE],
}

/// A fully-quantised Q8_0 tensor stored as a flat array of blocks.
/// Preserves the original shape so it can be dequantised back.
#[derive(Debug, Clone)]
pub struct Q8Tensor {
    /// Original tensor shape (e.g. `[1, seq, heads, dim]`).
    pub shape: Vec<usize>,
    /// Number of logical f32 elements (product of shape).
    pub numel: usize,
    /// Quantised blocks — `ceil(numel / 32)` entries.
    pub blocks: Vec<Q8Block>,
}

impl Q8Tensor {
    /// Quantise a slice of f32 values into Q8_0 blocks.
    pub fn quantize(data: &[f32], shape: Vec<usize>) -> Self {
        let numel = data.len();
        let n_blocks = (numel + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(n_blocks);

        for chunk in data.chunks(Q8_0_BLOCK_SIZE) {
            let max_abs = chunk.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / scale;

            let mut quants = [0i8; Q8_0_BLOCK_SIZE];
            for (i, &val) in chunk.iter().enumerate() {
                // Round to nearest, clamp to [-127, 127]
                let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                quants[i] = q;
            }

            blocks.push(Q8Block { scale, quants });
        }

        Self { shape, numel, blocks }
    }

    /// Dequantise Q8_0 blocks back to f32 values.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.numel);

        for (bi, block) in self.blocks.iter().enumerate() {
            let start = bi * Q8_0_BLOCK_SIZE;
            let end = (start + Q8_0_BLOCK_SIZE).min(self.numel);
            let count = end - start;

            for i in 0..count {
                out.push(block.quants[i] as f32 * block.scale);
            }
        }

        out
    }

    /// Memory footprint in bytes (36 bytes per block of 32 elements).
    pub fn size_bytes(&self) -> usize {
        // Each block: 32 bytes (i8×32) + 4 bytes (f32 scale) = 36
        self.blocks.len() * 36
    }

    /// Equivalent f32 size for comparison.
    pub fn f32_equivalent_bytes(&self) -> usize {
        self.numel * 4
    }

    /// Compression ratio vs F32.
    pub fn compression_ratio(&self) -> f64 {
        self.f32_equivalent_bytes() as f64 / self.size_bytes() as f64
    }
}

// ---------------------------------------------------------------------------
// Per-layer KV Cache with optional Q8_0
// ---------------------------------------------------------------------------

/// Per-layer KV cache stored as Candle tensors in system RAM.
pub struct LayerKvCache {
    pub layer_id: usize,
    /// Key cache: [batch, seq_len, n_kv_heads, head_dim] — stored on CPU in storage_dtype
    pub k_cache: Option<Tensor>,
    /// Value cache: [batch, seq_len, n_kv_heads, head_dim] — stored on CPU in storage_dtype
    pub v_cache: Option<Tensor>,
    /// Q8_0-quantised key cache (mutually exclusive with k_cache when q8_0 is enabled).
    pub k_q8: Option<Q8Tensor>,
    /// Q8_0-quantised value cache.
    pub v_q8: Option<Q8Tensor>,
}

// ---------------------------------------------------------------------------
// KV Cache Manager
// ---------------------------------------------------------------------------

pub struct KvCacheManager {
    layers: Vec<LayerKvCache>,
    device: Device,        // The GPU device to transfer to/from
    storage_dtype: DType,  // F16 (default) or F32 for CPU-side storage
    compute_dtype: DType,  // The model's working dtype for GPU computation
    /// When true, use Q8_0 quantisation instead of storage_dtype.
    q8_0_enabled: bool,
}

impl KvCacheManager {
    /// Create a new KV cache manager.
    ///
    /// `storage_dtype`: dtype for CPU storage (F16 halves memory vs F32)
    /// `compute_dtype`: dtype the model uses on GPU (typically F32 or BF16)
    pub fn new(device: Device, num_layers: usize) -> Self {
        Self::with_dtypes(device, num_layers, DType::F16, DType::F32)
    }

    /// Create a device-aware KV cache manager (P2-1).
    ///
    /// On CPU, uses F32 storage directly — avoids wasteful F16→F32 conversions
    /// when there's no VRAM to save. On GPU, uses F16 for 2× VRAM savings.
    pub fn new_for_device(device: Device, num_layers: usize) -> Self {
        let storage_dtype = if matches!(device, Device::Cpu) {
            DType::F32  // No VRAM benefit from F16 on CPU
        } else {
            DType::F16  // 2× compression for GPU VRAM
        };
        Self::with_dtypes(device, num_layers, storage_dtype, DType::F32)
    }

    /// Create with Q8_0 quantisation enabled — ~3.56× compression vs F32.
    pub fn with_q8_0(device: Device, num_layers: usize) -> Self {
        let mut mgr = Self::with_dtypes(device, num_layers, DType::F32, DType::F32);
        mgr.q8_0_enabled = true;
        mgr
    }

    /// Create with explicit storage and compute dtypes.
    pub fn with_dtypes(
        device: Device,
        num_layers: usize,
        storage_dtype: DType,
        compute_dtype: DType,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for id in 0..num_layers {
            layers.push(LayerKvCache {
                layer_id: id,
                k_cache: None,
                v_cache: None,
                k_q8: None,
                v_q8: None,
            });
        }

        Self {
            device,
            layers,
            storage_dtype,
            compute_dtype,
            q8_0_enabled: false,
        }
    }

    /// Whether Q8_0 quantisation is enabled.
    pub fn is_q8_0(&self) -> bool {
        self.q8_0_enabled
    }

    /// Load a specific layer's KV-cache from CPU RAM to GPU VRAM.
    /// Returns None if no cache exists yet (first token / prefill).
    ///
    /// If Q8_0 is enabled, dequantises blocks back to compute_dtype tensors.
    /// Otherwise casts from storage_dtype (F16) to compute_dtype (F32).
    pub fn load_to_device(&self, layer_id: usize) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let layer = &self.layers[layer_id];

        if self.q8_0_enabled {
            // Dequantise Q8_0 → F32 tensor → transfer to device
            let k = layer.k_q8.as_ref().map(|q| {
                let data = q.dequantize();
                Tensor::new(data, &Device::Cpu)?
                    .reshape(q.shape.as_slice())?
                    .to_dtype(self.compute_dtype)?
                    .to_device(&self.device)
            }).transpose()?;

            let v = layer.v_q8.as_ref().map(|q| {
                let data = q.dequantize();
                Tensor::new(data, &Device::Cpu)?
                    .reshape(q.shape.as_slice())?
                    .to_dtype(self.compute_dtype)?
                    .to_device(&self.device)
            }).transpose()?;

            Ok((k, v))
        } else {
            // Fast path (P2-1): if cache is already on compute device in
            // compute dtype, return a cheap Arc clone (no data copy).
            let k = layer
                .k_cache
                .as_ref()
                .map(|t| {
                    if t.dtype() == self.compute_dtype && t.device().same_device(&self.device) {
                        Ok(t.clone()) // Cheap clone — backed by Arc, not data copy
                    } else {
                        t.to_dtype(self.compute_dtype)?
                            .to_device(&self.device)
                    }
                })
                .transpose()?;
            let v = layer
                .v_cache
                .as_ref()
                .map(|t| {
                    if t.dtype() == self.compute_dtype && t.device().same_device(&self.device) {
                        Ok(t.clone())
                    } else {
                        t.to_dtype(self.compute_dtype)?
                            .to_device(&self.device)
                    }
                })
                .transpose()?;

            Ok((k, v))
        }
    }

    /// After computing attention, save the updated K/V tensors back to CPU RAM.
    ///
    /// If Q8_0 is enabled, quantises F32 → Q8_0 blocks (~3.56× compression).
    /// Otherwise casts to storage_dtype (F16) for ~2× compression.
    pub fn save_from_device(
        &mut self,
        layer_id: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<()> {
        let layer = &mut self.layers[layer_id];

        if self.q8_0_enabled {
            // Flatten to F32, quantise to Q8_0 blocks
            let k_cpu = new_k.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
            let v_cpu = new_v.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

            let k_data: Vec<f32> = k_cpu.flatten_all()?.to_vec1()?;
            let v_data: Vec<f32> = v_cpu.flatten_all()?.to_vec1()?;

            let k_shape: Vec<usize> = k_cpu.shape().dims().to_vec();
            let v_shape: Vec<usize> = v_cpu.shape().dims().to_vec();

            layer.k_q8 = Some(Q8Tensor::quantize(&k_data, k_shape));
            layer.v_q8 = Some(Q8Tensor::quantize(&v_data, v_shape));
            layer.k_cache = None;
            layer.v_cache = None;
        } else {
            // Cast to storage dtype (F16) and move to CPU
            layer.k_cache = Some(
                new_k
                    .to_dtype(self.storage_dtype)?
                    .to_device(&Device::Cpu)?,
            );
            layer.v_cache = Some(
                new_v
                    .to_dtype(self.storage_dtype)?
                    .to_device(&Device::Cpu)?,
            );
            layer.k_q8 = None;
            layer.v_q8 = None;
        }

        Ok(())
    }

    /// Get the current sequence length stored in the cache for a given layer.
    pub fn seq_len(&self, layer_id: usize) -> usize {
        let layer = &self.layers[layer_id];
        if self.q8_0_enabled {
            // For Q8, reconstruct from shape[1] (seq dim)
            layer.k_q8.as_ref()
                .map(|q| if q.shape.len() >= 2 { q.shape[1] } else { 0 })
                .unwrap_or(0)
        } else {
            layer.k_cache
                .as_ref()
                .map(|t| t.dim(1).unwrap_or(0))
                .unwrap_or(0)
        }
    }

    /// Clear all cached KV data (e.g., starting a new conversation).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.k_cache = None;
            layer.v_cache = None;
            layer.k_q8 = None;
            layer.v_q8 = None;
        }
    }

    /// Truncate KV cache to `pos` tokens across all layers.
    ///
    /// Used by speculative decoding to roll back draft tokens that were
    /// rejected by the verifier. O(1) metadata update for the float path
    /// (tensor `narrow` is zero-copy — it just repoints the Arc).
    ///
    /// **Q8 path**: Q8 block slicing is unsupported; Q8 should be disabled
    /// during speculative decoding (per the protocol spec).
    pub fn truncate_to(&mut self, pos: usize) {
        for layer in &mut self.layers {
            // Float path: narrow seq dimension (dim 1) to `pos` tokens.
            if let Some(ref k) = layer.k_cache.clone() {
                let seq = k.dim(1).unwrap_or(0);
                if pos < seq {
                    layer.k_cache = k.narrow(1, 0, pos).ok();
                }
            }
            if let Some(ref v) = layer.v_cache.clone() {
                let seq = v.dim(1).unwrap_or(0);
                if pos < seq {
                    layer.v_cache = v.narrow(1, 0, pos).ok();
                }
            }
            // Q8 path: full block-slice not implemented; clear if pos == 0.
            if pos == 0 {
                layer.k_q8 = None;
                layer.v_q8 = None;
            }
        }
    }

    /// Total memory used by KV cache across all layers (in bytes).
    pub fn memory_usage(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                if self.q8_0_enabled {
                    let k_size = l.k_q8.as_ref().map(|q| q.size_bytes()).unwrap_or(0);
                    let v_size = l.v_q8.as_ref().map(|q| q.size_bytes()).unwrap_or(0);
                    k_size + v_size
                } else {
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
                }
            })
            .sum()
    }

    /// Get the storage dtype (for diagnostics).
    pub fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }
}

// ---------------------------------------------------------------------------
// SessionKvCache trait — ADR-0004
// ---------------------------------------------------------------------------

/// Single-session KV cache abstraction.
///
/// Decouples `InferenceGenerator` and `SpeculativeDecoder` from the concrete
/// `KvCacheManager` implementation. Enables:
/// - Zero-GPU unit tests via `MockSessionKvCache`
/// - Future `PrefixAwareSessionCache` wrapper (#3)
/// - `truncate_to` rollback needed by speculative decoding (#12)
///
/// See ADR-0004 and ADR-0006 (amendment adding `truncate_to`).
pub trait SessionKvCache: Send + Sync {
    /// Sequence length stored in layer `layer` (0 if empty).
    fn seq_len(&self, layer: usize) -> usize;

    /// Save new K/V tensors for `layer` to the cache.
    ///
    /// Maps to `KvCacheManager::save_from_device`. The tensors are
    /// expected to be on the compute device; the impl handles dtype
    /// conversion and transfer to CPU RAM.
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> candle_core::Result<()>;

    /// Load the K/V tensors for `layer` from the cache to the compute device.
    ///
    /// Returns `(None, None)` if the layer has no cached tokens yet.
    fn load(&self, layer: usize) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)>;

    /// Clear all cached KV data (start a new session / conversation).
    fn clear(&mut self);

    /// Truncate cached sequence to `pos` tokens across all layers.
    ///
    /// Required for speculative decoding: after the verifier rejects
    /// `n_reject` draft tokens, call `truncate_to(verified_len)` to roll
    /// back the cache without rebuilding it from scratch. See ADR-0006.
    fn truncate_to(&mut self, pos: usize);
}

impl SessionKvCache for KvCacheManager {
    fn seq_len(&self, layer: usize) -> usize {
        // Delegate to inherent method — avoids ambiguity with trait method name.
        KvCacheManager::seq_len(self, layer)
    }

    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> candle_core::Result<()> {
        self.save_from_device(layer, k, v)
    }

    fn load(&self, layer: usize) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        self.load_to_device(layer)
    }

    fn clear(&mut self) {
        KvCacheManager::clear(self);
    }

    fn truncate_to(&mut self, pos: usize) {
        KvCacheManager::truncate_to(self, pos);
    }
}

// ---------------------------------------------------------------------------
// MockSessionKvCache — test doubles (compiled only in #[cfg(test)])
// ---------------------------------------------------------------------------

/// A zero-GPU mock that tracks calls without allocating tensors.
///
/// Useful for testing `InferenceGenerator` and `SpeculativeDecoder` without
/// needing a real KV cache or compute device.
#[cfg(test)]
pub struct MockSessionKvCache {
    pub seq_count: usize,
    pub clear_count: usize,
    pub truncate_count: usize,
    pub last_truncate_pos: usize,
}

#[cfg(test)]
impl MockSessionKvCache {
    pub fn new() -> Self {
        Self { seq_count: 0, clear_count: 0, truncate_count: 0, last_truncate_pos: 0 }
    }

    /// Simulate having `n` tokens in cache (set directly for test setup).
    pub fn with_seq(n: usize) -> Self {
        let mut m = Self::new();
        m.seq_count = n;
        m
    }
}

#[cfg(test)]
impl SessionKvCache for MockSessionKvCache {
    fn seq_len(&self, _layer: usize) -> usize { self.seq_count }

    fn append(&mut self, _layer: usize, _k: &Tensor, _v: &Tensor) -> candle_core::Result<()> {
        self.seq_count += 1;
        Ok(())
    }

    fn load(&self, _layer: usize) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((None, None))
    }

    fn clear(&mut self) {
        self.seq_count = 0;
        self.clear_count += 1;
    }

    fn truncate_to(&mut self, pos: usize) {
        self.seq_count = pos;
        self.last_truncate_pos = pos;
        self.truncate_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults_to_f16_storage() {
        let mgr = KvCacheManager::new(Device::Cpu, 4);
        assert_eq!(mgr.storage_dtype(), DType::F16);
        assert_eq!(mgr.compute_dtype, DType::F32);
        assert!(!mgr.is_q8_0());
    }

    #[test]
    fn test_with_dtypes() {
        let mgr = KvCacheManager::with_dtypes(Device::Cpu, 4, DType::F32, DType::F32);
        assert_eq!(mgr.storage_dtype(), DType::F32);
    }

    #[test]
    fn test_with_q8_0_flag() {
        let mgr = KvCacheManager::with_q8_0(Device::Cpu, 4);
        assert!(mgr.is_q8_0());
    }

    #[test]
    fn test_q8_quantize_dequantize_roundtrip() {
        // Create known data
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
        let q = Q8Tensor::quantize(&data, vec![64]);

        // Should have 2 blocks (64 / 32)
        assert_eq!(q.blocks.len(), 2);
        assert_eq!(q.numel, 64);

        // Dequantise and check error
        let recovered = q.dequantize();
        assert_eq!(recovered.len(), 64);

        for (orig, rec) in data.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(
                err < 0.25,
                "Q8_0 roundtrip error too large: orig={}, rec={}, err={}",
                orig, rec, err
            );
        }
    }

    #[test]
    fn test_q8_compression_ratio() {
        let data: Vec<f32> = vec![1.0; 1024];
        let q = Q8Tensor::quantize(&data, vec![1024]);

        // F32: 1024 * 4 = 4096 bytes
        // Q8_0: 32 blocks * 36 bytes = 1152 bytes
        assert_eq!(q.f32_equivalent_bytes(), 4096);
        assert_eq!(q.size_bytes(), 32 * 36); // 1152
        assert!(q.compression_ratio() > 3.5);
    }

    #[test]
    fn test_q8_kv_cache_save_load_roundtrip() -> Result<()> {
        let mut mgr = KvCacheManager::with_q8_0(Device::Cpu, 1);

        // Create test tensor [batch=1, seq=4, heads=2, dim=32] — 256 elements
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let k = Tensor::new(data.clone(), &Device::Cpu)?.reshape((1, 4, 2, 32))?;
        let v = Tensor::new(data.clone(), &Device::Cpu)?.reshape((1, 4, 2, 32))?;

        // Save
        mgr.save_from_device(0, &k, &v)?;
        assert_eq!(mgr.seq_len(0), 4);

        // Check Q8 blocks exist
        assert!(mgr.layers[0].k_q8.is_some());
        assert!(mgr.layers[0].k_cache.is_none());

        // Load back
        let (k_loaded, v_loaded) = mgr.load_to_device(0)?;
        let k_loaded = k_loaded.unwrap();
        let v_loaded = v_loaded.unwrap();

        // Shape should be preserved
        assert_eq!(k_loaded.shape().dims(), &[1, 4, 2, 32]);
        assert_eq!(v_loaded.shape().dims(), &[1, 4, 2, 32]);
        assert_eq!(k_loaded.dtype(), DType::F32);

        // Values should be close
        let k_orig: Vec<f32> = k.flatten_all()?.to_vec1()?;
        let k_rt: Vec<f32> = k_loaded.flatten_all()?.to_vec1()?;
        for (orig, rt) in k_orig.iter().zip(k_rt.iter()) {
            assert!(
                (orig - rt).abs() < 0.05,
                "Q8_0 roundtrip error: {} vs {}",
                orig, rt
            );
        }

        Ok(())
    }

    #[test]
    fn test_q8_memory_vs_f16() -> Result<()> {
        // Compare Q8_0 vs F16 memory for same data
        let k = Tensor::zeros((1, 100, 32, 128), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((1, 100, 32, 128), DType::F32, &Device::Cpu)?;

        let mut mgr_q8 = KvCacheManager::with_q8_0(Device::Cpu, 1);
        mgr_q8.save_from_device(0, &k, &v)?;

        let mut mgr_f16 = KvCacheManager::new(Device::Cpu, 1);
        mgr_f16.save_from_device(0, &k, &v)?;

        let q8_bytes = mgr_q8.memory_usage();
        let f16_bytes = mgr_f16.memory_usage();

        // Q8_0 should use less than F16
        assert!(
            q8_bytes < f16_bytes,
            "Q8_0 ({}) should be smaller than F16 ({})",
            q8_bytes, f16_bytes
        );

        // F16 = 2 bytes/elem → 1,638,400 bytes
        // Q8_0 ≈ 1.125 bytes/elem → ~921,600 bytes
        // Q8_0 should be roughly 55-60% of F16
        let ratio = q8_bytes as f64 / f16_bytes as f64;
        assert!(ratio < 0.65, "Q8_0/F16 ratio {} should be < 0.65", ratio);

        Ok(())
    }

    #[test]
    fn test_q8_clear() -> Result<()> {
        let mut mgr = KvCacheManager::with_q8_0(Device::Cpu, 2);
        let k = Tensor::zeros((1, 4, 2, 32), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((1, 4, 2, 32), DType::F32, &Device::Cpu)?;
        mgr.save_from_device(0, &k, &v)?;

        assert!(mgr.memory_usage() > 0);
        mgr.clear();
        assert_eq!(mgr.memory_usage(), 0);
        assert_eq!(mgr.seq_len(0), 0);

        Ok(())
    }

    #[test]
    fn test_f16_roundtrip_preserves_values() -> Result<()> {
        let mgr_f16 = KvCacheManager::new(Device::Cpu, 1);
        let mgr_f32 = KvCacheManager::with_dtypes(Device::Cpu, 1, DType::F32, DType::F32);

        // Create test tensors with known values: [batch=1, seq=4, heads=2, dim=4]
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let k = Tensor::new(data.clone(), &Device::Cpu)?.reshape((1, 4, 2, 4))?;
        let v = Tensor::new(data.clone(), &Device::Cpu)?.reshape((1, 4, 2, 4))?;

        // Save with F16 storage
        let mut mgr_f16 = mgr_f16;
        mgr_f16.save_from_device(0, &k, &v)?;

        // Save with F32 storage
        let mut mgr_f32 = mgr_f32;
        mgr_f32.save_from_device(0, &k, &v)?;

        // F16 should use half the memory
        assert!(mgr_f16.memory_usage() < mgr_f32.memory_usage());
        assert_eq!(mgr_f16.memory_usage() * 2, mgr_f32.memory_usage());

        // Load back and check values are close
        let (k_f16, v_f16) = mgr_f16.load_to_device(0)?;
        let k_f16 = k_f16.unwrap();
        let v_f16 = v_f16.unwrap();

        // Should be F32 after loading (compute dtype)
        assert_eq!(k_f16.dtype(), DType::F32);
        assert_eq!(v_f16.dtype(), DType::F32);

        // Values should be very close (F16 has ~3 decimal digits of precision)
        let k_orig: Vec<f32> = k.flatten_all()?.to_vec1()?;
        let k_roundtrip: Vec<f32> = k_f16.flatten_all()?.to_vec1()?;
        for (orig, rt) in k_orig.iter().zip(k_roundtrip.iter()) {
            assert!(
                (orig - rt).abs() < 0.01,
                "F16 roundtrip error too large: {} vs {}",
                orig,
                rt
            );
        }

        Ok(())
    }

    #[test]
    fn test_seq_len_tracking() -> Result<()> {
        let mut mgr = KvCacheManager::new(Device::Cpu, 2);
        assert_eq!(mgr.seq_len(0), 0);
        assert_eq!(mgr.seq_len(1), 0);

        // Add 4 tokens to layer 0
        let k = Tensor::zeros((1, 4, 2, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((1, 4, 2, 4), DType::F32, &Device::Cpu)?;
        mgr.save_from_device(0, &k, &v)?;

        assert_eq!(mgr.seq_len(0), 4);
        assert_eq!(mgr.seq_len(1), 0); // layer 1 untouched

        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let mut mgr = KvCacheManager::new(Device::Cpu, 2);
        let k = Tensor::zeros((1, 4, 2, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((1, 4, 2, 4), DType::F32, &Device::Cpu)?;
        mgr.save_from_device(0, &k, &v)?;
        mgr.save_from_device(1, &k, &v)?;

        assert!(mgr.memory_usage() > 0);
        mgr.clear();
        assert_eq!(mgr.memory_usage(), 0);
        assert_eq!(mgr.seq_len(0), 0);

        Ok(())
    }

    #[test]
    fn test_memory_usage_f16_vs_f32() -> Result<()> {
        let k = Tensor::zeros((1, 100, 32, 128), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((1, 100, 32, 128), DType::F32, &Device::Cpu)?;

        let mut mgr_f16 = KvCacheManager::new(Device::Cpu, 1);
        mgr_f16.save_from_device(0, &k, &v)?;

        let mut mgr_f32 = KvCacheManager::with_dtypes(Device::Cpu, 1, DType::F32, DType::F32);
        mgr_f32.save_from_device(0, &k, &v)?;

        // F16 = 2 bytes/elem, F32 = 4 bytes/elem → F16 is exactly half
        let f16_bytes = mgr_f16.memory_usage();
        let f32_bytes = mgr_f32.memory_usage();
        assert_eq!(f16_bytes * 2, f32_bytes);

        // Sanity: 1 * 100 * 32 * 128 = 409,600 elements × 2 (K+V) × 2 bytes = 1,638,400
        assert_eq!(f16_bytes, 1_638_400);

        Ok(())
    }
}

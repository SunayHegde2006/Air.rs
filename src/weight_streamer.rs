//! Weight Streamer — S.L.I.P. core: mmap + per-layer QMatMul streaming.
//!
//! Achieves RSS ≪ file_size by:
//! 1. mmap the entire GGUF file (Virtual = file_size, RSS ≈ 0)
//! 2. Per-layer: read quantized bytes → QMatMul (RSS += 1 layer)
//! 3. After forward pass: drop QBlockWeights (RSS -= 1 layer)
//! 4. madvise(DONTNEED) / VirtualUnlock to release mmap pages from physical RAM
//!
//! Double-buffer pipelining (P1-1):
//! The `PipelinedLoader` overlaps IO for layer N+1 with compute on layer N.
//! ```text
//! Sequential: [IO_L0][Compute_L0][IO_L1][Compute_L1]...
//! Pipelined:  [IO_L0][Compute_L0 | IO_L1][Compute_L1 | IO_L2]...
//! ```
//!
//! Steady-state RSS ≈ Layer_active + Layer_prefetch + embeddings + KV_cache

use crate::model::QBlockWeights;
use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

// ─── Windows FFI for PrefetchVirtualMemory / VirtualUnlock ──────────────
//
// PrefetchVirtualMemory is the Windows equivalent of madvise(MADV_WILLNEED).
// Available on Windows 8+ / Server 2012+. We call it via raw FFI to avoid
// pulling in the full `windows` crate (200+ MB compile-time dependency).
//
// VirtualUnlock is used as the rough equivalent of madvise(MADV_DONTNEED).
// It tells the OS that the pages can be paged out on memory pressure.

#[cfg(target_os = "windows")]
mod win_prefetch {
    use std::ffi::c_void;

    /// WIN32_MEMORY_RANGE_ENTRY for PrefetchVirtualMemory.
    #[repr(C)]
    pub struct MemoryRangeEntry {
        pub virtual_address: *mut c_void,
        pub number_of_bytes: usize,
    }

    // We use GetCurrentProcess() which returns a pseudo-handle (-1).
    // This is always valid and doesn't need to be closed.
    const CURRENT_PROCESS: isize = -1;

    extern "system" {
        /// Prefetch virtual memory pages into the working set.
        /// Available: Windows 8+ / Server 2012+.
        fn PrefetchVirtualMemory(
            h_process: isize,
            number_of_entries: usize,
            virtual_addresses: *const MemoryRangeEntry,
            flags: u32,
        ) -> i32;

        /// Unlock pages previously locked with VirtualLock.
        /// Also useful as a hint to the OS that pages can be evicted.
        fn VirtualUnlock(
            lp_address: *const c_void,
            dw_size: usize,
        ) -> i32;
    }

    /// Ask Windows to prefetch the given byte range into physical RAM.
    /// Best-effort: failure is silently ignored.
    pub fn prefetch(base_ptr: *const u8, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let entry = MemoryRangeEntry {
            virtual_address: unsafe { base_ptr.add(offset) as *mut c_void },
            number_of_bytes: len,
        };
        unsafe {
            // Return value: nonzero = success. We ignore errors —
            // worst case, the OS page-faults the data in on access.
            PrefetchVirtualMemory(CURRENT_PROCESS, 1, &entry, 0);
        }
    }

    /// Hint to Windows that these pages can be evicted on memory pressure.
    /// Best-effort: failure is silently ignored.
    pub fn release(base_ptr: *const u8, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        unsafe {
            let ptr = base_ptr.add(offset) as *const c_void;
            // VirtualUnlock on non-locked pages returns ERROR_NOT_LOCKED,
            // which is fine — it still serves as a soft eviction hint
            // when combined with the working set trimmer.
            VirtualUnlock(ptr, len);
        }
    }
}

/// Streams model weights from an mmap'd GGUF file one layer at a time.
///
/// The entire file is mapped into virtual address space (VM = file_size),
/// but physical RAM (RSS) only grows when tensors are materialized via
/// `load_layer()`, and shrinks when they're dropped + `release_layer()`.
pub struct WeightStreamer {
    mmap: Mmap,
    content: gguf_file::Content,
    n_layers: usize,
}

impl WeightStreamer {
    /// Open a GGUF file and mmap it for streaming.
    ///
    /// After this call: Virtual memory = file_size, RSS ≈ 0.
    /// No tensor data is loaded until explicitly requested.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;

        // Safety: file must not be modified while mapped
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| "Failed to mmap GGUF file")?;

        // Parse GGUF header from mmap'd bytes
        let mut cursor = Cursor::new(mmap.as_ref());
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {e}"))?;

        // Extract layer count from metadata
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| {
                if let gguf_file::Value::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("llama");

        let n_layers = content
            .metadata
            .get(&format!("{arch}.block_count"))
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                gguf_file::Value::U64(n) => Some(*n as usize),
                gguf_file::Value::I32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(32);

        println!(
            "⚡ WeightStreamer: mmap'd {} ({:.2} GB virtual, RSS ≈ 0)",
            path.display(),
            mmap.len() as f64 / 1_073_741_824.0
        );
        println!(
            "   {} layers, {} tensors — streaming on demand",
            n_layers,
            content.tensor_infos.len()
        );

        Ok(Self {
            mmap,
            content,
            n_layers,
        })
    }

    /// Number of transformer layers in the model.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Access the parsed GGUF content (for metadata/tokenizer extraction).
    pub fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    // ─── Tensor Loading ──────────────────────────────────────────────

    /// Load one transformer layer's weights as QBlockWeights.
    ///
    /// RSS increases by ~130 MB (7B Q4_K_M) per call.
    /// Drop the returned QBlockWeights to free that memory.
    pub fn load_layer(&self, layer_id: usize, device: &Device) -> Result<QBlockWeights> {
        let mut cursor = Cursor::new(self.mmap.as_ref());
        let pfx = format!("blk.{layer_id}");

        Ok(QBlockWeights {
            attn_norm: self.read_dequantized(
                &mut cursor,
                &format!("{pfx}.attn_norm.weight"),
                device,
            )?,
            wq: self.read_qmatmul(&mut cursor, &format!("{pfx}.attn_q.weight"), device)?,
            wk: self.read_qmatmul(&mut cursor, &format!("{pfx}.attn_k.weight"), device)?,
            wv: self.read_qmatmul(&mut cursor, &format!("{pfx}.attn_v.weight"), device)?,
            wo: self.read_qmatmul(
                &mut cursor,
                &format!("{pfx}.attn_output.weight"),
                device,
            )?,
            // C1: Probe for optional QKV biases (Qwen 2.5/3, QwQ, DeepSeek-R1-Distill Qwen-based).
            // GGUF tensor names: blk.N.attn_q.bias / attn_k.bias / attn_v.bias
            // Returns None for bias-free architectures (Llama, Mistral, Phi-3+).
            q_bias: self
                .read_dequantized(&mut cursor, &format!("{pfx}.attn_q.bias"), device)
                .ok(),
            k_bias: self
                .read_dequantized(&mut cursor, &format!("{pfx}.attn_k.bias"), device)
                .ok(),
            v_bias: self
                .read_dequantized(&mut cursor, &format!("{pfx}.attn_v.bias"), device)
                .ok(),
            // LayerNorm bias tensors (Falcon). Optional — None for all RMSNorm architectures.
            attn_norm_bias: self
                .read_dequantized(&mut cursor, &format!("{pfx}.attn_norm.bias"), device)
                .ok(),
            ffn_norm_bias: self
                .read_dequantized(&mut cursor, &format!("{pfx}.ffn_norm.bias"), device)
                .ok(),
            ffn_norm: self.read_dequantized(
                &mut cursor,
                &format!("{pfx}.ffn_norm.weight"),
                device,
            )?,
            w_gate: self.read_qmatmul(&mut cursor, &format!("{pfx}.ffn_gate.weight"), device)?,
            w_up: self.read_qmatmul(&mut cursor, &format!("{pfx}.ffn_up.weight"), device)?,
            w_down: self.read_qmatmul(&mut cursor, &format!("{pfx}.ffn_down.weight"), device)?,
        })
    }

    /// Load the token embedding table (dequantized to F32).
    ///
    /// Kept in memory for the entire session — every token lookup needs it.
    pub fn load_embedding(&self, device: &Device) -> Result<Tensor> {
        let mut cursor = Cursor::new(self.mmap.as_ref());
        self.read_dequantized(&mut cursor, "token_embd.weight", device)
    }

    /// Load the final output norm and lm_head projection.
    ///
    /// Returns (norm_weight, lm_head_qmatmul). Kept in memory for the session.
    ///
    /// Handles multiple GGUF naming conventions:
    /// - `output.weight` — standard (Llama 7B+, Mistral, etc.)
    /// - `lm_head.weight` — alternative naming (some community quantizations)
    /// - Tied embeddings — `token_embd.weight` reused as lm_head (Llama 3.2-3B, Phi-3, etc.)
    pub fn load_output(&self, device: &Device) -> Result<(Tensor, QMatMul)> {
        let mut cursor = Cursor::new(self.mmap.as_ref());
        let norm = self.read_dequantized(&mut cursor, "output_norm.weight", device)?;

        // Try standard names first, fall back to tied embeddings
        let lm_head_names = ["output.weight", "lm_head.weight", "token_embd.weight"];
        let mut lm_head = None;
        for name in &lm_head_names {
            match self.read_qmatmul(&mut cursor, name, device) {
                Ok(q) => {
                    if *name == "token_embd.weight" {
                        println!("   ℹ Using tied embeddings (token_embd.weight) as lm_head");
                    }
                    lm_head = Some(q);
                    break;
                }
                Err(_) => continue,
            }
        }

        let lm_head = lm_head.ok_or_else(|| {
            anyhow::anyhow!(
                "Cannot find lm_head weights — tried: {}. \
                 This model's GGUF may use an unsupported naming convention.",
                lm_head_names.join(", ")
            )
        })?;

        Ok((norm, lm_head))
    }

    // ─── Page Management (S.L.I.P. RSS Control) ─────────────────────

    /// Prefetch next layer's mmap pages into OS page cache.
    ///
    /// Called while GPU is busy with current layer — hides SSD latency.
    pub fn prefetch_layer(&self, layer_id: usize) {
        if layer_id >= self.n_layers {
            return;
        }
        let (start, len) = self.layer_byte_range(layer_id);
        if len == 0 {
            return;
        }

        #[cfg(unix)]
        unsafe {
            let ptr = self.mmap.as_ptr().add(start) as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }

        #[cfg(target_os = "windows")]
        {
            win_prefetch::prefetch(self.mmap.as_ptr(), start, len);
        }
    }

    /// Advise OS to reclaim physical RAM pages for a completed layer.
    ///
    /// After QBlockWeights is dropped (heap freed), this also releases
    /// the mmap page cache pages, keeping RSS minimal.
    pub fn release_layer(&self, layer_id: usize) {
        if layer_id >= self.n_layers {
            return;
        }
        let (start, len) = self.layer_byte_range(layer_id);
        if len == 0 {
            return;
        }

        #[cfg(unix)]
        unsafe {
            let ptr = self.mmap.as_ptr().add(start) as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_DONTNEED);
        }

        #[cfg(target_os = "windows")]
        {
            win_prefetch::release(self.mmap.as_ptr(), start, len);
        }
    }

    // ─── Internal Helpers ────────────────────────────────────────────

    /// Read a tensor as QMatMul (quantized matmul, no F32 expansion).
    fn read_qmatmul(
        &self,
        cursor: &mut Cursor<&[u8]>,
        name: &str,
        device: &Device,
    ) -> Result<QMatMul> {
        let qtensor = self
            .content
            .tensor(cursor, name, device)
            .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}': {e}"))?;
        Ok(QMatMul::QTensor(Arc::new(qtensor)))
    }

    /// Read a tensor and dequantize to F32 Tensor.
    /// Used for small tensors (norm weights) that need element-wise ops.
    fn read_dequantized(
        &self,
        cursor: &mut Cursor<&[u8]>,
        name: &str,
        device: &Device,
    ) -> Result<Tensor> {
        let qtensor = self
            .content
            .tensor(cursor, name, device)
            .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}': {e}"))?;
        qtensor
            .dequantize(device)
            .map_err(|e| anyhow::anyhow!("Failed to dequantize '{name}': {e}"))
    }

    /// Compute the byte range in the mmap for all tensors in a given layer.
    fn layer_byte_range(&self, layer_id: usize) -> (usize, usize) {
        let prefix = format!("blk.{layer_id}.");
        let mut min_start = usize::MAX;
        let mut max_end = 0usize;

        for (name, info) in &self.content.tensor_infos {
            if name.starts_with(&prefix) {
                let abs_offset = (self.content.tensor_data_offset + info.offset) as usize;
                let elem_count = info.shape.elem_count();
                let block_size = info.ggml_dtype.block_size();
                let type_size = info.ggml_dtype.type_size();
                let size = (elem_count / block_size) * type_size;
                let end = abs_offset + size;

                min_start = min_start.min(abs_offset);
                max_end = max_end.max(end);
            }
        }

        if min_start == usize::MAX {
            return (0, 0);
        }

        (min_start, max_end - min_start)
    }
}

// ---------------------------------------------------------------------------
// Double-Buffered Pipeline Loader (P1-1)
// ---------------------------------------------------------------------------

/// Overlaps weight loading of layer N+1 with compute on layer N.
///
/// Usage:
/// ```text
/// let loader = PipelinedLoader::new(&streamer, n_layers, &device);
/// for layer_id in 0..n_layers {
///     let weights = loader.take_current(layer_id, &streamer, &device)?;
///     // Compute with `weights` — next layer is loading in background
///     // (PipelinedLoader::take_current handles the join + spawn internally)
/// }
/// ```
///
/// When IO ≥ Compute, this nearly doubles throughput. The pipeline
/// efficiency ρ should approach 1.0.
pub struct PipelinedLoader {
    /// Pre-loaded weights for the current layer.
    /// `None` after `take_current()` has been called.
    buffered: Option<QBlockWeights>,
    /// Total number of layers.
    n_layers: usize,
}

impl PipelinedLoader {
    /// Create a new pipelined loader and pre-load layer 0.
    pub fn new(streamer: &WeightStreamer, device: &Device) -> Result<Self> {
        // Eagerly load layer 0 — no background thread for the first layer
        let weights = streamer.load_layer(0, device)?;
        Ok(Self {
            buffered: Some(weights),
            n_layers: streamer.n_layers(),
        })
    }

    /// Take the pre-loaded weights for `layer_id` and kick off background
    /// loading of `layer_id + 1` (if it exists).
    ///
    /// This must be called in order: 0, 1, 2, ..., n_layers-1.
    ///
    /// Returns `(current_weights, next_loader_state)` — the caller should
    /// compute with `current_weights`, then call `collect_next()` after.
    pub fn take_current(&mut self, layer_id: usize) -> Result<QBlockWeights> {
        self.buffered
            .take()
            .ok_or_else(|| anyhow::anyhow!(
                "PipelinedLoader: no buffered weights for layer {} (double-take or out-of-order call)",
                layer_id
            ))
    }

    /// Load the next layer synchronously and buffer it.
    ///
    /// Call this after compute on the current layer completes.
    /// In the future, this can be moved to a background thread using
    /// `std::thread::scope`.
    ///
    /// For now, we use the simpler approach of prefetch + sync load:
    /// - `prefetch_layer(N+1)` issues madvise WILLNEED (async page-in)
    /// - After compute on layer N finishes, `load_layer(N+1)` materializes
    ///   the prefetched pages — which should be a fast page-table hit
    pub fn prepare_next(
        &mut self,
        next_layer_id: usize,
        streamer: &WeightStreamer,
        device: &Device,
    ) -> Result<()> {
        if next_layer_id < self.n_layers {
            // prefetch was already called by the caller before compute
            let weights = streamer.load_layer(next_layer_id, device)?;
            self.buffered = Some(weights);
        }
        Ok(())
    }

    /// Run a full layer loop with pipelined loading.
    ///
    /// `compute_fn` is called for each layer with (layer_id, weights).
    /// While `compute_fn` runs for layer N, layer N+1's mmap pages are being
    /// prefetched. After compute, the next layer is materialized from the
    /// warm page cache.
    ///
    /// This is the recommended way to use PipelinedLoader for maximum overlap.
    pub fn run_layers<F>(
        streamer: &WeightStreamer,
        device: &Device,
        n_layers: usize,
        mut compute_fn: F,
    ) -> Result<()>
    where
        F: FnMut(usize, QBlockWeights) -> Result<()>,
    {
        let mut loader = Self::new(streamer, device)?;

        for layer_id in 0..n_layers {
            // Issue prefetch hint for next layer (async page-in)
            if layer_id + 1 < n_layers {
                streamer.prefetch_layer(layer_id + 1);
            }

            // Take pre-loaded weights
            let weights = loader.take_current(layer_id)?;

            // Compute on current layer
            compute_fn(layer_id, weights)?;

            // Materialize next layer from (hopefully warm) page cache
            if layer_id + 1 < n_layers {
                loader.prepare_next(layer_id + 1, streamer, device)?;
            }

            // Release previous layer pages
            if layer_id > 0 {
                streamer.release_layer(layer_id - 1);
            }
        }

        // Release the last layer
        if n_layers > 0 {
            streamer.release_layer(n_layers - 1);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipelined_loader_struct_creation() {
        // Verify PipelinedLoader can be created with buffered weights set to None
        let loader = PipelinedLoader {
            buffered: None,
            n_layers: 32,
        };
        assert_eq!(loader.n_layers, 32);
        assert!(loader.buffered.is_none());
    }

    #[test]
    fn test_pipelined_loader_take_current_error_on_empty() {
        let mut loader = PipelinedLoader {
            buffered: None,
            n_layers: 32,
        };
        let result = loader.take_current(0);
        match result {
            Err(e) => assert!(
                e.to_string().contains("no buffered weights"),
                "Expected 'no buffered weights' error, got: {}",
                e
            ),
            Ok(_) => panic!("Expected error from take_current on empty buffer"),
        }
    }
}

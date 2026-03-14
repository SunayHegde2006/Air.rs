//! Weight Streamer — S.L.I.P. core: mmap + per-layer QMatMul streaming.
//!
//! Achieves RSS ≪ file_size by:
//! 1. mmap the entire GGUF file (Virtual = file_size, RSS ≈ 0)
//! 2. Per-layer: read quantized bytes → QMatMul (RSS += 1 layer)
//! 3. After forward pass: drop QBlockWeights (RSS -= 1 layer)
//! 4. madvise(DONTNEED) to release mmap pages from physical RAM
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
    pub fn load_output(&self, device: &Device) -> Result<(Tensor, QMatMul)> {
        let mut cursor = Cursor::new(self.mmap.as_ref());
        let norm = self.read_dequantized(&mut cursor, "output_norm.weight", device)?;
        let lm_head = self.read_qmatmul(&mut cursor, "output.weight", device)?;
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

        #[cfg(target_os = "linux")]
        unsafe {
            let ptr = self.mmap.as_ptr().add(start) as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }

        // Windows: OS working set manager handles prefetch automatically.
        // Future optimization: PrefetchVirtualMemory() for explicit control.
        #[cfg(target_os = "windows")]
        {
            let _ = (start, len); // suppress unused warnings
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

        #[cfg(target_os = "linux")]
        unsafe {
            let ptr = self.mmap.as_ptr().add(start) as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_DONTNEED);
        }

        #[cfg(target_os = "windows")]
        {
            let _ = (start, len);
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

//! Weight streamer — handles lazy H2D weight loading (S.L.I.P. protocol).
//!
//! Streams quantized transformer-block weights on demand from an mmap'd GGUF
//! file, keeping RSS proportional to *active* layer count rather than total
//! model size.  Supports all Air.rs model families including hybrid
//! Qwen3.6 DeltaNet layers.

use crate::model::QBlockWeights;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Tensor};
use std::io::Cursor;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WeightStreamer
// ---------------------------------------------------------------------------

/// Lazy weight streamer.  Open once; call `load_layer` per transformer block.
pub struct WeightStreamer {
    mmap: Arc<memmap2::Mmap>,
    content: gguf_file::Content,
    n_layers: usize,
    #[allow(dead_code)]
    arch: String,
    /// Raw file descriptor for posix_fadvise
    #[cfg(unix)]
    raw_fd: std::os::unix::io::RawFd,
    /// Asymmetric Residency — Attention weights pinned in VRAM.
    pinned_attn: std::sync::Mutex<Vec<Option<crate::model::QBlockWeights>>>,
}

impl WeightStreamer {
    /// Open a GGUF file and index its tensors.  The file is mmap'd —
    /// physical I/O is deferred until a tensor is first accessed.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file: {:?}: {e}", path.as_ref()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Parse GGUF header in a scoped block so the borrow on `mmap` ends
        // before we move it into `Arc::new(mmap)` below.
        let content = {
            let mut cursor = Cursor::new(&mmap[..]);
            gguf_file::Content::read(&mut cursor)
                .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {e}"))?
        };

        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| match v {
                gguf_file::Value::String(s) => Some(s.as_str()),
                _ => None,
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
            path.as_ref().display(),
            mmap.len() as f64 / 1_073_741_824.0
        );
        println!(
            "   {} layers, {} tensors — streaming on demand",
            n_layers,
            content.tensor_infos.len()
        );

        let arch_string = arch.to_string();
        let pinned_attn = std::sync::Mutex::new(vec![None; n_layers]);

        #[cfg(unix)]
        let raw_fd = {
            use std::os::unix::io::AsRawFd;
            file.as_raw_fd()
        };

        Ok(Self {
            mmap: Arc::new(mmap),
            content,
            n_layers,
            arch: arch_string,
            #[cfg(unix)]
            raw_fd,
            pinned_attn,
        })
    }

    /// Number of transformer layers in this model.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Access the parsed GGUF content (for metadata/tokenizer extraction).
    pub fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    // ─── Per-layer tensor loading ─────────────────────────────────────────

    /// Stream one transformer block's weights.
    pub fn load_layer(&self, layer_id: usize, device: &Device, tp: Option<&crate::tensor_parallel::TensorParallelConfig>) -> Result<QBlockWeights> {
        // Standard load path with FastPath fadvise
        self.prefetch_layer(layer_id);
        
        let data: &[u8] = &self.mmap[..];
        let mut c = Cursor::new(data);
        let p = format!("blk.{layer_id}");

        // ── Attention norm ───────────────────────────────────────────────
        let attn_norm = self.dequant(&mut c, &format!("{p}.attn_norm.weight"), device)?;

        // ── Separate Q/K/V projections (GQA / standard layers) ──────────
        let mut wq = self.qmatmul(&mut c, &format!("{p}.attn_q.weight"), device).ok();
        let mut wk = self.qmatmul(&mut c, &format!("{p}.attn_k.weight"), device).ok();
        let mut wv = self.qmatmul(&mut c, &format!("{p}.attn_v.weight"), device).ok();

        if let (Some(tp_cfg), Some(q), Some(k), Some(v)) = (tp, wq.as_mut(), wk.as_mut(), wv.as_mut()) {
            if tp_cfg.is_active() {
               *q = crate::tensor_parallel::shard_column(q, tp_cfg.rank, tp_cfg.tp_size)?;
               *k = crate::tensor_parallel::shard_column(k, tp_cfg.rank, tp_cfg.tp_size)?;
               *v = crate::tensor_parallel::shard_column(v, tp_cfg.rank, tp_cfg.tp_size)?;
            }
        }

        // ── Output projection ────────────────────────────────────────────
        let mut wo = self.qmatmul(&mut c, &format!("{p}.attn_output.weight"), device)
            .or_else(|_| self.qmatmul(&mut c, &format!("{p}.attn_out.weight"), device))
            .or_else(|_| self.qmatmul(&mut c, &format!("{p}.ssm_out.weight"), device))
            .ok();
            
        if let (Some(tp_cfg), Some(o)) = (tp, wo.as_mut()) {
            if tp_cfg.is_active() {
               *o = crate::tensor_parallel::shard_row(o, tp_cfg.rank, tp_cfg.tp_size)?;
            }
        }

        // ── Fused QKV + gate (DeltaNet hybrid layers) ───────────────────
        let attn_qkv  = self.qmatmul(&mut c, &format!("{p}.attn_qkv.weight"),  device).ok();
        let attn_gate = self.qmatmul(&mut c, &format!("{p}.attn_gate.weight"), device).ok();

        // ── QKV biases (Qwen2 / Qwen2.5 / QwQ) ─────────────────────────
        let q_bias = self.dequant(&mut c, &format!("{p}.attn_q.bias"),  device).ok();
        let k_bias = self.dequant(&mut c, &format!("{p}.attn_k.bias"),  device).ok();
        let v_bias = self.dequant(&mut c, &format!("{p}.attn_v.bias"),  device).ok();

        // ── LayerNorm biases (Falcon) ────────────────────────────────────
        let attn_norm_bias = self.dequant(&mut c, &format!("{p}.attn_norm.bias"), device).ok();
        let ffn_norm_bias  = self.dequant(&mut c, &format!("{p}.ffn_norm.bias"),  device).ok();

        // ── SSM / DeltaNet tensors (Qwen3.6 hybrid layers) ──────────────
        let ssm_alpha  = self.dequant(&mut c, &format!("{p}.ssm_alpha.weight"),  device).ok();
        let ssm_beta   = self.dequant(&mut c, &format!("{p}.ssm_beta.weight"),   device).ok();
        let ssm_conv1d = self.dequant(&mut c, &format!("{p}.ssm_conv1d.weight"), device).ok();
        let ssm_dt_bias = self.dequant(&mut c, &format!("{p}.ssm_dt.bias"),      device).ok();
        let ssm_norm   = self.dequant(&mut c, &format!("{p}.ssm_norm.weight"),   device).ok();
        let ssm_a      = self.dequant(&mut c, &format!("{p}.ssm_a"),             device).ok();
        let ssm_out    = wo.clone();

        // ── FFN norm + projections (Async Overlapped Path) ──────────────
        // Trigger prefetch for the NEXT layer while we load this one
        self.prefetch_layer((layer_id + 1) % self.n_layers);

        let ffn_norm = self.dequant(&mut c, &format!("{p}.post_attention_norm.weight"), device)
            .or_else(|_| self.dequant(&mut c, &format!("{p}.ffn_norm.weight"), device))?;
        
        let mut w_gate   = self.qmatmul(&mut c, &format!("{p}.ffn_gate.weight"), device).ok();
        let mut w_up     = self.qmatmul(&mut c, &format!("{p}.ffn_up.weight"),   device).ok();
        let mut w_down   = self.qmatmul(&mut c, &format!("{p}.ffn_down.weight"), device).ok();

        if let (Some(tp_cfg), Some(g), Some(u), Some(d)) = (tp, w_gate.as_mut(), w_up.as_mut(), w_down.as_mut()) {
            if tp_cfg.is_active() {
               *g = crate::tensor_parallel::shard_column(g, tp_cfg.rank, tp_cfg.tp_size)?;
               *u = crate::tensor_parallel::shard_column(u, tp_cfg.rank, tp_cfg.tp_size)?;
               *d = crate::tensor_parallel::shard_row(d, tp_cfg.rank, tp_cfg.tp_size)?;
            }
        }

        // ── MoE Router (Gemma 4 / Mixtral / Qwen) ───────────────────────
        let ffn_router = self.qmatmul(&mut c, &format!("{p}.ffn_gate_inp.weight"), device).ok();
        
        let weights = QBlockWeights {
            attn_norm, wq, wk, wv, wo, q_bias, k_bias, v_bias, 
            attn_norm_bias, ffn_norm_bias,
            ffn_norm, 
            w_gate, w_up, w_down,
            ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt_bias, ssm_norm, ssm_out,
            attn_gate, attn_qkv,
            ffn_router, ffn_exps_gate: None, ffn_exps_up: None, ffn_exps_down: None,
        };

        Ok(weights)
    }

    /// Load the routing logits weight for a MoE layer.
    pub fn load_router(&self, layer_id: usize, device: &Device) -> Result<QMatMul> {
        let mut c = Cursor::new(&self.mmap[..]);
        self.qmatmul(&mut c, &format!("blk.{layer_id}.ffn_gate_inp.weight"), device)
    }

    /// Load a specific expert from a MoE layer.
    pub fn load_expert(
        &self, 
        layer_id: usize, 
        expert_id: usize, 
        device: &Device
    ) -> Result<crate::moe::ExpertWeights> {
        let mut c = Cursor::new(&self.mmap[..]);
        let p = format!("blk.{layer_id}");
        
        // Gemma 4 / Mixtral naming convention: blk.N.ffn_{gate,up,down}_exps.E
        let w_gate = self.qmatmul(&mut c, &format!("{p}.ffn_gate_exps.{expert_id}.weight"), device)?;
        let w_up   = self.qmatmul(&mut c, &format!("{p}.ffn_up_exps.{expert_id}.weight"),   device)?;
        let w_down = self.qmatmul(&mut c, &format!("{p}.ffn_down_exps.{expert_id}.weight"), device)?;
        
        Ok(crate::moe::ExpertWeights { w_gate, w_up, w_down })
    }

    /// Load the token-embedding table (dequantized F16).
    pub fn load_embedding(&self, device: &Device) -> Result<Tensor> {
        let mut c = Cursor::new(&self.mmap[..]);
        // Always try to put on CUDA to save system RAM
        let target = device;
        self.dequant(&mut c, "token_embd.weight", target)
    }

    /// Load the final RMSNorm weight (dequantized F16).
    pub fn load_final_norm(&self, device: &Device) -> Result<Tensor> {
        let mut c = Cursor::new(&self.mmap[..]);
        self.dequant(&mut c, "output_norm.weight", device)
    }

    /// Load the LM-head weight (quantized).
    pub fn load_lm_head(&self, device: &Device) -> Result<QMatMul> {
        let mut c = Cursor::new(&self.mmap[..]);
        self.qmatmul(&mut c, "output.weight", device)
            .or_else(|_| self.qmatmul(&mut c, "lm_head.weight", device))
    }

    pub fn load_output(&self, device: &Device) -> Result<(Tensor, QMatMul)> {
        let norm = self.load_final_norm(device)?;
        let head = self.load_lm_head(device)?;
        Ok((norm, head))
    }

    /// Load an arbitrary dequantized F16 tensor by name.
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let mut c = Cursor::new(&self.mmap[..]);
        self.dequant(&mut c, name, device)
    }

    /// Prefetch hint — layer `layer_id` will be needed soon.
    pub fn prefetch_layer(&self, layer_id: usize) {
        let p = format!("blk.{layer_id}.");
        for (name, info) in self.content.tensor_infos.iter() {
            if name.starts_with(&p) {
                let offset = self.content.tensor_data_offset + info.offset;
                let nelements: usize = info.shape.dims().iter().product();
                let length = (nelements * info.ggml_dtype.type_size()) / info.ggml_dtype.block_size();
                
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::posix_fadvise(
                        self.raw_fd, 
                        offset as libc::off_t, 
                        length as libc::off_t, 
                        libc::POSIX_FADV_WILLNEED
                    );
                }
            }
        }
    }

    pub fn release_layer(&self, layer_id: usize) {
        let p = format!("blk.{layer_id}.");
        for (name, info) in self.content.tensor_infos.iter() {
            if name.starts_with(&p) {
                let offset = self.content.tensor_data_offset + info.offset;
                let nelements: usize = info.shape.elem_count();
                let length = (nelements * info.ggml_dtype.type_size()) / info.ggml_dtype.block_size();
                
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::posix_fadvise(
                        self.raw_fd, 
                        offset as libc::off_t, 
                        length as libc::off_t, 
                        libc::POSIX_FADV_DONTNEED
                    );
                }
            }
        }
    }

    /// Load only the specific rows of the FFN weights as predicted by A2WS.
    pub fn load_layer_sparse(
        &self, 
        layer_id: usize, 
        mask: &crate::sparsity_predictor::SparseWeightMask,
        device: &Device
    ) -> Result<QBlockWeights> {
        // For now, this fallback to load_layer but simulates the density
        // in a real production environment with custom kernels.
        self.load_layer(layer_id, device, None)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    fn qmatmul(
        &self,
        cursor: &mut Cursor<&[u8]>,
        name: &str,
        device: &Device,
    ) -> Result<QMatMul> {
        let qt_cpu = self.content
            .tensor(cursor, name, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}' to CPU: {e}"))?;
        
        let dtype = qt_cpu.dtype();
        match dtype {
            candle_core::quantized::GgmlDType::F16 | candle_core::quantized::GgmlDType::Q8K => {
                let t = qt_cpu.dequantize_f16(&Device::Cpu)?
                    .to_device(device)
                    .map_err(|e| anyhow::anyhow!("Failed to move '{name}' to CUDA: {e}"))?;
                Ok(QMatMul::TensorF16(t))
            }
            candle_core::quantized::GgmlDType::F32 => {
                let t = qt_cpu.dequantize(&Device::Cpu)?
                    .to_device(device)?;
                Ok(QMatMul::Tensor(t))
            }
            _ => {
                let qt = self.content
                    .tensor(cursor, name, device)
                    .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}' to device: {e}"))?;
                QMatMul::from_arc(Arc::new(qt))
                    .map_err(|e| anyhow::anyhow!("Failed to create QMatMul for '{name}': {e}"))
            }
        }
    }

    fn dequant(
        &self,
        cursor: &mut Cursor<&[u8]>,
        name: &str,
        device: &Device,
    ) -> Result<Tensor> {
        let qt = self.content
            .tensor(cursor, name, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}' to CPU: {e}"))?;
        
        let dtype = qt.dtype();
        if device.is_cuda() {
            qt.dequantize_f16(device)
                .map_err(|e| anyhow::anyhow!("Failed to dequantize_f16 '{name}' on CUDA: {e}"))
        } else {
            qt.dequantize_f16(device)
                .map_err(|e| anyhow::anyhow!("Failed to dequantize_f16 '{name}': {e}"))
        }
    }
}

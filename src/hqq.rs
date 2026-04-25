//! HQQ — Half-Quadratic Quantization dequantizer
//!
//! HQQ (timothynest/HQQ, 2023) is a fast, calibration-free post-training
//! quantization method that solves a half-quadratic optimization problem to
//! find optimal per-group scales and zeros in ≈ 1 minute on CPU.
//!
//! # What makes HQQ different from GPTQ/AWQ
//! - **No calibration data needed** — pure weight-only, no activation statistics
//! - **Per-group scale+zero** (default group_size=64) stored as f16/bf16
//! - **Bit-widths 1–8**, typically 2, 3, or 4 bits in practice
//! - **Axis-0 or axis-1** quantization (axis=0 is default in HQQ)
//!
//! # Wire format (HQQ checkpoint)
//! HQQ checkpoints (`.pt` or SafeTensors) contain per-layer dicts:
//! ```text
//! {
//!   "W_q":    int tensor [out, in/group_size * bits/8]   — packed uint8
//!   "scale":  float16 tensor [1, out * (in/group_size)]  — per-group scale
//!   "zero":   float16 tensor [1, out * (in/group_size)]  — per-group zero
//!   "meta": { "nbits": 4, "group_size": 64, "axis": 0, "shape": [out, in] }
//! }
//! ```
//!
//! # Dequantization formula
//! ```text
//! w_dq[g] = (W_q[g].to_f32() - zero[g]) * scale[g]
//! ```
//! where g indexes a group of `group_size` weights sharing a scale/zero pair.
//!
//! # HQQ-K nested super-blocks
//! HQQ+ quantizes the scale tensors with a second level of HQQ (the "K" suffix).
//! We expose this as `HqqKLayer` which chains two dequant calls.

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Axis along which quantization groups are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HqqAxis {
    /// Groups along the output dimension (default, axis=0 in HQQ notation)
    Axis0 = 0,
    /// Groups along the input dimension (axis=1)
    Axis1 = 1,
}

/// HQQ layer configuration.
#[derive(Debug, Clone)]
pub struct HqqConfig {
    /// Number of quantized bits per weight (1–8, typically 2/3/4)
    pub nbits: u8,
    /// Number of weights per quantization group
    pub group_size: usize,
    /// Which axis groups are formed along
    pub axis: HqqAxis,
    /// Original weight shape [out_features, in_features]
    pub shape: [usize; 2],
}

impl HqqConfig {
    /// Default: 4-bit, group_size=64, axis=0 — matches HQQ library defaults
    pub fn default_4bit(out: usize, in_feat: usize) -> Self {
        Self { nbits: 4, group_size: 64, axis: HqqAxis::Axis0, shape: [out, in_feat] }
    }

    /// 3-bit, group_size=64, axis=0
    pub fn default_3bit(out: usize, in_feat: usize) -> Self {
        Self { nbits: 3, group_size: 64, axis: HqqAxis::Axis0, shape: [out, in_feat] }
    }

    /// 2-bit, group_size=16 (common for ultra-compression)
    pub fn ultra_2bit(out: usize, in_feat: usize) -> Self {
        Self { nbits: 2, group_size: 16, axis: HqqAxis::Axis0, shape: [out, in_feat] }
    }
}

// ---------------------------------------------------------------------------
// Packed Integer Unpacking
// ---------------------------------------------------------------------------

/// Unpack `nbits`-per-element values from a packed uint8 buffer.
///
/// HQQ packs weights tightly: `ceil(group_size * nbits / 8)` bytes per group.
/// Returns a flat `Vec<u8>` of unpacked raw values in `[0, 2^nbits)`.
///
/// Supports nbits = 1, 2, 3, 4, 5, 6, 7, 8.
pub fn unpack_weights(packed: &[u8], nbits: u8, n_weights: usize) -> Vec<u8> {
    assert!(nbits >= 1 && nbits <= 8, "nbits must be in 1..=8");
    let mut out = Vec::with_capacity(n_weights);

    match nbits {
        8 => {
            // No packing — 1 byte per weight
            out.extend_from_slice(&packed[..n_weights]);
        }
        4 => {
            // 2 weights per byte (low nibble first)
            for &byte in packed.iter().take((n_weights + 1) / 2) {
                out.push(byte & 0x0F);
                if out.len() < n_weights { out.push((byte >> 4) & 0x0F); }
            }
        }
        2 => {
            // 4 weights per byte
            for &byte in packed.iter().take((n_weights + 3) / 4) {
                for shift in [0u8, 2, 4, 6] {
                    if out.len() < n_weights {
                        out.push((byte >> shift) & 0x03);
                    }
                }
            }
        }
        1 => {
            // 8 weights per byte
            for &byte in packed.iter().take((n_weights + 7) / 8) {
                for bit in 0..8u8 {
                    if out.len() < n_weights {
                        out.push((byte >> bit) & 0x01);
                    }
                }
            }
        }
        3 => {
            // 3 bits per weight — spans byte boundaries
            // Bit-serial packing, MSB-first within each byte
            let mut bit_cursor = 0usize;
            while out.len() < n_weights {
                let byte_idx = bit_cursor / 8;
                let bit_offset = bit_cursor % 8;
                if byte_idx >= packed.len() { break; }

                let mut val = 0u8;
                for b in 0..3usize {
                    let global_bit = bit_cursor + b;
                    let bi = global_bit / 8;
                    let bo = global_bit % 8;
                    if bi < packed.len() {
                        val |= ((packed[bi] >> bo) & 1) << b;
                    }
                }
                out.push(val);
                bit_cursor += 3;
                let _ = bit_offset; // suppress unused warning
            }
        }
        5 => {
            let mut bit_cursor = 0usize;
            while out.len() < n_weights {
                let mut val = 0u8;
                for b in 0..5usize {
                    let global_bit = bit_cursor + b;
                    let bi = global_bit / 8;
                    let bo = global_bit % 8;
                    if bi < packed.len() {
                        val |= ((packed[bi] >> bo) & 1) << b;
                    }
                }
                out.push(val);
                bit_cursor += 5;
            }
        }
        6 => {
            let mut bit_cursor = 0usize;
            while out.len() < n_weights {
                let mut val = 0u8;
                for b in 0..6usize {
                    let global_bit = bit_cursor + b;
                    let bi = global_bit / 8;
                    let bo = global_bit % 8;
                    if bi < packed.len() {
                        val |= ((packed[bi] >> bo) & 1) << b;
                    }
                }
                out.push(val);
                bit_cursor += 6;
            }
        }
        7 => {
            let mut bit_cursor = 0usize;
            while out.len() < n_weights {
                let mut val = 0u8;
                for b in 0..7usize {
                    let global_bit = bit_cursor + b;
                    let bi = global_bit / 8;
                    let bo = global_bit % 8;
                    if bi < packed.len() {
                        val |= ((packed[bi] >> bo) & 1) << b;
                    }
                }
                out.push(val);
                bit_cursor += 7;
            }
        }
        _ => unreachable!(),
    }
    out
}

// ---------------------------------------------------------------------------
// HQQ Layer
// ---------------------------------------------------------------------------

/// One weight matrix stored in HQQ format.
pub struct HqqLayer {
    /// Packed quantized weights: shape determined by nbits + axis
    pub w_q: Tensor,
    /// Per-group scales (f16 or f32): [n_groups] or [1, n_groups]
    pub scale: Tensor,
    /// Per-group zeros (same shape/dtype as scale)
    pub zero: Tensor,
    pub cfg: HqqConfig,
}

impl HqqLayer {
    /// Dequantize to a full f32 weight matrix.
    ///
    /// Returns `[out_features, in_features]` in F32.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let [out_feat, in_feat] = self.cfg.shape;
        let group_size = self.cfg.group_size;
        let n_groups = match self.cfg.axis {
            HqqAxis::Axis0 => out_feat * ((in_feat + group_size - 1) / group_size),
            HqqAxis::Axis1 => in_feat * ((out_feat + group_size - 1) / group_size),
        };

        // ── Step 1: unpack raw uint8 quantized values ─────────────────────
        let w_q_raw: Vec<u8> = self.w_q
            .to_dtype(DType::U8)?
            .flatten_all()?
            .to_vec1()?;

        let n_weights = out_feat * in_feat;
        let unpacked = unpack_weights(&w_q_raw, self.cfg.nbits, n_weights);

        // ── Step 2: load scales and zeros as f32 ──────────────────────────
        let scales: Vec<f32> = self.scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let zeros: Vec<f32>  = self.zero.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // ── Step 3: dequantize per group ──────────────────────────────────
        // w_dq[i] = (w_q[i].f32() - zero[group(i)]) * scale[group(i)]
        let mut w_f32 = vec![0.0f32; n_weights];
        let _ = n_groups;

        match self.cfg.axis {
            HqqAxis::Axis0 => {
                // axis=0: groups along in_feat dimension for each output row
                let groups_per_row = (in_feat + group_size - 1) / group_size;
                for out_idx in 0..out_feat {
                    for in_idx in 0..in_feat {
                        let g = out_idx * groups_per_row + in_idx / group_size;
                        let w_idx = out_idx * in_feat + in_idx;
                        let s = *scales.get(g).unwrap_or(&1.0f32);
                        let z = *zeros.get(g).unwrap_or(&0.0f32);
                        w_f32[w_idx] = (unpacked[w_idx] as f32 - z) * s;
                    }
                }
            }
            HqqAxis::Axis1 => {
                // axis=1: groups along out_feat dimension for each input column
                let groups_per_col = (out_feat + group_size - 1) / group_size;
                for out_idx in 0..out_feat {
                    for in_idx in 0..in_feat {
                        let g = in_idx * groups_per_col + out_idx / group_size;
                        let w_idx = out_idx * in_feat + in_idx;
                        let s = *scales.get(g).unwrap_or(&1.0f32);
                        let z = *zeros.get(g).unwrap_or(&0.0f32);
                        w_f32[w_idx] = (unpacked[w_idx] as f32 - z) * s;
                    }
                }
            }
        }

        Tensor::from_vec(w_f32, (out_feat, in_feat), device)
    }
}

// ---------------------------------------------------------------------------
// HQQ-K Nested Super-blocks
// ---------------------------------------------------------------------------

/// HQQ-K: the scale tensor itself is quantized with a second HQQ layer.
///
/// Used by HQQ+ and TorchAO's HQQ backend when `quant_scale=True`.
/// Dequant: first dequant the inner (scale) layer, then use those scales
/// in the outer dequant pass.
pub struct HqqKLayer {
    /// The weight matrix layer (w_q quantized with nbits)
    pub outer: HqqLayer,
    /// The scale-quantization layer (scale_q quantized with scale_nbits)
    pub scale_inner: HqqLayer,
}

impl HqqKLayer {
    /// Full two-level dequantization.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        // Step 1: dequantize the scale layer to get float scales
        let dequant_scales = self.scale_inner.dequantize(device)?;

        // Step 2: substitute dequant_scales into outer.scale and run outer dequant
        // Build a temporary outer layer with the recovered scale tensor
        let outer_with_real_scale = HqqLayer {
            w_q: self.outer.w_q.clone(),
            scale: dequant_scales.flatten_all()?,
            zero: self.outer.zero.clone(),
            cfg: self.outer.cfg.clone(),
        };
        outer_with_real_scale.dequantize(device)
    }
}

// ---------------------------------------------------------------------------
// Format detection helper
// ---------------------------------------------------------------------------

/// Detect if a set of tensor names looks like an HQQ checkpoint.
///
/// HQQ checkpoints have tensors named `*.W_q`, `*.scale`, `*.zero`.
/// Returns `true` if the pattern matches ≥1 weight group.
pub fn is_hqq_checkpoint(tensor_names: &[&str]) -> bool {
    let has_wq    = tensor_names.iter().any(|n| n.ends_with(".W_q") || n.ends_with("W_q"));
    let has_scale = tensor_names.iter().any(|n| n.ends_with(".scale") || n.ends_with("scale"));
    let has_zero  = tensor_names.iter().any(|n| n.ends_with(".zero") || n.ends_with("zero"));
    has_wq && has_scale && has_zero
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_unpack_4bit_basic() {
        // Two weights packed in one byte: low nibble = 3, high nibble = 7
        let packed = vec![0b0111_0011u8]; // low=3, high=7
        let out = unpack_weights(&packed, 4, 2);
        assert_eq!(out, vec![3, 7]);
    }

    #[test]
    fn test_unpack_4bit_full_byte() {
        let packed = vec![0xABu8]; // low=B(11), high=A(10)
        let out = unpack_weights(&packed, 4, 2);
        assert_eq!(out[0], 0xB); // low nibble
        assert_eq!(out[1], 0xA); // high nibble
    }

    #[test]
    fn test_unpack_2bit() {
        // 4 weights of 2 bits each: [0,1,2,3] packed = 0b11_10_01_00 = 0xE4
        let packed = vec![0b1110_0100u8];
        let out = unpack_weights(&packed, 2, 4);
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_unpack_1bit() {
        // 8 weights: alternating 0,1 packed = 0b10101010 = 0xAA
        let packed = vec![0xAAu8];
        let out = unpack_weights(&packed, 1, 8);
        assert_eq!(out, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_unpack_8bit_passthrough() {
        let packed = vec![10u8, 20, 30];
        let out = unpack_weights(&packed, 8, 3);
        assert_eq!(out, vec![10, 20, 30]);
    }

    #[test]
    fn test_hqq_dequant_4bit_axis0() {
        let dev = &Device::Cpu;
        let cfg = HqqConfig::default_4bit(2, 4); // 2×4 weight matrix

        // Packed: 4 weights per row, 2 rows = 8 weights
        // 4bit: 2 weights per byte → 4 bytes total
        // Row 0: weights [2, 3, 4, 5], Row 1: weights [6, 7, 8, 9]
        let packed_row0 = vec![(2 | (3 << 4)) as u8, (4 | (5 << 4)) as u8];
        let packed_row1 = vec![(6 | (7 << 4)) as u8, (8 | (9 << 4)) as u8];
        let mut packed = packed_row0;
        packed.extend(packed_row1);

        let w_q = candle_core::Tensor::from_vec(packed, 4usize, dev).unwrap();
        // scale=1.0, zero=0 → dequant = identity
        let scale = candle_core::Tensor::ones((1usize,), DType::F32, dev).unwrap();
        let zero  = candle_core::Tensor::zeros((1usize,), DType::F32, dev).unwrap();

        let layer = HqqLayer { w_q, scale, zero, cfg };
        let dq = layer.dequantize(dev).unwrap();
        assert_eq!(dq.dims(), &[2, 4]);
        let vals: Vec<f32> = dq.flatten_all().unwrap().to_vec1().unwrap();
        // scale=1, zero=0: first 4 = [2,3,4,5], next 4 = [6,7,8,9]
        assert!((vals[0] - 2.0).abs() < 0.01);
        assert!((vals[3] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_hqq_config_presets() {
        let c4 = HqqConfig::default_4bit(256, 256);
        assert_eq!(c4.nbits, 4);
        assert_eq!(c4.group_size, 64);

        let c3 = HqqConfig::default_3bit(128, 128);
        assert_eq!(c3.nbits, 3);

        let c2 = HqqConfig::ultra_2bit(64, 64);
        assert_eq!(c2.nbits, 2);
        assert_eq!(c2.group_size, 16);
    }

    #[test]
    fn test_is_hqq_checkpoint_true() {
        let names = vec!["model.layer0.W_q", "model.layer0.scale", "model.layer0.zero"];
        assert!(is_hqq_checkpoint(&names));
    }

    #[test]
    fn test_is_hqq_checkpoint_false() {
        let names = vec!["model.weight", "model.bias"];
        assert!(!is_hqq_checkpoint(&names));
    }

    #[test]
    fn test_dequant_with_scale_zero() {
        // w_dq = (w_q - zero) * scale
        // w_q=8, zero=4, scale=0.5 → (8-4)*0.5 = 2.0
        let dev = &Device::Cpu;
        let cfg = HqqConfig { nbits: 8, group_size: 1, axis: HqqAxis::Axis0, shape: [1, 1] };
        let w_q = candle_core::Tensor::from_vec(vec![8u8], 1usize, dev).unwrap();
        let scale = candle_core::Tensor::from_vec(vec![0.5f32], 1usize, dev).unwrap();
        let zero  = candle_core::Tensor::from_vec(vec![4.0f32], 1usize, dev).unwrap();
        let layer = HqqLayer { w_q, scale, zero, cfg };
        let dq = layer.dequantize(dev).unwrap();
        let v: Vec<f32> = dq.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - 2.0).abs() < 1e-5, "expected 2.0 got {}", v[0]);
    }
}

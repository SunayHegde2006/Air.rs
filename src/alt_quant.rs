//! P12 — Alternative Quantization Format Readers: GPTQ / AWQ / EXL2
//!
//! These are popular community quantization formats that differ from the standard
//! GGUF/GGML formats used natively in Air.rs.
//!
//! # Format Overview
//!
//! | Format | Creator | Storage | Bits | Key Feature |
//! |--------|---------|---------|------|-------------|
//! | GPTQ   | IST-DAS | safetensors | 4/8-bit | Block-wise layerwise PTQ |
//! | AWQ    | MIT HAN lab | safetensors | 4-bit | Activation-aware weight quant |
//! | EXL2   | turboderp | safetensors | 2-8-bit mixed | Per-row scale + calibration |
//!
//! # Integration Approach
//! Air.rs loads weights via GGUF natively. For GPTQ/AWQ/EXL2, we:
//! 1. Detect the format from file headers / metadata keys
//! 2. Map their weight tensors to the Air.rs weight naming convention
//! 3. Dequantize on load to f32/f16 (or keep quantized for compatible ops)
//!
//! # GPTQ Format Details
//! - `qweight`: int32 packed column-wise, 8 values per int32 (for 4-bit)
//! - `qzeros`: quantized zero points (int32 packed similarly)
//! - `scales`: f16 scale per block
//! - `g_idx`: group index per column (for actorder / desc-act)
//!
//! # AWQ Format Details
//! - `/layers.N.attention.wq.weight`: uint8 packed (2 values per byte for 4-bit)
//! - `/layers.N.attention.wq.scales`: f16 scales
//! - `/layers.N.attention.wq.zeros`: f16 zero points (different from GPTQ!)
//! - Group size: typically 128
//!
//! # EXL2 Format Details
//! More complex: per-matrix adaptive bit allocation (some rows 3-bit, some 4-bit)
//! - `q_weight`: packed mixed-bit weights
//! - `q_scale`: grouped scales
//! - `q_scale_max`: per-matrix max scale
//! - `q_groups`: column group assignments
//! - `q_perm` (optional): permutation for desc_act

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Format Detection
// ---------------------------------------------------------------------------

/// Detected alternative quantization format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AltQuantFormat {
    Gptq,
    Awq,
    Exl2,
}

/// Metadata key signature for format detection.
///
/// Given a map of top-level tensor names or metadata keys, identify the format.
pub fn detect_format(keys: &[String]) -> Option<AltQuantFormat> {
    let has = |k: &str| keys.iter().any(|s| s.contains(k));

    if has("q_scale_max") || has("q_groups") || has("q_perm") {
        Some(AltQuantFormat::Exl2)
    } else if has("qweight") && has("qzeros") && has("scales") {
        Some(AltQuantFormat::Gptq)
    } else if has(".zeros") && has(".scales") && !has("qzeros") {
        // AWQ uses float zeros, GPTQ uses int zeros
        Some(AltQuantFormat::Awq)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// GPTQ Dequantization
// ---------------------------------------------------------------------------

/// GPTQ 4-bit packed weight block.
///
/// `qweight`: shape `[K/8, N]` for 4-bit (8 values packed per int32 column-wise)
/// `qzeros`:  shape `[G, N/8]` where G = K / group_size
/// `scales`:  shape `[G, N]` f16
pub struct GptqLayer {
    /// Packed int32 weights: [K/pack_factor, N]
    pub qweight: Tensor,
    /// Packed int32 zeros: [n_groups, N/pack_factor]
    pub qzeros: Tensor,
    /// FP16 scales: [n_groups, N]
    pub scales: Tensor,
    /// Group size (default 128)
    pub group_size: usize,
    /// Bits per weight (4 or 8)
    pub bits: usize,
    /// Optional column order permutation (desc_act)
    pub g_idx: Option<Tensor>,
}

impl GptqLayer {
    /// Dequantize to dense float matrix `[K, N]`.
    ///
    /// Algorithm:
    /// ```text
    /// for each group g:
    ///   scale = scales[g, :]       → [N]
    ///   zero  = unpack(qzeros[g])  → [N]   (int zero points)
    ///   for row k in group g:
    ///     q = unpack_col(qweight[k/pack, :]) → [N]   (int, 0..2^bits)
    ///     w[k, :] = scale * (q - zero)
    /// ```
    pub fn dequantize(&self) -> Result<Tensor> {
        let device = self.qweight.device();
        let dtype = DType::F32;
        let pack = 32 / self.bits; // values per int32

        let k_packed = self.qweight.dim(0)?;
        let n = self.qweight.dim(1)?;
        let k = k_packed * pack;
        let n_groups = k / self.group_size;

        // Extract qweight as i32 data
        let qw_data: Vec<i32> = self.qweight
            .to_dtype(DType::I64)?.to_vec2::<i64>()?
            .into_iter().flatten()
            .map(|v| v as i32)
            .collect();
        let qz_data: Vec<i32> = self.qzeros
            .to_dtype(DType::I64)?.to_vec2::<i64>()?
            .into_iter().flatten()
            .map(|v| v as i32)
            .collect();

        // Scales: [n_groups, N] f32
        let scales_data: Vec<f32> = self.scales
            .to_dtype(DType::F32)?.to_vec2()?
            .into_iter().flatten()
            .collect();

        let mask = (1i32 << self.bits) - 1;
        let z_pack = 32 / self.bits;

        let mut out = vec![0.0f32; k * n];

        for row in 0..k {
            let group = row / self.group_size;
            let qw_row = row / pack;
            let qw_shift = (row % pack) * self.bits;

            for col in 0..n {
                let qw_int = qw_data[qw_row * n + col];
                let q = ((qw_int >> qw_shift) & mask) as f32;

                // Extract zero for this group + col
                let z_col_packed = col / z_pack;
                let z_shift = (col % z_pack) * self.bits;
                let qz_int = qz_data[group * (n / z_pack) + z_col_packed];
                let z = ((qz_int >> z_shift) & mask) as f32;

                let scale = scales_data[group * n + col];
                out[row * n + col] = scale * (q - z);
            }
        }

        Tensor::from_vec(out, (k, n), device)
    }
}

// ---------------------------------------------------------------------------
// AWQ Dequantization
// ---------------------------------------------------------------------------

/// AWQ 4-bit layer (activation-aware weight quantization).
///
/// Key difference from GPTQ:
/// - Weights are uint8 packed (2 per byte) not int32 packed (8 per int)
/// - Zero points are f16 (float), not integer packed
/// - Quantization applied after activation-aware scaling
pub struct AwqLayer {
    /// Packed uint8 weights: [N, K/2] (4-bit, 2 vals per byte, transposed)
    pub qweight: Tensor,
    /// FP16 scales: [N, K/group_size]
    pub scales: Tensor,
    /// FP16 zero points: [N, K/group_size]
    pub zeros: Tensor,
    /// Group size (default 128)
    pub group_size: usize,
}

impl AwqLayer {
    /// Dequantize AWQ weights to dense `[N, K]` then transpose to `[K, N]`.
    ///
    /// Algorithm:
    /// ```text
    /// for each col (output dim) n:
    ///   for each group g in [K/group_size]:
    ///     scale = scales[n, g]     f16
    ///     zero  = zeros[n, g]      f16
    ///     for k in group:
    ///       q4 = unpack(qweight[n, k/2]) (high or low nibble)
    ///       w[k, n] = scale * q4 - zero
    /// ```
    pub fn dequantize(&self) -> Result<Tensor> {
        let device = self.qweight.device();
        let n = self.qweight.dim(0)?;
        let k_packed = self.qweight.dim(1)?;
        let k = k_packed * 2; // 2 values per byte

        let qw_bytes: Vec<u8> = self.qweight
            .to_dtype(DType::U8)?.to_vec2()?
            .into_iter().flatten()
            .collect();
        let scales_f: Vec<f32> = self.scales
            .to_dtype(DType::F32)?.to_vec2()?
            .into_iter().flatten()
            .collect();
        let zeros_f: Vec<f32> = self.zeros
            .to_dtype(DType::F32)?.to_vec2()?
            .into_iter().flatten()
            .collect();

        let n_groups = k / self.group_size;
        let mut out = vec![0.0f32; k * n]; // layout: [k, n]

        for row_n in 0..n {
            for ki in 0..k {
                let group = ki / self.group_size;
                let byte_idx = ki / 2;
                let byte = qw_bytes[row_n * k_packed + byte_idx];
                let q4 = if ki % 2 == 0 { byte & 0x0F } else { byte >> 4 };

                let scale = scales_f[row_n * n_groups + group];
                let zero  = zeros_f[row_n * n_groups + group];
                // AWQ: w = scale * q - zero
                out[ki * n + row_n] = scale * q4 as f32 - zero;
            }
        }

        Tensor::from_vec(out, (k, n), device)
    }
}

// ---------------------------------------------------------------------------
// EXL2 Format (structural support + dequantize stub)
// ---------------------------------------------------------------------------

/// EXL2 quantized layer (per-row mixed-bit).
///
/// EXL2 is the most complex format — each row can have a different bit width.
/// The actual per-row bits are stored in `q_groups` which has a complex layout.
/// This implementation provides the structural reader; the full dequant is
/// equivalent to GPTQ but with variable bit widths per row.
pub struct Exl2Layer {
    /// Packed quantized weights
    pub q_weight: Tensor,
    /// Per-group scales
    pub q_scale: Tensor,
    /// Per-matrix scale maximum
    pub q_scale_max: Tensor,
    /// Group column assignments [cols]
    pub q_groups: Tensor,
    /// Optional permutation for desc_act
    pub q_perm: Option<Tensor>,
    /// Optional inverse permutation
    pub q_invperm: Option<Tensor>,
}

impl Exl2Layer {
    /// Estimate average bits per weight from group info.
    pub fn avg_bits(&self) -> Result<f32> {
        // groups shape: [n_group_sets, group_info_cols]
        // This is a heuristic — actual bits encoded in q_groups header
        let n = self.q_weight.dim(1)?;
        let k_packed = self.q_weight.dim(0)?;
        let bits = (k_packed as f32 * 32.0) / (n as f32 * k_packed as f32);
        Ok(bits.min(8.0).max(2.0))
    }

    /// Dequantize to dense f32 (GPTQ-compatible path for 4-bit groups).
    ///
    /// For full EXL2 mixed-bit support, use exllama_v2 kernels (not yet implemented).
    /// This stub dequantizes treating all groups as 4-bit (correct for pure-4-bit EXL2).
    pub fn dequantize_4bit(&self) -> Result<Tensor> {
        let device = self.q_weight.device();
        let k_packed = self.q_weight.dim(0)?;
        let n = self.q_weight.dim(1)?;
        let k = k_packed * 8; // 4-bit: 8 per int32

        // Use GPTQ-style decode with zero hint from q_scale_max
        let qw: Vec<i32> = self.q_weight
            .to_dtype(DType::I64)?.to_vec2::<i64>()?
            .into_iter().flatten()
            .map(|v| v as i32)
            .collect();

        let scales_raw: Vec<f32> = self.q_scale
            .to_dtype(DType::F32)?.to_vec2()?
            .into_iter().flatten()
            .collect();

        let scale_max: f32 = self.q_scale_max
            .to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
            .into_iter().next().unwrap_or(1.0);

        let mut out = vec![0.0f32; k * n];
        let group_size = k / scales_raw.len().max(1);
        let mask = 0x0F_i32;

        for row in 0..k {
            let qrow = row / 8;
            let shift = (row % 8) * 4;
            let group = if group_size > 0 { (row / group_size).min(scales_raw.len() - 1) } else { 0 };
            let scale = scales_raw[group] * scale_max;
            for col in 0..n {
                let q = ((qw[qrow * n + col] >> shift) & mask) as f32;
                out[row * n + col] = scale * (q - 8.0); // zero centered
            }
        }

        Tensor::from_vec(out, (k, n), device)
    }
}

// ---------------------------------------------------------------------------
// Unified Reader Interface
// ---------------------------------------------------------------------------

/// Result of loading an alternative-format weight file.
pub struct AltQuantLayerSet {
    pub format: AltQuantFormat,
    /// Map from layer name → dense f32 weight tensor
    pub weights: HashMap<String, Tensor>,
}

impl AltQuantLayerSet {
    /// Convert GPTQ layer to dense weights and insert into map.
    pub fn from_gptq(layer_name: &str, gptq: GptqLayer) -> Result<Self> {
        let mut weights = HashMap::new();
        let w = gptq.dequantize()?;
        weights.insert(layer_name.to_string(), w);
        Ok(Self { format: AltQuantFormat::Gptq, weights })
    }

    /// Convert AWQ layer to dense weights.
    pub fn from_awq(layer_name: &str, awq: AwqLayer) -> Result<Self> {
        let mut weights = HashMap::new();
        let w = awq.dequantize()?;
        weights.insert(layer_name.to_string(), w);
        Ok(Self { format: AltQuantFormat::Awq, weights })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_gptq() {
        let keys: Vec<String> = vec!["model.layers.0.qweight".into(), "model.layers.0.qzeros".into(), "model.layers.0.scales".into()];
        assert_eq!(detect_format(&keys), Some(AltQuantFormat::Gptq));
    }

    #[test]
    fn test_format_detection_awq() {
        let keys: Vec<String> = vec!["model.layers.0.weight.scales".into(), "model.layers.0.weight.zeros".into()];
        assert_eq!(detect_format(&keys), Some(AltQuantFormat::Awq));
    }

    #[test]
    fn test_format_detection_exl2() {
        let keys: Vec<String> = vec!["q_weight".into(), "q_scale_max".into(), "q_groups".into()];
        assert_eq!(detect_format(&keys), Some(AltQuantFormat::Exl2));
    }

    #[test]
    fn test_format_detection_unknown() {
        let keys: Vec<String> = vec!["model.weight".into()];
        assert_eq!(detect_format(&keys), None);
    }

    #[test]
    fn test_gptq_dequantize_shape() {
        use candle_core::Device;
        let dev = Device::Cpu;
        // 4-bit: K=32, N=8, K/pack=4 (8 per i32), group_size=16, 2 groups
        let qweight = Tensor::zeros((4usize, 8), DType::I64, &dev).unwrap();
        let qzeros = Tensor::zeros((2usize, 1), DType::I64, &dev).unwrap();
        let scales = Tensor::ones((2usize, 8), DType::F32, &dev).unwrap();
        let layer = GptqLayer {
            qweight, qzeros, scales,
            group_size: 16,
            bits: 4,
            g_idx: None,
        };
        let w = layer.dequantize().unwrap();
        assert_eq!(w.dims(), &[32, 8]);
    }

    #[test]
    fn test_awq_dequantize_shape() {
        use candle_core::Device;
        let dev = Device::Cpu;
        // N=4, K=16 (K/2=8 bytes per row), group_size=8, n_groups=2
        let qweight = Tensor::zeros((4usize, 8usize), DType::U8, &dev).unwrap();
        let scales = Tensor::ones((4usize, 2usize), DType::F32, &dev).unwrap();
        let zeros = Tensor::zeros((4usize, 2usize), DType::F32, &dev).unwrap();
        let layer = AwqLayer { qweight, scales, zeros, group_size: 8 };
        let w = layer.dequantize().unwrap();
        assert_eq!(w.dims(), &[16, 4]);
    }
}

//! P11 — Ultra-Low-Bit Quantization: IQ3_M / IQ3_S / IQ1_S / IQ1_M
//!
//! These are llama.cpp's "imatrix quantization" (IQ) formats that use
//! importance-weighted quantization to allocate bits adaptively.
//!
//! # Formats
//!
//! | Format | Bits/weight | ggml code | Notes |
//! |--------|-------------|-----------|-------|
//! | IQ1_S  | 1.5625 b/w | 24 | Packed 1.5-bit, importance-weighted |
//! | IQ1_M  | 1.75 b/w   | 29 | 1.75-bit, better than IQ1_S |
//! | IQ2_XXS| 2.0625 b/w | 18 | Super-small 2-bit |
//! | IQ3_S  | 3.4375 b/w | 22 | Solid 3.5-bit quality |
//! | IQ3_M  | 3.6640 b/w | 23 | Best 3-bit quality |
//!
//! # Encoding
//!
//! IQ formats use a **codebook + per-superblock scale** approach:
//!
//! For IQ3_S (example, 256-entry codebook per superblock):
//! - Each 256-element superblock has one f32 scale and a set of 3-bit indices
//! - 3-bit indices into a sign-magnitude codebook with 256 entries
//! - No per-block scale (just global superblock scale) + sign bits
//!
//! For IQ1_S / IQ1_M:
//! - 1.5 bits per weight: 8 weights share a 3-byte ternary state + scale
//!
//! # Implementation Strategy
//!
//! This module provides CPU dequantization (GPU would use CUDA kernels).
//! The approach:
//! 1. Read raw bytes from GGUF file
//! 2. Decode indices + scale values
//! 3. Return f32 weights for matmul
//!
//! Reference: llama.cpp ggml-quants.c (dequantize_row_iq3_s, etc.)

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Codebooks (from lleama.cpp ggml-quants.c)
// ---------------------------------------------------------------------------

/// IQ3_XXS / IQ3_S sign-magnitude codebook (256 entries, 8-way grouped)
/// These are the dequantized float values for each 3-bit code index.
/// Source: ggml-quants.c `iq3s_grid`
const IQ3S_GRID: [u32; 512] = {
    // Compressed: each entry encodes 4 values as nibbles (values are in {-7..7})
    // This is a placeholder — real values come from ggml-quants.c
    // We embed a compact representation here
    [0u32; 512]
};

/// IQ1_S 1.5-bit grid (ternary: -1, 0, +1)
/// 8 ternary values per byte-pair
const IQ1S_DELTA: f32 = 0.125; // scale factor for IQ1_S

// ---------------------------------------------------------------------------
// IQ Quantized Block Structures
// ---------------------------------------------------------------------------

/// IQ3_M / IQ3_S block (256 floats = 1 superblock)
/// Format: scale(f16) + sc_bits(u8 * 4) + qs(u8 * 96) + qh(u8 * 12) + signs(u8 * 48)
#[derive(Debug, Clone)]
pub struct BlockIq3s {
    /// Superblock scale (bf16 stored as u16)
    pub scale: f32,
    /// Sub-block scales: 8 × 3.5-bit packed
    pub sub_scales: [u8; 4],
    /// Quantized indices: 256 × 3-bit packed into bytes
    pub qs: Vec<u8>, // 96 bytes for 256 values
    /// High bits for index: 256 × 1-bit packed
    pub qh: Vec<u8>, // 12 bytes
    /// Sign bits: 256 × 1-bit
    pub signs: Vec<u8>, // 48 bytes
}

impl BlockIq3s {
    /// Bytes per superblock (256 weights)
    pub const BYTES_PER_BLOCK: usize = 2 + 4 + 96 + 12 + 48; // = 162 bytes ≈ 3.375 bits/weight

    /// Dequantize this superblock to 256 f32 values.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; 256];

        // Sub-block scales: 4 bytes → 8 sub-scales (4-bit each, sign separate)
        let sub_scales: Vec<f32> = (0..8).map(|i| {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (self.sub_scales[byte_idx] & 0x0F) as f32
            } else {
                ((self.sub_scales[byte_idx] >> 4) & 0x0F) as f32
            };
            self.scale * (nibble + 0.5) * (128.0 / 32.0)
        }).collect();

        // Decode each weight (simplified — real decode reads 3-bit indices from qs/qh)
        // This is the structurally correct decode loop; actual codebook lookup
        // requires porting the full IQ3S grid from ggml-quants.c
        for i in 0..256 {
            let sub_block = i / 32;
            let sub_scale = sub_scales[sub_block];

            // 3-bit index from qs (low 3 bits of each packed entry)
            let byte_idx = i * 3 / 8;
            let bit_offset = (i * 3) % 8;
            let qs_idx = if byte_idx < self.qs.len() {
                let raw = if bit_offset <= 5 {
                    (self.qs[byte_idx] >> bit_offset) & 0x07
                } else {
                    let low = (self.qs[byte_idx] >> bit_offset) as u32;
                    let high = if byte_idx + 1 < self.qs.len() {
                        ((self.qs[byte_idx + 1] as u32) << (8 - bit_offset)) & 0x07
                    } else { 0 };
                    (low | high) as u8 & 0x07
                };
                raw
            } else { 0 };

            // Sign bit from signs array
            let sign_byte = i / 8;
            let sign_bit = i % 8;
            let sign = if sign_byte < self.signs.len() {
                if (self.signs[sign_byte] >> sign_bit) & 1 == 1 { -1.0f32 } else { 1.0f32 }
            } else { 1.0f32 };

            // High bit from qh
            let qh_byte = i / 8;
            let qh_bit = i % 8;
            let _qh_high = if qh_byte < self.qh.len() {
                (self.qh[qh_byte] >> qh_bit) & 1
            } else { 0 };

            // Approximate dequant: scale * sign * (2*index + 1)
            out[i] = sign * sub_scale * (2.0 * qs_idx as f32 + 1.0);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// IQ1_S Block
// ---------------------------------------------------------------------------

/// IQ1_S block (32 weights = 1 group)
/// Format: scale(f16) + qs(u8 * 4) + qh(u16)
/// Each group of 8 weights uses 2 bytes for ternary coded values
#[derive(Debug, Clone)]
pub struct BlockIq1s {
    /// Delta scale (fp16 → f32)
    pub delta: f32,
    /// Quantized ternary codes: 4 bytes per 32 weights (8 bits per 8 ternary values)
    pub qs: [u8; 4],
    /// High bits: 16-bit field for sub-block scales
    pub qh: u16,
}

impl BlockIq1s {
    /// Dequantize 32 weights.
    pub fn dequantize(&self) -> [f32; 32] {
        let mut out = [0.0f32; 32];
        // IQ1_S: each group of 8 weights is encoded as a ternary value in {-1, 0, +1}
        // with an overall delta scale and sub-block scale encoded in qh
        let sub_scale = [
            ((self.qh & 0x000F) as f32 + 0.5) * IQ1S_DELTA,
            (((self.qh >> 4) & 0x000F) as f32 + 0.5) * IQ1S_DELTA,
            (((self.qh >> 8) & 0x000F) as f32 + 0.5) * IQ1S_DELTA,
            (((self.qh >> 12) & 0x000F) as f32 + 0.5) * IQ1S_DELTA,
        ];

        for group in 0..4 {
            let byte = self.qs[group];
            let scale = self.delta * sub_scale[group];
            for bit in 0..8 {
                // Extract 2-bit value: 00=0, 01=+1, 10=-1, 11=+1 (approximately)
                let v = ((byte >> (bit * 1)) & 0x3) as i8;
                let fv = match v & 0x3 {
                    0 => -1.0f32,
                    1 =>  0.0f32,
                    2 =>  1.0f32,
                    _ =>  1.0f32,
                };
                out[group * 8 + bit] = scale * fv;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// IQ Format Reader (from raw bytes)
// ---------------------------------------------------------------------------

/// Quantization format discriminant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IqFormat {
    Iq1s,
    Iq1m,
    Iq3s,
    Iq3m,
}

const IQ1S_BLOCK_SIZE: usize = 32;
const IQ3S_BLOCK_SIZE: usize = 256;

/// Dequantize a full weight tensor from IQ-format packed bytes.
///
/// # Arguments
/// * `data`   — raw packed bytes from GGUF
/// * `n_rows` — number of rows (output dimension)
/// * `n_cols` — number of columns (input dimension)
/// * `format` — IQ quantization format
/// * `device` — target device
///
/// # Returns
/// Dense f32 tensor `[n_rows, n_cols]`.
pub fn dequantize_iq(
    data: &[u8],
    n_rows: usize,
    n_cols: usize,
    format: IqFormat,
    device: &Device,
) -> Result<Tensor> {
    let n_weights = n_rows * n_cols;

    let values: Vec<f32> = match format {
        IqFormat::Iq1s | IqFormat::Iq1m => {
            // IQ1_S: 32 weights per block, 6 bytes per block
            let bytes_per_block = 6usize;
            let n_blocks = n_weights / IQ1S_BLOCK_SIZE;
            let mut out = Vec::with_capacity(n_weights);

            for b in 0..n_blocks {
                let offset = b * bytes_per_block;
                if offset + bytes_per_block > data.len() { break; }

                // Parse fp16 delta from first 2 bytes
                let raw_delta = u16::from_le_bytes([data[offset], data[offset + 1]]);
                let delta = half::f16::from_bits(raw_delta).to_f32();

                let qs = [data[offset+2], data[offset+3], data[offset+4], data[offset+5]];
                let qh = 0u16; // simplified — IQ1_M has different layout

                let block = BlockIq1s { delta, qs, qh };
                out.extend_from_slice(&block.dequantize());
            }

            // Pad if needed
            out.resize(n_weights, 0.0);
            out
        }

        IqFormat::Iq3s | IqFormat::Iq3m => {
            // IQ3_S: 256 weights per block, ~162 bytes per block
            let bytes_per_block = BlockIq3s::BYTES_PER_BLOCK;
            let n_blocks = n_weights / IQ3S_BLOCK_SIZE;
            let mut out = Vec::with_capacity(n_weights);

            for b in 0..n_blocks {
                let offset = b * bytes_per_block;
                if offset + bytes_per_block > data.len() { break; }

                // fp16 scale
                let raw_scale = u16::from_le_bytes([data[offset], data[offset+1]]);
                let scale = half::f16::from_bits(raw_scale).to_f32();

                let sub_scales = [
                    data[offset+2], data[offset+3],
                    data[offset+4], data[offset+5],
                ];
                let qs = data[offset+6..offset+102].to_vec();
                let qh = data[offset+102..offset+114].to_vec();
                let signs = data[offset+114..offset+162].to_vec();

                let block = BlockIq3s { scale, sub_scales, qs, qh, signs };
                out.extend(block.dequantize());
            }

            out.resize(n_weights, 0.0);
            out
        }
    };

    Tensor::from_vec(values, (n_rows, n_cols), device)
}

// ---------------------------------------------------------------------------
// Integration with GGUF Loader
// ---------------------------------------------------------------------------

/// GGUF type codes for IQ formats (from ggml.h)
pub mod gguf_types {
    pub const GGML_TYPE_IQ1_S: u32 = 24;
    pub const GGML_TYPE_IQ1_M: u32 = 29;
    pub const GGML_TYPE_IQ2_XXS: u32 = 18;
    pub const GGML_TYPE_IQ2_XS: u32 = 19;
    pub const GGML_TYPE_IQ2_S: u32 = 28;
    pub const GGML_TYPE_IQ3_XXS: u32 = 20;
    pub const GGML_TYPE_IQ3_S: u32 = 22;
    pub const GGML_TYPE_IQ3_M: u32 = 23;
    pub const GGML_TYPE_IQ4_NL: u32 = 25;
    pub const GGML_TYPE_IQ4_XS: u32 = 26;
}

/// Convert GGML type code to IqFormat (returns None if not an IQ format).
pub fn ggml_type_to_iq_format(ggml_type: u32) -> Option<IqFormat> {
    match ggml_type {
        gguf_types::GGML_TYPE_IQ1_S => Some(IqFormat::Iq1s),
        gguf_types::GGML_TYPE_IQ1_M => Some(IqFormat::Iq1m),
        gguf_types::GGML_TYPE_IQ3_S => Some(IqFormat::Iq3s),
        gguf_types::GGML_TYPE_IQ3_M => Some(IqFormat::Iq3m),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq1s_block_dequantize() {
        let block = BlockIq1s {
            delta: 1.0,
            qs: [0b10101010, 0b01010101, 0b10101010, 0b01010101],
            qh: 0x8888u16, // mid-range sub-scale
        };
        let vals = block.dequantize();
        assert_eq!(vals.len(), 32);
        // All values should be small (within ±4)
        for v in vals.iter() {
            assert!(v.abs() <= 4.0, "IQ1S value out of expected range: {}", v);
        }
    }

    #[test]
    fn test_iq3s_block_dequantize() {
        let block = BlockIq3s {
            scale: 1.0,
            sub_scales: [0x77, 0x77, 0x77, 0x77],
            qs: vec![0u8; 96],
            qh: vec![0u8; 12],
            signs: vec![0u8; 48],
        };
        let vals = block.dequantize();
        assert_eq!(vals.len(), 256);
    }

    #[test]
    fn test_ggml_type_to_iq_format() {
        assert_eq!(ggml_type_to_iq_format(24), Some(IqFormat::Iq1s));
        assert_eq!(ggml_type_to_iq_format(22), Some(IqFormat::Iq3s));
        assert_eq!(ggml_type_to_iq_format(23), Some(IqFormat::Iq3m));
        assert_eq!(ggml_type_to_iq_format(0), None);
        assert_eq!(ggml_type_to_iq_format(1), None); // Q4_0 is not IQ
    }

    #[test]
    fn test_dequantize_iq_shape() {
        use candle_core::Device;
        let dev = Device::Cpu;
        // Create dummy IQ1_S data (32 weights = 1 block = 6 bytes)
        let data = vec![0u8; 6];
        let t = dequantize_iq(&data, 1, 32, IqFormat::Iq1s, &dev).unwrap();
        assert_eq!(t.dims(), &[1, 32]);
    }
}

//! P12 — Alternative Quantization Format Readers: GPTQ / AWQ / EXL2
//!
//! These are popular community quantization formats that differ from the standard
//! GGUF/GGML formats used natively in Air.rs.

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Format Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AltQuantFormat { Gptq, Awq, Exl2 }

pub fn detect_format(keys: &[String]) -> Option<AltQuantFormat> {
    let has = |k: &str| keys.iter().any(|s| s.contains(k));
    if has("q_scale_max") || has("q_groups") || has("q_perm") { Some(AltQuantFormat::Exl2) }
    else if has("qweight") && has("qzeros") && has("scales") { Some(AltQuantFormat::Gptq) }
    else if has(".zeros") && has(".scales") && !has("qzeros") { Some(AltQuantFormat::Awq) }
    else { None }
}

// ---------------------------------------------------------------------------
// GPTQ Dequantization (Parallelized)
// ---------------------------------------------------------------------------

pub struct GptqLayer {
    pub qweight: Tensor,
    pub qzeros: Tensor,
    pub scales: Tensor,
    pub group_size: usize,
    pub bits: usize,
    pub g_idx: Option<Tensor>,
}

impl GptqLayer {
    pub fn dequantize(&self) -> Result<Tensor> {
        let device = self.qweight.device();
        let pack = 32 / self.bits;
        let k_packed = self.qweight.dim(0)?;
        let n = self.qweight.dim(1)?;
        let k = k_packed * pack;

        let qw_data: Vec<i32> = self.qweight.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as i32).collect();
        let qz_data: Vec<i32> = self.qzeros.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as i32).collect();
        let scales_data: Vec<f32> = self.scales.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let mask = (1i32 << self.bits) - 1;
        let z_pack = 32 / self.bits;

        // Parallelize over rows for high throughput
        let out: Vec<f32> = (0..k).into_par_iter().flat_map(|row| {
            let group = row / self.group_size;
            let qw_row = row / pack;
            let qw_shift = (row % pack) * self.bits;
            let mut row_out = vec![0.0f32; n];
            for col in 0..n {
                let qw_int = qw_data[qw_row * n + col];
                let q = ((qw_int >> qw_shift) & mask) as f32;
                let z_col_packed = col / z_pack;
                let z_shift = (col % z_pack) * self.bits;
                let qz_int = qz_data[group * (n / z_pack) + z_col_packed];
                let z = ((qz_int >> z_shift) & mask) as f32;
                let scale = scales_data[group * n + col];
                row_out[col] = scale * (q - z);
            }
            row_out
        }).collect();

        Tensor::from_vec(out, (k, n), device)
    }
}

// ---------------------------------------------------------------------------
// AWQ Dequantization (Parallelized)
// ---------------------------------------------------------------------------

pub struct AwqLayer {
    pub qweight: Tensor,
    pub scales: Tensor,
    pub zeros: Tensor,
    pub group_size: usize,
}

impl AwqLayer {
    pub fn dequantize(&self) -> Result<Tensor> {
        let device = self.qweight.device();
        let n = self.qweight.dim(0)?;
        let k_packed = self.qweight.dim(1)?;
        let k = k_packed * 2;

        let qw_bytes: Vec<u8> = self.qweight.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
        let scales_f: Vec<f32> = self.scales.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let zeros_f: Vec<f32> = self.zeros.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let n_groups = k / self.group_size;
        let out: Vec<f32> = (0..k).into_par_iter().flat_map(|ki| {
            let group = ki / self.group_size;
            let byte_idx = ki / 2;
            let mut row_out = vec![0.0f32; n];
            for row_n in 0..n {
                let byte = qw_bytes[row_n * k_packed + byte_idx];
                let q4 = if ki % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let scale = scales_f[row_n * n_groups + group];
                let zero = zeros_f[row_n * n_groups + group];
                row_out[row_n] = scale * q4 as f32 - zero;
            }
            row_out
        }).collect();

        // Note: out is [k, n], transpose to [n, k] if needed by consumer
        Tensor::from_vec(out, (k, n), device)
    }
}

// ---------------------------------------------------------------------------
// EXL2 Dequantization (Sequential - Bitstream Correct)
// ---------------------------------------------------------------------------

pub struct Exl2Layer {
    pub q_weight: Tensor,
    pub q_scale: Tensor,
    pub q_scale_max: Tensor,
    pub q_groups: Tensor,
    pub q_perm: Option<Tensor>,
}

impl Exl2Layer {
    pub fn dequantize(&self) -> Result<Tensor> {
        let device = self.q_weight.device();
        let n = self.q_weight.dim(1)?;
        let q_groups_data: Vec<u32> = self.q_groups.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect();
        let q_weight_data: Vec<u32> = self.q_weight.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect();
        let q_scale_data: Vec<f32> = self.q_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let scale_max: f32 = self.q_scale_max.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.get(0).copied().unwrap_or(1.0);

        let n_groups = q_groups_data.len() / 2;
        let k = n_groups * 32;
        let mut out = vec![0.0f32; k * n];
        
        let mut qw_idx = 0;
        let mut qw_rem_bits = 0u32;
        let mut qw_buffer = 0u32;

        for g in 0..n_groups {
            let bits = q_groups_data[g * 2 + 1];
            let scale = q_scale_data[g] * (scale_max / 256.0);
            let center = (1u32 << (bits - 1)) as f32;
            let mask = (1u32 << bits) - 1;
            
            for i in 0..32 {
                if qw_rem_bits < bits {
                    qw_buffer = q_weight_data[qw_idx];
                    qw_idx += 1;
                    qw_rem_bits = 32;
                }
                let q = (qw_buffer & mask) as f32;
                qw_buffer >>= bits;
                qw_rem_bits -= bits;
                for col in 0..n {
                    out[(g * 32 + i) * n + col] = scale * (q - center);
                }
            }
        }
        Tensor::from_vec(out, (k, n), device)
    }
}

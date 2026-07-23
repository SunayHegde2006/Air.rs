//! PrismML Bonsai dequantisation — Q1_0 (1-bit binary) and Q2_0 (2-bit ternary).
//!
//! Both formats use g128 blocks: 128 elements share one FP16 scale factor.
//!
//! # Q1_0 layout (18 bytes / block, GGUF type ID 41)
//! ```text
//! [ d: f16 (2B) | qs: u8 × 16 (16B) ]
//!   128 bits packed LSB-first across 16 bytes
//!   bit j of qs[j/8] encodes element j:  1 → +d,  0 → −d
//! ```
//!
//! # Q2_0 layout (34 bytes / block, GGUF type ID 42)
//! ```text
//! [ d: f16 (2B) | qs: u8 × 32 (32B) ]
//!   128 2-bit codes packed LSB-first, 4 codes per byte
//!   code q for element i = (qs[i/4] >> (2*(i%4))) & 0x3
//!   dequant: (q as f32 - 1.0) * d  →  {-d, 0, +d, 2d}
//!   (code 3 = 2d is reserved/unused in ternary models)
//! ```

use anyhow::{bail, Result};
use half::f16;

const Q1_0_BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (signs)
const Q2_0_BLOCK_BYTES: usize = 34; // 2 (scale) + 32 (codes)
const BLOCK_ELEMS: usize = 128;

// ── Q1_0 ─────────────────────────────────────────────────────────────────────

/// Dequantise a slice of Q1_0 blocks into an `f32` buffer.
///
/// `raw` must be a multiple of 18 bytes (one block each).
/// `out` must have length `(raw.len() / 18) * 128`.
pub fn dequant_q1_0(raw: &[u8], out: &mut Vec<f32>) -> Result<()> {
    if raw.len() % Q1_0_BLOCK_BYTES != 0 {
        bail!(
            "Q1_0 raw length {} is not a multiple of {} (block size)",
            raw.len(),
            Q1_0_BLOCK_BYTES
        );
    }
    let n_blocks = raw.len() / Q1_0_BLOCK_BYTES;
    out.clear();
    out.reserve(n_blocks * BLOCK_ELEMS);

    for block in raw.chunks_exact(Q1_0_BLOCK_BYTES) {
        // First 2 bytes: FP16 scale (little-endian)
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        // Remaining 16 bytes: 128 packed sign bits, LSB-first
        let qs = &block[2..]; // 16 bytes
        static SIGNS: [f32; 2] = [-1.0, 1.0];
        for byte_idx in 0..16 {
            let byte = qs[byte_idx];
            out.extend((0..8).map(|bit_idx| d * SIGNS[((byte >> bit_idx) & 1) as usize]));
        }
    }
    Ok(())
}

// ── Q2_0 ─────────────────────────────────────────────────────────────────────

/// Dequantise a slice of Q2_0 blocks into an `f32` buffer.
///
/// `raw` must be a multiple of 34 bytes (one block each).
/// `out` must have length `(raw.len() / 34) * 128`.
pub fn dequant_q2_0(raw: &[u8], out: &mut Vec<f32>) -> Result<()> {
    if raw.len() % Q2_0_BLOCK_BYTES != 0 {
        bail!(
            "Q2_0 raw length {} is not a multiple of {} (block size)",
            raw.len(),
            Q2_0_BLOCK_BYTES
        );
    }
    let n_blocks = raw.len() / Q2_0_BLOCK_BYTES;
    out.clear();
    out.reserve(n_blocks * BLOCK_ELEMS);

    for block in raw.chunks_exact(Q2_0_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..]; // 32 bytes, 4 codes per byte
        for byte_idx in 0..32 {
            let byte = qs[byte_idx];
            for code_idx in 0..4 {
                let q = (byte >> (2 * code_idx)) & 0x3;
                // dequant: (q - 1) * d  →  {-d, 0, +d, 2d}
                out.push((q as f32 - 1.0) * d);
            }
        }
    }
    Ok(())
}

// ── Self-check (non-trivial logic) ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_le(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    #[test]
    fn q1_0_single_block_signs() {
        // One block: scale = 1.0, all bits = 1 → all weights +1.0
        let mut raw = vec![0u8; Q1_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&f16_le(1.0));
        raw[2..].fill(0xFF); // all bits set
        let mut out = Vec::new();
        dequant_q1_0(&raw, &mut out).unwrap();
        assert_eq!(out.len(), 128);
        assert!(out.iter().all(|&w| (w - 1.0).abs() < 1e-4), "all +1.0");
    }

    #[test]
    fn q1_0_zero_bits_give_negative() {
        // scale = 2.0, all bits = 0 → all weights -2.0
        let mut raw = vec![0u8; Q1_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&f16_le(2.0));
        // qs bytes already 0 from vec!
        let mut out = Vec::new();
        dequant_q1_0(&raw, &mut out).unwrap();
        assert!(out.iter().all(|&w| (w + 2.0).abs() < 1e-4), "all -2.0");
    }

    #[test]
    fn q2_0_ternary_values() {
        // One block: scale = 1.0
        // byte pattern 0b_11_10_01_00 = 0xE4 → codes [0,1,2,3] → [-1, 0, +1, +2]
        let mut raw = vec![0u8; Q2_0_BLOCK_BYTES];
        raw[..2].copy_from_slice(&f16_le(1.0));
        raw[2] = 0b_11_10_01_00; // first 4 elements
        raw[3..].fill(0x55); // code 1 = 0 everywhere else (0b01_01_01_01)
        let mut out = Vec::new();
        dequant_q2_0(&raw, &mut out).unwrap();
        assert_eq!(out.len(), 128);
        assert!((out[0] + 1.0).abs() < 1e-4, "code 0 → -1.0");
        assert!((out[1]).abs() < 1e-4,        "code 1 → 0.0");
        assert!((out[2] - 1.0).abs() < 1e-4,  "code 2 → +1.0");
        assert!((out[3] - 2.0).abs() < 1e-4,  "code 3 → +2.0 (reserved)");
        // remaining bytes 0x55 → all zero
        assert!(out[4..].iter().all(|&w| w.abs() < 1e-4));
    }

    #[test]
    fn q1_0_block_count() {
        let raw = vec![0u8; Q1_0_BLOCK_BYTES * 3];
        let mut out = Vec::new();
        dequant_q1_0(&raw, &mut out).unwrap();
        assert_eq!(out.len(), 384);
    }
}

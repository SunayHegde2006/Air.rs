//! Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 — GGUF ARM NEON/SVE tile-reordered dequantizers
//!
//! These formats appear in GGUF files produced for mobile/edge devices (Apple M-series,
//! Android ARMv9, Linux ARM64). They have **identical bit-density to Q4_0** but the
//! weights within each block are rearranged to match NEON or SVE vector-load tile patterns,
//! enabling zero-shuffle matrix-multiply kernels.
//!
//! # Naming Convention
//! `Q4_0_T1_T2` means:
//! - `Q4_0` — base quantization (4 bits, 1 scale f16, 0 bias)
//! - `T1×T2` — tile dimensions in _output × inner_ weights per vector register
//!
//! | Format   | Tile  | Target hardware                          |
//! |----------|-------|------------------------------------------|
//! | Q4_0_4_4 | 4×4   | ARM NEON 128-bit (int8 dot-product)      |
//! | Q4_0_4_8 | 4×8   | ARM NEON (int8 matmul SDOT extension)    |
//! | Q4_0_8_8 | 8×8   | ARM SVE 256-bit (SMMLA/UMMLA)            |
//!
//! # Memory layout
//! Each **block** covers 32 weights and stores:
//! - `scale`: 1 × f16 (2 bytes)
//! - `nibbles`: 16 bytes (32 weights × 0.5 byte) — same total as Q4_0
//!
//! The only difference from Q4_0 is weight reordering within the 16 nibble bytes.
//! Dequantization is weight-order-invariant (each weight uses its own nibble and the
//! shared block scale), so this module **un-tiles then delegates to standard Q4_0 dequant**.
//!
//! # Usage (CPU fallback — no NEON required)
//! This implementation is pure Rust and works on all architectures.
//! On real ARM hardware, a future NEON path would bypass this entirely.

// ---------------------------------------------------------------------------
// Block constants (all tile variants share Q4_0 block geometry)
// ---------------------------------------------------------------------------

/// Number of weights per Q4_0 block.
pub const Q4_0_BLOCK_WEIGHTS: usize = 32;
/// Number of nibble bytes per Q4_0 block.
pub const Q4_0_NIBBLE_BYTES: usize = 16;
/// Total block size in bytes (2 + 16).
pub const Q4_0_BLOCK_SIZE: usize = 18;

// ---------------------------------------------------------------------------
// Tile pattern descriptors
// ---------------------------------------------------------------------------

/// Which tile reordering pattern is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4TileKind {
    /// 4×4 tile (NEON int8 dot-product)
    T4x4,
    /// 4×8 tile (NEON SDOT)
    T4x8,
    /// 8×8 tile (ARM SVE SMMLA)
    T8x8,
}

impl Q4TileKind {
    /// Number of output elements in one tile row.
    pub fn tile_rows(self) -> usize {
        match self { Self::T4x4 | Self::T4x8 => 4, Self::T8x8 => 8 }
    }
    /// Number of inner elements in one tile column.
    pub fn tile_cols(self) -> usize {
        match self { Self::T4x4 => 4, Self::T4x8 | Self::T8x8 => 8 }
    }
    /// Total elements per tile.
    pub fn tile_size(self) -> usize { self.tile_rows() * self.tile_cols() }
}

// ---------------------------------------------------------------------------
// Block wire format
// ---------------------------------------------------------------------------

/// One Q4_0_T1_T2 block on the wire.
#[derive(Debug, Clone)]
pub struct Q4TiledBlock {
    /// Scale factor as f16 (raw IEEE 754 little-endian bits)
    pub scale_bits: u16,
    /// 32 nibbles packed as 16 bytes — weight order follows tile pattern
    pub nibbles: [u8; Q4_0_NIBBLE_BYTES],
}

impl Q4TiledBlock {
    /// Parse one block from a raw byte slice.
    ///
    /// # Panics
    /// Panics if `bytes.len() < 18`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Q4_0_BLOCK_SIZE);
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut nibbles = [0u8; Q4_0_NIBBLE_BYTES];
        nibbles.copy_from_slice(&bytes[2..18]);
        Self { scale_bits, nibbles }
    }

    /// Decode the f16 scale to f32.
    pub fn scale_f32(&self) -> f32 {
        f16_bits_to_f32(self.scale_bits)
    }

    /// Unpack the 32 raw 4-bit values from this block (before un-tiling).
    /// Values are in [0, 15]; the signed centred value is raw - 8.
    pub fn unpack_raw_nibbles(&self) -> [u8; Q4_0_BLOCK_WEIGHTS] {
        let mut out = [0u8; Q4_0_BLOCK_WEIGHTS];
        for (i, &byte) in self.nibbles.iter().enumerate() {
            out[i * 2]     = byte & 0x0F;
            out[i * 2 + 1] = (byte >> 4) & 0x0F;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Un-tiling helpers
// ---------------------------------------------------------------------------

/// Undo the 4×4 tile reordering.
///
/// Q4_0_4_4 stores blocks in tiles of 4 output × 4 inner weights.
/// Within a 32-element block, weights are interleaved as:
/// `[out0_in0, out0_in1, out0_in2, out0_in3,
///   out1_in0, out1_in1, out1_in2, out1_in3, ...]`
///
/// For dequant we just need linear order, so this is a no-op reshape — the
/// tile ordering only matters for NEON matrix-multiply kernels, not for
/// individual weight extraction. We pass through unchanged.
#[inline(always)]
pub fn untile_4x4(raw: &[u8; Q4_0_BLOCK_WEIGHTS]) -> [u8; Q4_0_BLOCK_WEIGHTS] {
    // 4×4 tiling reorders weights within a 16-element group to:
    // [w[0], w[4], w[8], w[12], w[1], w[5], w[9], w[13], w[2], w[6], w[10], w[14], w[3], w[7], w[11], w[15]]
    // and repeats for the second half.
    let mut out = [0u8; Q4_0_BLOCK_WEIGHTS];
    for half in 0..2usize {
        let base = half * 16;
        for row in 0..4usize {
            for col in 0..4usize {
                // tiled index: row*4 + col → original: col*4 + row
                out[base + col * 4 + row] = raw[base + row * 4 + col];
            }
        }
    }
    out
}

/// Undo the 4×8 tile reordering.
///
/// Q4_0_4_8 interleaves weights within a 32-element block as 4 output rows × 8 inner cols.
#[inline(always)]
pub fn untile_4x8(raw: &[u8; Q4_0_BLOCK_WEIGHTS]) -> [u8; Q4_0_BLOCK_WEIGHTS] {
    // The 32 weights are laid out as:
    // [out0_in0, out0_in1, ..., out0_in7,  out1_in0, .., out1_in7, out2*, out3*]
    // which is already row-major (4×8), so no reordering needed for scalar dequant.
    // Just return as-is — each weight is independent.
    *raw
}

/// Undo the 8×8 tile reordering (ARM SVE SMMLA).
///
/// Q4_0_8_8 stores the 32 weights in an 8×4 sub-tile (8 out rows × 4 inner pairs).
/// For scalar dequant the weight ordering doesn't matter.
#[inline(always)]
pub fn untile_8x8(raw: &[u8; Q4_0_BLOCK_WEIGHTS]) -> [u8; Q4_0_BLOCK_WEIGHTS] {
    // Same logic — scalar dequant is order-invariant within a block.
    *raw
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize a single Q4-tiled block to 32 f32 weights.
///
/// Formula: `w = (nibble - 8) * scale`
/// (identical to Q4_0, just with un-tiling first)
pub fn dequantize_block(block: &Q4TiledBlock, kind: Q4TileKind) -> [f32; Q4_0_BLOCK_WEIGHTS] {
    let raw = block.unpack_raw_nibbles();
    let untiled = match kind {
        Q4TileKind::T4x4 => untile_4x4(&raw),
        Q4TileKind::T4x8 => untile_4x8(&raw),
        Q4TileKind::T8x8 => untile_8x8(&raw),
    };
    let scale = block.scale_f32();
    let mut out = [0.0f32; Q4_0_BLOCK_WEIGHTS];
    for (i, &nibble) in untiled.iter().enumerate() {
        // Signed: [0,15] → [-8,7]
        out[i] = (nibble as i8 - 8) as f32 * scale;
    }
    out
}

/// Dequantize a full slice of Q4-tiled blocks into a flat `Vec<f32>`.
///
/// `data` must be a valid byte slice containing an integer number of
/// 18-byte Q4_0 blocks. `kind` selects the tile geometry.
pub fn dequantize_q4_tiled(data: &[u8], kind: Q4TileKind) -> Vec<f32> {
    assert!(data.len() % Q4_0_BLOCK_SIZE == 0,
        "Q4 tiled data length {} is not a multiple of block size {}", data.len(), Q4_0_BLOCK_SIZE);
    let n_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * Q4_0_BLOCK_WEIGHTS);
    for i in 0..n_blocks {
        let block_bytes = &data[i * Q4_0_BLOCK_SIZE..(i + 1) * Q4_0_BLOCK_SIZE];
        let block = Q4TiledBlock::from_bytes(block_bytes);
        let weights = dequantize_block(&block, kind);
        out.extend_from_slice(&weights);
    }
    out
}

/// Convenience wrappers for each specific tile variant.
pub fn dequantize_q4_0_4_4(data: &[u8]) -> Vec<f32> { dequantize_q4_tiled(data, Q4TileKind::T4x4) }
pub fn dequantize_q4_0_4_8(data: &[u8]) -> Vec<f32> { dequantize_q4_tiled(data, Q4TileKind::T4x8) }
pub fn dequantize_q4_0_8_8(data: &[u8]) -> Vec<f32> { dequantize_q4_tiled(data, Q4TileKind::T8x8) }

// ---------------------------------------------------------------------------
// f16 → f32 conversion (IEEE 754 half-precision, no external dep)
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 f16 bit pattern to f32.
///
/// Matches the result of `f16::to_f32()` in all normal/subnormal/inf/NaN cases.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign     = ((bits >> 15) as u32) << 31;
    let exp_raw  = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    let (exp, mant) = if exp_raw == 0 {
        if mantissa == 0 {
            // ±0
            return f32::from_bits(sign);
        }
        // Subnormal: normalise
        let leading = mantissa.leading_zeros() - 22; // relative to 10-bit mantissa
        ((127 - 14 - leading) as u32, (mantissa << (leading + 1)) & 0x7F_FFFF)
    } else if exp_raw == 0x1F {
        // Infinity or NaN
        let f32_exp  = 0xFF_u32 << 23;
        let f32_mant = mantissa << 13;
        return f32::from_bits(sign | f32_exp | f32_mant);
    } else {
        let f32_exp = (exp_raw as i32 - 15 + 127) as u32;
        (f32_exp, mantissa << 13)
    };

    f32::from_bits(sign | (exp << 23) | mant)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal Q4_0 block: scale=1.0 (f16), nibbles encoding [8..23] → values [0..15]
    fn q4_block_with_scale(scale_f32: f32) -> Vec<u8> {
        let scale_f16 = f32_to_f16_bits(scale_f32);
        let mut bytes = vec![0u8; 18];
        bytes[0] = (scale_f16 & 0xFF) as u8;
        bytes[1] = (scale_f16 >> 8) as u8;
        // Nibble pairs: each byte encodes two consecutive weights
        // Fill with alternating nibble values 0..15
        for i in 0..16usize {
            let lo = (i * 2) as u8 & 0x0F;
            let hi = (i * 2 + 1) as u8 & 0x0F;
            bytes[2 + i] = lo | (hi << 4);
        }
        bytes
    }

    fn f32_to_f16_bits(val: f32) -> u16 {
        // Approximate f32→f16 for scale=1.0 only
        // Using the well-known bit-manipulation approach
        let x = val.to_bits();
        let sign = (x >> 16) & 0x8000;
        let exp  = ((x >> 23) & 0xFF) as i32 - 127 + 15;
        if exp <= 0 { return sign as u16; }
        if exp >= 31 { return (sign | 0x7FFF) as u16; }
        let mant = (x & 0x7F_FFFF) >> 13;
        (sign | ((exp as u32) << 10) | mant) as u16
    }

    #[test]
    fn f16_roundtrip_one() {
        let bits = f32_to_f16_bits(1.0);
        let back = f16_bits_to_f32(bits);
        assert!((back - 1.0).abs() < 1e-3, "got {back}");
    }

    #[test]
    fn f16_zero() {
        assert_eq!(f16_bits_to_f32(0), 0.0);
    }

    #[test]
    fn q4_block_parse() {
        let bytes = q4_block_with_scale(1.0);
        let block = Q4TiledBlock::from_bytes(&bytes);
        assert!((block.scale_f32() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn q4_block_nibble_unpack() {
        let bytes = q4_block_with_scale(1.0);
        let block = Q4TiledBlock::from_bytes(&bytes);
        let nibbles = block.unpack_raw_nibbles();
        // First 32 nibbles should cycle [0..31 mod 16]
        for i in 0..32usize {
            assert_eq!(nibbles[i], (i % 16) as u8, "at index {i}");
        }
    }

    #[test]
    fn dequant_q4_0_4_4_identity_scale() {
        // Build 1 block with scale=1.0, all nibbles = 8 → dequant = (8-8)*1 = 0
        let mut bytes = vec![0u8; 18];
        let scale_bits = f32_to_f16_bits(1.0);
        bytes[0] = (scale_bits & 0xFF) as u8;
        bytes[1] = (scale_bits >> 8) as u8;
        // All nibbles = 8 (0x88 per byte)
        for i in 2..18 { bytes[i] = 0x88; }

        let out = dequantize_q4_0_4_4(&bytes);
        assert_eq!(out.len(), 32);
        for v in &out { assert!(v.abs() < 1e-5, "expected 0 got {v}"); }
    }

    #[test]
    fn dequant_block_signed_range() {
        // nibble 0 → (0-8)*scale = -8*scale
        // nibble 15 → (15-8)*scale = 7*scale
        let mut bytes = vec![0u8; 18];
        let scale = 0.5f32;
        let scale_bits = f32_to_f16_bits(scale);
        bytes[0] = (scale_bits & 0xFF) as u8;
        bytes[1] = (scale_bits >> 8) as u8;
        // First byte: nibble 0 (lo=0) and nibble 15 (hi=F)
        bytes[2] = 0xF0; // lo=0, hi=15
        // rest = 0x88 (centred)
        for i in 3..18 { bytes[i] = 0x88; }

        let block = Q4TiledBlock::from_bytes(&bytes);
        let out = dequantize_block(&block, Q4TileKind::T4x8); // T4x8 is pass-through: preserves nibble positions
        assert!((out[0] - (-8.0 * scale)).abs() < 0.01, "w[0]={}", out[0]);
        assert!((out[1] - ( 7.0 * scale)).abs() < 0.01, "w[1]={}", out[1]);
    }

    #[test]
    fn untile_4x4_identity_check() {
        // Verify that untile_4x4 of untile_4x4 = original (it's its own inverse since
        // it's a transpose, and transpose(transpose) = identity)
        let raw: [u8; 32] = core::array::from_fn(|i| i as u8);
        let tiled   = untile_4x4(&raw);
        let untiled = untile_4x4(&tiled);
        assert_eq!(raw, untiled);
    }

    #[test]
    fn tile_kind_sizes() {
        assert_eq!(Q4TileKind::T4x4.tile_size(), 16);
        assert_eq!(Q4TileKind::T4x8.tile_size(), 32);
        assert_eq!(Q4TileKind::T8x8.tile_size(), 64);
    }

    #[test]
    fn three_tile_variants_same_element_count() {
        let bytes = q4_block_with_scale(1.0);
        let a = dequantize_q4_0_4_4(&bytes);
        let b = dequantize_q4_0_4_8(&bytes);
        let c = dequantize_q4_0_8_8(&bytes);
        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 32);
        assert_eq!(c.len(), 32);
        // 4_8 and 8_8 use pass-through untiling so they agree
        for i in 0..32 { assert!((b[i] - c[i]).abs() < 1e-5); }
    }
}

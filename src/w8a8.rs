//! W8A8 Direct Matmul — INT8 weight × INT8 activation multiply-accumulate.
//!
//! Inner k-loop uses AVX-512 VNNI (`vpdpbusd`) when the CPU supports it
//! (compile with `RUSTFLAGS="-C target-cpu=native"` or `-C target-feature=+avx512vnni`).
//! Falls back to scalar tiled accumulation on non-AVX-512 targets.
//!
//! Tile size mirrors an MMA warp tile (16×16 per warp fragment):
//!   - m_tile = 16, n_tile = 16, k_tile = 16
//! Accumulator kept as i32 per element (avoids saturation until writeback).
//! Final scaling: out_f32 = (acc_i32 × scale_w × scale_x) / 127²

#[cfg(target_feature = "avx512vnni")]
use std::arch::x86_64::{
    __m512i,
    _mm512_setzero_si512,
    _mm512_dpbusd_epi32,
    _mm512_loadu_si512,
    _mm512_reduce_add_epi32,
};

/// Output of a W8A8 matmul call.
pub struct W8A8Output {
    /// Dequantized f32 output, shape [m, n], row-major.
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// W8A8 matmul: `out[m,n] = lhs[m,k] × rhs[k,n]` in INT8, dequantized to f32.
///
/// # Arguments
/// * `lhs`     — INT8 activations, shape [m, k], row-major.
/// * `rhs`     — INT8 weights, shape [k, n], row-major.
/// * `m,k,n`   — dimensions.
/// * `scale_x` — activation scale (float16 absmax / 127).
/// * `scale_w` — weight scale (float16 absmax / 127).
///
/// # Panics
/// Panics if `lhs.len() != m*k` or `rhs.len() != k*n`.
pub fn w8a8_matmul(
    lhs: &[i8],
    rhs: &[i8],
    m: usize,
    k: usize,
    n: usize,
    scale_x: f32,
    scale_w: f32,
) -> W8A8Output {
    assert_eq!(lhs.len(), m * k, "lhs size mismatch");
    assert_eq!(rhs.len(), k * n, "rhs size mismatch");

    const TILE: usize = 16; // MMA warp tile width
    let mut out = vec![0.0f32; m * n];
    let dequant = scale_x * scale_w / (127.0 * 127.0);

    let m_tiles = m.div_ceil(TILE);
    let n_tiles = n.div_ceil(TILE);
    let k_tiles = k.div_ceil(TILE);

    for mt in 0..m_tiles {
        for nt in 0..n_tiles {
            let mut acc = [0i32; TILE * TILE];

            for kt in 0..k_tiles {
                let m0 = mt * TILE;
                let n0 = nt * TILE;
                let k0 = kt * TILE;

                let m_end = (m0 + TILE).min(m);
                let n_end = (n0 + TILE).min(n);
                let k_end = (k0 + TILE).min(k);

                for i in m0..m_end {
                    let ai = i - m0;
                    // --- AVX-512 VNNI path -------------------------------------------
                    // vpdpbusd: acc += u8(a) * i8(b), 4 elements per lane, 16 lanes.
                    // Processes the k dimension in chunks of 64 bytes (16×i32 lanes × 4).
                    // Requires lhs values to be non-negative (u8 domain). W8A8 uses
                    // symmetric int8 weights; activations can be negative. We shift
                    // activations by 128 to u8 and subtract the corrective term after.
                    //
                    // For each output column j in [n0, n_end):
                    //   acc[ai][j-n0] += sum_p( lhs[i,p] * rhs[p,j] )
                    //
                    // The VNNI loop processes the n dimension one column at a time and
                    // the k dimension in 64-element strides using a 512-bit register.
                    // Remaining k elements fall through to the scalar tail.
                    #[cfg(target_feature = "avx512vnni")]
                    {
                        for j in n0..n_end {
                            let ji = j - n0;
                            let mut k_idx = k0;
                            // Accumulate into a single 512-bit __m512i register.
                            // Each i32 lane accumulates 4 i8 pairs via vpdpbusd.
                            let mut vacc: __m512i = unsafe { _mm512_setzero_si512() };
                            // Process 64 k-elements per iteration (16 lanes × 4 bytes).
                            while k_idx + 64 <= k_end {
                                // Build 64-byte a-vector: shift lhs i8 → u8 by +128.
                                let mut a_buf = [0u8; 64];
                                let mut b_buf = [0i8; 64];
                                for q in 0..64 {
                                    // u8 domain: add 128 to make non-negative
                                    a_buf[q] = (lhs[(i) * k + k_idx + q] as i16 + 128) as u8;
                                    b_buf[q] = rhs[(k_idx + q) * n + j];
                                }
                                let va = unsafe {
                                    _mm512_loadu_si512(a_buf.as_ptr() as *const __m512i)
                                };
                                let vb = unsafe {
                                    _mm512_loadu_si512(b_buf.as_ptr() as *const __m512i)
                                };
                                vacc = unsafe { _mm512_dpbusd_epi32(vacc, va, vb) };
                                k_idx += 64;
                            }
                            // Horizontal reduce the 16 i32 lanes.
                            let vnni_sum = unsafe { _mm512_reduce_add_epi32(vacc) };
                            // Corrective term: sum_p( 128 * rhs[p,j] ) for the 128-shift
                            let mut corr = 0i32;
                            for q in k0..(k0 + (k_idx - k0)) {
                                corr += 128 * rhs[q * n + j] as i32;
                            }
                            acc[ai * TILE + ji] += vnni_sum - corr;
                            // Scalar tail for remaining k < 64
                            for p in k_idx..k_end {
                                acc[ai * TILE + ji] +=
                                    lhs[i * k + p] as i32 * rhs[p * n + j] as i32;
                            }
                        }
                    }
                    // --- Scalar path (non-AVX-512 targets) ---------------------------
                    #[cfg(not(target_feature = "avx512vnni"))]
                    for p in k0..k_end {
                        let a = lhs[i * k + p] as i32;
                        for j in n0..n_end {
                            acc[ai * TILE + (j - n0)] += a * rhs[p * n + j] as i32;
                        }
                    }
                }
            }

            // Writeback dequantized
            let m0 = mt * TILE;
            let n0 = nt * TILE;
            let m_end = (m0 + TILE).min(m);
            let n_end = (n0 + TILE).min(n);
            for i in m0..m_end {
                for j in n0..n_end {
                    out[i * n + j] = acc[(i - m0) * TILE + (j - n0)] as f32 * dequant;
                }
            }
        }
    }

    W8A8Output { data: out, rows: m, cols: n }
}

/// Quantize a f32 slice to INT8 using symmetric per-tensor quantization.
///
/// Returns (quantized bytes, scale = absmax/127).
pub fn quantize_f32_to_i8(src: &[f32]) -> (Vec<i8>, f32) {
    let absmax = src.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
    if absmax == 0.0 {
        return (vec![0i8; src.len()], 1.0);
    }
    let scale = absmax / 127.0;
    let inv = 1.0 / scale;
    let q = src.iter().map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect();
    (q, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matmul() {
        // 2×2 × 2×2 identity: [[1,0],[0,1]] × [[2,0],[0,3]] = [[2,0],[0,3]]
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![2, 0, 0, 3];
        let out = w8a8_matmul(&a, &b, 2, 2, 2, 127.0, 127.0);
        // dequant = 127*127/(127*127) = 1.0
        assert!((out.data[0] - 2.0).abs() < 0.01);
        assert!((out.data[1]).abs() < 0.01);
        assert!((out.data[2]).abs() < 0.01);
        assert!((out.data[3] - 3.0).abs() < 0.01);
    }

    #[test]
    fn quantize_round_trip() {
        let src = vec![1.0f32, -0.5, 0.25, -0.125];
        let (q, scale) = quantize_f32_to_i8(&src);
        let restored: Vec<f32> = q.iter().map(|&x| x as f32 * scale).collect();
        for (a, b) in src.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.01, "round-trip error: {a} vs {b}");
        }
    }

    #[test]
    fn large_tile_no_panic() {
        // Exercises the tiling path (33 > 16 tile width)
        let m = 33; let k = 17; let n = 20;
        let a = vec![1i8; m * k];
        let b = vec![1i8; k * n];
        let out = w8a8_matmul(&a, &b, m, k, n, 1.0, 1.0);
        assert_eq!(out.data.len(), m * n);
    }
}

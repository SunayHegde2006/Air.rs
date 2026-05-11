//! FlashDecoding++ — v0.5.0
//!
//! Correct parallel softmax reduction across split-k KV chunks for the
//! decode phase (seq_len=1, kv_len ≥ 4096).
//!
//! # Research Basis
//!
//! - **FlashDecoding** (Dao et al., 2023): splits the KV sequence into P chunks,
//!   each chunk computes partial attention scores in parallel, then reduces via
//!   the log-sum-exp trick. Enables decode-phase parallelism across KV length
//!   rather than batch.
//!
//! - **FlashDecoding++** (Hong et al., ICLR 2024): adds unified max statistics
//!   (flat GEMV instead of sequential scan), achieving 40% additional latency
//!   reduction vs FlashDecoding on A100 for long-context decode.
//!
//! # Algorithm (CPU reference implementation)
//!
//! For each query vector q ∈ ℝ^d and KV cache K,V ∈ ℝ^{n×d}:
//!
//! ```text
//! 1. Split K,V into P chunks of size chunk_size.
//!    Chunk p: K_p ∈ ℝ^{C×d}, V_p ∈ ℝ^{C×d}
//!
//! 2. Per chunk p (parallelisable):
//!    s_p = K_p · q / √d          (scores, shape C)
//!    m_p = max(s_p)              (local max for numerical stability)
//!    a_p = exp(s_p - m_p)       (unnormalised attention weights)
//!    l_p = sum(a_p)             (partial sum of exp)
//!    o_p = a_p^T · V_p / l_p    (partial output, shape d)
//!
//! 3. Global log-sum-exp reduction:
//!    m* = max_p(m_p)
//!    a*_p = exp(m_p - m*) · l_p  (rescaled sum)
//!    L = sum_p(a*_p)
//!    o = sum_p(a*_p · o_p) / L   (final output)
//! ```
//!
//! # CUDA/Metal Note
//!
//! This file provides a numerically correct CPU reference. The production
//! CUDA kernel would map each chunk to one thread block; the reduction
//! step is a small all-reduce across P blocks.

// ── Constants ──────────────────────────────────────────────────────────────

/// Default number of KV tokens per split-k chunk.
pub const DEFAULT_CHUNK_SIZE: usize = 64;

// ── Core Computation ───────────────────────────────────────────────────────

/// Partial attention output for one KV chunk.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    /// Partial output vector o_p.
    pub output: Vec<f32>,
    /// Local maximum score m_p (for log-sum-exp reduction).
    pub local_max: f32,
    /// Sum of unnormalised attention weights l_p = Σ exp(s - m_p).
    pub local_sum: f32,
}

/// Compute partial attention for one KV chunk.
///
/// # Parameters
/// - `q`: query vector, length `head_dim`
/// - `k_chunk`: key chunk, length `chunk_len × head_dim` (row-major)
/// - `v_chunk`: value chunk, same shape as `k_chunk`
/// - `head_dim`: head dimension d
pub fn chunk_attention(
    q: &[f32],
    k_chunk: &[f32],
    v_chunk: &[f32],
    head_dim: usize,
) -> ChunkOutput {
    assert_eq!(q.len(), head_dim);
    let chunk_len = k_chunk.len() / head_dim;
    assert_eq!(k_chunk.len(), chunk_len * head_dim);
    assert_eq!(v_chunk.len(), k_chunk.len());

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Step 1: s_p = K_p · q / √d
    let scores: Vec<f32> = (0..chunk_len)
        .map(|i| {
            let k_row = &k_chunk[i * head_dim..(i + 1) * head_dim];
            let dot: f32 = q.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
            dot * scale
        })
        .collect();

    // Step 2: m_p = max(s_p)
    let local_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Step 3: a_p = exp(s_p - m_p), l_p = Σ a_p
    let attn_weights: Vec<f32> = scores.iter().map(|&s| (s - local_max).exp()).collect();
    let local_sum: f32 = attn_weights.iter().sum();

    // Step 4: o_p = (a_p / l_p) · V_p
    let mut output = vec![0.0f32; head_dim];
    for (i, &w) in attn_weights.iter().enumerate() {
        let v_row = &v_chunk[i * head_dim..(i + 1) * head_dim];
        for (o, &v) in output.iter_mut().zip(v_row.iter()) {
            *o += (w / local_sum) * v;
        }
    }

    ChunkOutput { output, local_max, local_sum }
}

/// Reduce multiple `ChunkOutput`s into a single attention output vector.
///
/// Uses the numerically stable log-sum-exp combination from FlashDecoding.
pub fn reduce_chunks(chunks: &[ChunkOutput], head_dim: usize) -> Vec<f32> {
    assert!(!chunks.is_empty(), "cannot reduce zero chunks");

    // Global max
    let global_max = chunks
        .iter()
        .map(|c| c.local_max)
        .fold(f32::NEG_INFINITY, f32::max);

    // Rescaled sums: a*_p = exp(m_p - m*) · l_p
    let rescaled_sums: Vec<f32> = chunks
        .iter()
        .map(|c| (c.local_max - global_max).exp() * c.local_sum)
        .collect();

    let total_sum: f32 = rescaled_sums.iter().sum();
    assert!(total_sum > 1e-8, "total_sum near zero — all scores -inf?");

    // Weighted sum of partial outputs
    let mut output = vec![0.0f32; head_dim];
    for (chunk, &rs) in chunks.iter().zip(rescaled_sums.iter()) {
        let weight = rs / total_sum;
        for (o, &v) in output.iter_mut().zip(chunk.output.iter()) {
            *o += weight * v;
        }
    }

    output
}

// ── Flash Decode ───────────────────────────────────────────────────────────

/// Decode-phase attention: q ∈ ℝ^d, KV ∈ ℝ^{n×d} → output ∈ ℝ^d.
///
/// This is the O(n) FlashDecoding++ CPU reference: split KV into P chunks,
/// compute partial outputs in parallel, reduce via log-sum-exp.
///
/// In production, replace with a CUDA/Metal kernel where each chunk
/// maps to one thread block.
pub fn flash_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k_cache.len(), kv_len * head_dim);
    assert_eq!(v_cache.len(), kv_len * head_dim);
    assert!(chunk_size > 0);

    let chunks: Vec<ChunkOutput> = k_cache
        .chunks(chunk_size * head_dim)
        .zip(v_cache.chunks(chunk_size * head_dim))
        .map(|(k_chunk, v_chunk)| chunk_attention(q, k_chunk, v_chunk, head_dim))
        .collect();

    reduce_chunks(&chunks, head_dim)
}

/// Naive O(n) reference implementation for correctness validation.
///
/// Standard softmax attention without chunking. Used to verify
/// `flash_decode` produces numerically equivalent output.
pub fn reference_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores: Vec<f32> = (0..kv_len)
        .map(|i| {
            let k = &k_cache[i * head_dim..(i + 1) * head_dim];
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale
        })
        .collect();

    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut out = vec![0.0f32; head_dim];
    for (i, &w) in weights.iter().enumerate() {
        let v = &v_cache[i * head_dim..(i + 1) * head_dim];
        for (o, &vi) in out.iter_mut().zip(v.iter()) {
            *o += w * vi;
        }
    }
    out
}

/// L∞ (max absolute) error between two vectors.
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
    }

    fn randn(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg(&mut s)).collect()
    }

    const HD: usize = 64;

    #[test]
    fn flash_decode_matches_reference_small() {
        // 128 KV tokens, chunk_size=32
        let kv_len = 128;
        let q = randn(HD, 1);
        let k = randn(kv_len * HD, 2);
        let v = randn(kv_len * HD, 3);

        let flash_out = flash_decode(&q, &k, &v, kv_len, HD, 32);
        let ref_out = reference_attention(&q, &k, &v, kv_len, HD);

        let err = max_abs_error(&flash_out, &ref_out);
        assert!(err < 1e-4, "L∞ error={err} > 1e-4 (small case)");
    }

    #[test]
    fn flash_decode_matches_reference_large() {
        // 4096 KV tokens, chunk_size=64 (main decode perf scenario)
        let kv_len = 4096;
        let q = randn(HD, 10);
        let k = randn(kv_len * HD, 11);
        let v = randn(kv_len * HD, 12);

        let flash_out = flash_decode(&q, &k, &v, kv_len, HD, DEFAULT_CHUNK_SIZE);
        let ref_out = reference_attention(&q, &k, &v, kv_len, HD);

        let err = max_abs_error(&flash_out, &ref_out);
        assert!(err < 1e-3, "L∞ error={err} > 1e-3 (large case)");
    }

    #[test]
    fn flash_decode_single_chunk_equals_reference() {
        // chunk_size ≥ kv_len → one chunk → should exactly match
        let kv_len = 32;
        let q = randn(HD, 20);
        let k = randn(kv_len * HD, 21);
        let v = randn(kv_len * HD, 22);

        let flash_out = flash_decode(&q, &k, &v, kv_len, HD, kv_len);
        let ref_out = reference_attention(&q, &k, &v, kv_len, HD);

        let err = max_abs_error(&flash_out, &ref_out);
        assert!(err < 1e-5, "single-chunk L∞ error={err}");
    }

    #[test]
    fn flash_decode_output_length_correct() {
        let kv_len = 256;
        let q = randn(HD, 5);
        let k = randn(kv_len * HD, 6);
        let v = randn(kv_len * HD, 7);
        let out = flash_decode(&q, &k, &v, kv_len, HD, DEFAULT_CHUNK_SIZE);
        assert_eq!(out.len(), HD);
    }

    #[test]
    fn chunk_outputs_local_sum_positive() {
        let kv_len = 16;
        let q = randn(HD, 99);
        let k = randn(kv_len * HD, 100);
        let v = randn(kv_len * HD, 101);
        let chunk = chunk_attention(&q, &k, &v, HD);
        assert!(chunk.local_sum > 0.0, "local_sum must be positive");
    }

    #[test]
    fn reduce_single_chunk_passthrough() {
        let output = vec![1.0, 2.0, 3.0];
        let chunk = ChunkOutput { output: output.clone(), local_max: 0.0, local_sum: 1.0 };
        let reduced = reduce_chunks(&[chunk], 3);
        for (a, b) in reduced.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "single chunk passthrough failed");
        }
    }
}

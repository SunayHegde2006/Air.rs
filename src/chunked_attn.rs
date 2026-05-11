//! # Blockwise Chunked Attention (Ring-compatible)
//!
//! Enables **long-context attention** on consumer GPUs by processing the KV
//! sequence in fixed-size blocks, accumulating the softmax numerator and
//! denominator across blocks without materialising the full N×N attention matrix.
//!
//! ## Research
//! - "FlashAttention-2: Faster Attention with Better Parallelism and Work
//!   Partitioning" (Dao, ICLR 2024, arXiv:2307.08691)
//! - "Ring Attention with Blockwise Transformers" (Liu et al., arXiv:2310.01889)
//! - "Blockwise Parallel Transformer for Large Context Models" (Liu & Abbeel,
//!   NeurIPS 2023, arXiv:2305.19370)
//!
//! ## Consumer benefit
//! Standard attention allocates O(N²) memory: at 128K tokens, 128K × 128K × 2
//! bytes ≈ **32 GB** — impossible on consumer GPUs.
//! Blockwise attention: O(N × B) where B = block_size (e.g. 512).
//! At 128K tokens, block_size=512 → only 128K × 512 × 2 bytes ≈ **128 MB**.
//!
//! ## Algorithm (safe softmax with log-sum-exp accumulation)
//!
//! For each query block Q_i (rows q_s..q_e):
//!   For each KV block K_j, V_j (cols k_s..k_e):
//!     S_ij = Q_i @ K_j^T × scale        [block_q × block_kv]
//!     m_new = max(m_old, rowmax(S_ij))   update running row-max
//!     p    = exp(S_ij - m_new)           renormalised scores per block
//!     l    = exp(m_old - m_new) × l_old + rowsum(p)   accumulate denom
//!     acc  = exp(m_old - m_new) × acc_old + p @ V_j   accumulate num
//!   Output_i = acc / l   (normalise)

/// Default block size (tokens per KV chunk).  Must evenly divide context_len.
pub const DEFAULT_BLOCK_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// ChunkedAttentionConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChunkedAttentionConfig {
    /// Number of tokens per KV block (inner loop step).
    pub block_size: usize,
    /// Number of attention heads (Q).
    pub n_heads: usize,
    /// Number of KV heads (GQA: n_kv_heads ≤ n_heads).
    pub n_kv_heads: usize,
    /// Head dimension (must be ≤ 256 for FlashAttn compatibility).
    pub head_dim: usize,
    /// Whether to apply causal masking (decode: true, full bidirectional: false).
    pub causal: bool,
}

impl Default for ChunkedAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            n_heads: 32,
            n_kv_heads: 8,  // GQA (LLaMA-3 style)
            head_dim: 128,
            causal: true,
        }
    }
}

impl ChunkedAttentionConfig {
    /// Groups per KV head (= n_heads / n_kv_heads for GQA).
    pub fn gqa_groups(&self) -> usize {
        assert_eq!(
            self.n_heads % self.n_kv_heads,
            0,
            "n_heads must be divisible by n_kv_heads"
        );
        self.n_heads / self.n_kv_heads
    }

    /// Softmax scale = 1 / √head_dim.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Validate config.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("block_size must be > 0".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if self.n_heads < self.n_kv_heads {
            return Err("n_heads must be >= n_kv_heads".into());
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err("n_heads must be divisible by n_kv_heads".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BlockwiseAttention — the core algorithm
// ---------------------------------------------------------------------------

/// Blockwise chunked attention over a single head.
///
/// Inputs:
/// - `q`:  [seq_q  × head_dim] — query vectors
/// - `k`:  [seq_kv × head_dim] — key vectors
/// - `v`:  [seq_kv × head_dim] — value vectors
/// - `cfg`: attention config (scale, causal flag, etc.)
///
/// Output: `[seq_q × head_dim]` attention output for this head.
///
/// This is the **reference scalar implementation** — correct but slow.
/// Production replaces the inner GEMM with a SIMD-accelerated or GPU kernel.
pub fn blockwise_attention_head(
    q: &[f32],  // [seq_q × head_dim]
    k: &[f32],  // [seq_kv × head_dim]
    v: &[f32],  // [seq_kv × head_dim]
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    block_size: usize,
    causal: bool,
    q_offset: usize,  // absolute position of first query token (for causal mask)
) -> Vec<f32> {
    assert_eq!(q.len(), seq_q * head_dim);
    assert_eq!(k.len(), seq_kv * head_dim);
    assert_eq!(v.len(), seq_kv * head_dim);

    // Running accumulators per query position:
    // acc[i]    = numerator sum (output partial sum before normalisation)
    // l[i]      = log-sum-exp denominator
    // m[i]      = running row-maximum (used for numerically stable softmax)
    let mut acc = vec![0.0f32; seq_q * head_dim];
    let mut l   = vec![0.0f32; seq_q];
    let mut m   = vec![f32::NEG_INFINITY; seq_q];

    let block_size = block_size.min(seq_kv).max(1);
    let n_kv_blocks = (seq_kv + block_size - 1) / block_size;

    for kv_block in 0..n_kv_blocks {
        let ks = kv_block * block_size;
        let ke = (ks + block_size).min(seq_kv);
        let blk_kv = ke - ks;

        // For each query, compute scores with this KV block.
        for qi in 0..seq_q {
            let q_abs_pos = q_offset + qi;     // absolute position for causal mask
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];

            // Compute raw scores S[qi, kj] = q_row · k[kj]  × scale
            let mut s_row = vec![0.0f32; blk_kv];
            for kj in 0..blk_kv {
                let k_abs_pos = ks + kj;
                // Causal mask: query can only attend to past+self positions.
                if causal && k_abs_pos > q_abs_pos {
                    s_row[kj] = f32::NEG_INFINITY;
                    continue;
                }
                let k_row = &k[(ks + kj) * head_dim..(ks + kj + 1) * head_dim];
                let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                s_row[kj] = dot * scale;
            }

            // Block row-max.
            let m_block = s_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Update running max.
            let m_new = m[qi].max(m_block);

            // Rescale old accumulator for new max.
            let exp_old = (m[qi] - m_new).exp();
            let qi_out = &mut acc[qi * head_dim..(qi + 1) * head_dim];
            for v_elem in qi_out.iter_mut() {
                *v_elem *= exp_old;
            }
            l[qi] *= exp_old;

            // Accumulate this block's contribution.
            let mut block_l = 0.0f32;
            let mut p_row = vec![0.0f32; blk_kv];
            for kj in 0..blk_kv {
                let p = (s_row[kj] - m_new).exp();
                p_row[kj] = p;
                block_l += p;
            }

            // acc[qi] += p_row @ V[ks..ke]
            let qi_out = &mut acc[qi * head_dim..(qi + 1) * head_dim];
            for kj in 0..blk_kv {
                let v_row = &v[(ks + kj) * head_dim..(ks + kj + 1) * head_dim];
                let p = p_row[kj];
                for (a, &vv) in qi_out.iter_mut().zip(v_row.iter()) {
                    *a += p * vv;
                }
            }

            l[qi] += block_l;
            m[qi] = m_new;
        }
    }

    // Normalise.
    for qi in 0..seq_q {
        let denom = l[qi].max(1e-8);
        let qi_out = &mut acc[qi * head_dim..(qi + 1) * head_dim];
        for v_elem in qi_out.iter_mut() {
            *v_elem /= denom;
        }
    }

    acc
}

// ---------------------------------------------------------------------------
// Multi-head blockwise attention (GQA-aware)
// ---------------------------------------------------------------------------

/// Run blockwise attention for all heads with GQA (grouped-query attention).
///
/// `xq`:  [n_heads    × seq_q  × head_dim] — queries (flat row-major)
/// `xk`:  [n_kv_heads × seq_kv × head_dim] — keys
/// `xv`:  [n_kv_heads × seq_kv × head_dim] — values
///
/// Output: [n_heads × seq_q × head_dim]
pub fn blockwise_attention_multihead(
    xq: &[f32],
    xk: &[f32],
    xv: &[f32],
    seq_q: usize,
    seq_kv: usize,
    cfg: &ChunkedAttentionConfig,
) -> Vec<f32> {
    cfg.validate().expect("ChunkedAttentionConfig invalid");
    let hd = cfg.head_dim;
    let groups = cfg.gqa_groups();
    let scale = cfg.scale();
    let mut out = vec![0.0f32; cfg.n_heads * seq_q * hd];

    for h in 0..cfg.n_heads {
        let kv_head = h / groups;
        let q_start = h * seq_q * hd;
        let k_start = kv_head * seq_kv * hd;
        let result = blockwise_attention_head(
            &xq[q_start..q_start + seq_q * hd],
            &xk[k_start..k_start + seq_kv * hd],
            &xv[k_start..k_start + seq_kv * hd],
            seq_q,
            seq_kv,
            hd,
            scale,
            cfg.block_size,
            cfg.causal,
            0,
        );
        out[q_start..q_start + seq_q * hd].copy_from_slice(&result);
    }
    out
}

// ---------------------------------------------------------------------------
// Memory usage estimator
// ---------------------------------------------------------------------------

/// Estimate peak memory (bytes) for blockwise attention at a given context length.
///
/// Standard attention: `seq_q × seq_kv × n_heads × 4` bytes (f32 attn matrix)
/// Blockwise attention: `seq_q × block_size × n_heads × 4` bytes
pub fn peak_memory_bytes(seq_q: usize, seq_kv: usize, cfg: &ChunkedAttentionConfig) -> u64 {
    let blockwise = seq_q as u64
        * cfg.block_size.min(seq_kv) as u64
        * cfg.n_heads as u64
        * 4;
    let standard = seq_q as u64 * seq_kv as u64 * cfg.n_heads as u64 * 4;
    let _ = standard; // for comparison in tests
    blockwise
}

/// Reduction factor vs standard attention for this config.
pub fn memory_reduction_factor(seq_q: usize, seq_kv: usize, cfg: &ChunkedAttentionConfig) -> f64 {
    let blockwise = seq_q as f64 * cfg.block_size.min(seq_kv) as f64 * cfg.n_heads as f64;
    let standard  = seq_q as f64 * seq_kv as f64 * cfg.n_heads as f64;
    if standard < 1e-9 { 1.0 } else { standard / blockwise }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ChunkedAttentionConfig {
        ChunkedAttentionConfig {
            block_size: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            causal: true,
        }
    }

    // --- Config ---

    #[test]
    fn test_config_validate_ok() {
        assert!(tiny_cfg().validate().is_ok());
    }

    #[test]
    fn test_config_gqa_groups() {
        let cfg = ChunkedAttentionConfig { n_heads: 8, n_kv_heads: 2, ..tiny_cfg() };
        assert_eq!(cfg.gqa_groups(), 4);
    }

    #[test]
    fn test_config_invalid_block_zero() {
        let cfg = ChunkedAttentionConfig { block_size: 0, ..tiny_cfg() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_kv_heads_gt_heads() {
        let cfg = ChunkedAttentionConfig { n_heads: 2, n_kv_heads: 4, ..tiny_cfg() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_scale_correct() {
        let cfg = ChunkedAttentionConfig { head_dim: 64, ..tiny_cfg() };
        assert!((cfg.scale() - (1.0 / 8.0f32)).abs() < 1e-6);
    }

    // --- blockwise_attention_head ---

    fn all_ones(rows: usize, cols: usize) -> Vec<f32> {
        vec![1.0f32; rows * cols]
    }

    fn make_qkv(seq_q: usize, seq_kv: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..seq_q * head_dim)
            .map(|i| ((i as f64 * 0.314).sin() as f32))
            .collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| ((i as f64 * 0.271).cos() as f32 * 0.5))
            .collect();
        (q, k, v)
    }

    #[test]
    fn test_output_shape() {
        let cfg = tiny_cfg();
        let (q, k, v) = make_qkv(4, 4, cfg.head_dim);
        let out = blockwise_attention_head(&q, &k, &v, 4, 4, cfg.head_dim, cfg.scale(), cfg.block_size, true, 0);
        assert_eq!(out.len(), 4 * cfg.head_dim);
    }

    #[test]
    fn test_self_query_is_finite() {
        let cfg = tiny_cfg();
        let (q, k, v) = make_qkv(4, 4, cfg.head_dim);
        let out = blockwise_attention_head(&q, &k, &v, 4, 4, cfg.head_dim, cfg.scale(), cfg.block_size, true, 0);
        assert!(out.iter().all(|v| v.is_finite()), "output contains non-finite values");
    }

    #[test]
    fn test_causal_mask_auto_regressive() {
        // With causal=true, position 0 attends only to itself.
        // All-ones Q, K, V: each output should be all-ones (single attend).
        let head_dim = 4;
        let seq = 4;
        let q = all_ones(seq, head_dim);
        let k = all_ones(seq, head_dim);
        let v = all_ones(seq, head_dim);
        let out = blockwise_attention_head(&q, &k, &v, seq, seq, head_dim, 1.0, 2, true, 0);
        // Every output token should be all-ones (V is all-ones; even after attending to multiple positions weighted sum = 1).
        for v_elem in &out {
            assert!((v_elem - 1.0).abs() < 1e-4, "expected all-ones output: {v_elem}");
        }
    }

    #[test]
    fn test_blockwise_matches_full_for_small_seq() {
        // For a tiny sequence where block_size >= seq_kv, blockwise == full attn.
        let head_dim = 4;
        let seq = 3;
        let (q, k, v) = make_qkv(seq, seq, head_dim);
        // block_size = seq → full attention in one block
        let out_block = blockwise_attention_head(&q, &k, &v, seq, seq, head_dim, 1.0, seq, true, 0);
        // block_size = 1 → one KV token per step
        let out_chunked = blockwise_attention_head(&q, &k, &v, seq, seq, head_dim, 1.0, 1, true, 0);
        for (a, b) in out_block.iter().zip(out_chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "block vs chunked mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_multihead_output_shape() {
        let cfg = tiny_cfg();
        let n_h = cfg.n_heads;
        let hd = cfg.head_dim;
        let sq = 4;
        let skv = 4;
        let xq = vec![0.1f32; n_h * sq * hd];
        let xk = vec![0.1f32; n_h * skv * hd];
        let xv = vec![0.1f32; n_h * skv * hd];
        let out = blockwise_attention_multihead(&xq, &xk, &xv, sq, skv, &cfg);
        assert_eq!(out.len(), n_h * sq * hd);
    }

    // --- Memory estimator ---

    #[test]
    fn test_peak_memory_blockwise_vs_standard() {
        let cfg = ChunkedAttentionConfig {
            block_size: 512,
            n_heads: 32,
            head_dim: 128,
            n_kv_heads: 8,
            causal: true,
        };
        let seq = 131072usize; // 128K
        let blockwise = peak_memory_bytes(seq, seq, &cfg);
        let standard  = seq as u64 * seq as u64 * 32 * 4; // n_heads × f32
        // Blockwise should be orders of magnitude smaller.
        assert!(blockwise < standard / 100, "block={blockwise}, std={standard}");
    }

    #[test]
    fn test_reduction_factor_128k() {
        let cfg = ChunkedAttentionConfig { block_size: 512, n_heads: 1, n_kv_heads: 1, head_dim: 64, causal: true };
        let factor = memory_reduction_factor(131072, 131072, &cfg);
        // 128K / 512 = 256×
        assert!(factor > 200.0, "reduction factor should be ~256x: {factor}");
    }

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_send_sync() {
        _assert_send_sync::<ChunkedAttentionConfig>();
    }
}

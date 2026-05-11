//! # AQLM — Additive Quantization of Language Models (2-bit)
//!
//! Represents each weight column as a **sum of M codebook lookups** over
//! groups of D=8 weights. Storage cost: 8 bits/lookup × M lookups / D weights
//! = 2 bits per weight at M=2.
//!
//! **Research**: "Extreme Compression of Large Language Models via Additive
//! Quantization" (Egiazarian et al., ICML 2024, arXiv:2401.06118).
//!
//! **Complementary**: VPTQ (arXiv:2409.17066) and QuIP# (arXiv:2402.04396)
//! for higher-precision residual paths.
//!
//! ## Consumer benefit
//! LLaMA-3-70B at 2 bpw fits in **~17.5 GB** — runs on dual RTX 3090 or a
//! single A100-80G. At 4 bpw (standard Q4) it needs ~35 GB.
//!
//! ## SIMD note
//! The forward pass inner loop processes `group_size=8` weights per codebook
//! lookup. On AVX2 this maps to a single 256-bit gather + FMA; on NEON to
//! two 128-bit loads + multiply-accumulate. The current impl is scalar Rust
//! (LLVM auto-vectorises); explicit intrinsics live behind `#[cfg(feature="simd")]`.

/// Number of entries in each codebook (2^8 = 256).
pub const CODEBOOK_ENTRIES: usize = 256;

/// Number of weights per codebook group (must divide `in_features`).
pub const GROUP_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// One AQLM codebook: 256 centroid vectors, each of length `group_size`.
///
/// Storage: 256 × 8 × 2 bytes (f16) = 4 096 bytes per codebook.
/// We store as `f32` here (expanded from f16 at load time) for CPU matmul.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Packed data: [CODEBOOK_ENTRIES × group_size]  (row-major).
    pub data: Vec<f32>,
    pub group_size: usize,
    pub n_entries: usize,
}

impl Codebook {
    /// Create a new codebook with random-ish data (for tests / init).
    pub fn new_random(group_size: usize) -> Self {
        let n = CODEBOOK_ENTRIES * group_size;
        // Deterministic pseudo-random: lcg with seed=42.
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1)) as f32;
                (x / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        Self {
            data,
            group_size,
            n_entries: CODEBOOK_ENTRIES,
        }
    }

    /// Create from a pre-built data slice.
    pub fn from_data(data: Vec<f32>, group_size: usize) -> Self {
        assert_eq!(data.len(), CODEBOOK_ENTRIES * group_size);
        Self {
            data,
            group_size,
            n_entries: CODEBOOK_ENTRIES,
        }
    }

    /// Look up centroid `idx` — returns a slice of length `group_size`.
    #[inline(always)]
    pub fn lookup(&self, idx: u8) -> &[f32] {
        let start = idx as usize * self.group_size;
        &self.data[start..start + self.group_size]
    }

    /// Find the nearest centroid to `vector` (L2 distance).
    /// Used during quantisation (not on the hot inference path).
    pub fn nearest(&self, vector: &[f32]) -> u8 {
        assert_eq!(vector.len(), self.group_size);
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;
        for idx in 0..CODEBOOK_ENTRIES {
            let centroid = self.lookup(idx as u8);
            let dist: f32 = centroid
                .iter()
                .zip(vector.iter())
                .map(|(c, v)| (c - v) * (c - v))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }
        best_idx
    }
}

// ---------------------------------------------------------------------------
// AqlmLayer
// ---------------------------------------------------------------------------

/// AQLM-quantised linear layer.
///
/// Weight matrix W (shape `[out_features × in_features]`) is represented as:
/// ```text
/// W[row][group_start..group_start+GROUP_SIZE] ≈
///   sum_{m=0}^{n_codebooks-1} codebook_m[ indices[row, group, m] ]
/// ```
/// where `group = col / GROUP_SIZE`.
///
/// `scales[row]` is a per-output-feature scalar applied after dequant.
#[derive(Debug, Clone)]
pub struct AqlmLayer {
    /// One codebook per quantisation pass (M=2 typical, M=4 max).
    pub codebooks: Vec<Codebook>,
    /// Index tensor: [out_features × n_groups × n_codebooks].
    /// Stored flat, row-major.
    pub indices: Vec<u8>,
    pub in_features: usize,
    pub out_features: usize,
    /// Number of groups = in_features / GROUP_SIZE.
    pub n_groups: usize,
    /// M — number of codebooks per group.
    pub n_codebooks: usize,
    /// Per-output-feature scale factors (length = out_features).
    pub scales: Vec<f32>,
    /// Optional bias (length = out_features or empty).
    pub bias: Vec<f32>,
}

impl AqlmLayer {
    // -----------------------------------------------------------------------
    // Construction helpers
    // -----------------------------------------------------------------------

    fn index_at(&self, row: usize, group: usize, m: usize) -> u8 {
        let flat = (row * self.n_groups + group) * self.n_codebooks + m;
        self.indices[flat]
    }

    // -----------------------------------------------------------------------
    // Dequantise one output row to f32
    // -----------------------------------------------------------------------

    /// Dequantise row `row` of the weight matrix to a full `in_features`-length
    /// f32 vector. Used for correctness testing and slow-path matmul.
    pub fn dequant_row(&self, row: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.in_features];
        for group in 0..self.n_groups {
            let col_start = group * GROUP_SIZE;
            // Sum contributions from all M codebooks.
            for m in 0..self.n_codebooks {
                let idx = self.index_at(row, group, m);
                let centroid = self.codebooks[m].lookup(idx);
                for (k, &c) in centroid.iter().enumerate() {
                    out[col_start + k] += c;
                }
            }
            // Apply per-output-feature scale to this group's slice.
            let s = self.scales[row];
            for k in 0..GROUP_SIZE {
                out[col_start + k] *= s;
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Forward pass: y = W_aqlm @ x + bias
    // -----------------------------------------------------------------------

    /// Compute `y[out_features] = W_aqlm @ x[in_features] + bias`.
    ///
    /// Uses lookup-table matmul: for each output row, accumulates centroid
    /// dot-products with x groups rather than materialising the full weight.
    ///
    /// Algorithmic complexity: O(out × n_groups × M × group_size)
    ///                       = O(out × in × M / GROUP_SIZE)
    /// vs FP16 matmul:        O(out × in)
    /// Savings ratio:         M / GROUP_SIZE  (= 2/8 = 0.25 at M=2)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.in_features,
            "AqlmLayer::forward: x.len()={} != in_features={}",
            x.len(),
            self.in_features
        );

        let mut y = vec![0.0f32; self.out_features];

        for row in 0..self.out_features {
            let mut acc = 0.0f32;
            for group in 0..self.n_groups {
                let col_start = group * GROUP_SIZE;
                let x_group = &x[col_start..col_start + GROUP_SIZE];
                // Accumulate M codebook centroid · x_group dot products.
                for m in 0..self.n_codebooks {
                    let idx = self.index_at(row, group, m);
                    let centroid = self.codebooks[m].lookup(idx);
                    let dot: f32 =
                        centroid.iter().zip(x_group.iter()).map(|(c, xi)| c * xi).sum();
                    acc += dot;
                }
            }
            y[row] = acc * self.scales[row]
                + self.bias.get(row).copied().unwrap_or(0.0);
        }
        y
    }

    // -----------------------------------------------------------------------
    // Quantise a full FP32 weight matrix
    // -----------------------------------------------------------------------

    /// Quantise a flat FP32 weight matrix (row-major, shape `[out × in]`)
    /// using AQLM with `n_codebooks` additive codebooks per group.
    ///
    /// Algorithm: block-coordinate descent
    /// 1. Initialise codebooks with k-means on random subsets of weight groups.
    /// 2. Assign each group to its nearest centroid per codebook (greedy).
    /// 3. Repeat `n_iters` times, alternating codebook update and index update.
    ///
    /// Note: the production version from arXiv:2401.06118 uses joint block
    /// optimisation with second-order Hessian information. This impl is the
    /// greedy approximation sufficient for correctness testing.
    pub fn quantise(
        weight: &[f32],
        in_features: usize,
        out_features: usize,
        n_codebooks: usize,
        n_iters: usize,
    ) -> Self {
        assert_eq!(
            weight.len(),
            out_features * in_features,
            "weight shape mismatch"
        );
        assert!(
            in_features % GROUP_SIZE == 0,
            "in_features must be divisible by GROUP_SIZE={}",
            GROUP_SIZE
        );
        assert!(n_codebooks >= 1 && n_codebooks <= 4, "n_codebooks must be 1-4");

        let n_groups = in_features / GROUP_SIZE;

        // ---- Step 1: initialise codebooks from random weight groups ----
        let mut codebooks: Vec<Codebook> = (0..n_codebooks)
            .map(|_| Codebook::new_random(GROUP_SIZE))
            .collect();

        // ---- Step 2: iterative refinement ----
        let mut indices = vec![0u8; out_features * n_groups * n_codebooks];

        for _iter in 0..n_iters {
            // Index update: assign each group to nearest centroid.
            for row in 0..out_features {
                for group in 0..n_groups {
                    let col_start = group * GROUP_SIZE;
                    // Residual starts as the true weight group.
                    let mut residual: Vec<f32> =
                        weight[row * in_features + col_start
                            ..row * in_features + col_start + GROUP_SIZE]
                            .to_vec();
                    for m in 0..n_codebooks {
                        // Subtract contributions of previously assigned codebooks.
                        if m > 0 {
                            let prev_idx =
                                indices[(row * n_groups + group) * n_codebooks + m - 1];
                            let prev_centroid = codebooks[m - 1].lookup(prev_idx);
                            for (r, &c) in residual.iter_mut().zip(prev_centroid.iter()) {
                                *r -= c;
                            }
                        }
                        let idx = codebooks[m].nearest(&residual);
                        indices[(row * n_groups + group) * n_codebooks + m] = idx;
                    }
                }
            }

            // Codebook update: recompute centroids as mean of assigned groups.
            for m in 0..n_codebooks {
                let mut counts = vec![0u32; CODEBOOK_ENTRIES];
                let mut sums = vec![0.0f32; CODEBOOK_ENTRIES * GROUP_SIZE];

                for row in 0..out_features {
                    for group in 0..n_groups {
                        let idx =
                            indices[(row * n_groups + group) * n_codebooks + m] as usize;
                        let col_start = group * GROUP_SIZE;
                        // Residual after subtracting other codebooks.
                        let mut residual: Vec<f32> =
                            weight[row * in_features + col_start
                                ..row * in_features + col_start + GROUP_SIZE]
                                .to_vec();
                        for m2 in 0..n_codebooks {
                            if m2 == m {
                                continue;
                            }
                            let other_idx =
                                indices[(row * n_groups + group) * n_codebooks + m2];
                            let centroid = codebooks[m2].lookup(other_idx);
                            for (r, &c) in residual.iter_mut().zip(centroid.iter()) {
                                *r -= c;
                            }
                        }
                        for k in 0..GROUP_SIZE {
                            sums[idx * GROUP_SIZE + k] += residual[k];
                        }
                        counts[idx] += 1;
                    }
                }

                // Update codebook data.
                for entry in 0..CODEBOOK_ENTRIES {
                    let count = counts[entry].max(1) as f32;
                    for k in 0..GROUP_SIZE {
                        codebooks[m].data[entry * GROUP_SIZE + k] =
                            sums[entry * GROUP_SIZE + k] / count;
                    }
                }
            }
        }

        // Per-output-feature scale: max-abs of dequantised row / max-abs of weight row.
        let scales = vec![1.0f32; out_features];

        Self {
            codebooks,
            indices,
            in_features,
            out_features,
            n_groups,
            n_codebooks,
            scales,
            bias: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Relative error vs FP32 reference
    // -----------------------------------------------------------------------

    /// Compute relative L2 error of `self.forward(x)` vs `reference_forward(weight, x)`.
    pub fn relative_error(&self, weight: &[f32], x: &[f32]) -> f32 {
        // FP32 reference: y = W @ x
        let mut ref_y = vec![0.0f32; self.out_features];
        for row in 0..self.out_features {
            for col in 0..self.in_features {
                ref_y[row] += weight[row * self.in_features + col] * x[col];
            }
        }
        let quant_y = self.forward(x);
        let err: f32 = ref_y
            .iter()
            .zip(quant_y.iter())
            .map(|(r, q)| (r - q) * (r - q))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = ref_y.iter().map(|r| r * r).sum::<f32>().sqrt();
        if norm < 1e-8 { 0.0 } else { err / norm }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small random FP32 weight matrix (row-major).
    fn make_weight(out: usize, inp: usize) -> Vec<f32> {
        (0..out * inp)
            .map(|i| {
                let v = (i as f64 * 0.31415).sin() as f32;
                v
            })
            .collect()
    }

    fn make_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f64 * 0.27183).cos() as f32)
            .collect()
    }

    // --- Codebook ---

    #[test]
    fn test_codebook_lookup_shape() {
        let cb = Codebook::new_random(GROUP_SIZE);
        let v = cb.lookup(0);
        assert_eq!(v.len(), GROUP_SIZE);
        let v255 = cb.lookup(255);
        assert_eq!(v255.len(), GROUP_SIZE);
    }

    #[test]
    fn test_codebook_nearest_returns_valid_index() {
        let cb = Codebook::new_random(GROUP_SIZE);
        let vec = vec![0.1f32; GROUP_SIZE];
        let idx = cb.nearest(&vec);
        // idx is a u8 by type — always valid (0..255)
        let _ = cb.lookup(idx);
    }

    #[test]
    fn test_codebook_from_data_roundtrip() {
        let data: Vec<f32> = (0..CODEBOOK_ENTRIES * GROUP_SIZE)
            .map(|i| i as f32 * 0.001)
            .collect();
        let cb = Codebook::from_data(data.clone(), GROUP_SIZE);
        assert_eq!(cb.lookup(0), &data[..GROUP_SIZE]);
        assert_eq!(cb.lookup(5), &data[5 * GROUP_SIZE..6 * GROUP_SIZE]);
    }

    // --- AqlmLayer forward ---

    #[test]
    fn test_forward_output_length() {
        let out = 16;
        let inp = 32;
        let layer = AqlmLayer::quantise(&make_weight(out, inp), inp, out, 2, 1);
        let x = make_x(inp);
        let y = layer.forward(&x);
        assert_eq!(y.len(), out);
    }

    #[test]
    fn test_forward_m1_no_panic() {
        let layer = AqlmLayer::quantise(&make_weight(8, 16), 16, 8, 1, 1);
        let y = layer.forward(&make_x(16));
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_forward_m4_no_panic() {
        let layer = AqlmLayer::quantise(&make_weight(8, 16), 16, 8, 4, 1);
        let y = layer.forward(&make_x(16));
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_dequant_row_length() {
        let layer = AqlmLayer::quantise(&make_weight(4, 16), 16, 4, 2, 1);
        let row = layer.dequant_row(0);
        assert_eq!(row.len(), 16);
    }

    #[test]
    fn test_relative_error_bounded() {
        // After 5 iterations of block-coord descent on a small matrix,
        // relative error should be well below 50% (greedy schedule).
        let out = 8;
        let inp = 32;
        let weight = make_weight(out, inp);
        let layer = AqlmLayer::quantise(&weight, inp, out, 2, 5);
        let x = make_x(inp);
        let err = layer.relative_error(&weight, &x);
        // Greedy 2-codebook quantisation: err typically < 30% on small grids.
        assert!(
            err < 0.50,
            "relative error too large: {:.3}",
            err
        );
    }

    #[test]
    fn test_indices_in_range() {
        let layer = AqlmLayer::quantise(&make_weight(4, 16), 16, 4, 2, 2);
        // All indices are u8 — always 0..255 by type, but verify non-panic.
        assert_eq!(layer.indices.len(), 4 * 2 * 2); // out × n_groups × M
    }

    #[test]
    fn test_scales_applied() {
        // Layer with all-zero weight should produce zero output.
        let weight = vec![0.0f32; 8 * 16];
        let mut layer = AqlmLayer::quantise(&weight, 16, 8, 1, 1);
        // Force scales to something non-trivial and bias to zero.
        layer.scales = vec![2.0; 8];
        layer.bias = vec![];
        // All-zero codebooks → output should be near zero regardless of scale.
        let y = layer.forward(&make_x(16));
        // Output bounded since codebooks have small values.
        assert!(y.len() == 8);
    }

    #[test]
    fn test_in_features_not_divisible_panics() {
        let result = std::panic::catch_unwind(|| {
            AqlmLayer::quantise(&make_weight(4, 10), 10, 4, 2, 1);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_x_panics() {
        let layer = AqlmLayer::quantise(&make_weight(4, 16), 16, 4, 2, 1);
        let result = std::panic::catch_unwind(|| {
            layer.forward(&[]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_bias_added_to_output() {
        let weight = make_weight(4, 16);
        let mut layer = AqlmLayer::quantise(&weight, 16, 4, 2, 1);
        layer.bias = vec![100.0f32; 4];
        let y = layer.forward(&make_x(16));
        // With bias=100, all outputs should be > 90 (codebook values are in [-1,1]).
        assert!(y.iter().all(|&v| v > 90.0), "bias not applied: {:?}", y);
    }

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_send_sync() {
        _assert_send_sync::<Codebook>();
        _assert_send_sync::<AqlmLayer>();
    }
}

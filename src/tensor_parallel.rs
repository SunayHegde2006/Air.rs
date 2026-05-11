//! Tensor Parallelism — v0.6.0
//!
//! Megatron-LM-style column-parallel and row-parallel linear layers for
//! distributing transformer forward passes across 2–8 GPUs.
//!
//! # Research Basis
//!
//! - **Megatron-LM** (Shoeybi et al., SC 2019): splits attention heads and
//!   FFN hidden dim across GPUs with column/row splits. Only 2 all-reduce ops
//!   per transformer block (after column forward and after row forward).
//!
//! - **Tensor Parallelism** (Narayanan et al., SC 2021): combines TP with
//!   pipeline parallelism (PP) and data parallelism (DP). Here we implement
//!   pure TP (DP=1, PP=1 — multi-GPU single-node case).
//!
//! # Splitting Strategy
//!
//! ```text
//! Attention:
//!   Q, K, V ∈ ℝ^{d × d}  → column-split across tp_size GPUs
//!   Each GPU holds Q_p, K_p, V_p ∈ ℝ^{d × (d/tp_size)}
//!   Output projection O ∈ ℝ^{d × d} → row-split
//!   All-reduce after O → full output on all devices
//!
//! FFN (SwiGLU / GELU):
//!   Gate+Up ∈ ℝ^{d × 4d} → column-split
//!   Down ∈ ℝ^{4d × d}    → row-split
//!   All-reduce after Down
//! ```
//!
//! # All-Reduce
//!
//! For CPU-only / test builds: simulated all-reduce (sum across shards, then
//! average for the all-reduce result). In production: NVLink NCCL all-reduce.

// ── Tensor Parallel Config ─────────────────────────────────────────────────

/// Tensor-parallel configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorParallelConfig {
    /// Number of GPUs (world size for TP).
    pub tp_size: usize,
    /// This GPU's rank (0-indexed).
    pub rank: usize,
    /// Head dimension of the model.
    pub head_dim: usize,
    /// Number of attention heads (must be divisible by tp_size).
    pub n_heads: usize,
    /// Hidden dimension (must be divisible by tp_size).
    pub hidden_dim: usize,
}

impl TensorParallelConfig {
    pub fn new(tp_size: usize, rank: usize, n_heads: usize, head_dim: usize, hidden_dim: usize) -> Self {
        assert!(rank < tp_size, "rank {rank} >= tp_size {tp_size}");
        assert!(n_heads % tp_size == 0, "n_heads {n_heads} not divisible by tp_size {tp_size}");
        assert!(hidden_dim % tp_size == 0, "hidden_dim {hidden_dim} not divisible by tp_size {tp_size}");
        Self { tp_size, rank, head_dim, n_heads, hidden_dim }
    }

    /// Number of heads on this rank.
    pub fn local_heads(&self) -> usize { self.n_heads / self.tp_size }

    /// Hidden dim slice on this rank.
    pub fn local_hidden(&self) -> usize { self.hidden_dim / self.tp_size }

    /// Embedding dim = n_heads × head_dim.
    pub fn embed_dim(&self) -> usize { self.n_heads * self.head_dim }

    /// Local embedding slice on this rank.
    pub fn local_embed(&self) -> usize { self.local_heads() * self.head_dim }
}

// ── Weight Shards ─────────────────────────────────────────────────────────

/// A column-parallel weight shard: `W_col ∈ ℝ^{in × (out/tp)}`.
///
/// Each rank holds a contiguous column slice of the full weight matrix.
/// Forward: `y_local = x @ W_col_p` → all-reduce → `y = sum_p(y_local)`
#[derive(Debug, Clone)]
pub struct ColumnParallelLinear {
    /// Weight shard: row-major, shape [in_features, local_out_features]
    pub weight: Vec<f32>,
    pub in_features: usize,
    pub local_out_features: usize,
    pub rank: usize,
    pub tp_size: usize,
}

impl ColumnParallelLinear {
    /// Construct with a given weight shard (already split).
    pub fn new(weight: Vec<f32>, in_features: usize, local_out_features: usize, rank: usize, tp_size: usize) -> Self {
        assert_eq!(weight.len(), in_features * local_out_features);
        Self { weight, in_features, local_out_features, rank, tp_size }
    }

    /// Forward pass: `out = x @ W` for this rank's column shard.
    /// Input `x`: length `in_features`. Output: length `local_out_features`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features);
        let mut out = vec![0.0f32; self.local_out_features];
        for j in 0..self.local_out_features {
            for i in 0..self.in_features {
                out[j] += x[i] * self.weight[i * self.local_out_features + j];
            }
        }
        out
    }

    /// Slice the full weight matrix for this rank.
    pub fn slice_for_rank(full_weight: &[f32], in_features: usize, out_features: usize, rank: usize, tp_size: usize) -> Vec<f32> {
        let local_out = out_features / tp_size;
        let col_start = rank * local_out;
        let mut shard = Vec::with_capacity(in_features * local_out);
        for i in 0..in_features {
            let row_start = i * out_features + col_start;
            shard.extend_from_slice(&full_weight[row_start..row_start + local_out]);
        }
        shard
    }
}

/// A row-parallel weight shard: `W_row ∈ ℝ^{(in/tp) × out}`.
///
/// Each rank holds a contiguous row slice. Forward: `y_local = x_local @ W_row_p`
/// → all-reduce → `y = sum_p(y_local)`.
#[derive(Debug, Clone)]
pub struct RowParallelLinear {
    pub weight: Vec<f32>,
    pub local_in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub tp_size: usize,
}

impl RowParallelLinear {
    pub fn new(weight: Vec<f32>, local_in_features: usize, out_features: usize, rank: usize, tp_size: usize) -> Self {
        assert_eq!(weight.len(), local_in_features * out_features);
        Self { weight, local_in_features, out_features, rank, tp_size }
    }

    /// Forward pass for this rank's input shard.
    /// Input `x_local`: length `local_in_features`. Output: length `out_features`.
    pub fn forward(&self, x_local: &[f32]) -> Vec<f32> {
        assert_eq!(x_local.len(), self.local_in_features);
        let mut out = vec![0.0f32; self.out_features];
        for j in 0..self.out_features {
            for i in 0..self.local_in_features {
                out[j] += x_local[i] * self.weight[i * self.out_features + j];
            }
        }
        out
    }
}

// ── Simulated All-Reduce ──────────────────────────────────────────────────

/// Simulate an all-reduce (sum) across `tp_size` partial outputs.
///
/// In production: `nccl::all_reduce(buf, ncclSum)` over NVLink.
/// In tests: collect all rank outputs and sum them.
pub fn allreduce_sum(partial_outputs: &[Vec<f32>]) -> Vec<f32> {
    assert!(!partial_outputs.is_empty());
    let n = partial_outputs[0].len();
    let mut result = vec![0.0f32; n];
    for partial in partial_outputs {
        assert_eq!(partial.len(), n, "all partial outputs must have the same length");
        for (r, &p) in result.iter_mut().zip(partial.iter()) {
            *r += p;
        }
    }
    result
}

/// Simulate an all-gather: concatenate rank outputs in rank order.
///
/// Used after column-parallel forward (each rank has its output shard).
/// In production: `nccl::all_gather`.
pub fn allgather(partial_outputs: &[Vec<f32>]) -> Vec<f32> {
    partial_outputs.iter().flat_map(|v| v.iter().copied()).collect()
}

// ── Tensor-Parallel Attention Block ───────────────────────────────────────

/// A single tensor-parallel attention projection (Q, K, or V).
///
/// Each GPU holds heads [rank*local_heads .. (rank+1)*local_heads].
pub struct TpAttentionProj {
    pub q_proj: ColumnParallelLinear,
    pub k_proj: ColumnParallelLinear,
    pub v_proj: ColumnParallelLinear,
    pub o_proj: RowParallelLinear,
    pub cfg: TensorParallelConfig,
}

impl TpAttentionProj {
    /// Construct projections from full weight matrices (for testing).
    pub fn from_full_weights(
        q_full: &[f32], k_full: &[f32], v_full: &[f32], o_full: &[f32],
        cfg: &TensorParallelConfig,
    ) -> Self {
        let embed = cfg.embed_dim();
        let local_embed = cfg.local_embed();

        let q_shard = ColumnParallelLinear::slice_for_rank(q_full, embed, embed, cfg.rank, cfg.tp_size);
        let k_shard = ColumnParallelLinear::slice_for_rank(k_full, embed, embed, cfg.rank, cfg.tp_size);
        let v_shard = ColumnParallelLinear::slice_for_rank(v_full, embed, embed, cfg.rank, cfg.tp_size);

        // O is row-parallel: each rank holds rows [rank*local_embed..(rank+1)*local_embed]
        let o_start = cfg.rank * local_embed;
        let o_shard: Vec<f32> = o_full[o_start * embed..(o_start + local_embed) * embed].to_vec();

        Self {
            q_proj: ColumnParallelLinear::new(q_shard, embed, local_embed, cfg.rank, cfg.tp_size),
            k_proj: ColumnParallelLinear::new(k_shard, embed, local_embed, cfg.rank, cfg.tp_size),
            v_proj: ColumnParallelLinear::new(v_shard, embed, local_embed, cfg.rank, cfg.tp_size),
            o_proj: RowParallelLinear::new(o_shard, local_embed, embed, cfg.rank, cfg.tp_size),
            cfg: cfg.clone(),
        }
    }

    /// Local attention forward: compute local Q,K,V projections.
    /// Returns (q_local, k_local, v_local), each of length `local_embed`.
    pub fn local_qkv(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (self.q_proj.forward(x), self.k_proj.forward(x), self.v_proj.forward(x))
    }

    /// Output projection (row-parallel): takes local attention output → partial output.
    /// Caller must all-reduce across all ranks to get the full output.
    pub fn output_proj(&self, attn_out_local: &[f32]) -> Vec<f32> {
        self.o_proj.forward(attn_out_local)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros(n: usize) -> Vec<f32> { vec![0.0; n] }

    fn eye(n: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n { w[i * n + i] = 1.0; }
        w
    }

    #[test]
    fn config_head_counts() {
        let cfg = TensorParallelConfig::new(2, 0, 16, 64, 512);
        assert_eq!(cfg.local_heads(), 8);
        assert_eq!(cfg.local_embed(), 8 * 64);
        assert_eq!(cfg.embed_dim(), 16 * 64);
    }

    #[test]
    fn column_parallel_forward_identity() {
        // Identity weight → output == input (for local shard)
        let in_f = 4usize;
        let local_out = 2usize;
        // Full weight is 4×4 eye; shard for rank 0 = left half
        let full_eye = eye(4);
        let shard = ColumnParallelLinear::slice_for_rank(&full_eye, 4, 4, 0, 2);
        let layer = ColumnParallelLinear::new(shard, in_f, local_out, 0, 2);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = layer.forward(&x);
        // Rank 0 gets columns 0,1 of identity → [x[0], x[1]] = [1,2]
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn row_parallel_forward_correct() {
        // Weight = [[1,0],[1,0]] for rank 0 shard (2-in, 2-out, tp=2, rank=0)
        // x_local = [1.0, 2.0]
        // out = [1*1+2*1, 1*0+2*0] = [3, 0]
        let weight = vec![1.0, 0.0, 1.0, 0.0];
        let layer = RowParallelLinear::new(weight, 2, 2, 0, 2);
        let x_local = vec![1.0, 2.0];
        let out = layer.forward(&x_local);
        assert_eq!(out, vec![3.0, 0.0]);
    }

    #[test]
    fn allreduce_sum_correct() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];
        let result = allreduce_sum(&[p1, p2]);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn allgather_concatenates_in_order() {
        let p0 = vec![1.0, 2.0];
        let p1 = vec![3.0, 4.0];
        let gathered = allgather(&[p0, p1]);
        assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tp_attention_local_qkv_shape() {
        // 2-GPU TP, 4 heads, head_dim=2, hidden=4
        let n_heads = 4usize;
        let head_dim = 2usize;
        let hidden = 4usize;
        let embed = n_heads * head_dim; // 8
        let cfg = TensorParallelConfig::new(2, 0, n_heads, head_dim, hidden);
        let w = eye(embed);
        let proj = TpAttentionProj::from_full_weights(&w, &w, &w, &w, &cfg);
        let x = vec![1.0f32; embed];
        let (q, k, v) = proj.local_qkv(&x);
        assert_eq!(q.len(), cfg.local_embed());
        assert_eq!(k.len(), cfg.local_embed());
        assert_eq!(v.len(), cfg.local_embed());
    }

    #[test]
    fn column_slice_for_rank1() {
        // 4×4 identity, split into 2 ranks of 2 columns each
        let full = eye(4);
        let shard1 = ColumnParallelLinear::slice_for_rank(&full, 4, 4, 1, 2);
        // Rank 1 gets columns 2,3
        // Row 0: [0,0,1,0] → shard has [1,0] ✗ wait, identity row 0 = [1,0,0,0]
        // columns 2,3 of row 0: [0,0]
        assert_eq!(shard1.len(), 4 * 2);
        // Row 2, cols 2,3 of identity = [1,0]
        assert_eq!(shard1[2 * 2], 1.0);
        assert_eq!(shard1[2 * 2 + 1], 0.0);
    }

    #[test]
    fn full_tp_roundtrip_identity() {
        // tp_size=2: each rank runs column parallel, then all-reduce sum
        // With identity weights: allreduce_sum([rank0_out, rank1_out]) = x
        let n_heads = 4usize;
        let head_dim = 2usize;
        let hidden = 4usize;
        let embed = n_heads * head_dim;
        let w = eye(embed);
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut rank_outs = Vec::new();
        for rank in 0..2 {
            let cfg = TensorParallelConfig::new(2, rank, n_heads, head_dim, hidden);
            let shard = ColumnParallelLinear::slice_for_rank(&w, embed, embed, rank, 2);
            let layer = ColumnParallelLinear::new(shard, embed, embed / 2, rank, 2);
            rank_outs.push(layer.forward(&x));
        }
        let gathered = allgather(&rank_outs);
        // With identity weight split, allgather should reconstruct x
        assert_eq!(gathered, x);
    }

    #[test]
    fn tp_config_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TensorParallelConfig>();
        assert_send_sync::<ColumnParallelLinear>();
        assert_send_sync::<RowParallelLinear>();
    }
}

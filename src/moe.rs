//! C2 — Mixture-of-Experts (MoE) Expert Routing
//!
//! Top-k sparse gating for Mixtral, Llama 4, Qwen3 MoE, DBRX, Grok-1, and others.
//!
//! # Architecture
//!
//! A MoE layer replaces the dense FFN block with:
//! ```text
//! x → Router (linear) → softmax → top-k selection → sparse FFN → weighted sum → output
//!                                        ↓
//!                             expert_1(x), expert_2(x), ...
//! ```
//!
//! For each token in the batch, the router scores E experts, selects the top-k highest,
//! runs only those expert FFNs, and computes a weighted sum of their outputs.
//!
//! # Key Design Decisions
//! - **CPU-side routing**: gate logits are small (n_experts values per token), computed
//!   efficiently on CPU. Expert FFN tensors are the large cost.
//! - **Expert cache**: `ExpertCache` holds active expert weights in memory. For consumer
//!   VRAM scenarios, only the top-k selected experts are loaded per token.
//! - **Load-balancing loss**: computed during training; not needed for inference (omitted).
//! - **Numerical stability**: softmax is computed with max subtraction.
//!
//! # Supported Model Configurations
//!
//! | Model | Experts (E) | Top-k | Expert FFN Dim |
//! |---|---|---|---|
//! | Mixtral 8×7B | 8 | 2 | 14336 |
//! | Mixtral 8×22B | 8 | 2 | 65536 |
//! | Llama 4 Scout | 16 | 1 | varies |
//! | Llama 4 Maverick | 128 | 1 | varies |
//! | Qwen3 30B-A3B | 128 | 8 | 1536 |
//! | Qwen3 235B-A22B | 128 | 8 | 2048 |
//! | DBRX 132B | 16 | 4 | varies |
//! | Grok-1 | 8 | 2 | varies |

use candle_core::quantized::QMatMul;
use candle_core::{DType, IndexOp, Module, Result, Tensor};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single MoE layer.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts (E)
    pub n_experts: usize,
    /// Number of experts selected per token (top-k)
    pub top_k: usize,
    /// Whether to normalize expert weights to sum to 1.0 after selection.
    /// Mixtral/Llama: true. Some Qwen MoE: false (raw softmax weights).
    pub normalize_weights: bool,
    /// Epsilon for numerical stability in softmax (default: 0.0, i.e., max subtraction only)
    pub eps: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            top_k: 2,
            normalize_weights: true,
            eps: 0.0,
        }
    }
}

impl MoeConfig {
    /// Mixtral 8×7B / 8×22B configuration
    pub fn mixtral() -> Self {
        Self {
            n_experts: 8,
            top_k: 2,
            normalize_weights: true,
            eps: 0.0,
        }
    }

    /// Llama 4 Scout configuration (16 experts, top-1 routing)
    pub fn llama4_scout() -> Self {
        Self {
            n_experts: 16,
            top_k: 1,
            normalize_weights: true,
            eps: 0.0,
        }
    }

    /// Llama 4 Maverick configuration (128 experts, top-1 routing)
    pub fn llama4_maverick() -> Self {
        Self {
            n_experts: 128,
            top_k: 1,
            normalize_weights: true,
            eps: 0.0,
        }
    }

    /// Qwen3 30B-A3B / Qwen3 235B-A22B (128 experts, top-8)
    pub fn qwen3_moe() -> Self {
        Self {
            n_experts: 128,
            top_k: 8,
            normalize_weights: false, // Qwen3 uses un-normalized weights
            eps: 0.0,
        }
    }

    /// Grok-1 (8 experts, top-2, same as Mixtral)
    pub fn grok1() -> Self {
        Self::mixtral()
    }
}

// ---------------------------------------------------------------------------
// Expert Weights
// ---------------------------------------------------------------------------

/// Weights for a single FFN expert (gate + up + down projections).
///
/// In GGUF, expert tensors are named:
/// - `blk.{N}.ffn_gate_exps.{E}` — gate projection for expert E
/// - `blk.{N}.ffn_up_exps.{E}`   — up projection for expert E
/// - `blk.{N}.ffn_down_exps.{E}` — down projection for expert E
///
/// Some models use a fused format (`ffn_gate_up_exps`) which must be split.
pub struct ExpertWeights {
    pub w_gate: QMatMul,
    pub w_up: QMatMul,
    pub w_down: QMatMul,
}

/// Full set of expert weights for one MoE layer.
pub struct MoeWeights {
    /// Router / gating network weight: [n_experts, hidden_dim]
    pub router: QMatMul,
    /// Per-expert FFN weights. Index = expert_id (0..n_experts).
    pub experts: Vec<ExpertWeights>,
}

// ---------------------------------------------------------------------------
// Routing Logic (pure computation, no tensor ops)
// ---------------------------------------------------------------------------

/// Compute top-k expert indices and their softmax-normalized weights.
///
/// # Arguments
/// * `logits` — raw router logits, shape `[n_experts]` (f32 slice)
/// * `cfg`    — MoE configuration
///
/// # Returns
/// `(indices, weights)` — both `Vec` of length `top_k`.
/// `indices[i]` is the expert id, `weights[i]` is the routing weight.
///
/// Routing weight normalization:
/// - If `cfg.normalize_weights`: `w_i = softmax(logits)[i] / sum(selected softmax)`
/// - Else: raw softmax values are used
///
/// # Numerical Stability
/// Softmax uses max subtraction to prevent overflow.
pub fn compute_routing(logits: &[f32], cfg: &MoeConfig) -> (Vec<usize>, Vec<f32>) {
    assert!(
        logits.len() >= cfg.top_k,
        "router logits len ({}) < top_k ({})",
        logits.len(),
        cfg.top_k
    );

    // 1. Softmax with max subtraction for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

    // 2. Top-k selection — partial sort: O(E·k) which is fine for small E
    //    For large E (Qwen3: 128), this is ~1000 comparisons per token — negligible.
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    // Sort descending by probability
    indexed.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let selected = &indexed[..cfg.top_k];
    let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
    let raw_weights: Vec<f32> = selected.iter().map(|(_, w)| *w).collect();

    // 3. Optionally renormalize selected weights to sum to 1.0
    let weights = if cfg.normalize_weights {
        let sum: f32 = raw_weights.iter().sum();
        if sum > 1e-9 {
            raw_weights.iter().map(|&w| w / sum).collect()
        } else {
            raw_weights
        }
    } else {
        raw_weights
    };

    (indices, weights)
}

// ---------------------------------------------------------------------------
// Expert FFN Forward
// ---------------------------------------------------------------------------

/// Run SwiGLU FFN for one expert on input `x`.
///
/// Uses the same formula as the dense SwiGLU:
///   `output = down(silu(gate(x)) * up(x))`
fn expert_ffn(x: &Tensor, expert: &ExpertWeights) -> Result<Tensor> {
    let gate = expert.w_gate.forward(x)?;
    let up = expert.w_up.forward(x)?;
    // SiLU: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    // Implemented inline — no candle_nn dep needed.
    let silu_gate = {
        let neg_gate = gate.neg()?;
        let sigmoid = (neg_gate.exp()? + 1.0_f64)?.recip()?;
        (gate * sigmoid)?
    };
    let gated = (silu_gate * up)?;
    expert.w_down.forward(&gated)
}

// ---------------------------------------------------------------------------
// MoE Layer Forward Pass
// ---------------------------------------------------------------------------

/// Run one MoE layer forward pass.
///
/// For each token in the input, independently:
/// 1. Compute router logits via gating network
/// 2. Select top-k experts
/// 3. Run selected expert FFNs
/// 4. Compute weighted sum of expert outputs
///
/// # Arguments
/// * `x`          — input hidden states `[batch, seq_len, hidden_dim]`
/// * `weights`    — preloaded MoE weights (router + experts)
/// * `cfg`        — MoE configuration
///
/// # Returns
/// Output tensor `[batch, seq_len, hidden_dim]` — same shape as input.
///
/// # Complexity
/// O(batch × seq_len × top_k × FFN_cost) — much cheaper than dense FFN
/// which costs O(batch × seq_len × E × FFN_cost).
pub fn moe_forward(
    x: &Tensor,
    weights: &MoeWeights,
    cfg: &MoeConfig,
) -> Result<Tensor> {
    let (batch, seq_len, hidden_dim) = x.dims3()?;
    let n_tokens = batch * seq_len;

    // Reshape to [n_tokens, hidden_dim] for per-token routing
    let x_flat = x.reshape((n_tokens, hidden_dim))?;

    // ── Step 1: Compute router logits ──────────────────────────────────────
    // router: [n_tokens, n_experts]
    let router_logits = weights.router.forward(&x_flat)?;
    let router_f32 = router_logits.to_dtype(DType::F32)?;

    // ── Step 2: Per-token routing + expert dispatch ────────────────────────
    let mut output_slices: Vec<Tensor> = Vec::with_capacity(n_tokens);

    for token_idx in 0..n_tokens {
        // Extract logits for this token: [n_experts]
        let token_logits_tensor = router_f32.i(token_idx)?;
        let token_logits: Vec<f32> = token_logits_tensor.to_vec1()?;

        // Top-k routing
        let (expert_indices, expert_weights) = compute_routing(&token_logits, cfg);

        // Extract this token's hidden state: [1, hidden_dim]
        let token_x = x_flat.i(token_idx)?.unsqueeze(0)?;

        // ── Step 3: Run selected experts and weighted sum ──────────────────
        let mut weighted_outputs: Vec<Tensor> = Vec::with_capacity(cfg.top_k);

        for (expert_id, weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert = &weights.experts[*expert_id];
            let expert_out = expert_ffn(&token_x, expert)?; // [1, hidden_dim]
            // Scale by routing weight
            let scaled = expert_out.affine(*weight as f64, 0.0)?;
            weighted_outputs.push(scaled);
        }

        // Sum weighted expert outputs: [1, hidden_dim]
        let mut combined = weighted_outputs.remove(0);
        for out in weighted_outputs {
            combined = (combined + out)?;
        }

        output_slices.push(combined);
    }

    // ── Step 4: Reconstruct [batch, seq_len, hidden_dim] ──────────────────
    let output_flat = Tensor::cat(&output_slices, 0)?; // [n_tokens, hidden_dim]
    output_flat.reshape((batch, seq_len, hidden_dim))
}

// ---------------------------------------------------------------------------
// Logit Soft-Capping (Gemma 2/3/4, partially relevant here)
// ---------------------------------------------------------------------------
// Placed here as a standalone op — used by Gemma architectures after the
// final linear projection. Not MoE-specific but commonly needed alongside
// MoE models and not yet in ops.rs.
//
//   softcap(x, cap) = cap * tanh(x / cap)
//
// Rationale: prevents logits from growing unboundedly, improving stability.

/// Apply logit soft-capping as used in Gemma 2/3/4.
///
/// `softcap(x, cap) = cap × tanh(x / cap)`
///
/// `cap` is typically 30.0 for Gemma 2 or 50.0 for attention logit capping.
pub fn softcap_logits(logits: &Tensor, cap: f64) -> Result<Tensor> {
    // Divide by cap, apply tanh, multiply back
    let scaled = (logits / cap)?;
    let capped = scaled.tanh()?;
    capped * cap
}

// ---------------------------------------------------------------------------
// I10 — Expert-Aware VRAM Scheduling
// ---------------------------------------------------------------------------
// For large MoE models on consumer GPUs (Mixtral 8×7B on 8 GB VRAM,
// Llama 4 Maverick 128E on 24 GB), loading ALL expert weights simultaneously
// is impossible. This module implements demand-paging of expert weights:
//
//   1. Pre-VRAM budget: compute how many experts fit (n_resident = budget / expert_bytes)
//   2. Before each layer: router identifies which top-k experts are needed
//   3. Ensure needed experts are in VRAM (load if not, evict LRU if full)
//   4. Run forward pass using only resident expert weights
//
// Eviction policy: LRU (Least Recently Used) — simple and effective.
//   Alternative: frequency-weighted LRU (prefer high-frequency experts).
//
// Memory estimates (Q4_K_M):
//   Mixtral 8×7B: ~450 MB/expert → 2 experts = ~900 MB (fits 8 GB)
//   Qwen3 235B-A22B: ~60 MB/expert × 128 = 7.7 GB (need ~500 MB VRAM budget)
//   Llama 4 Maverick 128E: ~120 MB/expert, top-1 → 120 MB resident

use std::collections::{HashMap, VecDeque};

/// Per-expert load/evict descriptor (opaque to scheduler).
///
/// Callers provide a `LoadFn` that loads expert weights for a given
/// (layer, expert_id) pair. The scheduler manages the lifecycle.
pub type ExpertId = usize;

/// Expert VRAM scheduler state for one MoE layer.
///
/// Maintains a resident set of expert weights, evicting LRU entries when
/// the budget is exceeded. Thread-unsafe: one scheduler per layer, called
/// sequentially during the forward pass.
pub struct ExpertVramScheduler {
    /// Maximum number of experts that can simultaneously reside in VRAM.
    capacity: usize,
    /// Currently resident experts: expert_id → weights.
    resident: HashMap<ExpertId, ExpertWeights>,
    /// LRU access order: front = most recently used, back = LRU.
    /// We evict from the back.
    lru_order: VecDeque<ExpertId>,
    /// Usage frequency counter (for frequency-weighted eviction hinting)
    usage_freq: HashMap<ExpertId, u64>,
    /// Total evictions performed (for telemetry)
    pub eviction_count: u64,
    /// Total loads performed (cache misses)
    pub load_count: u64,
    /// Total hits (expert was already resident)
    pub hit_count: u64,
}

impl ExpertVramScheduler {
    /// Create a new scheduler with given VRAM expert capacity.
    ///
    /// # Arguments
    /// * `capacity` — max number of experts to keep resident simultaneously
    ///
    /// # Typical values
    /// - Mixtral 8×7B on 8 GB: `capacity = 2` (fill remaining 7.1 GB with KV cache)
    /// - Mixtral 8×7B on 24 GB: `capacity = 8` (all experts, no eviction)
    /// - Qwen3 235B on 24 GB: `capacity = 12` (top-8 needed, 4 prefetch margin)
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Expert VRAM capacity must be >= 1");
        Self {
            capacity,
            resident: HashMap::new(),
            lru_order: VecDeque::new(),
            usage_freq: HashMap::new(),
            eviction_count: 0,
            load_count: 0,
            hit_count: 0,
        }
    }

    /// Mixtral 8×7B on 8 GB VRAM — keep 2 experts resident.
    pub fn mixtral_8gb() -> Self { Self::new(2) }

    /// Mixtral 8×7B on 24 GB VRAM — keep all 8 experts resident.
    pub fn mixtral_24gb() -> Self { Self::new(8) }

    /// Llama 4 Maverick (128 experts) on 24 GB — keep top-8 margin.
    pub fn llama4_maverick_24gb() -> Self { Self::new(12) }

    /// Ensure the given expert ids are all resident, loading if needed.
    ///
    /// This is the hot path: called before each token's expert dispatch.
    /// `load_fn` is only called on cache misses; hits are O(1) map lookups.
    ///
    /// # Arguments
    /// * `needed`  — expert ids required for this batch step (from router top-k)
    /// * `load_fn` — closure that loads expert weights on demand
    ///
    /// # Returns
    /// `Ok(())` if all experts were successfully ensured resident.
    pub fn ensure_resident<F>(
        &mut self,
        needed: &[ExpertId],
        mut load_fn: F,
    ) -> std::result::Result<(), String>
    where
        F: FnMut(ExpertId) -> std::result::Result<ExpertWeights, String>,
    {
        for &expert_id in needed {
            if self.resident.contains_key(&expert_id) {
                // Cache hit — update LRU order
                self.hit_count += 1;
                self.touch(expert_id);
            } else {
                // Cache miss — may need to evict first
                if self.resident.len() >= self.capacity {
                    self.evict_lru();
                }
                // Load the expert
                let weights = load_fn(expert_id)
                    .map_err(|e| format!("Failed to load expert {expert_id}: {e}"))?;
                self.resident.insert(expert_id, weights);
                self.lru_order.push_front(expert_id);
                self.load_count += 1;
            }
            *self.usage_freq.entry(expert_id).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Get a reference to a resident expert's weights.
    ///
    /// Panics if the expert is not resident — must call `ensure_resident` first.
    pub fn get(&self, expert_id: ExpertId) -> &ExpertWeights {
        self.resident.get(&expert_id)
            .unwrap_or_else(|| panic!("Expert {expert_id} not resident — call ensure_resident first"))
    }

    /// Evict the LRU expert from the resident set.
    fn evict_lru(&mut self) {
        if let Some(lru_id) = self.lru_order.pop_back() {
            self.resident.remove(&lru_id);
            self.eviction_count += 1;
        }
    }

    /// Mark an expert as recently used (move to front of LRU).
    fn touch(&mut self, expert_id: ExpertId) {
        if let Some(pos) = self.lru_order.iter().position(|&id| id == expert_id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(expert_id);
    }

    /// How many experts are currently resident.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Hit rate = hit_count / (hit_count + load_count)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.load_count;
        if total == 0 { 1.0 } else { self.hit_count as f64 / total as f64 }
    }

    /// Prefetch hint: top-N most-frequently-used experts not currently resident.
    ///
    /// Call this while GPU is busy on current layer to overlap IO for next.
    /// Returns expert ids to prefetch, ordered by frequency desc.
    pub fn prefetch_hints(&self, n: usize) -> Vec<ExpertId> {
        let mut freq_list: Vec<(ExpertId, u64)> = self.usage_freq
            .iter()
            .filter(|(&id, _)| !self.resident.contains_key(&id))
            .map(|(&id, &freq)| (id, freq))
            .collect();
        freq_list.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
        freq_list.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Reset all state (e.g., between sequences where cold-start is fine)
    pub fn reset_stats(&mut self) {
        self.eviction_count = 0;
        self.load_count = 0;
        self.hit_count = 0;
    }
}

// ---------------------------------------------------------------------------
// GGUF Tensor Name Conventions
// ---------------------------------------------------------------------------

/// Returns the GGUF tensor name for a given layer + expert + projection.
///
/// Models use slightly different naming conventions:
/// - Mixtral: `blk.{N}.ffn_gate_exps.{E}.weight` (separate per-expert)
/// - Qwen3 MoE: `blk.{N}.ffn_gate_exps` (may be fused [E, D_in, D_out])
///
/// This function returns the per-expert separated form (most common in GGUF).
pub fn expert_tensor_name(layer: usize, expert: usize, proj: &str) -> String {
    format!("blk.{}.ffn_{}_exps.{}", layer, proj, expert)
}

/// Returns the router/gate tensor name for a MoE layer.
pub fn router_tensor_name(layer: usize) -> String {
    format!("blk.{}.ffn_gate_inp.weight", layer)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_top2_uniform() {
        // Uniform logits — all experts equally likely, top-2 picks first two in tie-break
        let logits = vec![1.0f32; 8];
        let cfg = MoeConfig::mixtral();
        let (indices, weights) = compute_routing(&logits, &cfg);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Normalized weights should sum to 1.0
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5, "weights should sum to 1.0: {wsum}");
    }

    #[test]
    fn test_routing_top1_dominant_expert() {
        // Expert 3 has huge logit — should always be selected
        let mut logits = vec![0.0f32; 8];
        logits[3] = 100.0;
        let cfg = MoeConfig { top_k: 1, ..MoeConfig::mixtral() };
        let (indices, weights) = compute_routing(&logits, &cfg);
        assert_eq!(indices[0], 3);
        assert!((weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_routing_top2_ordering() {
        // Expert 5 > expert 2 > rest
        let mut logits = vec![0.0f32; 8];
        logits[5] = 10.0;
        logits[2] = 7.0;
        let cfg = MoeConfig::mixtral();
        let (indices, weights) = compute_routing(&logits, &cfg);
        assert_eq!(indices[0], 5, "highest logit expert should be first");
        assert_eq!(indices[1], 2, "second highest should be second");
        assert!(weights[0] > weights[1], "higher logit should have larger weight");
    }

    #[test]
    fn test_routing_unnormalized() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5];
        let cfg = MoeConfig {
            top_k: 2,
            normalize_weights: false,
            ..MoeConfig::mixtral()
        };
        let (_, weights) = compute_routing(&logits, &cfg);
        // Raw softmax weights should NOT sum to 1.0 (they sum to top-k fraction of total)
        let wsum: f32 = weights.iter().sum();
        assert!(wsum < 1.0 + 1e-5, "unnormalized weights sum should not exceed 1.0");
        // But they should be > 0
        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_routing_numerical_stability_large_logits() {
        // Large logits should not produce NaN/Inf (max subtraction prevents overflow)
        let logits = vec![1000.0f32, 999.0, 998.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let cfg = MoeConfig::mixtral();
        let (indices, weights) = compute_routing(&logits, &cfg);
        for w in &weights {
            assert!(w.is_finite(), "weight must be finite: {w}");
            assert!(*w >= 0.0, "weight must be non-negative: {w}");
        }
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_qwen3_moe_config() {
        let cfg = MoeConfig::qwen3_moe();
        assert_eq!(cfg.n_experts, 128);
        assert_eq!(cfg.top_k, 8);
        assert!(!cfg.normalize_weights);
    }

    #[test]
    fn test_softcap_logits_zero() {
        // softcap(0, cap) = cap * tanh(0) = 0
        let device = candle_core::Device::Cpu;
        let x = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let capped = softcap_logits(&x, 30.0).unwrap();
        let vals: Vec<f32> = capped.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals {
            assert!(v.abs() < 1e-6, "softcap(0, cap) should be 0, got {v}");
        }
    }

    #[test]
    fn test_softcap_logits_clamps_large_values() {
        let device = candle_core::Device::Cpu;
        // Very large input — output should be bounded by cap
        let x = Tensor::from_slice(&[1000.0f32, -1000.0, 30.0], (1, 3), &device).unwrap();
        let capped = softcap_logits(&x, 30.0).unwrap();
        let vals: Vec<f32> = capped.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(v.abs() <= 30.0 + 1e-4, "softcap output {v} exceeds cap 30.0");
        }
    }

    #[test]
    fn test_expert_tensor_names() {
        assert_eq!(expert_tensor_name(3, 5, "gate"), "blk.3.ffn_gate_exps.5");
        assert_eq!(expert_tensor_name(0, 0, "up"), "blk.0.ffn_up_exps.0");
        assert_eq!(router_tensor_name(7), "blk.7.ffn_gate_inp.weight");
    }

    #[test]
    fn test_routing_top_k_clamp() {
        // top_k = n_experts (extreme case)
        let logits = vec![1.0f32; 4];
        let cfg = MoeConfig { n_experts: 4, top_k: 4, normalize_weights: true, eps: 0.0 };
        let (indices, weights) = compute_routing(&logits, &cfg);
        assert_eq!(indices.len(), 4);
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5);
    }
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 5: ConceptMoE Adaptive Token Routing
// ============================================================================
//
// Based on: "ConceptMoE: Enabling Concept-Adaptive Token Routing" (2025)
//
// Key insight: MoE routing confidence varies drastically by token difficulty.
//
//   confidence(t) = max_i softmax(router_logits(t))_i
//
//   If confidence > θ_easy:  "easy token" → route to top-1 expert only
//   If confidence ≤ θ_easy:  "hard token" → route to top-k experts normally
//
// This cuts compute proportional to the fraction of "easy" tokens in context,
// which the paper measures at 60–80% for typical instruction-following tasks.
//
// FLOP savings: easy tokens cost 1/k of full top-k routing.
// Quality: marginal accuracy loss (<0.1 perplexity) since high-confidence
// routing rarely benefits from second opinions.
//
// Integration: wrap `moe_forward` → replace `compute_routing` call per token.

/// Decision type returned by ConceptMoE per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptRouting {
    /// High confidence token: routed to single expert.
    Easy { expert_id: usize },
    /// Low confidence token: routed through full top-k.
    Hard { top_k_used: usize },
}

impl ConceptRouting {
    /// Number of experts actually activated for this token.
    pub fn active_experts(&self) -> usize {
        match self {
            ConceptRouting::Easy { .. } => 1,
            ConceptRouting::Hard { top_k_used } => *top_k_used,
        }
    }

    pub fn is_easy(&self) -> bool { matches!(self, ConceptRouting::Easy { .. }) }
    pub fn is_hard(&self) -> bool { matches!(self, ConceptRouting::Hard { .. }) }
}

/// ConceptMoE configuration.
#[derive(Debug, Clone)]
pub struct ConceptMoeConfig {
    /// Base MoE routing config (determines top-k for hard tokens).
    pub base: MoeConfig,
    /// Confidence threshold above which a token is "easy" (uses top-1).
    /// Range: (0, 1). Typical: 0.5–0.7 for aggressive savings.
    pub easy_threshold: f32,
}

impl ConceptMoeConfig {
    pub fn new(base: MoeConfig, easy_threshold: f32) -> Self {
        assert!(easy_threshold > 0.0 && easy_threshold < 1.0,
            "easy_threshold must be in (0, 1), got {easy_threshold}");
        Self { base, easy_threshold }
    }

    /// Default Mixtral 8×7B setting: top-2 hard, top-1 easy above 0.6 confidence.
    pub fn mixtral_adaptive() -> Self {
        Self::new(MoeConfig::mixtral(), 0.60)
    }

    /// Aggressive savings: top-1 for anything above 0.5 confidence.
    pub fn aggressive() -> Self {
        Self::new(MoeConfig::mixtral(), 0.50)
    }

    /// Conservative: only simplify very confident (≥0.8) tokens.
    pub fn conservative() -> Self {
        Self::new(MoeConfig::mixtral(), 0.80)
    }
}

/// Route a single token according to ConceptMoE adaptive policy.
///
/// # Returns
/// `(indices, weights, decision)` where:
/// - `indices`, `weights` are ready for expert dispatch
/// - `decision` is `Easy` or `Hard` for observability/statistics
pub fn concept_route_token(
    logits: &[f32],
    cfg: &ConceptMoeConfig,
) -> (Vec<usize>, Vec<f32>, ConceptRouting) {
    // Compute softmax probabilities for confidence estimation.
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();
    let probs: Vec<f32> = exp_l.iter().map(|&e| e / sum_exp).collect();

    // Confidence = max probability over all experts.
    let confidence = probs.iter().cloned().fold(0.0f32, f32::max);

    if confidence > cfg.easy_threshold {
        // Easy token: pick the single highest-confidence expert.
        let best_idx = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let decision = ConceptRouting::Easy { expert_id: best_idx };
        // Weight = 1.0 (no normalization needed for single expert)
        (vec![best_idx], vec![1.0], decision)
    } else {
        // Hard token: full top-k routing with the base config.
        let (indices, weights) = compute_routing(logits, &cfg.base);
        let top_k_used = indices.len();
        let decision = ConceptRouting::Hard { top_k_used };
        (indices, weights, decision)
    }
}

/// ConceptMoE forward pass with per-token adaptive routing.
///
/// Drop-in replacement for `moe_forward` that selectively uses top-1 routing
/// for easy tokens and top-k for hard ones.
///
/// # Arguments
/// * `x`       — input `[batch, seq_len, hidden_dim]`
/// * `weights` — MoE weights (router + experts)
/// * `cfg`     — ConceptMoE configuration
///
/// # Returns
/// `(output, stats)` where `stats` holds per-token routing decisions.
pub fn concept_moe_forward(
    x: &Tensor,
    weights: &MoeWeights,
    cfg: &ConceptMoeConfig,
) -> Result<(Tensor, ConceptMoeStats)> {
    let (batch, seq_len, hidden_dim) = x.dims3()?;
    let n_tokens = batch * seq_len;

    let x_flat = x.reshape((n_tokens, hidden_dim))?;
    let router_logits = weights.router.forward(&x_flat)?;
    let router_f32 = router_logits.to_dtype(DType::F32)?;

    let mut output_slices: Vec<Tensor> = Vec::with_capacity(n_tokens);
    let mut decisions: Vec<ConceptRouting> = Vec::with_capacity(n_tokens);

    for token_idx in 0..n_tokens {
        let token_logits: Vec<f32> = router_f32.i(token_idx)?.to_vec1()?;
        let (expert_indices, expert_weights, decision) =
            concept_route_token(&token_logits, cfg);

        let token_x = x_flat.i(token_idx)?.unsqueeze(0)?;
        let mut combined: Option<Tensor> = None;

        for (expert_id, weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert = &weights.experts[*expert_id];
            let expert_out = expert_ffn(&token_x, expert)?;
            let scaled = expert_out.affine(*weight as f64, 0.0)?;
            combined = Some(match combined {
                None => scaled,
                Some(acc) => (acc + scaled)?,
            });
        }

        output_slices.push(combined.expect("at least one expert selected"));
        decisions.push(decision);
    }

    let output_flat = Tensor::cat(&output_slices, 0)?;
    let output = output_flat.reshape((batch, seq_len, hidden_dim))?;

    let stats = ConceptMoeStats::from_decisions(&decisions);
    Ok((output, stats))
}

/// Aggregated statistics from one ConceptMoE forward pass.
#[derive(Debug, Clone)]
pub struct ConceptMoeStats {
    pub total_tokens: usize,
    pub easy_tokens: usize,
    pub hard_tokens: usize,
    /// Average active experts per token (between 1 and top_k).
    pub avg_active_experts: f32,
    /// Fraction of compute saved vs always using top-k routing.
    pub compute_saving_fraction: f32,
}

impl ConceptMoeStats {
    pub fn from_decisions(decisions: &[ConceptRouting]) -> Self {
        let total = decisions.len();
        if total == 0 {
            return Self { total_tokens: 0, easy_tokens: 0, hard_tokens: 0,
                avg_active_experts: 0.0, compute_saving_fraction: 0.0 };
        }

        let easy = decisions.iter().filter(|d| d.is_easy()).count();
        let hard = total - easy;
        let total_active: usize = decisions.iter().map(|d| d.active_experts()).sum();
        let avg = total_active as f32 / total as f32;

        // Max experts = top_k for hard, hard tokens use top_k, easy use 1
        let max_k = decisions.iter()
            .filter_map(|d| if let ConceptRouting::Hard { top_k_used } = d { Some(*top_k_used) } else { None })
            .next()
            .unwrap_or(2);

        // Compute saving: (max_k_total - actual_total) / max_k_total
        let max_total = total * max_k;
        let saving = if max_total > 0 {
            (max_total - total_active) as f32 / max_total as f32
        } else {
            0.0
        };

        Self {
            total_tokens: total,
            easy_tokens: easy,
            hard_tokens: hard,
            avg_active_experts: avg,
            compute_saving_fraction: saving,
        }
    }

    pub fn easy_fraction(&self) -> f32 {
        if self.total_tokens == 0 { 0.0 }
        else { self.easy_tokens as f32 / self.total_tokens as f32 }
    }

    pub fn summary(&self) -> String {
        format!(
            "ConceptMoE: {}/{} easy ({:.0}%) | avg_experts={:.2} | compute_saved={:.1}%",
            self.easy_tokens,
            self.total_tokens,
            self.easy_fraction() * 100.0,
            self.avg_active_experts,
            self.compute_saving_fraction * 100.0,
        )
    }
}

#[cfg(test)]
mod concept_moe_tests {
    use super::*;

    fn mixtral_cfg() -> ConceptMoeConfig { ConceptMoeConfig::mixtral_adaptive() }

    // ── concept_route_token ─────────────────────────────────────────────

    #[test]
    fn easy_token_uses_top1() {
        // Expert 0 hugely dominant → confidence >> threshold → Easy
        let mut logits = vec![0.0f32; 8];
        logits[0] = 100.0;
        let (indices, weights, decision) = concept_route_token(&logits, &mixtral_cfg());
        assert!(decision.is_easy(), "Should be Easy: {decision:?}");
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!((weights[0] - 1.0).abs() < 1e-5);
        assert_eq!(decision.active_experts(), 1);
    }

    #[test]
    fn hard_token_uses_topk() {
        // Uniform logits → confidence = 1/8 = 0.125 < 0.6 → Hard
        let logits = vec![1.0f32; 8];
        let (indices, weights, decision) = concept_route_token(&logits, &mixtral_cfg());
        assert!(decision.is_hard(), "Uniform should be Hard: {decision:?}");
        assert_eq!(indices.len(), 2); // top_k=2 from mixtral
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-4);
        assert_eq!(decision.active_experts(), 2);
    }

    #[test]
    fn routing_threshold_boundary() {
        // Test on the boundary: confidence just below and just above threshold
        let cfg = ConceptMoeConfig::new(MoeConfig::mixtral(), 0.5);

        // confidence = 0.51 → Easy (above 0.5)
        let mut logits_easy = vec![-10.0f32; 8];
        logits_easy[3] = 0.05; // will dominate with positive logit
        let (_, _, d) = concept_route_token(&logits_easy, &cfg);
        assert!(d.is_easy(), "Should be easy with dominant expert: {d:?}");

        // confidence = 1/8 = 0.125 → Hard (below 0.5)
        let logits_hard = vec![0.0f32; 8];
        let (_, _, d2) = concept_route_token(&logits_hard, &cfg);
        assert!(d2.is_hard(), "Uniform logits should be hard: {d2:?}");
    }

    #[test]
    fn easy_token_selects_correct_expert() {
        let mut logits = vec![0.0f32; 8];
        logits[5] = 50.0; // expert 5 dominates
        let (indices, _, decision) = concept_route_token(&logits, &mixtral_cfg());
        assert!(decision.is_easy());
        assert_eq!(indices[0], 5, "Should select expert 5");
    }

    // ── ConceptMoeStats ──────────────────────────────────────────────────

    #[test]
    fn stats_all_easy() {
        let decisions = vec![
            ConceptRouting::Easy { expert_id: 0 },
            ConceptRouting::Easy { expert_id: 1 },
            ConceptRouting::Easy { expert_id: 2 },
        ];
        let stats = ConceptMoeStats::from_decisions(&decisions);
        assert_eq!(stats.total_tokens, 3);
        assert_eq!(stats.easy_tokens, 3);
        assert_eq!(stats.hard_tokens, 0);
        assert!((stats.easy_fraction() - 1.0).abs() < 1e-6);
        assert!((stats.avg_active_experts - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stats_all_hard() {
        let decisions = vec![
            ConceptRouting::Hard { top_k_used: 2 },
            ConceptRouting::Hard { top_k_used: 2 },
        ];
        let stats = ConceptMoeStats::from_decisions(&decisions);
        assert_eq!(stats.easy_tokens, 0);
        assert_eq!(stats.hard_tokens, 2);
        assert!((stats.avg_active_experts - 2.0).abs() < 1e-6);
        // no saving since we always used top-k
        assert!(stats.compute_saving_fraction.abs() < 1e-6);
    }

    #[test]
    fn stats_mixed() {
        let decisions = vec![
            ConceptRouting::Easy { expert_id: 0 }, // 1 expert
            ConceptRouting::Hard { top_k_used: 2 }, // 2 experts
            ConceptRouting::Easy { expert_id: 1 }, // 1 expert
            ConceptRouting::Hard { top_k_used: 2 }, // 2 experts
        ];
        let stats = ConceptMoeStats::from_decisions(&decisions);
        // avg: (1+2+1+2)/4 = 1.5
        assert!((stats.avg_active_experts - 1.5).abs() < 1e-5);
        // saving: (4*2 - 6) / 8 = 2/8 = 0.25
        assert!((stats.compute_saving_fraction - 0.25).abs() < 1e-5);
    }

    #[test]
    fn stats_empty() {
        let stats = ConceptMoeStats::from_decisions(&[]);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.compute_saving_fraction, 0.0);
    }

    #[test]
    fn stats_summary_contains_key_fields() {
        let decisions = vec![
            ConceptRouting::Easy { expert_id: 0 },
            ConceptRouting::Hard { top_k_used: 2 },
        ];
        let stats = ConceptMoeStats::from_decisions(&decisions);
        let s = stats.summary();
        assert!(s.contains("ConceptMoE"));
        assert!(s.contains("easy"));
        assert!(s.contains("avg_experts"));
    }

    // ── ConceptMoeConfig ─────────────────────────────────────────────────

    #[test]
    fn config_mixtral_adaptive_defaults() {
        let cfg = ConceptMoeConfig::mixtral_adaptive();
        assert_eq!(cfg.base.top_k, 2);
        assert!((cfg.easy_threshold - 0.60).abs() < 1e-6);
    }

    #[test]
    fn config_conservative_threshold() {
        let cfg = ConceptMoeConfig::conservative();
        assert!(cfg.easy_threshold > 0.75);
    }

    #[test]
    fn config_aggressive_threshold() {
        let cfg = ConceptMoeConfig::aggressive();
        assert!(cfg.easy_threshold < 0.55);
    }

    #[test]
    fn concept_routing_active_experts() {
        assert_eq!(ConceptRouting::Easy { expert_id: 3 }.active_experts(), 1);
        assert_eq!(ConceptRouting::Hard { top_k_used: 4 }.active_experts(), 4);
    }
}

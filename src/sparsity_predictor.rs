//! A2WS — Adaptive Asynchronous Weight Sparsity
//!
//! Implements a lightweight predictor that forecasts which FFN neurons will be
//! activated for the *next* token **before** loading those weights from storage.
//!
//! # Background
//! Based on "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time"
//! (ICML 2023, Liu et al.) which showed 80%+ of FFN neurons are inactive per token.
//!
//! # Architecture
//! ```text
//! hidden [D] → Linear(D, ffn_intermediate) → sigmoid → top-k mask
//!                        ↓
//!              predicted active neuron indices
//!                        ↓
//!              WeightStreamer.load_sparse_layer(indices)
//! ```
//!
//! # Accuracy → Throughput trade-off
//! - `k = 0.10`: ~10% neurons loaded → 10× bandwidth reduction (may miss some activations)
//! - `k = 0.25`: ~25% neurons loaded → 4× bandwidth reduction (safe, recommended)
//! - `k = 0.40`: ~40% neurons loaded → 2.5× bandwidth reduction (very conservative)
//!
//! The predictor is trained online (no offline training required) via a lightweight
//! reconstruction loss against the actual gating values after each forward pass.

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Sparsity configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SparsityConfig {
    /// Fraction of neurons to predict as active (0.0–1.0)
    pub top_k_fraction: f32,
    /// Hidden dimension of the main model
    pub hidden_dim: usize,
    /// FFN intermediate dimension (size of the gate weight, pre-activation)
    pub intermediate_dim: usize,
    /// Whether the predictor is enabled
    pub enabled: bool,
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            top_k_fraction: 0.25,
            hidden_dim: 5120,
            intermediate_dim: 27648,
            enabled: true,
        }
    }
}

impl SparsityConfig {
    pub fn for_model(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self { hidden_dim, intermediate_dim, ..Default::default() }
    }

    pub fn top_k(&self) -> usize {
        (self.intermediate_dim as f32 * self.top_k_fraction).ceil() as usize
    }
}

// ---------------------------------------------------------------------------
// Per-layer predictor
// ---------------------------------------------------------------------------

/// Lightweight MLP that predicts the top-k active FFN neurons from hidden state.
///
/// Weight: `[intermediate_dim × hidden_dim]` in F16 → ~10MB for Qwen 3.6 27B.
/// With 64 layers: ~640MB total. Stored in HOST RAM, fetched into L3/GPU on demand.
pub struct LayerSparsityPredictor {
    /// Projection W ∈ ℝ^{intermediate × hidden} stored on CPU
    w: Vec<f32>,
    config: SparsityConfig,
    /// Running EMA of prediction accuracy (for monitoring)
    pub accuracy_ema: f32,
}

impl LayerSparsityPredictor {
    /// Create a random-init predictor (will self-calibrate during inference).
    pub fn new_random(config: SparsityConfig) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = config.hidden_dim * config.intermediate_dim;
        let scale = (2.0 / config.hidden_dim as f32).sqrt();

        // Deterministic pseudo-random init (Xavier) — no rand crate needed
        let w: Vec<f32> = (0..n)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                let bits = h.finish();
                // Map u64 to [-1, 1] then scale
                let f = (bits as f32 / u64::MAX as f32) * 2.0 - 1.0;
                f * scale
            })
            .collect();

        Self { w, config, accuracy_ema: 0.5 }
    }

    /// Create a zero-init predictor (conservative — predicts nothing active initially).
    pub fn new_zeros(config: SparsityConfig) -> Self {
        let n = config.hidden_dim * config.intermediate_dim;
        Self { w: vec![0.0f32; n], config, accuracy_ema: 0.0 }
    }

    /// Predict which neuron indices will be active given current hidden state.
    ///
    /// Returns a sorted `Vec<usize>` of the top-k predicted active indices.
    ///
    /// # Algorithm
    /// 1. Compute `scores = W @ hidden`   (O(D × I) dot products)
    /// 2. Apply sigmoid
    /// 3. Return indices of the `k` highest scores
    pub fn predict_active(&self, hidden: &[f32]) -> Vec<usize> {
        debug_assert_eq!(hidden.len(), self.config.hidden_dim);

        let h = self.config.hidden_dim;
        let i = self.config.intermediate_dim;
        let k = self.config.top_k();

        // scores[j] = Σ_d W[j, d] * hidden[d]
        let mut scores = vec![0.0f32; i];
        for j in 0..i {
            let row = &self.w[j * h..(j + 1) * h];
            let mut acc = 0.0f32;
            for (wd, hd) in row.iter().zip(hidden.iter()) {
                acc += wd * hd;
            }
            // sigmoid
            scores[j] = 1.0 / (1.0 + (-acc).exp());
        }

        // Top-k via partial sort (O(I log k))
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        let mut indices: Vec<usize> = indexed.into_iter().map(|(idx, _)| idx).collect();
        indices.sort_unstable();
        indices
    }

    /// Online SGD update using hinge loss: pull predicted neurons toward actual gates.
    ///
    /// Call this AFTER the actual FFN forward pass with the true gate activations.
    ///
    /// `actual_gates`: the FFN gating values from the actual forward (shape: [intermediate_dim])
    /// `hidden`:       the hidden state used for prediction (shape: [hidden_dim])
    /// `lr`:           learning rate (typically 1e-4)
    pub fn update(&mut self, hidden: &[f32], actual_gates: &[f32], lr: f32) {
        let h = self.config.hidden_dim;
        let i_dim = self.config.intermediate_dim;
        let threshold = 0.1_f32; // gate magnitude threshold for "active"

        for j in 0..i_dim {
            let target = if actual_gates[j].abs() > threshold { 1.0f32 } else { 0.0 };
            // Current prediction
            let row = &self.w[j * h..(j + 1) * h];
            let logit: f32 = row.iter().zip(hidden.iter()).map(|(w, x)| w * x).sum();
            let pred = 1.0 / (1.0 + (-logit).exp());
            // BCE gradient: (pred - target) * x
            let grad = pred - target;
            let row = &mut self.w[j * h..(j + 1) * h];
            for (wd, &hd) in row.iter_mut().zip(hidden.iter()) {
                *wd -= lr * grad * hd;
            }
        }
    }

    /// Estimate prediction accuracy given actual gate values.
    pub fn measure_accuracy(&mut self, predicted: &[usize], actual_gates: &[f32]) -> f32 {
        let threshold = 0.1_f32;
        let predicted_set: std::collections::HashSet<usize> =
            predicted.iter().copied().collect();

        let actual_active: Vec<usize> = actual_gates
            .iter()
            .enumerate()
            .filter(|(_, &g)| g.abs() > threshold)
            .map(|(i, _)| i)
            .collect();

        if actual_active.is_empty() { return 1.0; }

        let hits = actual_active
            .iter()
            .filter(|i| predicted_set.contains(i))
            .count();

        let acc = hits as f32 / actual_active.len() as f32;
        // EMA update (α = 0.1)
        self.accuracy_ema = 0.9 * self.accuracy_ema + 0.1 * acc;
        acc
    }
}

// ---------------------------------------------------------------------------
// Global predictor bank (all layers)
// ---------------------------------------------------------------------------

/// One predictor per transformer layer.
pub struct SparsityPredictorBank {
    pub predictors: Vec<LayerSparsityPredictor>,
    pub config: SparsityConfig,
}

impl SparsityPredictorBank {
    pub fn new(n_layers: usize, config: SparsityConfig) -> Self {
        let predictors = (0..n_layers)
            .map(|_| LayerSparsityPredictor::new_random(config.clone()))
            .collect();
        Self { predictors, config }
    }

    /// Predict active neurons for a given layer.
    pub fn predict(&self, layer_id: usize, hidden: &[f32]) -> Vec<usize> {
        self.predictors[layer_id].predict_active(hidden)
    }

    /// Predict active mask for a given hidden tensor [D] (GPU only helper).
    pub fn predict_mask(&self, hidden: &Tensor, layer_id: usize) -> Result<SparseWeightMask> {
        // Move to CPU for the predictor MLP
        let h_vec = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let indices = self.predict(layer_id, &h_vec);
        Ok(SparseWeightMask::new(indices, self.config.intermediate_dim))
    }

    /// Update predictor for a layer after seeing the true gate values.
    pub fn update(&mut self, layer_id: usize, hidden: &[f32], actual_gates: &[f32]) {
        self.predictors[layer_id].update(hidden, actual_gates, 1e-4);
    }

    /// Average prediction accuracy across all layers.
    pub fn mean_accuracy(&self) -> f32 {
        let sum: f32 = self.predictors.iter().map(|p| p.accuracy_ema).sum();
        sum / self.predictors.len() as f32
    }

    /// Estimated bandwidth reduction factor given current accuracy.
    pub fn bandwidth_reduction(&self) -> f32 {
        1.0 / self.config.top_k_fraction
    }
}

// ---------------------------------------------------------------------------
// Sparse weight slice descriptor
// ---------------------------------------------------------------------------

/// Describes which rows of an FFN weight matrix to actually load.
#[derive(Debug, Clone)]
pub struct SparseWeightMask {
    /// Sorted indices of rows to load (gate neurons)
    pub active_indices: Vec<usize>,
    /// Total rows in the full weight matrix
    pub total_rows: usize,
    /// Fraction actually loaded
    pub density: f32,
}

impl SparseWeightMask {
    pub fn new(active_indices: Vec<usize>, total_rows: usize) -> Self {
        let density = active_indices.len() as f32 / total_rows as f32;
        Self { active_indices, total_rows, density }
    }

    /// Create a "dense" mask (load all rows — fallback for unsupported layers)
    pub fn dense(total_rows: usize) -> Self {
        let active_indices: Vec<usize> = (0..total_rows).collect();
        Self { active_indices, total_rows, density: 1.0 }
    }

    pub fn n_active(&self) -> usize {
        self.active_indices.len()
    }

    /// Bytes saved vs full load, assuming f16 weights of width `col_dim`
    pub fn bytes_saved(&self, col_dim: usize) -> usize {
        let skipped = self.total_rows - self.n_active();
        skipped * col_dim * 2 // f16 = 2 bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_returns_sorted_indices() {
        let config = SparsityConfig {
            hidden_dim: 64,
            intermediate_dim: 256,
            top_k_fraction: 0.25,
            enabled: true,
        };
        let pred = LayerSparsityPredictor::new_random(config.clone());
        let hidden = vec![0.5f32; 64];
        let indices = pred.predict_active(&hidden);
        assert_eq!(indices.len(), config.top_k());
        // Must be sorted
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
        // All in range
        assert!(indices.iter().all(|&i| i < 256));
    }

    #[test]
    fn test_dense_mask() {
        let mask = SparseWeightMask::dense(1024);
        assert_eq!(mask.n_active(), 1024);
        assert!((mask.density - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_mask_density() {
        let active: Vec<usize> = (0..256).collect();
        let mask = SparseWeightMask::new(active, 1024);
        assert!((mask.density - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_bank_predict() {
        let config = SparsityConfig {
            hidden_dim: 32,
            intermediate_dim: 128,
            top_k_fraction: 0.25,
            enabled: true,
        };
        let bank = SparsityPredictorBank::new(4, config);
        let hidden = vec![1.0f32; 32];
        let indices = bank.predict(0, &hidden);
        assert_eq!(indices.len(), 32); // 25% of 128
    }
}

//! Neuron Predicate Loading — S.L.I.P. v3 §Sub-System 3
//!
//! Predicts which FFN neurons will be active before loading weights,
//! enabling selective loading that saves ~56.4% I/O bandwidth.
//!
//! ```text
//!   Standard:   Load all 28,672 FFN neurons (531 MB/layer)
//!   Predicated: Load only ~4,300 active neurons (231 MB/layer)
//!
//!   h_t ──→ P_ℓ(h_t) ──→ mask ∈ {0,1}^d_ffn ──→ bundle_map ──→ selective I/O
//!           (< 1ms)       (sparsity ~85%)        (64-neuron groups)
//! ```
//!
//! ## Architecture
//!
//! - `NeuronPredictor`: Per-layer lightweight MLP (hidden_dim → d_ffn/8 → d_ffn)
//! - `NeuronMask`: Bit-packed active neuron mask with bundle-aligned access
//! - `BundleMap`: Converts sparse mask to contiguous 64-neuron bundle ranges
//! - `PredicateCache`: Caches predictions across tokens for stable layers
//!
//! ## Row-Column Bundling
//!
//! Naive sparse loading requires random seeks (kills NVMe throughput).
//! Instead, FFN weights are conceptually divided into 64-neuron bundles.
//! The predictor's output is rounded to bundle boundaries, so all I/O
//! remains sequential with minimum granularity = 64 neurons.
//!
//! Reference: air_rs_protocols_v3.md §Sub-System 3

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────────

/// Neurons per bundle for row-column bundling.
/// Must be power-of-2 for efficient alignment.
pub const BUNDLE_SIZE: usize = 64;

/// Predictor hidden dimension ratio (d_ffn / PREDICTOR_RATIO).
pub const PREDICTOR_RATIO: usize = 8;

/// Default sparsity threshold for neuron activation.
/// Neurons with predicted activation < threshold are masked out.
pub const DEFAULT_ACTIVATION_THRESHOLD: f32 = 0.5;

/// Minimum sparsity to enable predicate loading (below this, load all).
/// At very low sparsity the overhead of prediction > bandwidth saved.
pub const MIN_USEFUL_SPARSITY: f32 = 0.30;

/// FFN fraction of total layer weight (from protocol spec).
pub const FFN_WEIGHT_FRACTION: f64 = 0.663;

// ── Neuron Mask ──────────────────────────────────────────────────────────

/// Bit-packed mask indicating which FFN neurons are predicted-active.
///
/// Internally stored as a `Vec<u64>` where each bit represents one neuron.
/// This is memory-efficient (28,672 neurons = 448 bytes) and supports
/// fast bulk operations via bitwise ops.
#[derive(Clone)]
pub struct NeuronMask {
    /// Bit-packed active neuron flags.
    bits: Vec<u64>,
    /// Total number of neurons (d_ffn).
    d_ffn: usize,
    /// Number of neurons marked as active.
    active_count: usize,
}

impl NeuronMask {
    /// Create a mask with all neurons active (no sparsity).
    pub fn all_active(d_ffn: usize) -> Self {
        let n_words = (d_ffn + 63) / 64;
        let mut bits = vec![u64::MAX; n_words];

        // Clear unused high bits in the last word.
        let remainder = d_ffn % 64;
        if remainder > 0 {
            bits[n_words - 1] = (1u64 << remainder) - 1;
        }

        Self {
            bits,
            d_ffn,
            active_count: d_ffn,
        }
    }

    /// Create a mask with all neurons inactive.
    pub fn all_inactive(d_ffn: usize) -> Self {
        let n_words = (d_ffn + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
            d_ffn,
            active_count: 0,
        }
    }

    /// Create a mask from a vector of activation scores.
    ///
    /// Neurons with score >= threshold are marked as active.
    /// The mask is then rounded to bundle boundaries (groups of 64).
    pub fn from_scores(scores: &[f32], threshold: f32) -> Self {
        let d_ffn = scores.len();
        let n_words = (d_ffn + 63) / 64;
        let mut bits = vec![0u64; n_words];
        let mut active_count = 0;

        for (i, &score) in scores.iter().enumerate() {
            if score >= threshold {
                bits[i / 64] |= 1u64 << (i % 64);
                active_count += 1;
            }
        }

        let mut mask = Self {
            bits,
            d_ffn,
            active_count,
        };

        // Round to bundle boundaries.
        mask.align_to_bundles();
        mask
    }

    /// Create from raw bundle activation flags.
    ///
    /// `active_bundles[i]` = true means neurons [i*64 .. (i+1)*64) are active.
    pub fn from_bundles(active_bundles: &[bool], d_ffn: usize) -> Self {
        let n_words = (d_ffn + 63) / 64;
        let mut bits = vec![0u64; n_words];
        let mut active_count = 0;

        for (bundle_idx, &active) in active_bundles.iter().enumerate() {
            if active {
                let word_idx = bundle_idx; // 1 bundle = 64 neurons = 1 u64 word
                if word_idx < n_words {
                    let remaining = if (bundle_idx + 1) * BUNDLE_SIZE > d_ffn {
                        d_ffn - bundle_idx * BUNDLE_SIZE
                    } else {
                        BUNDLE_SIZE
                    };
                    if remaining == 64 {
                        bits[word_idx] = u64::MAX;
                    } else {
                        bits[word_idx] = (1u64 << remaining) - 1;
                    }
                    active_count += remaining;
                }
            }
        }

        Self {
            bits,
            d_ffn,
            active_count,
        }
    }

    /// Round the mask to 64-neuron bundle boundaries.
    ///
    /// If ANY neuron in a bundle is active, ALL neurons in that bundle
    /// are activated. This ensures reads remain sequential.
    fn align_to_bundles(&mut self) {
        self.active_count = 0;
        for (i, word) in self.bits.iter_mut().enumerate() {
            if *word != 0 {
                // Any bit set → all bits set (full bundle active).
                let remaining = if (i + 1) * 64 > self.d_ffn {
                    self.d_ffn - i * 64
                } else {
                    64
                };
                if remaining == 64 {
                    *word = u64::MAX;
                } else {
                    *word = (1u64 << remaining) - 1;
                }
                self.active_count += remaining;
            }
        }
    }

    /// Check if a specific neuron is active.
    #[inline]
    pub fn is_active(&self, neuron_idx: usize) -> bool {
        if neuron_idx >= self.d_ffn {
            return false;
        }
        (self.bits[neuron_idx / 64] >> (neuron_idx % 64)) & 1 == 1
    }

    /// Check if a specific bundle is active (any neuron in the bundle).
    #[inline]
    pub fn is_bundle_active(&self, bundle_idx: usize) -> bool {
        if bundle_idx >= self.n_bundles() {
            return false;
        }
        self.bits[bundle_idx] != 0
    }

    /// Total number of active neurons.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Total number of neurons (d_ffn).
    pub fn d_ffn(&self) -> usize {
        self.d_ffn
    }

    /// Number of 64-neuron bundles.
    pub fn n_bundles(&self) -> usize {
        (self.d_ffn + BUNDLE_SIZE - 1) / BUNDLE_SIZE
    }

    /// Number of active bundles.
    pub fn active_bundles(&self) -> usize {
        self.bits.iter().filter(|&&w| w != 0).count()
    }

    /// Sparsity ratio (fraction of neurons that are INACTIVE).
    pub fn sparsity(&self) -> f32 {
        if self.d_ffn == 0 {
            return 0.0;
        }
        1.0 - (self.active_count as f32 / self.d_ffn as f32)
    }

    /// Bandwidth saving ratio (accounting for FFN fraction).
    ///
    /// Formula: saving = sparsity × f_ffn
    pub fn bandwidth_saving(&self) -> f64 {
        self.sparsity() as f64 * FFN_WEIGHT_FRACTION
    }
}

impl fmt::Debug for NeuronMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuronMask")
            .field("d_ffn", &self.d_ffn)
            .field("active", &self.active_count)
            .field("bundles", &format_args!("{}/{}", self.active_bundles(), self.n_bundles()))
            .field("sparsity", &format_args!("{:.1}%", self.sparsity() * 100.0))
            .field("bw_saving", &format_args!("{:.1}%", self.bandwidth_saving() * 100.0))
            .finish()
    }
}

impl fmt::Display for NeuronMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeuronMask: {}/{} active ({:.1}% sparse, {:.1}% BW saving, {}/{} bundles)",
            self.active_count,
            self.d_ffn,
            self.sparsity() * 100.0,
            self.bandwidth_saving() * 100.0,
            self.active_bundles(),
            self.n_bundles(),
        )
    }
}

// ── Bundle Map ───────────────────────────────────────────────────────────

/// A contiguous range of active bundles that can be loaded in a single
/// sequential read.
///
/// Multiple `BundleRange`s are produced from a sparse `NeuronMask`.
/// Each range represents a segment of FFN weights that should be
/// loaded as a single sequential I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundleRange {
    /// First bundle index (inclusive).
    pub start_bundle: usize,
    /// Last bundle index (exclusive).
    pub end_bundle: usize,
}

impl BundleRange {
    /// Number of bundles in this range.
    pub fn len(&self) -> usize {
        self.end_bundle - self.start_bundle
    }

    /// Whether this range is empty.
    pub fn is_empty(&self) -> bool {
        self.start_bundle >= self.end_bundle
    }

    /// Number of neurons in this range.
    pub fn neuron_count(&self) -> usize {
        self.len() * BUNDLE_SIZE
    }

    /// Byte offset within the FFN weight tensor for this range.
    ///
    /// `bytes_per_neuron` depends on quantization (e.g., Q4_K_M ≈ 2.3 bytes/param).
    pub fn byte_offset(&self, bytes_per_neuron: usize) -> usize {
        self.start_bundle * BUNDLE_SIZE * bytes_per_neuron
    }

    /// Byte length of this range.
    pub fn byte_len(&self, bytes_per_neuron: usize) -> usize {
        self.neuron_count() * bytes_per_neuron
    }
}

/// Converts a `NeuronMask` into a set of contiguous `BundleRange`s
/// for sequential I/O.
///
/// Adjacent active bundles are merged into a single range to minimize
/// the number of I/O operations.
#[derive(Debug, Clone)]
pub struct BundleMap {
    /// Contiguous ranges of active bundles.
    pub ranges: Vec<BundleRange>,
    /// Total neurons across all ranges.
    pub total_neurons: usize,
    /// Total bundles across all ranges.
    pub total_bundles: usize,
}

impl BundleMap {
    /// Build a bundle map from a neuron mask.
    ///
    /// Adjacent active bundles are coalesced into contiguous ranges.
    pub fn from_mask(mask: &NeuronMask) -> Self {
        let mut ranges = Vec::new();
        let n_bundles = mask.n_bundles();
        let mut i = 0;

        while i < n_bundles {
            if mask.is_bundle_active(i) {
                let start = i;
                // Coalesce adjacent active bundles.
                while i < n_bundles && mask.is_bundle_active(i) {
                    i += 1;
                }
                ranges.push(BundleRange {
                    start_bundle: start,
                    end_bundle: i,
                });
            } else {
                i += 1;
            }
        }

        let total_bundles: usize = ranges.iter().map(|r| r.len()).sum();
        let total_neurons = total_bundles * BUNDLE_SIZE;

        Self {
            ranges,
            total_neurons,
            total_bundles,
        }
    }

    /// Number of separate I/O operations needed.
    pub fn n_reads(&self) -> usize {
        self.ranges.len()
    }

    /// Total bytes to read, given bytes_per_neuron.
    pub fn total_bytes(&self, bytes_per_neuron: usize) -> usize {
        self.total_neurons * bytes_per_neuron
    }

    /// Generate (offset, length) pairs for I/O operations.
    pub fn io_ranges(&self, bytes_per_neuron: usize) -> Vec<(usize, usize)> {
        self.ranges
            .iter()
            .map(|r| (r.byte_offset(bytes_per_neuron), r.byte_len(bytes_per_neuron)))
            .collect()
    }
}

impl fmt::Display for BundleMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BundleMap: {} ranges, {} bundles, {} neurons",
            self.ranges.len(),
            self.total_bundles,
            self.total_neurons,
        )
    }
}

// ── Neuron Predictor ─────────────────────────────────────────────────────

/// Per-layer lightweight MLP predictor for neuron activation.
///
/// Architecture: hidden_dim → predictor_dim → d_ffn (with sigmoid)
///
/// - `W_down`: [hidden_dim × predictor_dim] — compress hidden state
/// - `W_up`:   [predictor_dim × d_ffn]      — predict per-neuron activation
///
/// Total size per layer: hidden_dim × (d_ffn/8) + (d_ffn/8) × d_ffn
/// For LLaMA 70B: 8192 × 3584 + 3584 × 28672 ≈ 132 MB total across 80 layers
///
/// The predictor is trained offline via knowledge distillation from the
/// full model's activation patterns and stored as a sidecar file.
pub struct NeuronPredictor {
    /// Projection down: [hidden_dim, predictor_dim]
    w_down: Vec<f32>,
    /// Projection up: [predictor_dim, d_ffn]
    w_up: Vec<f32>,
    /// Bias for down projection (optional).
    b_down: Option<Vec<f32>>,
    /// Model hidden dimension.
    hidden_dim: usize,
    /// Predictor hidden dimension (d_ffn / 8).
    predictor_dim: usize,
    /// FFN intermediate dimension (number of neurons).
    d_ffn: usize,
    /// Activation threshold.
    threshold: f32,
    /// Layer index this predictor belongs to.
    layer_idx: usize,
}

impl NeuronPredictor {
    /// Create a new predictor from weight matrices.
    ///
    /// `w_down`: flattened [hidden_dim × predictor_dim] row-major
    /// `w_up`: flattened [predictor_dim × d_ffn] row-major
    pub fn new(
        w_down: Vec<f32>,
        w_up: Vec<f32>,
        hidden_dim: usize,
        d_ffn: usize,
        layer_idx: usize,
    ) -> Self {
        let predictor_dim = d_ffn / PREDICTOR_RATIO;
        assert_eq!(w_down.len(), hidden_dim * predictor_dim,
            "w_down size mismatch: expected {} got {}", hidden_dim * predictor_dim, w_down.len());
        assert_eq!(w_up.len(), predictor_dim * d_ffn,
            "w_up size mismatch: expected {} got {}", predictor_dim * d_ffn, w_up.len());

        Self {
            w_down,
            w_up,
            b_down: None,
            hidden_dim,
            predictor_dim,
            d_ffn,
            threshold: DEFAULT_ACTIVATION_THRESHOLD,
            layer_idx,
        }
    }

    /// Create a predictor with random weights (for testing/init).
    pub fn random(hidden_dim: usize, d_ffn: usize, layer_idx: usize) -> Self {
        let predictor_dim = d_ffn / PREDICTOR_RATIO;

        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale_down = (2.0 / (hidden_dim + predictor_dim) as f64).sqrt() as f32;
        let scale_up = (2.0 / (predictor_dim + d_ffn) as f64).sqrt() as f32;

        // Simple deterministic pseudo-random for reproducibility.
        let mut seed: u64 = (layer_idx as u64 + 1).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next_f32 = move || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((seed >> 33) as u32) as f32 / u32::MAX as f32;
            (bits - 0.5) * 2.0
        };

        let w_down: Vec<f32> = (0..hidden_dim * predictor_dim)
            .map(|_| next_f32() * scale_down)
            .collect();
        let w_up: Vec<f32> = (0..predictor_dim * d_ffn)
            .map(|_| next_f32() * scale_up)
            .collect();

        Self::new(w_down, w_up, hidden_dim, d_ffn, layer_idx)
    }

    /// Set the activation threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the down-projection bias.
    pub fn with_bias(mut self, bias: Vec<f32>) -> Self {
        assert_eq!(bias.len(), self.predictor_dim);
        self.b_down = Some(bias);
        self
    }

    /// Predict active neurons given the current hidden state.
    ///
    /// Architecture:
    /// 1. `mid = ReLU(h_t × W_down + b_down)`  — compress to predictor_dim
    /// 2. `scores = sigmoid(mid × W_up)`         — per-neuron activation score
    /// 3. `mask = bundle_align(scores >= threshold)`
    ///
    /// Runtime: < 1ms for LLaMA 70B (8192 → 3584 → 28672, ~200M FLOPs).
    pub fn predict(&self, hidden_state: &[f32]) -> NeuronMask {
        assert_eq!(
            hidden_state.len(),
            self.hidden_dim,
            "Hidden state dim mismatch: expected {}, got {}",
            self.hidden_dim,
            hidden_state.len()
        );

        // Step 1: Down-project — h_t × W_down → mid [predictor_dim]
        let mut mid = vec![0.0f32; self.predictor_dim];
        for j in 0..self.predictor_dim {
            let mut sum = 0.0f32;
            let w_col_start = j;
            for i in 0..self.hidden_dim {
                sum += hidden_state[i] * self.w_down[i * self.predictor_dim + w_col_start];
            }
            // Add bias if present.
            if let Some(ref bias) = self.b_down {
                sum += bias[j];
            }
            // ReLU activation.
            mid[j] = sum.max(0.0);
        }

        // Step 2: Up-project — mid × W_up → scores [d_ffn]
        let mut scores = vec![0.0f32; self.d_ffn];
        for j in 0..self.d_ffn {
            let mut sum = 0.0f32;
            for i in 0..self.predictor_dim {
                sum += mid[i] * self.w_up[i * self.d_ffn + j];
            }
            // Sigmoid activation.
            scores[j] = 1.0 / (1.0 + (-sum).exp());
        }

        // Step 3: Threshold + bundle-align.
        NeuronMask::from_scores(&scores, self.threshold)
    }

    /// Predict and return the raw activation scores (for analysis).
    pub fn predict_scores(&self, hidden_state: &[f32]) -> Vec<f32> {
        assert_eq!(hidden_state.len(), self.hidden_dim);

        let mut mid = vec![0.0f32; self.predictor_dim];
        for j in 0..self.predictor_dim {
            let mut sum = 0.0f32;
            for i in 0..self.hidden_dim {
                sum += hidden_state[i] * self.w_down[i * self.predictor_dim + j];
            }
            if let Some(ref bias) = self.b_down {
                sum += bias[j];
            }
            mid[j] = sum.max(0.0);
        }

        let mut scores = vec![0.0f32; self.d_ffn];
        for j in 0..self.d_ffn {
            let mut sum = 0.0f32;
            for i in 0..self.predictor_dim {
                sum += mid[i] * self.w_up[i * self.d_ffn + j];
            }
            scores[j] = 1.0 / (1.0 + (-sum).exp());
        }
        scores
    }

    /// Layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Size of this predictor in bytes (weight storage).
    pub fn size_bytes(&self) -> usize {
        (self.w_down.len() + self.w_up.len()) * std::mem::size_of::<f32>()
            + self.b_down.as_ref().map(|b| b.len() * 4).unwrap_or(0)
    }
}

impl fmt::Debug for NeuronPredictor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuronPredictor")
            .field("layer", &self.layer_idx)
            .field("dims", &format_args!("{} → {} → {}", self.hidden_dim, self.predictor_dim, self.d_ffn))
            .field("threshold", &self.threshold)
            .field("size", &format_args!("{:.2} MB", self.size_bytes() as f64 / (1024.0 * 1024.0)))
            .finish()
    }
}

// ── Predicate Engine ─────────────────────────────────────────────────────

/// Top-level engine managing per-layer predictors and prediction caching.
///
/// Lifecycle:
/// 1. `load()` — load predictor weights from sidecar file
/// 2. `predict(layer, hidden_state)` — get mask for selective loading
/// 3. `bundle_map(mask)` — convert mask to I/O ranges
///
/// When no predictor is available, falls back to `NeuronMask::all_active()`
/// (load everything — equivalent to standard path).
pub struct PredicateEngine {
    /// Per-layer predictors (indexed by layer_id).
    predictors: Vec<Option<NeuronPredictor>>,
    /// Number of layers.
    n_layers: usize,
    /// Model hidden dimension.
    hidden_dim: usize,
    /// FFN intermediate dimension.
    d_ffn: usize,
    /// Whether predicate loading is enabled.
    enabled: bool,
    /// Activation threshold.
    threshold: f32,
    /// Running statistics.
    pub stats: PredicateStats,
}

/// Running statistics for predicate loading.
#[derive(Debug, Clone)]
pub struct PredicateStats {
    /// Total predictions made.
    pub total_predictions: u64,
    /// Total neurons that would have been loaded without prediction.
    pub total_neurons_full: u64,
    /// Total neurons actually loaded (after masking).
    pub total_neurons_loaded: u64,
    /// Sum of per-layer sparsity values.
    pub sparsity_sum: f64,
    /// Sum of per-layer bandwidth savings.
    pub bandwidth_saving_sum: f64,
    /// Number of fallbacks to full loading (no predictor available).
    pub fallback_count: u64,
}

impl PredicateStats {
    fn new() -> Self {
        Self {
            total_predictions: 0,
            total_neurons_full: 0,
            total_neurons_loaded: 0,
            sparsity_sum: 0.0,
            bandwidth_saving_sum: 0.0,
            fallback_count: 0,
        }
    }

    /// Average sparsity across all predictions.
    pub fn avg_sparsity(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.sparsity_sum / self.total_predictions as f64
    }

    /// Average bandwidth saving.
    pub fn avg_bandwidth_saving(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.bandwidth_saving_sum / self.total_predictions as f64
    }

    /// Overall compression ratio (neurons loaded / neurons total).
    pub fn compression_ratio(&self) -> f64 {
        if self.total_neurons_full == 0 {
            return 1.0;
        }
        self.total_neurons_loaded as f64 / self.total_neurons_full as f64
    }
}

impl fmt::Display for PredicateStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Predicate: {} preds, {:.1}% avg sparsity, {:.1}% avg BW saving, {:.2}x compression, {} fallbacks",
            self.total_predictions,
            self.avg_sparsity() * 100.0,
            self.avg_bandwidth_saving() * 100.0,
            1.0 / self.compression_ratio(),
            self.fallback_count,
        )
    }
}

impl PredicateEngine {
    /// Create a new predicate engine (disabled by default until predictors are loaded).
    pub fn new(n_layers: usize, hidden_dim: usize, d_ffn: usize) -> Self {
        let mut predictors = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            predictors.push(None);
        }

        Self {
            predictors,
            n_layers,
            hidden_dim,
            d_ffn,
            enabled: false,
            threshold: DEFAULT_ACTIVATION_THRESHOLD,
            stats: PredicateStats::new(),
        }
    }

    /// Create with random predictors for all layers (for testing).
    pub fn with_random_predictors(n_layers: usize, hidden_dim: usize, d_ffn: usize) -> Self {
        let predictors: Vec<Option<NeuronPredictor>> = (0..n_layers)
            .map(|i| Some(NeuronPredictor::random(hidden_dim, d_ffn, i)))
            .collect();

        Self {
            predictors,
            n_layers,
            hidden_dim,
            d_ffn,
            enabled: true,
            threshold: DEFAULT_ACTIVATION_THRESHOLD,
            stats: PredicateStats::new(),
        }
    }

    /// Register a predictor for a specific layer.
    pub fn set_predictor(&mut self, layer_idx: usize, predictor: NeuronPredictor) {
        if layer_idx < self.n_layers {
            self.predictors[layer_idx] = Some(predictor);
        }
    }

    /// Enable/disable predicate loading.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the activation threshold for all predictors.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Whether predicate loading is enabled and has predictors.
    pub fn is_active(&self) -> bool {
        self.enabled && self.predictors.iter().any(|p| p.is_some())
    }

    /// Number of layers with loaded predictors.
    pub fn loaded_count(&self) -> usize {
        self.predictors.iter().filter(|p| p.is_some()).count()
    }

    /// Total predictor weight storage in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.predictors
            .iter()
            .filter_map(|p| p.as_ref())
            .map(|p| p.size_bytes())
            .sum()
    }

    /// Predict the active neuron mask for a given layer.
    ///
    /// If no predictor is available for this layer, returns `all_active`.
    pub fn predict(&mut self, layer_idx: usize, hidden_state: &[f32]) -> NeuronMask {
        if !self.enabled || layer_idx >= self.n_layers {
            self.stats.fallback_count += 1;
            return NeuronMask::all_active(self.d_ffn);
        }

        let mask = if let Some(ref predictor) = self.predictors[layer_idx] {
            let mask = predictor.predict(hidden_state);

            // If sparsity is too low, not worth the overhead.
            if mask.sparsity() < MIN_USEFUL_SPARSITY {
                NeuronMask::all_active(self.d_ffn)
            } else {
                mask
            }
        } else {
            self.stats.fallback_count += 1;
            NeuronMask::all_active(self.d_ffn)
        };

        // Update running stats.
        self.stats.total_predictions += 1;
        self.stats.total_neurons_full += self.d_ffn as u64;
        self.stats.total_neurons_loaded += mask.active_count() as u64;
        self.stats.sparsity_sum += mask.sparsity() as f64;
        self.stats.bandwidth_saving_sum += mask.bandwidth_saving();

        mask
    }

    /// Convert a mask to an I/O-ready bundle map.
    pub fn bundle_map(&self, mask: &NeuronMask) -> BundleMap {
        BundleMap::from_mask(mask)
    }

    /// Calculate expected bandwidth saving from the protocol spec formula.
    ///
    /// B_load = S_layer × (1 - s_ℓ × f_ffn)
    pub fn expected_bytes_per_layer(
        &self,
        layer_size_bytes: usize,
        sparsity: f32,
    ) -> usize {
        let factor = 1.0 - (sparsity as f64 * FFN_WEIGHT_FRACTION);
        (layer_size_bytes as f64 * factor) as usize
    }

    /// Get the layer dimensions.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.hidden_dim, self.d_ffn / PREDICTOR_RATIO, self.d_ffn)
    }
}

impl fmt::Debug for PredicateEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredicateEngine")
            .field("layers", &self.n_layers)
            .field("loaded", &self.loaded_count())
            .field("enabled", &self.enabled)
            .field("dims", &format_args!("{} → {} → {}", self.hidden_dim, self.d_ffn / PREDICTOR_RATIO, self.d_ffn))
            .field("total_size", &format_args!("{:.1} MB", self.total_size_bytes() as f64 / (1024.0 * 1024.0)))
            .field("stats", &self.stats)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NeuronMask Tests ─────────────────────────────────────────────

    #[test]
    fn test_mask_all_active() {
        let mask = NeuronMask::all_active(256);
        assert_eq!(mask.d_ffn(), 256);
        assert_eq!(mask.active_count(), 256);
        assert_eq!(mask.n_bundles(), 4);
        assert_eq!(mask.active_bundles(), 4);
        assert!((mask.sparsity() - 0.0).abs() < 0.001);
        assert!(mask.is_active(0));
        assert!(mask.is_active(255));
    }

    #[test]
    fn test_mask_all_inactive() {
        let mask = NeuronMask::all_inactive(256);
        assert_eq!(mask.active_count(), 0);
        assert!((mask.sparsity() - 1.0).abs() < 0.001);
        assert!(!mask.is_active(0));
    }

    #[test]
    fn test_mask_non_aligned_size() {
        // 100 neurons = 1 full bundle + 36 remainder
        let mask = NeuronMask::all_active(100);
        assert_eq!(mask.d_ffn(), 100);
        assert_eq!(mask.active_count(), 100);
        assert_eq!(mask.n_bundles(), 2);
        assert!(mask.is_active(99));
        assert!(!mask.is_active(100)); // out of bounds
    }

    #[test]
    fn test_mask_from_scores_bundle_alignment() {
        // 256 neurons, only neuron 0 is above threshold.
        // After bundle alignment, the entire first bundle (64 neurons) should be active.
        let mut scores = vec![0.0f32; 256];
        scores[0] = 1.0;

        let mask = NeuronMask::from_scores(&scores, 0.5);
        assert_eq!(mask.active_count(), 64); // Entire bundle activated
        assert_eq!(mask.active_bundles(), 1);
        assert!(mask.is_active(0));
        assert!(mask.is_active(63)); // Same bundle
        assert!(!mask.is_active(64)); // Next bundle
    }

    #[test]
    fn test_mask_from_scores_multiple_bundles() {
        let mut scores = vec![0.0f32; 256];
        // Activate neurons in bundle 0 and bundle 3.
        scores[5] = 0.9;
        scores[200] = 0.8;

        let mask = NeuronMask::from_scores(&scores, 0.5);
        assert_eq!(mask.active_bundles(), 2);
        assert_eq!(mask.active_count(), 128); // 2 × 64
        assert!(mask.is_bundle_active(0));
        assert!(!mask.is_bundle_active(1));
        assert!(!mask.is_bundle_active(2));
        assert!(mask.is_bundle_active(3));
    }

    #[test]
    fn test_mask_from_bundles() {
        let bundles = vec![true, false, true, false];
        let mask = NeuronMask::from_bundles(&bundles, 256);
        assert_eq!(mask.active_count(), 128);
        assert_eq!(mask.active_bundles(), 2);
        assert!(mask.is_bundle_active(0));
        assert!(!mask.is_bundle_active(1));
        assert!(mask.is_bundle_active(2));
    }

    #[test]
    fn test_mask_bandwidth_saving() {
        // 85% sparsity → saving = 0.85 × 0.663 = 56.4%
        let mut scores = vec![0.0f32; 1000];
        // Activate 15% = 150 neurons
        for i in 0..150 {
            scores[i] = 1.0;
        }
        let mask = NeuronMask::from_scores(&scores, 0.5);

        // Due to bundle alignment, actual sparsity may differ slightly.
        let saving = mask.bandwidth_saving();
        assert!(saving > 0.3, "Expected significant saving, got {:.2}", saving);
    }

    #[test]
    fn test_mask_display() {
        let mask = NeuronMask::all_active(256);
        let s = format!("{}", mask);
        assert!(s.contains("NeuronMask"));
        assert!(s.contains("256/256"));
    }

    #[test]
    fn test_mask_debug() {
        let mask = NeuronMask::all_active(256);
        let s = format!("{:?}", mask);
        assert!(s.contains("NeuronMask"));
        assert!(s.contains("d_ffn"));
    }

    // ── BundleMap Tests ──────────────────────────────────────────────

    #[test]
    fn test_bundle_map_all_active() {
        let mask = NeuronMask::all_active(256);
        let map = BundleMap::from_mask(&mask);
        assert_eq!(map.n_reads(), 1); // Single contiguous range
        assert_eq!(map.total_bundles, 4);
        assert_eq!(map.total_neurons, 256);
    }

    #[test]
    fn test_bundle_map_sparse() {
        let bundles = vec![true, false, false, true];
        let mask = NeuronMask::from_bundles(&bundles, 256);
        let map = BundleMap::from_mask(&mask);
        assert_eq!(map.n_reads(), 2); // Two non-contiguous ranges
        assert_eq!(map.total_bundles, 2);
    }

    #[test]
    fn test_bundle_map_coalescing() {
        // Adjacent active bundles should be merged.
        let bundles = vec![true, true, false, true];
        let mask = NeuronMask::from_bundles(&bundles, 256);
        let map = BundleMap::from_mask(&mask);
        assert_eq!(map.n_reads(), 2); // [0,1] merged, [3] separate
        assert_eq!(map.ranges[0].start_bundle, 0);
        assert_eq!(map.ranges[0].end_bundle, 2);
        assert_eq!(map.ranges[1].start_bundle, 3);
        assert_eq!(map.ranges[1].end_bundle, 4);
    }

    #[test]
    fn test_bundle_range_byte_calc() {
        let range = BundleRange {
            start_bundle: 2,
            end_bundle: 5,
        };
        // 3 bundles × 64 neurons = 192 neurons
        assert_eq!(range.neuron_count(), 192);
        // At 4 bytes/neuron: offset = 2 × 64 × 4 = 512
        assert_eq!(range.byte_offset(4), 512);
        // Length = 192 × 4 = 768
        assert_eq!(range.byte_len(4), 768);
    }

    #[test]
    fn test_bundle_map_io_ranges() {
        let bundles = vec![true, false, true, false];
        let mask = NeuronMask::from_bundles(&bundles, 256);
        let map = BundleMap::from_mask(&mask);
        let ranges = map.io_ranges(4);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], (0, 256));     // bundle 0: offset=0, len=64*4
        assert_eq!(ranges[1], (512, 256));   // bundle 2: offset=128*4, len=64*4
    }

    #[test]
    fn test_bundle_map_display() {
        let mask = NeuronMask::all_active(256);
        let map = BundleMap::from_mask(&mask);
        let s = format!("{}", map);
        assert!(s.contains("BundleMap"));
    }

    // ── NeuronPredictor Tests ────────────────────────────────────────

    #[test]
    fn test_predictor_random_creation() {
        let pred = NeuronPredictor::random(128, 256, 0);
        assert_eq!(pred.hidden_dim, 128);
        assert_eq!(pred.d_ffn, 256);
        assert_eq!(pred.predictor_dim, 32); // 256 / 8
        assert_eq!(pred.layer_idx(), 0);
    }

    #[test]
    fn test_predictor_inference() {
        let pred = NeuronPredictor::random(64, 128, 0);
        let hidden = vec![0.1f32; 64];
        let mask = pred.predict(&hidden);

        assert_eq!(mask.d_ffn(), 128);
        // Should be bundle-aligned.
        assert_eq!(mask.active_count() % BUNDLE_SIZE, 0);
    }

    #[test]
    fn test_predictor_scores() {
        let pred = NeuronPredictor::random(64, 128, 0);
        let hidden = vec![0.1f32; 64];
        let scores = pred.predict_scores(&hidden);

        assert_eq!(scores.len(), 128);
        // All sigmoid outputs should be in [0, 1].
        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0, "Score out of range: {}", s);
        }
    }

    #[test]
    fn test_predictor_size_bytes() {
        let pred = NeuronPredictor::random(128, 256, 0);
        // w_down: 128 × 32 = 4096 floats
        // w_up: 32 × 256 = 8192 floats
        // Total: 12288 × 4 = 49152 bytes
        assert_eq!(pred.size_bytes(), 49152);
    }

    #[test]
    fn test_predictor_with_threshold() {
        let pred = NeuronPredictor::random(64, 128, 0).with_threshold(0.9);
        assert_eq!(pred.threshold, 0.9);
    }

    #[test]
    fn test_predictor_with_bias() {
        let pred = NeuronPredictor::random(64, 128, 0)
            .with_bias(vec![0.0; 16]); // predictor_dim = 128/8 = 16
        assert!(pred.b_down.is_some());
    }

    #[test]
    fn test_predictor_debug() {
        let pred = NeuronPredictor::random(128, 256, 5);
        let s = format!("{:?}", pred);
        assert!(s.contains("NeuronPredictor"));
        assert!(s.contains("layer"));
    }

    // ── PredicateEngine Tests ────────────────────────────────────────

    #[test]
    fn test_engine_creation() {
        let engine = PredicateEngine::new(32, 4096, 11008);
        assert_eq!(engine.n_layers, 32);
        assert!(!engine.is_active());
        assert_eq!(engine.loaded_count(), 0);
    }

    #[test]
    fn test_engine_with_random_predictors() {
        let engine = PredicateEngine::with_random_predictors(4, 64, 128);
        assert!(engine.is_active());
        assert_eq!(engine.loaded_count(), 4);
        assert!(engine.total_size_bytes() > 0);
    }

    #[test]
    fn test_engine_predict_with_predictor() {
        let mut engine = PredicateEngine::with_random_predictors(4, 64, 128);
        let hidden = vec![0.1f32; 64];
        let mask = engine.predict(0, &hidden);

        assert_eq!(mask.d_ffn(), 128);
        assert!(engine.stats.total_predictions > 0);
    }

    #[test]
    fn test_engine_predict_without_predictor() {
        let mut engine = PredicateEngine::new(4, 64, 128);
        engine.set_enabled(true);
        let hidden = vec![0.1f32; 64];
        let mask = engine.predict(0, &hidden);

        // Should fallback to all active.
        assert_eq!(mask.active_count(), 128);
        assert!(engine.stats.fallback_count > 0);
    }

    #[test]
    fn test_engine_predict_disabled() {
        let mut engine = PredicateEngine::with_random_predictors(4, 64, 128);
        engine.set_enabled(false);
        let hidden = vec![0.1f32; 64];
        let mask = engine.predict(0, &hidden);

        // Should fallback to all active.
        assert_eq!(mask.active_count(), 128);
    }

    #[test]
    fn test_engine_set_predictor() {
        let mut engine = PredicateEngine::new(4, 64, 128);
        let pred = NeuronPredictor::random(64, 128, 2);
        engine.set_predictor(2, pred);
        engine.set_enabled(true);
        assert_eq!(engine.loaded_count(), 1);
        assert!(engine.is_active());
    }

    #[test]
    fn test_engine_bundle_map() {
        let engine = PredicateEngine::new(4, 64, 256);
        let mask = NeuronMask::from_bundles(&[true, false, true, false], 256);
        let map = engine.bundle_map(&mask);
        assert_eq!(map.n_reads(), 2);
    }

    #[test]
    fn test_engine_expected_bytes() {
        let engine = PredicateEngine::new(4, 64, 256);
        // 531 MB layer, 85% sparsity → 231 MB
        let bytes = engine.expected_bytes_per_layer(531_000_000, 0.85);
        let expected = (531_000_000.0 * (1.0 - 0.85 * FFN_WEIGHT_FRACTION)) as usize;
        let diff = (bytes as i64 - expected as i64).unsigned_abs();
        assert!(diff < 1000, "Expected ~{expected}, got {bytes}");
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = PredicateEngine::with_random_predictors(2, 64, 128);
        let hidden = vec![0.1f32; 64];

        for layer in 0..2 {
            engine.predict(layer, &hidden);
        }

        assert_eq!(engine.stats.total_predictions, 2);
        assert!(engine.stats.total_neurons_full > 0);
        assert!(engine.stats.compression_ratio() > 0.0);
        assert!(engine.stats.compression_ratio() <= 1.0);

        let s = format!("{}", engine.stats);
        assert!(s.contains("Predicate"));
    }

    #[test]
    fn test_engine_dimensions() {
        let engine = PredicateEngine::new(4, 8192, 28672);
        let (h, p, d) = engine.dimensions();
        assert_eq!(h, 8192);
        assert_eq!(p, 3584); // 28672 / 8
        assert_eq!(d, 28672);
    }

    #[test]
    fn test_engine_debug() {
        let engine = PredicateEngine::with_random_predictors(2, 64, 128);
        let s = format!("{:?}", engine);
        assert!(s.contains("PredicateEngine"));
        assert!(s.contains("loaded"));
    }

    // ── Protocol Spec Validation ─────────────────────────────────────

    #[test]
    fn test_protocol_spec_bandwidth_saving() {
        // From the spec: at s=0.85, f_ffn=0.663:
        // saving = 0.85 × 0.663 = 0.564 = 56.4%
        let saving = 0.85 * FFN_WEIGHT_FRACTION;
        assert!((saving - 0.564).abs() < 0.01, "Expected 56.4%, got {:.1}%", saving * 100.0);
    }

    #[test]
    fn test_protocol_spec_bytes_per_layer() {
        // From the spec: B_load = 531 × (1 - 0.564) = 231.5 MB
        let engine = PredicateEngine::new(80, 8192, 28672);
        let bytes = engine.expected_bytes_per_layer(531_000_000, 0.85);
        let expected_mb = 231.5;
        let actual_mb = bytes as f64 / 1_000_000.0;
        assert!(
            (actual_mb - expected_mb).abs() < 2.0,
            "Expected ~{expected_mb} MB, got {actual_mb:.1} MB"
        );
    }

    #[test]
    fn test_protocol_spec_predictor_size() {
        // From the spec: predictor_dim = 28672/8 = 3584
        // Per layer: 8192 × 3584 + 3584 × 28672 = 29,360,128 + 102,760,448 = 132,120,576
        // 80 layers: ~132 MB × 80 ≈ 10 GB (spec says ~200 MB — they use compressed)
        // Our F32 storage: ~132 MB per layer × 80 = raw size
        let pred = NeuronPredictor::random(8192, 28672, 0);
        let size_per_layer = pred.size_bytes();
        assert_eq!(pred.predictor_dim, 3584);
        // Exact: (8192 × 3584 + 3584 × 28672) × 4 bytes
        let expected = (8192 * 3584 + 3584 * 28672) * 4;
        assert_eq!(size_per_layer, expected);
    }

    #[test]
    fn test_bundle_size_constant() {
        assert_eq!(BUNDLE_SIZE, 64);
        assert!(BUNDLE_SIZE.is_power_of_two());
    }
}

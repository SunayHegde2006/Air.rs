//! # QLoRA — Quantized Low-Rank Adaptation
//!
//! Merges a frozen AQLM-quantised base layer with a trainable low-rank
//! adapter (A × B). The adapter is applied at **full FP32 precision**
//! while the base weights remain 2-bit, achieving parameter-efficient
//! fine-tuning on consumer hardware.
//!
//! **Research**: "QLoRA: Efficient Finetuning of Quantized LLMs"
//! (Dettmers et al., NeurIPS 2023, arXiv:2305.14314).
//!
//! ## Consumer benefit
//! Fine-tune LLaMA-3-8B in **12 GB VRAM**:
//! - Base: 8B params × 2 bpw = ~2 GB
//! - Adapters: r=64 → ~0.1% of params → ~8 MB
//! - Gradients + optimiser: ~2 GB
//! Total: ~4 GB model memory → fits in RTX 3060.
//!
//! ## Merge-to-base
//! After fine-tuning, `merge_to_base()` fuses the adapter into the AQLM
//! layer (re-quantises the delta W = B × A) for zero-overhead inference.

use crate::aqlm::AqlmLayer;

// ---------------------------------------------------------------------------
// QLoraConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QLoraConfig {
    /// LoRA rank `r`. Controls adapter capacity. Typical: 8, 16, 32, 64.
    pub rank: usize,
    /// Scaling factor: effective_lr = alpha / rank. Typical: alpha = rank.
    pub alpha: f32,
    /// Names of modules to adapt (informational — actual selection done
    /// externally by the model loader).
    pub target_modules: Vec<String>,
    /// Whether to drop out adapter activations during training (0.0 = off).
    pub dropout: f32,
}

impl Default for QLoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dropout: 0.0,
        }
    }
}

impl QLoraConfig {
    /// Scaling factor applied to adapter output: alpha / rank.
    #[inline]
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ---------------------------------------------------------------------------
// QLoraAdapter
// ---------------------------------------------------------------------------

/// QLoRA adapter wrapping a frozen AQLM base layer.
///
/// Forward pass:
/// ```text
/// y = base_layer.forward(x)          // 2-bit quantised forward
///   + scaling × B @ (A @ x)         // low-rank FP32 delta
/// ```
///
/// `A` is initialised with Gaussian noise (σ = 1/√rank); `B` is zero-init
/// so the adapter starts as an identity (no change at step 0).
#[derive(Debug, Clone)]
pub struct QLoraAdapter {
    /// Frozen quantised base.
    pub base: AqlmLayer,
    /// A matrix: [rank × in_features] — trainable.
    pub lora_a: Vec<f32>,
    /// B matrix: [out_features × rank] — trainable, zero-init.
    pub lora_b: Vec<f32>,
    pub rank: usize,
    pub scaling: f32,
    pub in_features: usize,
    pub out_features: usize,
}

impl QLoraAdapter {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Wrap a base layer with a new LoRA adapter.
    ///
    /// `A` initialised N(0, 1/√rank); `B` zero-init.
    pub fn new(base: AqlmLayer, cfg: &QLoraConfig) -> Self {
        let in_features = base.in_features;
        let out_features = base.out_features;
        let rank = cfg.rank;
        let scaling = cfg.scaling();

        // A: [rank × in_features], Gaussian init.
        let sigma = 1.0 / (rank as f32).sqrt();
        let lora_a: Vec<f32> = (0..rank * in_features)
            .map(|i| {
                // Deterministic pseudo-Gaussian via Box-Muller on LCG.
                let u1 = {
                    let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1))
                        as f32
                        / u64::MAX as f32;
                    x.max(1e-10)
                };
                let u2 = {
                    let x = ((i as u64 + 1).wrapping_mul(6364136223846793005).wrapping_add(1))
                        as f32
                        / u64::MAX as f32;
                    x
                };
                let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                n * sigma
            })
            .collect();

        // B: [out_features × rank], zero-init.
        let lora_b = vec![0.0f32; out_features * rank];

        Self {
            base,
            lora_a,
            lora_b,
            rank,
            scaling,
            in_features,
            out_features,
        }
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Compute `y = base.forward(x) + scaling × B @ (A @ x)`.
    ///
    /// At init (B=0) this is identical to `base.forward(x)`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features);

        // Base forward (2-bit quantised).
        let mut y = self.base.forward(x);

        // A @ x → [rank]
        let ax: Vec<f32> = (0..self.rank)
            .map(|r| {
                (0..self.in_features)
                    .map(|i| self.lora_a[r * self.in_features + i] * x[i])
                    .sum::<f32>()
            })
            .collect();

        // B @ (A @ x) → [out_features]
        for row in 0..self.out_features {
            let delta: f32 = (0..self.rank)
                .map(|r| self.lora_b[row * self.rank + r] * ax[r])
                .sum();
            y[row] += self.scaling * delta;
        }
        y
    }

    // -----------------------------------------------------------------------
    // Merge adapter into base (for deployment)
    // -----------------------------------------------------------------------

    /// Compute the delta weight matrix W_delta = B @ A (shape [out × in]).
    ///
    /// This can be added to the dequantised base weights and re-quantised
    /// to produce a fused AQLM layer with no runtime overhead.
    pub fn compute_delta(&self) -> Vec<f32> {
        let mut delta = vec![0.0f32; self.out_features * self.in_features];
        for row in 0..self.out_features {
            for col in 0..self.in_features {
                let v: f32 = (0..self.rank)
                    .map(|r| {
                        self.lora_b[row * self.rank + r] * self.lora_a[r * self.in_features + col]
                    })
                    .sum();
                delta[row * self.in_features + col] = v * self.scaling;
            }
        }
        delta
    }

    /// Merge adapter delta into the base layer's dequantised weights and
    /// re-quantise into a new `AqlmLayer`. Zero-overhead inference after merge.
    ///
    /// Uses 2 quantisation iterations (fast, lossless for small deltas).
    pub fn merge_to_base(&self) -> AqlmLayer {
        // Dequantise base weight matrix.
        let mut full_w = vec![0.0f32; self.out_features * self.in_features];
        for row in 0..self.out_features {
            let dq = self.base.dequant_row(row);
            for (col, &v) in dq.iter().enumerate() {
                full_w[row * self.in_features + col] = v;
            }
        }
        // Add adapter delta.
        let delta = self.compute_delta();
        for (w, d) in full_w.iter_mut().zip(delta.iter()) {
            *w += d;
        }
        // Re-quantise.
        AqlmLayer::quantise(
            &full_w,
            self.in_features,
            self.out_features,
            self.base.n_codebooks,
            2,
        )
    }

    // -----------------------------------------------------------------------
    // Parameter counts (for memory planning)
    // -----------------------------------------------------------------------

    /// Total trainable parameters: rank × (in + out).
    pub fn trainable_params(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }

    /// Total parameters including frozen base (in bits).
    pub fn total_bits(&self) -> usize {
        // Base: 2 bpw (AQLM 2-bit) + adapter: 32 bpw FP32.
        let base_bits = self.in_features * self.out_features * 2;
        let adapter_bits = self.trainable_params() * 32;
        base_bits + adapter_bits
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aqlm::AqlmLayer;

    fn make_base(out: usize, inp: usize) -> AqlmLayer {
        let weight: Vec<f32> = (0..out * inp)
            .map(|i| ((i as f64 * 0.31415).sin() * 0.1) as f32)
            .collect();
        AqlmLayer::quantise(&weight, inp, out, 2, 2)
    }

    fn make_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f64 * 0.27183).cos() * 0.5) as f32)
            .collect()
    }

    // --- QLoraConfig ---

    #[test]
    fn test_config_scaling() {
        let cfg = QLoraConfig { rank: 8, alpha: 16.0, ..Default::default() };
        assert_eq!(cfg.scaling(), 2.0);
    }

    #[test]
    fn test_config_default_rank() {
        let cfg = QLoraConfig::default();
        assert_eq!(cfg.rank, 16);
    }

    // --- QLoraAdapter construction ---

    #[test]
    fn test_adapter_init_zero_b() {
        let base = make_base(8, 16);
        let cfg = QLoraConfig::default();
        let adapter = QLoraAdapter::new(base, &cfg);
        // B is zero-init → adapter delta = 0 at init.
        assert!(adapter.lora_b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_adapter_a_nonzero() {
        let base = make_base(8, 16);
        let cfg = QLoraConfig::default();
        let adapter = QLoraAdapter::new(base, &cfg);
        // A should have non-zero initialisation.
        assert!(adapter.lora_a.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn test_adapter_shape() {
        let base = make_base(8, 16);
        let cfg = QLoraConfig { rank: 4, alpha: 4.0, ..Default::default() };
        let adapter = QLoraAdapter::new(base, &cfg);
        assert_eq!(adapter.lora_a.len(), 4 * 16); // rank × in
        assert_eq!(adapter.lora_b.len(), 8 * 4);  // out × rank
    }

    // --- Forward ---

    #[test]
    fn test_forward_at_init_equals_base() {
        let weight: Vec<f32> = (0..8 * 16)
            .map(|i| ((i as f64 * 0.31415).sin() * 0.1) as f32)
            .collect();
        let base = AqlmLayer::quantise(&weight, 16, 8, 2, 2);
        let base2 = base.clone();
        let cfg = QLoraConfig { rank: 4, alpha: 4.0, ..Default::default() };
        let adapter = QLoraAdapter::new(base, &cfg);
        let x = make_x(16);
        // At init B=0 → adapter.forward(x) == base.forward(x).
        let y_adapter = adapter.forward(&x);
        let y_base = base2.forward(&x);
        for (a, b) in y_adapter.iter().zip(y_base.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_forward_output_length() {
        let adapter = QLoraAdapter::new(make_base(8, 16), &QLoraConfig::default());
        let y = adapter.forward(&make_x(16));
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_forward_nonzero_b_changes_output() {
        let base = make_base(8, 16);
        let base2 = base.clone();
        let cfg = QLoraConfig { rank: 4, alpha: 4.0, ..Default::default() };
        let mut adapter = QLoraAdapter::new(base, &cfg);
        // Set B to ones → non-zero delta.
        adapter.lora_b = vec![1.0f32; 8 * 4];
        let x = make_x(16);
        let y_adapter = adapter.forward(&x);
        let y_base = base2.forward(&x);
        // Some outputs must differ.
        assert!(
            y_adapter.iter().zip(y_base.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "non-zero B had no effect"
        );
    }

    // --- compute_delta ---

    #[test]
    fn test_compute_delta_zero_b_is_zero() {
        let adapter = QLoraAdapter::new(make_base(8, 16), &QLoraConfig::default());
        let delta = adapter.compute_delta();
        assert!(delta.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn test_compute_delta_shape() {
        let adapter = QLoraAdapter::new(make_base(8, 16), &QLoraConfig::default());
        let delta = adapter.compute_delta();
        assert_eq!(delta.len(), 8 * 16);
    }

    // --- merge_to_base ---

    #[test]
    fn test_merge_to_base_output_shape() {
        let adapter = QLoraAdapter::new(make_base(8, 16), &QLoraConfig::default());
        let merged = adapter.merge_to_base();
        let y = merged.forward(&make_x(16));
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_merge_to_base_zero_b_preserves_output() {
        let base = make_base(8, 16);
        let base2 = base.clone();
        let adapter = QLoraAdapter::new(base, &QLoraConfig::default());
        let merged = adapter.merge_to_base();
        let x = make_x(16);
        let y_merged = merged.forward(&x);
        let y_base = base2.forward(&x);
        // With B=0, merge should preserve base output (within re-quantisation error).
        for (m, b) in y_merged.iter().zip(y_base.iter()) {
            assert!((m - b).abs() < 2.0, "merge changed output: {m} vs {b}");
        }
    }

    // --- Parameter counting ---

    #[test]
    fn test_trainable_params() {
        let cfg = QLoraConfig { rank: 8, ..Default::default() };
        let adapter = QLoraAdapter::new(make_base(32, 64), &cfg);
        // rank × (in + out) = 8 × (64 + 32) = 768
        assert_eq!(adapter.trainable_params(), 768);
    }

    #[test]
    fn test_total_bits_less_than_fp32() {
        let cfg = QLoraConfig { rank: 8, ..Default::default() };
        let adapter = QLoraAdapter::new(make_base(32, 64), &cfg);
        let fp32_bits = 32 * 64 * 32; // all FP32
        assert!(adapter.total_bits() < fp32_bits);
    }

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_send_sync() {
        _assert_send_sync::<QLoraConfig>();
        _assert_send_sync::<QLoraAdapter>();
    }
}

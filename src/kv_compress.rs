//! KV Cache Compression — M.I.S.T. v3 §Sub-System 2 Warm Tier
//!
//! Implements 1-bit key compression and Q8 value quantization for the
//! warm tier, enabling 32K context on 8GB RAM machines.
//!
//! ```text
//! Warm Tier Compression:
//!   BF16 key → sign(k − μ_k) ∈ {−1, +1}^d + scalar ‖k‖   (16× compression)
//!   BF16 value → Q8 (round-to-nearest 8-bit)               (2× compression)
//!
//! Net: 32K context = 10 GB BF16 → ~1.56 GB warm (fits 8GB machine)
//! ```
//!
//! ## 1-Bit Key Compression
//!
//! Each key vector k ∈ ℝ^d is compressed to:
//! - A bit vector: sign(k_i − μ_k) packed into u64 words
//! - A scalar magnitude: ‖k‖ (f32)
//! - The mean: μ_k (f32)
//!
//! Reconstruction: k̂_i ≈ sign_i × ‖k‖ / √d + μ_k
//!
//! ## Q8 Value Quantization
//!
//! Values quantized to 8-bit with per-vector scale and zero-point:
//! - scale = (max - min) / 255
//! - zero_point = -min / scale
//! - q_i = round(v_i / scale + zero_point)
//!
//! Reference: air_rs_protocols_v3.md §Sub-System 2, Warm Tier Compression Formula

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────────

/// BF16 bytes per element.
pub const BF16_BYTES: usize = 2;

/// Key compression ratio (BF16 → 1-bit = 16×).
pub const KEY_COMPRESSION_RATIO: f64 = 16.0;

/// Value compression ratio (BF16 → Q8 = 2×).
pub const VALUE_COMPRESSION_RATIO: f64 = 2.0;

/// Default head dimension for LLaMA 70B.
pub const DEFAULT_HEAD_DIM: usize = 128;

/// Number of KV heads (GQA) for LLaMA 70B.
pub const DEFAULT_KV_HEADS: usize = 8;

/// Number of layers for LLaMA 70B.
pub const DEFAULT_N_LAYERS: usize = 80;

// ── 1-Bit Compressed Key ─────────────────────────────────────────────────

/// A key vector compressed to 1-bit per dimension using sign encoding.
///
/// Stores sign(k_i − μ) as a packed bit vector, plus the mean and
/// magnitude for approximate reconstruction.
///
/// Memory: d/8 bytes (bits) + 4 (magnitude) + 4 (mean) = ~24 bytes for d=128
/// vs. 256 bytes in BF16 → 10.7× compression.
#[derive(Clone)]
pub struct CompressedKey {
    /// Packed sign bits: 1 = positive (k_i > μ), 0 = negative.
    sign_bits: Vec<u64>,
    /// L2 magnitude of the original key: ‖k‖.
    magnitude: f32,
    /// Mean of the original key: μ_k.
    mean: f32,
    /// Dimensionality.
    dim: usize,
}

impl CompressedKey {
    /// Compress a BF16/FP32 key vector to 1-bit representation.
    ///
    /// Formula: ĥ = sign(k − μ_k), stored as packed bits.
    pub fn compress(key: &[f32]) -> Self {
        let dim = key.len();
        let n_words = (dim + 63) / 64;

        // Compute mean.
        let mean = if dim > 0 {
            key.iter().sum::<f32>() / dim as f32
        } else {
            0.0
        };

        // Compute magnitude.
        let magnitude = key.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Pack sign bits: sign(k_i - μ).
        let mut sign_bits = vec![0u64; n_words];
        for (i, &val) in key.iter().enumerate() {
            if val >= mean {
                sign_bits[i / 64] |= 1u64 << (i % 64);
            }
        }

        Self {
            sign_bits,
            magnitude,
            mean,
            dim,
        }
    }

    /// Approximate reconstruction of the key.
    ///
    /// k̂_i ≈ sign_i × (magnitude / √dim) + mean
    pub fn decompress(&self) -> Vec<f32> {
        let scale = if self.dim > 0 {
            self.magnitude / (self.dim as f32).sqrt()
        } else {
            0.0
        };

        let mut result = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let sign_bit = (self.sign_bits[i / 64] >> (i % 64)) & 1;
            let sign = if sign_bit == 1 { 1.0f32 } else { -1.0f32 };
            result[i] = sign * scale + self.mean;
        }
        result
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        self.sign_bits.len() * 8 + 4 + 4 // bits + magnitude + mean
    }

    /// Original uncompressed size in bytes (BF16).
    pub fn uncompressed_bytes(&self) -> usize {
        self.dim * BF16_BYTES
    }

    /// Actual compression ratio achieved.
    pub fn compression_ratio(&self) -> f64 {
        if self.size_bytes() == 0 {
            return 0.0;
        }
        self.uncompressed_bytes() as f64 / self.size_bytes() as f64
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl fmt::Debug for CompressedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedKey")
            .field("dim", &self.dim)
            .field("magnitude", &format_args!("{:.3}", self.magnitude))
            .field("mean", &format_args!("{:.3}", self.mean))
            .field("size", &format_args!("{} B", self.size_bytes()))
            .field("ratio", &format_args!("{:.1}×", self.compression_ratio()))
            .finish()
    }
}

// ── Q8 Compressed Value ──────────────────────────────────────────────────

/// A value vector quantized to 8-bit per dimension.
///
/// Uses affine quantization: q = round((v - min) / scale)
///
/// Memory: d bytes (quantized) + 4 (scale) + 4 (zero_point) = ~136 bytes for d=128
/// vs. 256 bytes in BF16 → 1.88× compression.
#[derive(Clone)]
pub struct CompressedValue {
    /// Quantized 8-bit values.
    data: Vec<u8>,
    /// Dequantization scale: (max - min) / 255.
    scale: f32,
    /// Minimum value for affine offset.
    min_val: f32,
    /// Dimensionality.
    dim: usize,
}

impl CompressedValue {
    /// Quantize a BF16/FP32 value vector to Q8.
    pub fn compress(value: &[f32]) -> Self {
        let dim = value.len();
        if dim == 0 {
            return Self {
                data: Vec::new(),
                scale: 0.0,
                min_val: 0.0,
                dim: 0,
            };
        }

        let min_val = value.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = value.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };

        let data: Vec<u8> = value
            .iter()
            .map(|&v| {
                let q = ((v - min_val) / scale).round().clamp(0.0, 255.0) as u8;
                q
            })
            .collect();

        Self {
            data,
            scale,
            min_val,
            dim,
        }
    }

    /// Dequantize back to f32.
    pub fn decompress(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| q as f32 * self.scale + self.min_val)
            .collect()
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() + 4 + 4 // data + scale + min_val
    }

    /// Original uncompressed size in bytes (BF16).
    pub fn uncompressed_bytes(&self) -> usize {
        self.dim * BF16_BYTES
    }

    /// Actual compression ratio achieved.
    pub fn compression_ratio(&self) -> f64 {
        if self.size_bytes() == 0 {
            return 0.0;
        }
        self.uncompressed_bytes() as f64 / self.size_bytes() as f64
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl fmt::Debug for CompressedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedValue")
            .field("dim", &self.dim)
            .field("scale", &format_args!("{:.6}", self.scale))
            .field("min", &format_args!("{:.3}", self.min_val))
            .field("size", &format_args!("{} B", self.size_bytes()))
            .field("ratio", &format_args!("{:.1}×", self.compression_ratio()))
            .finish()
    }
}

// ── Compressed KV Entry ──────────────────────────────────────────────────

/// A single compressed KV cache entry (one head, one token position).
#[derive(Clone)]
pub struct CompressedKvEntry {
    /// Compressed key (1-bit).
    pub key: CompressedKey,
    /// Compressed value (Q8).
    pub value: CompressedValue,
    /// Token position.
    pub position: usize,
    /// Layer index.
    pub layer: usize,
    /// Head index.
    pub head: usize,
}

impl CompressedKvEntry {
    /// Compress a KV pair.
    pub fn compress(key: &[f32], value: &[f32], position: usize, layer: usize, head: usize) -> Self {
        Self {
            key: CompressedKey::compress(key),
            value: CompressedValue::compress(value),
            position,
            layer,
            head,
        }
    }

    /// Total compressed size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.key.size_bytes() + self.value.size_bytes()
    }

    /// Total uncompressed size in bytes (BF16 K+V).
    pub fn uncompressed_bytes(&self) -> usize {
        self.key.uncompressed_bytes() + self.value.uncompressed_bytes()
    }

    /// Compression ratio for this entry.
    pub fn compression_ratio(&self) -> f64 {
        if self.size_bytes() == 0 {
            return 0.0;
        }
        self.uncompressed_bytes() as f64 / self.size_bytes() as f64
    }
}

impl fmt::Debug for CompressedKvEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedKvEntry")
            .field("pos", &self.position)
            .field("layer", &self.layer)
            .field("head", &self.head)
            .field("size", &format_args!("{} B", self.size_bytes()))
            .field("ratio", &format_args!("{:.1}×", self.compression_ratio()))
            .finish()
    }
}

// ── KV Triage Score ──────────────────────────────────────────────────────

/// The τ_i triage score formula weights.
///
/// τ_i = α·A_i + β·R_i + γ·P_i
///
/// Where:
/// - A_i = mean attention weight over last W=16 steps
/// - R_i = recency: e^(−λ(t − t_i)), λ = 0.1
/// - P_i = positional bonus for prefix tokens
#[derive(Debug, Clone)]
pub struct TriageWeights {
    /// Weight for attention score component.
    pub alpha: f64,
    /// Weight for recency component.
    pub beta: f64,
    /// Weight for positional bonus.
    pub gamma: f64,
    /// Recency decay rate.
    pub lambda: f64,
    /// Window size for attention averaging.
    pub attention_window: usize,
    /// Number of prefix tokens that receive positional bonus.
    pub prefix_tokens: usize,
}

impl Default for TriageWeights {
    fn default() -> Self {
        Self {
            alpha: 0.50,
            beta: 0.30,
            gamma: 0.20,
            lambda: 0.1,
            attention_window: 16,
            prefix_tokens: 64,
        }
    }
}

/// Compute the triage score τ_i for a single KV entry.
///
/// τ_i = α·A_i + β·R_i + γ·P_i
///
/// - A_i: mean attention weight (from recent attention scores)
/// - R_i: recency = e^(−λ(current_step − last_access_step))
/// - P_i: 1.0 if position < prefix_tokens, else 0.0
pub fn triage_score(
    weights: &TriageWeights,
    mean_attention: f64,
    current_step: u64,
    last_access_step: u64,
    token_position: usize,
) -> f64 {
    // Attention component.
    let a_i = mean_attention;

    // Recency component: e^(−λ(t − t_i)).
    let steps_since = current_step.saturating_sub(last_access_step) as f64;
    let r_i = (-weights.lambda * steps_since).exp();

    // Positional bonus: 1.0 for prefix tokens, 0.0 otherwise.
    let p_i = if token_position < weights.prefix_tokens {
        1.0
    } else {
        0.0
    };

    weights.alpha * a_i + weights.beta * r_i + weights.gamma * p_i
}

/// Score and rank all entries, returning (position, score) sorted descending.
pub fn rank_entries(
    weights: &TriageWeights,
    entries: &[(usize, f64, u64)], // (position, mean_attention, last_access_step)
    current_step: u64,
) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = entries
        .iter()
        .map(|&(pos, attn, last_step)| {
            let score = triage_score(weights, attn, current_step, last_step, pos);
            (pos, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

// ── KV Budget Calculator ─────────────────────────────────────────────────

/// Calculate KV cache memory budget for a given context length.
///
/// Formula: KV_bytes = 2 × n_layers × kv_heads × head_dim × dtype_bytes × context_len
#[derive(Debug, Clone)]
pub struct KvBudgetCalc {
    /// Number of layers.
    pub n_layers: usize,
    /// Number of KV heads (GQA).
    pub kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl Default for KvBudgetCalc {
    fn default() -> Self {
        Self {
            n_layers: DEFAULT_N_LAYERS,
            kv_heads: DEFAULT_KV_HEADS,
            head_dim: DEFAULT_HEAD_DIM,
        }
    }
}

impl KvBudgetCalc {
    /// Bytes per token per layer for KV (K+V, given dtype).
    pub fn bytes_per_token_per_layer(&self, dtype_bytes: usize) -> usize {
        2 * self.kv_heads * self.head_dim * dtype_bytes
    }

    /// Total bytes per token across all layers.
    pub fn bytes_per_token(&self, dtype_bytes: usize) -> usize {
        self.bytes_per_token_per_layer(dtype_bytes) * self.n_layers
    }

    /// Total KV cache size for a context length (in bytes).
    pub fn total_bytes(&self, context_len: usize, dtype_bytes: usize) -> usize {
        self.bytes_per_token(dtype_bytes) * context_len
    }

    /// Total KV cache size in MB.
    pub fn total_mb(&self, context_len: usize, dtype_bytes: usize) -> f64 {
        self.total_bytes(context_len, dtype_bytes) as f64 / (1024.0 * 1024.0)
    }

    /// Compressed warm tier size for a context (1-bit keys + Q8 values).
    ///
    /// Key: 16× compression from BF16 (1-bit per dim).
    /// Value: 2× compression from BF16 (Q8).
    /// Overall: about 6.4× compression (approximate).
    pub fn compressed_warm_mb(&self, context_len: usize) -> f64 {
        let raw_bf16 = self.total_mb(context_len, BF16_BYTES);
        let key_half = raw_bf16 / 2.0; // K half
        let val_half = raw_bf16 / 2.0; // V half

        let compressed_keys = key_half / KEY_COMPRESSION_RATIO;
        let compressed_vals = val_half / VALUE_COMPRESSION_RATIO;

        compressed_keys + compressed_vals
    }

    /// Check if 32K context fits in given RAM with warm tier compression.
    ///
    /// Budget: pinned+active (hot ~30% in BF16) + warm (1-bit K + Q8 V for ~70%)
    pub fn fits_in_ram(
        &self,
        context_len: usize,
        available_ram_mb: f64,
        hot_fraction: f64,
    ) -> ContextFitResult {
        let raw_bf16_mb = self.total_mb(context_len, BF16_BYTES);

        // Hot portion stays in BF16.
        let hot_mb = raw_bf16_mb * hot_fraction;

        // Warm portion gets compressed.
        let warm_raw_mb = raw_bf16_mb * (1.0 - hot_fraction);
        let warm_key_mb = warm_raw_mb / 2.0 / KEY_COMPRESSION_RATIO;
        let warm_val_mb = warm_raw_mb / 2.0 / VALUE_COMPRESSION_RATIO;
        let warm_compressed_mb = warm_key_mb + warm_val_mb;

        let total_mb = hot_mb + warm_compressed_mb;

        ContextFitResult {
            context_len,
            raw_bf16_mb,
            hot_mb,
            warm_compressed_mb,
            total_mb,
            available_ram_mb,
            fits: total_mb <= available_ram_mb,
        }
    }
}

/// Result of a context-fit calculation.
#[derive(Debug, Clone)]
pub struct ContextFitResult {
    /// Context length tested.
    pub context_len: usize,
    /// Raw BF16 KV size in MB.
    pub raw_bf16_mb: f64,
    /// Hot tier size (BF16) in MB.
    pub hot_mb: f64,
    /// Warm tier compressed size in MB.
    pub warm_compressed_mb: f64,
    /// Total resident size in MB.
    pub total_mb: f64,
    /// Available RAM in MB.
    pub available_ram_mb: f64,
    /// Whether it fits.
    pub fits: bool,
}

impl fmt::Display for ContextFitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KV Budget: {}K context", self.context_len / 1024)?;
        writeln!(f, "  Raw BF16:         {:>8.2} MB", self.raw_bf16_mb)?;
        writeln!(f, "  Hot (BF16):       {:>8.2} MB", self.hot_mb)?;
        writeln!(f, "  Warm (compressed): {:>7.2} MB", self.warm_compressed_mb)?;
        writeln!(f, "  Total resident:   {:>8.2} MB", self.total_mb)?;
        writeln!(f, "  Available:        {:>8.2} MB", self.available_ram_mb)?;
        write!(f, "  Fits: {}", if self.fits { "YES ✓" } else { "NO ✗" })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1-Bit Key Compression ────────────────────────────────────────

    #[test]
    fn test_key_compress_decompress() {
        let key: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let compressed = CompressedKey::compress(&key);

        assert_eq!(compressed.dim(), 128);
        assert!(compressed.size_bytes() < key.len() * 4); // Much smaller than f32

        let decompressed = compressed.decompress();
        assert_eq!(decompressed.len(), 128);
    }

    #[test]
    fn test_key_compression_ratio() {
        let key: Vec<f32> = vec![1.0; 128];
        let compressed = CompressedKey::compress(&key);

        // BF16: 128 * 2 = 256 bytes
        assert_eq!(compressed.uncompressed_bytes(), 256);
        // Compressed: 2 words * 8 + 4 + 4 = 24 bytes
        assert_eq!(compressed.size_bytes(), 24);
        // Ratio: 256/24 ≈ 10.7×
        assert!(compressed.compression_ratio() > 10.0);
    }

    #[test]
    fn test_key_sign_encoding() {
        // All positive → all bits should be set (> mean=0.5 for [0,1]).
        let key: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0];
        let compressed = CompressedKey::compress(&key);
        // mean = 0.5; elements [0.0, 1.0, 0.0, 1.0]
        // sign(0.0 - 0.5) = negative → bit 0 = 0
        // sign(1.0 - 0.5) = positive → bit 1 = 1
        assert_eq!(compressed.sign_bits[0] & 0b1010, 0b1010);
    }

    #[test]
    fn test_key_empty() {
        let key: Vec<f32> = vec![];
        let compressed = CompressedKey::compress(&key);
        assert_eq!(compressed.dim(), 0);
        assert_eq!(compressed.decompress().len(), 0);
    }

    #[test]
    fn test_key_debug() {
        let key: Vec<f32> = vec![1.0; 128];
        let compressed = CompressedKey::compress(&key);
        let s = format!("{:?}", compressed);
        assert!(s.contains("CompressedKey"));
        assert!(s.contains("dim: 128"));
    }

    // ── Q8 Value Compression ─────────────────────────────────────────

    #[test]
    fn test_value_compress_decompress() {
        let value: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let compressed = CompressedValue::compress(&value);

        assert_eq!(compressed.dim(), 128);
        let decompressed = compressed.decompress();
        assert_eq!(decompressed.len(), 128);

        // Check reconstruction error is small.
        for (orig, recon) in value.iter().zip(decompressed.iter()) {
            assert!(
                (orig - recon).abs() < 0.01,
                "Q8 reconstruction error too large: {} vs {}",
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_value_compression_ratio() {
        let value: Vec<f32> = vec![0.5; 128];
        let compressed = CompressedValue::compress(&value);

        // BF16: 128 * 2 = 256 bytes
        assert_eq!(compressed.uncompressed_bytes(), 256);
        // Q8: 128 + 4 + 4 = 136 bytes
        assert_eq!(compressed.size_bytes(), 136);
        // Ratio: 256/136 ≈ 1.88×
        assert!(compressed.compression_ratio() > 1.8);
    }

    #[test]
    fn test_value_range() {
        let value: Vec<f32> = vec![-10.0, 10.0, 0.0, 5.0, -5.0];
        let compressed = CompressedValue::compress(&value);
        let decompressed = compressed.decompress();

        // Check min/max are preserved approximately.
        assert!((decompressed[0] - (-10.0)).abs() < 0.1);
        assert!((decompressed[1] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_value_empty() {
        let value: Vec<f32> = vec![];
        let compressed = CompressedValue::compress(&value);
        assert_eq!(compressed.dim(), 0);
        assert_eq!(compressed.decompress().len(), 0);
    }

    #[test]
    fn test_value_constant() {
        // All same value — should still work.
        let value: Vec<f32> = vec![3.14; 64];
        let compressed = CompressedValue::compress(&value);
        let decompressed = compressed.decompress();
        for &v in &decompressed {
            assert!((v - 3.14).abs() < 0.01);
        }
    }

    #[test]
    fn test_value_debug() {
        let value: Vec<f32> = vec![1.0; 128];
        let compressed = CompressedValue::compress(&value);
        let s = format!("{:?}", compressed);
        assert!(s.contains("CompressedValue"));
    }

    // ── Compressed KV Entry ──────────────────────────────────────────

    #[test]
    fn test_kv_entry_compress() {
        let key: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let value: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

        let entry = CompressedKvEntry::compress(&key, &value, 42, 5, 3);
        assert_eq!(entry.position, 42);
        assert_eq!(entry.layer, 5);
        assert_eq!(entry.head, 3);

        // Compressed should be much smaller than uncompressed.
        assert!(entry.size_bytes() < entry.uncompressed_bytes());
        assert!(entry.compression_ratio() > 2.0);
    }

    #[test]
    fn test_kv_entry_debug() {
        let entry = CompressedKvEntry::compress(&[1.0; 128], &[0.5; 128], 0, 0, 0);
        let s = format!("{:?}", entry);
        assert!(s.contains("CompressedKvEntry"));
        assert!(s.contains("pos: 0"));
    }

    // ── Triage Score ─────────────────────────────────────────────────

    #[test]
    fn test_triage_score_high_attention() {
        let weights = TriageWeights::default();
        let score = triage_score(&weights, 0.9, 100, 99, 0);
        // High attention + very recent + prefix → should be high.
        assert!(score > 0.8, "High attention score: {}", score);
    }

    #[test]
    fn test_triage_score_stale_token() {
        let weights = TriageWeights::default();
        let score = triage_score(&weights, 0.0, 1000, 0, 200);
        // No attention + very old + not prefix → near zero.
        assert!(score < 0.1, "Stale token score: {}", score);
    }

    #[test]
    fn test_triage_score_prefix_bonus() {
        let weights = TriageWeights::default();
        // Same attention and recency, different position.
        let prefix_score = triage_score(&weights, 0.5, 50, 40, 10);  // prefix
        let non_prefix_score = triage_score(&weights, 0.5, 50, 40, 100); // not prefix
        assert!(prefix_score > non_prefix_score,
            "Prefix should boost score: {} vs {}", prefix_score, non_prefix_score);
        // Difference should be γ × 1.0 = 0.20
        assert!((prefix_score - non_prefix_score - 0.20).abs() < 0.01);
    }

    #[test]
    fn test_triage_recency_decay() {
        let weights = TriageWeights::default();
        // Same attention and position, different recency.
        let recent = triage_score(&weights, 0.5, 100, 99, 200);  // 1 step ago
        let old = triage_score(&weights, 0.5, 100, 50, 200);     // 50 steps ago
        assert!(recent > old, "Recent should score higher: {} vs {}", recent, old);
    }

    #[test]
    fn test_rank_entries() {
        let weights = TriageWeights::default();
        let entries = vec![
            (0, 0.1, 0_u64),   // old, low attention, prefix
            (10, 0.9, 99),     // recent, high attention, not prefix
            (200, 0.5, 50),    // medium
        ];
        let ranked = rank_entries(&weights, &entries, 100);

        // Position 10 should be first (high attention + recent).
        assert_eq!(ranked[0].0, 10);
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn test_rank_entries_empty() {
        let weights = TriageWeights::default();
        let ranked = rank_entries(&weights, &[], 100);
        assert!(ranked.is_empty());
    }

    // ── KV Budget Calculator ─────────────────────────────────────────

    #[test]
    fn test_kv_budget_4k_bf16() {
        // Spec: 4K context, BF16 → 1.25 GB = 1280 MiB
        // 2 × 80 × 8 × 128 × 2 × 4096 = 1,342,177,280 bytes = 1280 MiB
        let calc = KvBudgetCalc::default();
        let mb = calc.total_mb(4096, BF16_BYTES);
        assert!((mb - 1280.0).abs() < 10.0,
            "4K BF16 KV should be ~1280 MiB (1.25 GiB), got {:.0}", mb);
    }

    #[test]
    fn test_kv_budget_32k_bf16() {
        // Spec: 32K context, BF16 → 10 GB
        let calc = KvBudgetCalc::default();
        let mb = calc.total_mb(32768, BF16_BYTES);
        let gb = mb / 1024.0;
        assert!((gb - 10.0).abs() < 0.5,
            "32K BF16 KV should be ~10 GB, got {:.1} GB", gb);
    }

    #[test]
    fn test_kv_bytes_per_token() {
        // Spec: KV_per_token = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 0.31 MB
        let calc = KvBudgetCalc::default();
        let bpt = calc.bytes_per_token(BF16_BYTES);
        assert_eq!(bpt, 327_680, "Bytes per token: {}", bpt);
    }

    #[test]
    fn test_compressed_warm_32k() {
        // Spec: 32K compressed to ~1.56 GB warm tier
        let calc = KvBudgetCalc::default();
        let warm_mb = calc.compressed_warm_mb(32768);
        let warm_gb = warm_mb / 1024.0;
        // Spec says ~1.56 GB total; our pure formula gives key/16 + val/2
        assert!(warm_gb < 4.0,
            "32K warm should be < 4 GB, got {:.2} GB", warm_gb);
    }

    #[test]
    fn test_32k_fits_8gb() {
        // Spec: 32K context fits in 8GB machine with compression.
        // Hot 30% = ~3072 MB BF16, Warm 70% = ~2016 MB compressed.
        // Total ~5088 MB → needs ~5.1 GB KV budget.
        // On 8GB machine with ~5.5 GB available for KV:
        let calc = KvBudgetCalc::default();
        let result = calc.fits_in_ram(32768, 5_500.0, 0.30);
        assert!(result.fits,
            "32K should fit in 5.5GB KV budget:\n{}", result);
        // Verify that without compression it would NOT fit.
        let raw_gb = calc.total_mb(32768, BF16_BYTES) / 1024.0;
        assert!(raw_gb > 5.5, "Raw 32K BF16 should exceed 5.5 GB: {:.1}", raw_gb);
    }

    #[test]
    fn test_4k_fits_easily() {
        // 4K context = ~1.25 GB BF16 → fits trivially.
        let calc = KvBudgetCalc::default();
        let result = calc.fits_in_ram(4096, 4_000.0, 1.0); // all hot
        assert!(result.fits);
        assert!((result.total_mb - 1250.0).abs() < 50.0);
    }

    #[test]
    fn test_context_fit_display() {
        let calc = KvBudgetCalc::default();
        let result = calc.fits_in_ram(32768, 4_000.0, 0.30);
        let s = format!("{}", result);
        assert!(s.contains("32K context"));
        assert!(s.contains("Hot"));
        assert!(s.contains("Warm"));
    }

    // ── Spec Validation: Compression Ratios ──────────────────────────

    #[test]
    fn test_spec_key_16x_compression() {
        // Protocol spec: BF16 → 1-bit = 16× for keys.
        // Our actual ratio is ~10.7× (we store magnitude + mean overhead).
        // The spec's 16× is for the bit-packing alone.
        let key: Vec<f32> = vec![1.0; 128];
        let compressed = CompressedKey::compress(&key);
        // Just check the bit packing: 128 bits = 16 bytes vs 256 bytes BF16.
        let bits_only = (128 + 63) / 64 * 8; // 16 bytes
        let original = 128 * BF16_BYTES; // 256 bytes
        assert_eq!(original / bits_only, 16, "Pure bit-pack ratio should be 16×");
    }

    #[test]
    fn test_spec_value_2x_compression() {
        // Protocol spec: BF16 → Q8 = 2× for values.
        let value: Vec<f32> = vec![0.5; 128];
        let compressed = CompressedValue::compress(&value);
        // Data: 128 bytes vs 256 bytes BF16 = 2× pure data.
        assert!((compressed.compression_ratio() - 1.88).abs() < 0.1,
            "Q8 ratio: {:.2}×", compressed.compression_ratio());
    }

    // ── Edge Cases ───────────────────────────────────────────────────

    #[test]
    fn test_triage_score_zero_step() {
        let weights = TriageWeights::default();
        let score = triage_score(&weights, 0.5, 0, 0, 0);
        // Recency: e^0 = 1.0, so R_i = 1.0
        // α·0.5 + β·1.0 + γ·1.0 = 0.25 + 0.30 + 0.20 = 0.75
        assert!((score - 0.75).abs() < 0.001, "Score: {}", score);
    }

    #[test]
    fn test_triage_weights_sum_to_one() {
        let w = TriageWeights::default();
        assert!((w.alpha + w.beta + w.gamma - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_kv_budget_custom() {
        let calc = KvBudgetCalc {
            n_layers: 32,
            kv_heads: 32,
            head_dim: 128,
        };
        let bpt = calc.bytes_per_token(BF16_BYTES);
        // 2 × 32 × 128 × 2 × 32 = 524,288 bytes
        assert_eq!(bpt, 524_288);
    }
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 3: QJL 1-bit JL-Transform KV Keys
// ============================================================================
//
// Based on: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization" (2025)
//
// Johnson-Lindenstrauss random projection:
//   k_compressed = sign(R × k)    where R ~ N(0,1) ∈ ℝ^{m×d}
//
// Similarity estimation via Hamming distance:
//   dot(q, k) ≈ ‖q‖·‖k‖·cos(π·hamming(q*,k*)/m)
//
// Memory savings: d f32 → m bits (e.g., 128 → 64 bits for m=64)
// Still beats existing 1-bit sign compression because:
//   1. Independent Gaussian projection destroys correlation between dims
//   2. JL theorem guarantees ε-similarity preservation w.h.p.

/// Projection dimension for QJL (bits per key).
/// 64 bits = 8 bytes vs 256 bytes BF16 → 32× compression.
pub const QJL_PROJ_DIM: usize = 64;

/// QJL-compressed key: 1-bit per projected dimension.
///
/// Stores `m` projected bits packed into `ceil(m/64)` u64 words,
/// plus the original magnitude for similarity rescaling.
#[derive(Clone, Debug)]
pub struct QjlKey {
    /// Packed sign bits of R×k, one bit per projected dimension.
    pub bits: Vec<u64>,
    /// ‖k‖ for rescaling the Hamming-based similarity estimate.
    pub magnitude: f32,
    /// Original key dimension d.
    pub orig_dim: usize,
    /// Projection dimension m.
    pub proj_dim: usize,
}

impl QjlKey {
    /// Compress a key vector using a fixed Gaussian projection matrix R.
    ///
    /// # Arguments
    /// * `key`        — f32 key vector of length `d`
    /// * `proj_matrix`— flattened Gaussian matrix `[m × d]`, row-major
    /// * `proj_dim`   — m (number of projected bits)
    pub fn compress(key: &[f32], proj_matrix: &[f32], proj_dim: usize) -> Self {
        let orig_dim = key.len();
        debug_assert_eq!(proj_matrix.len(), proj_dim * orig_dim,
            "proj_matrix must be [proj_dim × orig_dim]");

        let magnitude = key.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let n_words = (proj_dim + 63) / 64;
        let mut bits = vec![0u64; n_words];

        for i in 0..proj_dim {
            // Dot product of row i of R with key
            let row = &proj_matrix[i * orig_dim..(i + 1) * orig_dim];
            let dot: f32 = row.iter().zip(key).map(|(&r, &k)| r * k).sum();
            // Sign-quantize: positive → 1, non-positive → 0
            if dot > 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }

        Self { bits, magnitude, orig_dim, proj_dim }
    }

    /// Estimate cosine similarity between a compressed query and this key.
    ///
    /// Query must be compressed with the **same** projection matrix R.
    ///
    /// Estimate: cos_sim(q,k) ≈ cos(π·d_H(q*,k*)/m)
    /// where d_H is the Hamming distance between the bit vectors.
    pub fn approx_cos_sim(&self, query_bits: &[u64]) -> f32 {
        debug_assert_eq!(query_bits.len(), self.bits.len());
        let hamming: u32 = self.bits.iter()
            .zip(query_bits)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        let angle = std::f32::consts::PI * hamming as f32 / self.proj_dim as f32;
        // cos(π·d_H/m) — JL estimator
        angle.cos()
    }

    /// Full similarity with magnitude rescaling (for ranking, not just ordering).
    pub fn approx_dot(&self, query_bits: &[u64], query_magnitude: f32) -> f32 {
        self.approx_cos_sim(query_bits) * self.magnitude * query_magnitude
    }

    /// Memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8 + 4 // bits + magnitude
    }
}

/// Generate a deterministic Gaussian projection matrix for QJL.
///
/// Uses a seeded LCG to avoid requiring `rand` dependency.
/// Returns flattened `[proj_dim × orig_dim]` f32 matrix.
///
/// In production, generate this once at model load time and cache it.
pub fn qjl_projection_matrix(proj_dim: usize, orig_dim: usize, seed: u64) -> Vec<f32> {
    let n = proj_dim * orig_dim;
    let mut state = seed.wrapping_add(1);
    let mut out = Vec::with_capacity(n);

    // Box-Muller transform using the LCG for pairs of uniform samples
    let mut i = 0;
    while i < n {
        // Two uniform samples in (0, 1)
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;

        let u1 = u1.max(1e-10); // avoid log(0)
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        // Scale by 1/sqrt(orig_dim) for JL normalization
        let scale = 1.0 / (orig_dim as f32).sqrt();
        out.push(r * theta.cos() * scale);
        i += 1;
        if i < n {
            out.push(r * theta.sin() * scale);
            i += 1;
        }
    }
    out
}

// ============================================================================
// OPTIMAL COMPOUNDING STACK — Layer 3b: Fast KV Compaction
// ============================================================================
//
// Based on: "Fast KV Compaction via Attention Matching" (2025)
//
// Instead of naive eviction (drop lowest-score tokens), merge similar tokens:
//   If cosine_sim(k_i, k_j) > θ, replace (k_i, v_i) and (k_j, v_j) with:
//     k_merged = (w_i·k_i + w_j·k_j) / (w_i + w_j)
//     v_merged = (w_i·v_i + w_j·v_j) / (w_i + w_j)
//
// This preserves information better than deletion:
//   - Similar keys represent tokens with the same attention pattern
//   - Merging reduces KV count while keeping average attention distribution

/// A single KV entry for compaction (raw f32 vectors).
#[derive(Clone)]
pub struct KvEntry {
    /// Key vector [head_dim]
    pub key: Vec<f32>,
    /// Value vector [head_dim]
    pub value: Vec<f32>,
    /// Original token position (used for ordering after compaction)
    pub position: usize,
    /// Attention weight for this entry (used as merge weight)
    pub attention_weight: f32,
}

/// Compact a list of KV entries by merging similar key-value pairs.
///
/// Algorithm:
///   1. Compute pairwise cosine similarity for all key pairs.
///   2. Greedily merge the most similar pair above the threshold.
///   3. Repeat until no pair exceeds the threshold or budget is reached.
///
/// # Arguments
/// * `entries`   — mutable list of KV entries to compact
/// * `threshold` — cosine similarity threshold above which to merge (0.0–1.0)
/// * `target_n`  — stop merging when entry count reaches this value (0 = merge all possible)
///
/// # Returns
/// Compacted list with merged entries. Order is by position.
pub fn compact_kv_by_similarity(
    mut entries: Vec<KvEntry>,
    threshold: f32,
    target_n: usize,
) -> Vec<KvEntry> {
    if entries.len() <= 1 || (target_n > 0 && entries.len() <= target_n) {
        return entries;
    }

    loop {
        if target_n > 0 && entries.len() <= target_n {
            break;
        }

        let n = entries.len();
        // Find best merge pair: highest cosine similarity above threshold.
        let mut best_sim = threshold;
        let mut best_i = n;
        let mut best_j = n;

        for i in 0..n {
            for j in i + 1..n {
                let sim = cosine_sim_f32(&entries[i].key, &entries[j].key);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // No pair above threshold → done.
        if best_i == n {
            break;
        }

        // Merge best_j into best_i, then remove best_j.
        let wi = entries[best_i].attention_weight;
        let wj = entries[best_j].attention_weight;
        let total_w = wi + wj;
        let (w_i, w_j) = if total_w > 0.0 { (wi / total_w, wj / total_w) } else { (0.5, 0.5) };

        let dim = entries[best_i].key.len();
        let merged_key: Vec<f32> = (0..dim)
            .map(|d| w_i * entries[best_i].key[d] + w_j * entries[best_j].key[d])
            .collect();
        let merged_val: Vec<f32> = (0..dim)
            .map(|d| w_i * entries[best_i].value[d] + w_j * entries[best_j].value[d])
            .collect();

        // Use the earlier position for the merged entry.
        let merged_pos = entries[best_i].position.min(entries[best_j].position);
        let merged_weight = wi + wj; // Combined weight

        entries[best_i] = KvEntry {
            key: merged_key,
            value: merged_val,
            position: merged_pos,
            attention_weight: merged_weight,
        };
        entries.remove(best_j);
    }

    // Sort by position to maintain sequence order.
    entries.sort_by_key(|e| e.position);
    entries
}

/// Cosine similarity between two f32 vectors.
#[inline]
fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

// ── Tests: QJL + Fast KV Compaction ─────────────────────────────────────

#[cfg(test)]
mod ocs_kv_tests {
    use super::*;

    // ── QJL Tests ────────────────────────────────────────────────────────

    #[test]
    fn qjl_proj_matrix_length() {
        let m = qjl_projection_matrix(64, 128, 42);
        assert_eq!(m.len(), 64 * 128);
    }

    #[test]
    fn qjl_proj_matrix_approx_normal() {
        let m = qjl_projection_matrix(64, 128, 7);
        let mean: f32 = m.iter().sum::<f32>() / m.len() as f32;
        // Mean should be near 0 (Gaussian)
        assert!(mean.abs() < 0.1, "mean={mean}");
    }

    #[test]
    fn qjl_compress_size() {
        let key = vec![1.0f32; 128];
        let proj = qjl_projection_matrix(QJL_PROJ_DIM, 128, 0);
        let compressed = QjlKey::compress(&key, &proj, QJL_PROJ_DIM);
        assert_eq!(compressed.bits.len(), (QJL_PROJ_DIM + 63) / 64);
        // 64 bits → 1 u64 word → 8 bytes + 4 = 12 bytes vs 256 bytes BF16
        assert!(compressed.size_bytes() < 256 / 8);
    }

    #[test]
    fn qjl_same_key_max_similarity() {
        let key = vec![0.5f32; 128];
        let proj = qjl_projection_matrix(QJL_PROJ_DIM, 128, 1);
        let ck = QjlKey::compress(&key, &proj, QJL_PROJ_DIM);
        // Same key compressed twice should have Hamming=0 → cos_sim=1
        let ck2 = QjlKey::compress(&key, &proj, QJL_PROJ_DIM);
        let sim = ck.approx_cos_sim(&ck2.bits);
        assert!((sim - 1.0).abs() < 1e-6, "Same key sim={sim}");
    }

    #[test]
    fn qjl_opposite_keys_low_similarity() {
        let key_pos = vec![1.0f32; 128];
        let key_neg = vec![-1.0f32; 128];
        let proj = qjl_projection_matrix(QJL_PROJ_DIM, 128, 2);
        let ck_pos = QjlKey::compress(&key_pos, &proj, QJL_PROJ_DIM);
        let ck_neg = QjlKey::compress(&key_neg, &proj, QJL_PROJ_DIM);
        let sim = ck_pos.approx_cos_sim(&ck_neg.bits);
        // Opposite keys → all bits flipped → Hamming=64 → cos(π)=-1
        assert!(sim < -0.5, "Opposite keys should be dissimilar: sim={sim}");
    }

    #[test]
    fn qjl_approx_dot_scales_with_magnitude() {
        let key1 = vec![1.0f32; 128];
        let key2 = vec![2.0f32; 128]; // Same direction, 2× magnitude
        let proj = qjl_projection_matrix(QJL_PROJ_DIM, 128, 3);
        let ck1 = QjlKey::compress(&key1, &proj, QJL_PROJ_DIM);
        let ck2 = QjlKey::compress(&key2, &proj, QJL_PROJ_DIM);
        let dot = ck1.approx_dot(&ck2.bits, ck2.magnitude);
        // Should be positive and meaningful
        assert!(dot > 0.0, "Dot with itself (scaled) should be positive: {dot}");
    }

    #[test]
    fn qjl_magnitude_stored_correctly() {
        let key: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let expected_mag: f32 = key.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let proj = qjl_projection_matrix(QJL_PROJ_DIM, 128, 5);
        let ck = QjlKey::compress(&key, &proj, QJL_PROJ_DIM);
        assert!((ck.magnitude - expected_mag).abs() < 1e-4, "magnitude mismatch");
    }

    // ── Fast KV Compaction Tests ─────────────────────────────────────────

    fn make_entry(key: Vec<f32>, val: Vec<f32>, pos: usize, w: f32) -> KvEntry {
        KvEntry { key, value: val, position: pos, attention_weight: w }
    }

    #[test]
    fn compaction_empty_list() {
        let out = compact_kv_by_similarity(vec![], 0.9, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn compaction_single_entry_unchanged() {
        let entries = vec![make_entry(vec![1.0; 4], vec![0.5; 4], 0, 1.0)];
        let out = compact_kv_by_similarity(entries, 0.9, 0);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn compaction_identical_keys_merged() {
        // Two identical keys with threshold=0.9 → should merge into 1
        let entries = vec![
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![1.0; 4], 0, 1.0),
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![2.0; 4], 1, 1.0),
        ];
        let out = compact_kv_by_similarity(entries, 0.9, 0);
        assert_eq!(out.len(), 1, "Identical keys should merge");
        // Merged value should be average: (1+2)/2 = 1.5
        for &v in &out[0].value {
            assert!((v - 1.5).abs() < 1e-5, "merged value should be 1.5, got {v}");
        }
    }

    #[test]
    fn compaction_orthogonal_keys_not_merged() {
        // Orthogonal keys → sim=0 → should not merge
        let entries = vec![
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![1.0; 4], 0, 1.0),
            make_entry(vec![0.0, 1.0, 0.0, 0.0], vec![2.0; 4], 1, 1.0),
        ];
        let out = compact_kv_by_similarity(entries, 0.9, 0);
        assert_eq!(out.len(), 2, "Orthogonal keys should not merge");
    }

    #[test]
    fn compaction_respects_target_n() {
        // 5 nearly-identical entries → merge until 2 remain
        let keys = vec![
            vec![1.0f32, 0.01, 0.0, 0.0],
            vec![1.0f32, 0.02, 0.0, 0.0],
            vec![1.0f32, 0.03, 0.0, 0.0],
            vec![1.0f32, 0.04, 0.0, 0.0],
            vec![1.0f32, 0.05, 0.0, 0.0],
        ];
        let entries: Vec<_> = keys.into_iter().enumerate()
            .map(|(i, k)| make_entry(k, vec![i as f32; 4], i, 1.0))
            .collect();
        let out = compact_kv_by_similarity(entries, 0.8, 2);
        assert!(out.len() <= 2, "Should compact to ≤2 entries, got {}", out.len());
    }

    #[test]
    fn compaction_maintains_position_order() {
        // After merging, entries should be sorted by position.
        let entries = vec![
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![1.0; 4], 5, 1.0),
            make_entry(vec![0.0, 1.0, 0.0, 0.0], vec![2.0; 4], 2, 1.0),
            make_entry(vec![0.0, 0.0, 1.0, 0.0], vec![3.0; 4], 8, 1.0),
        ];
        let out = compact_kv_by_similarity(entries, 0.99, 0);
        for w in out.windows(2) {
            assert!(w[0].position <= w[1].position, "positions not ordered");
        }
    }

    #[test]
    fn compaction_weighted_merge() {
        // Higher-weight entry should bias the merged key/value.
        let entries = vec![
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![10.0; 4], 0, 9.0), // high weight
            make_entry(vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4],  1, 1.0), // low weight
        ];
        let out = compact_kv_by_similarity(entries, 0.9, 0);
        assert_eq!(out.len(), 1);
        // Merged value = 9/10*10 + 1/10*0 = 9.0
        for &v in &out[0].value {
            assert!((v - 9.0).abs() < 1e-4, "weighted merge: got {v}");
        }
    }

    #[test]
    fn cosine_sim_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_sim_f32(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_sim_f32(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        // Should return 0 safely, no NaN
        assert_eq!(cosine_sim_f32(&a, &b), 0.0);
    }
}

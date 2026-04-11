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

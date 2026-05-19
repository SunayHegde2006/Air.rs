//! Dual RoPE (p-RoPE) — Per-Layer Positional Encoding for Gemma 4 (v0.10.0)
//!
//! Gemma 4 applies *proportional* RoPE, where local sliding-window layers and
//! global full-attention layers use **separate** base frequencies:
//!
//!   θ_local  — smaller base (high-frequency), applied to SW layers
//!   θ_global — larger base (low-frequency),  applied to global layers
//!
//! Reference:
//!   Gemma 4 Technical Report (Google DeepMind, 2025)
//!   GGUF metadata keys:
//!     `gemma4.attention.local_rope_theta`   (default: 10000.0)
//!     `gemma4.attention.global_rope_theta`  (default: 1000000.0)
//!
//! # Design
//! - `DualRopeCache` pre-computes inverse frequency vectors for both θ values
//! - `apply_rope_dual` dispatches the correct frequencies per layer-type
//! - Pre-computed tables are shared across all layers of the same type
//! - No allocation after construction; all buffers are pre-sized to `max_seq`
//!
//! # Relationship to existing `ops.rs` RoPE
//! The existing `apply_rope_embedding` takes a single `theta` scalar.
//! `DualRopeCache` wraps two instances and selects based on `AttentionBackend`.

use crate::attention_backend::AttentionBackend;

// ---------------------------------------------------------------------------
// RoPE frequency table
// ---------------------------------------------------------------------------

/// Pre-computed inverse frequency vector: `inv_freq[i] = 1 / θ^{2i/d}`.
///
/// Stored for `d/2` dimensions (half-dimension because RoPE pairs dims).
#[derive(Debug, Clone)]
pub struct RopeFreqTable {
    /// inv_freq[i] = 1 / theta^(2i/d)
    pub inv_freq: Vec<f64>,
    /// Base theta value used to compute this table.
    pub theta: f64,
    /// Head dimension this table was built for.
    pub head_dim: usize,
}

impl RopeFreqTable {
    /// Build a frequency table for given `theta` and `head_dim`.
    ///
    /// `head_dim` must be even (standard transformer constraint).
    pub fn new(theta: f64, head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        Self { inv_freq, theta, head_dim }
    }

    /// Compute cos/sin tables for positions `0..seq_len`.
    ///
    /// Returns `(cos_table, sin_table)` each of shape `[seq_len × half_dim]`.
    pub fn build_tables(&self, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim / 2;
        let mut cos_t = vec![0.0f32; seq_len * half];
        let mut sin_t = vec![0.0f32; seq_len * half];

        for pos in 0..seq_len {
            for (i, &inv_f) in self.inv_freq.iter().enumerate() {
                let angle = pos as f64 * inv_f;
                cos_t[pos * half + i] = angle.cos() as f32;
                sin_t[pos * half + i] = angle.sin() as f32;
            }
        }
        (cos_t, sin_t)
    }

    /// Apply RoPE to a query/key vector at a given position.
    ///
    /// Rotates pairs (x[2i], x[2i+1]) by the angle for dimension i at `pos`.
    ///
    /// In-place: `x` is modified.
    pub fn apply_inplace(&self, x: &mut [f32], pos: usize) {
        debug_assert_eq!(x.len(), self.head_dim, "vector length must equal head_dim");
        let half = self.head_dim / 2;
        for i in 0..half {
            let angle = (pos as f64 * self.inv_freq[i]) as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i]     = x0 * cos_a - x1 * sin_a;
            x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// DualRopeCache — main entry point
// ---------------------------------------------------------------------------

/// Dual RoPE frequency cache for Gemma 4.
///
/// Holds pre-computed inv_freq tables for both local (sliding-window)
/// and global (full) attention layers.
///
/// # GGUF loading
/// ```ignore
/// let cache = DualRopeCache::from_metadata(
///     metadata["gemma4.attention.local_rope_theta"].as_f64().unwrap_or(10_000.0),
///     metadata["gemma4.attention.global_rope_theta"].as_f64().unwrap_or(1_000_000.0),
///     head_dim,
/// );
/// ```
#[derive(Debug, Clone)]
pub struct DualRopeCache {
    /// RoPE table for `SlidingWindow` layers (local attention).
    pub local:  RopeFreqTable,
    /// RoPE table for `GlobalFull` layers.
    pub global: RopeFreqTable,
    /// Head dimension shared by both tables.
    pub head_dim: usize,
}

impl DualRopeCache {
    /// Construct from explicit theta values.
    ///
    /// Gemma 4 defaults: local=10_000.0, global=1_000_000.0.
    pub fn new(local_theta: f64, global_theta: f64, head_dim: usize) -> Self {
        Self {
            local:    RopeFreqTable::new(local_theta, head_dim),
            global:   RopeFreqTable::new(global_theta, head_dim),
            head_dim,
        }
    }

    /// Construct with Gemma 4 default theta values.
    pub fn gemma4_default(head_dim: usize) -> Self {
        Self::new(10_000.0, 1_000_000.0, head_dim)
    }

    /// Construct by reading GGUF metadata strings.
    ///
    /// Falls back to Gemma 4 defaults if keys are absent.
    pub fn from_metadata(
        metadata: &std::collections::HashMap<String, String>,
        head_dim: usize,
    ) -> Self {
        let local_theta = metadata
            .get("gemma4.attention.local_rope_theta")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(10_000.0);
        let global_theta = metadata
            .get("gemma4.attention.global_rope_theta")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1_000_000.0);
        Self::new(local_theta, global_theta, head_dim)
    }

    /// Select the correct freq table for a given `AttentionBackend`.
    ///
    /// - `SlidingWindow` → local table
    /// - `GlobalFull`    → global table
    /// - `Softmax` / others → global table (safe default for non-Gemma4 layers)
    pub fn table_for(&self, backend: AttentionBackend) -> &RopeFreqTable {
        match backend {
            AttentionBackend::SlidingWindow { .. } => &self.local,
            _ => &self.global,
        }
    }

    /// Apply the correct RoPE in-place for a given backend and position.
    ///
    /// Modifies `x` (a query or key vector of length `head_dim`).
    pub fn apply_inplace(&self, x: &mut [f32], pos: usize, backend: AttentionBackend) {
        self.table_for(backend).apply_inplace(x, pos);
    }

    /// Build cos/sin tables for a sequence length, for both local and global.
    ///
    /// Returns `(local_cos, local_sin, global_cos, global_sin)`,
    /// each `[seq_len × half_dim]`.
    pub fn build_all_tables(&self, seq_len: usize)
        -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let (lc, ls) = self.local.build_tables(seq_len);
        let (gc, gs) = self.global.build_tables(seq_len);
        (lc, ls, gc, gs)
    }

    /// VRAM cost of storing all four tables (bytes, f32).
    pub fn table_bytes(&self, seq_len: usize) -> usize {
        let half = self.head_dim / 2;
        4 * seq_len * half * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Batch RoPE application
// ---------------------------------------------------------------------------

/// Apply dual RoPE to a batch of query/key vectors.
///
/// # Arguments
/// * `qk`      — mutable slice `[n_tokens × n_heads × head_dim]`
/// * `positions` — token positions (absolute or relative)
/// * `n_heads` — number of attention heads
/// * `cache`   — `DualRopeCache` for this model
/// * `backend` — attention backend for this layer (selects local vs global)
pub fn apply_rope_batch(
    qk:        &mut [f32],
    positions: &[usize],
    n_heads:   usize,
    cache:     &DualRopeCache,
    backend:   AttentionBackend,
) {
    let d    = cache.head_dim;
    let freq = cache.table_for(backend);
    let n    = positions.len();

    debug_assert_eq!(qk.len(), n * n_heads * d);

    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            let offset = (t * n_heads + h) * d;
            let x = &mut qk[offset..offset + d];
            freq.apply_inplace(x, pos);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_freq_table_construction() {
        let t = RopeFreqTable::new(10_000.0, 64);
        assert_eq!(t.inv_freq.len(), 32, "half dim = 32");
        // inv_freq[0] = 1/10000^0 = 1.0
        assert!((t.inv_freq[0] - 1.0).abs() < 1e-9);
        // inv_freq[1] = 1/10000^(2/64) = 1/10000^(1/32)
        let expected = 1.0 / 10_000f64.powf(2.0 / 64.0);
        assert!((t.inv_freq[1] - expected).abs() < 1e-12, "inv_freq[1]: {}", t.inv_freq[1]);
    }

    #[test]
    fn test_rope_pos0_is_identity() {
        // At position 0: angle = 0 → cos=1, sin=0 → rotation is identity
        let t = RopeFreqTable::new(10_000.0, 8);
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        t.apply_inplace(&mut x, 0);
        for i in 0..8 {
            assert!((x[i] - orig[i]).abs() < 1e-6, "pos=0 should be identity at dim {i}");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation — it must preserve the L2 norm
        let t = RopeFreqTable::new(10_000.0, 16);
        let mut x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 + 0.5).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        t.apply_inplace(&mut x, 42);
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4,
            "norm must be preserved: before={norm_before}, after={norm_after}");
    }

    #[test]
    fn test_dual_cache_local_vs_global_differ() {
        let cache = DualRopeCache::gemma4_default(16);
        let mut xl: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut xg = xl.clone();
        cache.apply_inplace(&mut xl, 10, AttentionBackend::SlidingWindow { window: 4096 });
        cache.apply_inplace(&mut xg, 10, AttentionBackend::GlobalFull);
        // Local and global use different theta → outputs differ (for pos > 0)
        let differs = xl.iter().zip(&xg).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "local and global RoPE must produce different outputs");
    }

    #[test]
    fn test_dual_cache_from_metadata() {
        let mut meta = std::collections::HashMap::new();
        meta.insert("gemma4.attention.local_rope_theta".into(), "8000".into());
        meta.insert("gemma4.attention.global_rope_theta".into(), "500000".into());
        let cache = DualRopeCache::from_metadata(&meta, 32);
        assert!((cache.local.theta - 8000.0).abs() < 1.0);
        assert!((cache.global.theta - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_dual_cache_default_fallback() {
        let meta = std::collections::HashMap::new(); // empty
        let cache = DualRopeCache::from_metadata(&meta, 64);
        assert!((cache.local.theta - 10_000.0).abs() < 1.0);
        assert!((cache.global.theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_build_tables_shape() {
        let t = RopeFreqTable::new(10_000.0, 32);
        let (cos_t, sin_t) = t.build_tables(128);
        assert_eq!(cos_t.len(), 128 * 16); // seq × half_dim
        assert_eq!(sin_t.len(), 128 * 16);
    }

    #[test]
    fn test_build_tables_pos0_cos1_sin0() {
        let t = RopeFreqTable::new(10_000.0, 8);
        let (cos_t, sin_t) = t.build_tables(4);
        // pos=0: all cos=1, all sin=0
        for i in 0..4 {
            assert!((cos_t[i] - 1.0).abs() < 1e-6, "cos[0,{i}] should be 1");
            assert!((sin_t[i] - 0.0).abs() < 1e-6, "sin[0,{i}] should be 0");
        }
    }

    #[test]
    fn test_apply_rope_batch_shape() {
        let d  = 16;
        let nh = 4;
        let n  = 8;
        let cache = DualRopeCache::gemma4_default(d);
        let mut qk = vec![0.5f32; n * nh * d];
        let positions: Vec<usize> = (0..n).collect();
        apply_rope_batch(&mut qk, &positions, nh, &cache, AttentionBackend::GlobalFull);
        // Should complete without panic — output values already tested in per-table tests
        assert_eq!(qk.len(), n * nh * d);
    }

    #[test]
    fn test_table_bytes() {
        let cache = DualRopeCache::gemma4_default(64);
        // 4 tables × 1024 tokens × 32 half-dims × 4 bytes = 524288
        let b = cache.table_bytes(1024);
        assert_eq!(b, 4 * 1024 * 32 * 4);
    }

    #[test]
    fn test_table_for_dispatch() {
        let cache = DualRopeCache::gemma4_default(16);
        let local_ptr  = cache.table_for(AttentionBackend::SlidingWindow { window: 4096 }) as *const _;
        let global_ptr = cache.table_for(AttentionBackend::GlobalFull) as *const _;
        let softmax_ptr = cache.table_for(AttentionBackend::Softmax) as *const _;
        // SlidingWindow → local; GlobalFull and Softmax → global
        assert_ne!(local_ptr, global_ptr, "local and global should be different tables");
        assert_eq!(global_ptr, softmax_ptr, "Softmax falls back to global table");
    }
}

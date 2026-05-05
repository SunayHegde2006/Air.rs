//! Per-model prefix KV cache with CompressionScheme tagging — issue #3.
//!
//! Caches the KV state produced by a fixed system prompt so subsequent
//! requests for the same model can restore from the snapshot instead of
//! re-encoding the prompt from scratch.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────┐      store()       ┌──────────────────────────┐
//! │  SessionKvCache     │ ─────────────────► │  PrefixKvCache           │
//! │  (live session)     │                    │  model_id → PrefixEntry  │
//! └─────────────────────┘      restore()     └──────────────────────────┘
//!                          ◄─────────────────
//! ```
//!
//! # CompressionScheme
//! Every `PrefixEntry` carries a [`CompressionScheme`] tag describing how the
//! tensors are stored.  This lets consumers decide whether dequantization is
//! needed before the tensors can be used as cached KV.
//!
//! Currently all entries written by [`PrefixKvCache::store`] are tagged
//! `CompressionScheme::F16` (raw candle tensors, no extra compression).
//! Future work (issue #5): store `Q8` or `OneBit` compressed tensors to
//! save RAM; the tag ensures callers know which path to take.
//!
//! # Eviction
//! A simple FIFO policy evicts the oldest model_id when the cache is full.
//! For v0.3.0, most deployments serve 1–4 models so FIFO is sufficient; a
//! full LRU eviction can replace it without changing the public API.

use anyhow::{bail, Result};
use candle_core::Tensor;
use std::collections::{HashMap, VecDeque};

// ── CompressionScheme ─────────────────────────────────────────────────────────

/// Describes the storage encoding of tensors in a [`PrefixEntry`].
///
/// All variants are non-exhaustive (`#[non_exhaustive]`) so new schemes can
/// be added in minor releases without breaking match arms in callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompressionScheme {
    /// Raw FP16 tensors — no compression.  Default for stored entries.
    F16,
    /// BF16 tensors — slightly different numerical range from F16.
    Bf16,
    /// 8-bit asymmetric quantization (Q8_0 style).
    Q8,
    /// 1-bit key compression (M.I.S.T. §2 warm tier sign encoding).
    OneBit,
}

impl CompressionScheme {
    /// Approximate compression ratio vs raw F16 storage.
    pub fn compression_ratio(&self) -> f64 {
        match self {
            Self::F16   => 1.0,
            Self::Bf16  => 1.0,
            Self::Q8    => 2.0,
            Self::OneBit => 16.0,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::F16    => "F16",
            Self::Bf16   => "BF16",
            Self::Q8     => "Q8",
            Self::OneBit => "1-bit",
        }
    }
}

impl std::fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ── PrefixEntry ───────────────────────────────────────────────────────────────

/// A single cached prefix: the KV state produced by a fixed system prompt.
#[derive(Clone)]
pub struct PrefixEntry {
    /// Per-layer KV tensors: `layers[i] = (K_cache, V_cache)`.
    /// `None` when a layer has no cached tokens (empty prefix).
    pub layers: Vec<(Option<Tensor>, Option<Tensor>)>,
    /// How the tensors in this entry are encoded.
    pub scheme: CompressionScheme,
    /// Number of tokens encoded in this prefix (system prompt length).
    pub prefix_len: usize,
    /// Approximate RAM footprint in bytes (rough, for logging).
    pub size_bytes: u64,
}

impl PrefixEntry {
    fn estimate_size(layers: &[(Option<Tensor>, Option<Tensor>)]) -> u64 {
        layers
            .iter()
            .map(|(k, v)| {
                let k_bytes = k
                    .as_ref()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .unwrap_or(0);
                let v_bytes = v
                    .as_ref()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .unwrap_or(0);
                (k_bytes + v_bytes) as u64
            })
            .sum()
    }
}

// ── PrefixKvCache ─────────────────────────────────────────────────────────────

/// Per-model prefix KV cache.
///
/// Stores one [`PrefixEntry`] per model ID, keyed by `model_id: &str`.
/// When full, evicts the oldest entry (FIFO).
///
/// Thread-safety: wrap in `Arc<Mutex<PrefixKvCache>>` for multi-threaded use.
///
/// # Example
/// ```rust
/// use air_rs::prefix_kv::{PrefixKvCache, CompressionScheme};
///
/// let mut cache = PrefixKvCache::new(4);
/// assert!(cache.is_empty());
/// ```
pub struct PrefixKvCache {
    entries: HashMap<String, PrefixEntry>,
    /// Insertion-order queue for FIFO eviction.
    order: VecDeque<String>,
    /// Maximum number of model entries to hold simultaneously.
    max_entries: usize,
}

impl PrefixKvCache {
    /// Create a new prefix cache with the given capacity.
    ///
    /// `max_entries = 0` is treated as unlimited.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
        }
    }

    // ── Write ────────────────────────────────────────────────────────────

    /// Snapshot `n_layers` layers from `cache` and store under `model_id`.
    ///
    /// Calls `cache.load(layer)` for each layer 0..n_layers, clones the
    /// tensors, and stores them under `model_id` with the given `scheme` tag.
    ///
    /// If an entry for `model_id` already exists it is overwritten.
    ///
    /// If the cache is at capacity (`max_entries > 0`), the oldest entry is
    /// evicted before inserting the new one.
    ///
    /// # Errors
    /// Propagates any `candle_core::Error` from `cache.load()`.
    pub fn store(
        &mut self,
        model_id: &str,
        cache: &dyn crate::kv_cache::SessionKvCache,
        n_layers: usize,
        scheme: CompressionScheme,
        prefix_len: usize,
    ) -> Result<()> {
        if prefix_len == 0 {
            bail!("PrefixKvCache::store: prefix_len must be > 0 (nothing to store)");
        }

        // Snapshot all layers
        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let (k, v) = cache
                .load(layer)
                .map_err(|e| anyhow::anyhow!("prefix store: load layer {layer}: {e}"))?;

            // Clone tensors so this entry owns independent storage
            let k_owned = k.map(|t| t.clone());
            let v_owned = v.map(|t| t.clone());
            layers.push((k_owned, v_owned));
        }

        let size_bytes = PrefixEntry::estimate_size(&layers);

        // Evict oldest if at capacity
        if self.max_entries > 0
            && self.entries.len() >= self.max_entries
            && !self.entries.contains_key(model_id)
        {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }

        let entry = PrefixEntry { layers, scheme, prefix_len, size_bytes };

        // Update order: remove old position if overwriting
        self.order.retain(|id| id != model_id);
        self.order.push_back(model_id.to_string());
        self.entries.insert(model_id.to_string(), entry);

        Ok(())
    }

    // ── Read ─────────────────────────────────────────────────────────────

    /// Restore the prefix for `model_id` into `cache`.
    ///
    /// Clears `cache`, then calls `cache.append(layer, k, v)` for every layer
    /// in the stored entry.  Returns `Some(prefix_len)` if found, `None` if
    /// this model_id has no cached prefix.
    ///
    /// # Errors
    /// Propagates `candle_core::Error` from `cache.append()`.
    pub fn restore(
        &self,
        model_id: &str,
        cache: &mut dyn crate::kv_cache::SessionKvCache,
    ) -> Result<Option<usize>> {
        let Some(entry) = self.entries.get(model_id) else {
            return Ok(None);
        };

        cache.clear();

        for (layer, (k, v)) in entry.layers.iter().enumerate() {
            match (k, v) {
                (Some(k), Some(v)) => {
                    cache
                        .append(layer, k, v)
                        .map_err(|e| anyhow::anyhow!("prefix restore: append layer {layer}: {e}"))?;
                }
                _ => {
                    // Layer has no cached data — leave empty
                }
            }
        }

        Ok(Some(entry.prefix_len))
    }

    // ── Query ─────────────────────────────────────────────────────────────

    /// True if a prefix is cached for `model_id`.
    pub fn contains(&self, model_id: &str) -> bool {
        self.entries.contains_key(model_id)
    }

    /// Number of model prefixes currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no prefixes are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the `CompressionScheme` of the stored entry, if present.
    pub fn entry_scheme(&self, model_id: &str) -> Option<CompressionScheme> {
        self.entries.get(model_id).map(|e| e.scheme)
    }

    /// Returns the prefix length (token count) of the stored entry, if present.
    pub fn prefix_len(&self, model_id: &str) -> Option<usize> {
        self.entries.get(model_id).map(|e| e.prefix_len)
    }

    /// Approximate RAM footprint of the stored entry, in bytes.
    pub fn entry_size_bytes(&self, model_id: &str) -> Option<u64> {
        self.entries.get(model_id).map(|e| e.size_bytes)
    }

    // ── Eviction ─────────────────────────────────────────────────────────

    /// Explicitly evict the entry for `model_id`.
    ///
    /// Returns `true` if an entry existed and was removed.
    pub fn evict(&mut self, model_id: &str) -> bool {
        if self.entries.remove(model_id).is_some() {
            self.order.retain(|id| id != model_id);
            true
        } else {
            false
        }
    }

    /// Evict all entries.
    pub fn evict_all(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Human-readable summary for logging.
    pub fn summary(&self) -> String {
        let total_bytes: u64 = self.entries.values().map(|e| e.size_bytes).sum();
        format!(
            "PrefixKvCache: {}/{} entries | {:.2} MiB total",
            self.entries.len(),
            if self.max_entries == 0 { "∞".to_string() } else { self.max_entries.to_string() },
            total_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::MockSessionKvCache;

    // ── CompressionScheme ─────────────────────────────────────────────────

    #[test]
    fn compression_scheme_ratios() {
        assert!((CompressionScheme::F16.compression_ratio() - 1.0).abs() < f64::EPSILON);
        assert!((CompressionScheme::Q8.compression_ratio() - 2.0).abs() < f64::EPSILON);
        assert!((CompressionScheme::OneBit.compression_ratio() - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compression_scheme_display() {
        assert_eq!(CompressionScheme::F16.to_string(), "F16");
        assert_eq!(CompressionScheme::Q8.to_string(), "Q8");
        assert_eq!(CompressionScheme::OneBit.to_string(), "1-bit");
    }

    // ── PrefixKvCache basics ──────────────────────────────────────────────

    #[test]
    fn empty_cache() {
        let cache = PrefixKvCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains("llama-7b"));
    }

    #[test]
    fn store_and_contains() {
        let mut prefix = PrefixKvCache::new(4);
        let mock = MockSessionKvCache::new();
        // Empty mock → no tensors, but prefix_len=1 is still allowed only after append
        // Use store with prefix_len=1; layers will be (None,None) for empty mock
        prefix
            .store("llama-7b", &mock, 2, CompressionScheme::F16, 1)
            .unwrap();
        assert!(prefix.contains("llama-7b"));
        assert_eq!(prefix.len(), 1);
        assert_eq!(prefix.entry_scheme("llama-7b"), Some(CompressionScheme::F16));
        assert_eq!(prefix.prefix_len("llama-7b"), Some(1));
    }

    #[test]
    fn store_zero_prefix_len_errors() {
        let mut prefix = PrefixKvCache::new(4);
        let mock = MockSessionKvCache::new();
        let err = prefix.store("m", &mock, 2, CompressionScheme::F16, 0);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("prefix_len"));
    }

    #[test]
    fn restore_unknown_model_returns_none() {
        let prefix = PrefixKvCache::new(4);
        let mut mock = MockSessionKvCache::new();
        let result = prefix.restore("unknown", &mut mock).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn restore_clears_then_repopulates() {
        let mut prefix = PrefixKvCache::new(4);
        let src = MockSessionKvCache::new();
        prefix
            .store("model-a", &src, 2, CompressionScheme::F16, 8)
            .unwrap();

        let mut dst = MockSessionKvCache::new();
        let prefix_len = prefix.restore("model-a", &mut dst).unwrap();
        assert_eq!(prefix_len, Some(8));
    }

    #[test]
    fn evict_removes_entry() {
        let mut prefix = PrefixKvCache::new(4);
        let mock = MockSessionKvCache::new();
        prefix
            .store("m1", &mock, 1, CompressionScheme::Q8, 4)
            .unwrap();
        assert!(prefix.evict("m1"));
        assert!(!prefix.contains("m1"));
        assert!(!prefix.evict("m1")); // second evict returns false
    }

    #[test]
    fn fifo_eviction_at_capacity() {
        let mut prefix = PrefixKvCache::new(2); // capacity=2
        let mock = MockSessionKvCache::new();
        prefix.store("m1", &mock, 1, CompressionScheme::F16, 1).unwrap();
        prefix.store("m2", &mock, 1, CompressionScheme::F16, 1).unwrap();
        assert!(prefix.contains("m1"));
        assert!(prefix.contains("m2"));

        // Adding m3 should evict m1 (oldest)
        prefix.store("m3", &mock, 1, CompressionScheme::F16, 1).unwrap();
        assert!(!prefix.contains("m1"), "m1 should have been evicted");
        assert!(prefix.contains("m2"));
        assert!(prefix.contains("m3"));
    }

    #[test]
    fn overwrite_does_not_grow_beyond_capacity() {
        let mut prefix = PrefixKvCache::new(2);
        let mock = MockSessionKvCache::new();
        prefix.store("m1", &mock, 1, CompressionScheme::F16, 1).unwrap();
        prefix.store("m2", &mock, 1, CompressionScheme::F16, 1).unwrap();
        // Overwrite m1 — should NOT evict m2
        prefix.store("m1", &mock, 1, CompressionScheme::Q8, 5).unwrap();
        assert!(prefix.contains("m1"));
        assert!(prefix.contains("m2"));
        assert_eq!(prefix.len(), 2);
        assert_eq!(prefix.entry_scheme("m1"), Some(CompressionScheme::Q8));
    }

    #[test]
    fn evict_all_clears_cache() {
        let mut prefix = PrefixKvCache::new(4);
        let mock = MockSessionKvCache::new();
        prefix.store("m1", &mock, 1, CompressionScheme::F16, 1).unwrap();
        prefix.store("m2", &mock, 1, CompressionScheme::F16, 1).unwrap();
        prefix.evict_all();
        assert!(prefix.is_empty());
    }

    #[test]
    fn summary_contains_entry_count() {
        let mut prefix = PrefixKvCache::new(4);
        let mock = MockSessionKvCache::new();
        prefix.store("m1", &mock, 1, CompressionScheme::F16, 1).unwrap();
        let s = prefix.summary();
        assert!(s.contains("1/"), "summary: {s}");
        assert!(s.contains("MiB"), "summary: {s}");
    }
}

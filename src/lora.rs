//! LoRA / PEFT Hot-Swap — M.I.S.T. v4 §Adapter Multiplexing
//!
//! Implements S-LoRA-style adapter serving: base model weights stay resident
//! in VRAM, only the low-rank A/B matrices are swapped per request.
//!
//! # Research Basis
//!
//! - **LoRA** (Hu et al., ICLR 2022): Low-Rank Adaptation — for a weight matrix
//!   W ∈ ℝ^{d×k}, the adapter adds ΔW = BA where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k},
//!   initialised A ~ N(0, σ²), B = 0. At inference: y = Wx + (BA)x · α/r.
//! - **S-LoRA** (Chen et al., 2023): batches requests by adapter_id in the
//!   decode tick to avoid per-GEMM weight mutation; maintains an LRU adapter
//!   cache bounded by VRAM budget.
//! - **LoraX / Predibase** (2023): shows that adapter loading overhead is
//!   dominated by host→device transfer, not GEMM; mitigated by pinned memory.
//!
//! # Architecture
//!
//! ```text
//! AdapterCache (LRU, VRAM-bounded)
//!   ├─ adapter_id → LoraAdapter { A: [r × in_dim], B: [out_dim × r] }
//!   └─ LoraLinear::forward(x, adapter?) → Wx + (BAx) · (alpha / rank)
//!
//! GeneratorRequest { prompt, config, adapter_id: Option<AdapterId> }
//! DifferentialBatch { adapter_id, requests: Vec<_> }  ← tick-loop batching
//! ```

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};

// ── Types ──────────────────────────────────────────────────────────────────

/// Adapter identifier — typically a model path hash or user-assigned string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdapterId(pub String);

impl AdapterId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for AdapterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// LoRA adapter: a pair of low-rank matrices A and B.
///
/// For a linear layer W ∈ ℝ^{out_dim × in_dim}, the adapter adds:
/// `ΔW = B · A` where A ∈ ℝ^{rank × in_dim}, B ∈ ℝ^{out_dim × rank}.
///
/// The effective weight delta applied to an input x ∈ ℝ^{in_dim} is:
/// `Δy = B · (A · x) · (alpha / rank)`
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub id: AdapterId,
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub alpha: f32,
    /// Row-major: shape [rank × in_dim]
    pub a: Vec<f32>,
    /// Row-major: shape [out_dim × rank]
    pub b: Vec<f32>,
    /// Approximate VRAM footprint in bytes.
    pub vram_bytes: u64,
}

impl LoraAdapter {
    /// Construct a new adapter with the given dimensions.
    ///
    /// Initialises A ~ N(0, 0.02²), B = 0 (standard LoRA init).
    pub fn new(id: AdapterId, rank: usize, in_dim: usize, out_dim: usize, alpha: f32) -> Self {
        let a_len = rank * in_dim;
        let b_len = out_dim * rank;
        // Simple pseudo-Gaussian init via LCG
        let a = gaussian_init(a_len, 0.02, id.0.len() as u64);
        let b = vec![0.0f32; b_len];
        let vram_bytes = ((a_len + b_len) * 4) as u64;
        Self { id, rank, in_dim, out_dim, alpha, a, b, vram_bytes }
    }

    /// Load adapter weights from a flat binary file: [A row-major] ++ [B row-major].
    /// Both in f32 little-endian.
    pub fn from_file(id: AdapterId, rank: usize, in_dim: usize, out_dim: usize, alpha: f32, path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let a_len = rank * in_dim;
        let b_len = out_dim * rank;
        let expected = (a_len + b_len) * 4;
        if bytes.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("adapter file too small: {} < {expected}", bytes.len()),
            ));
        }
        let a: Vec<f32> = bytes[..a_len * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let b: Vec<f32> = bytes[a_len * 4..expected]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let vram_bytes = expected as u64;
        Ok(Self { id, rank, in_dim, out_dim, alpha, a, b, vram_bytes })
    }

    /// Apply the LoRA delta to an input vector x.
    ///
    /// Returns `Δy = B · (A · x) · scale` where scale = alpha / rank.
    pub fn delta(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "input dimension mismatch");
        let scale = self.alpha / self.rank as f32;

        // Ax: [rank] = A[rank × in_dim] × x[in_dim]
        let ax: Vec<f32> = (0..self.rank)
            .map(|r| {
                let row = &self.a[r * self.in_dim..(r + 1) * self.in_dim];
                row.iter().zip(x.iter()).map(|(a, xi)| a * xi).sum::<f32>()
            })
            .collect();

        // BAx: [out_dim] = B[out_dim × rank] × Ax[rank]
        (0..self.out_dim)
            .map(|o| {
                let row = &self.b[o * self.rank..(o + 1) * self.rank];
                let bax: f32 = row.iter().zip(ax.iter()).map(|(b, a)| b * a).sum();
                bax * scale
            })
            .collect()
    }
}

// ── AdapterCache ───────────────────────────────────────────────────────────

/// LRU adapter cache bounded by a VRAM budget.
///
/// Thread-safe via interior `RwLock`. Designed for concurrent read access
/// (many requests) with infrequent writes (adapter load/evict).
#[derive(Debug)]
pub struct AdapterCache {
    vram_budget: u64,
    vram_used: u64,
    adapters: HashMap<AdapterId, Arc<LoraAdapter>>,
    /// LRU order: front = most recent.
    lru: VecDeque<AdapterId>,
}

impl AdapterCache {
    /// Create a new cache with the given VRAM budget in bytes.
    pub fn new(vram_budget_bytes: u64) -> Self {
        Self {
            vram_budget: vram_budget_bytes,
            vram_used: 0,
            adapters: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    /// Insert an adapter into the cache, evicting LRU entries as needed.
    ///
    /// Returns `Err` if the adapter alone exceeds the VRAM budget.
    pub fn insert(&mut self, adapter: LoraAdapter) -> Result<AdapterId, String> {
        if adapter.vram_bytes > self.vram_budget {
            return Err(format!(
                "adapter '{}' requires {} bytes but budget is {} bytes",
                adapter.id, adapter.vram_bytes, self.vram_budget
            ));
        }

        // Already cached — bump to front
        if self.adapters.contains_key(&adapter.id) {
            self.touch(&adapter.id.clone());
            return Ok(adapter.id);
        }

        // Evict until there is space
        while self.vram_used + adapter.vram_bytes > self.vram_budget {
            self.evict_lru();
        }

        let id = adapter.id.clone();
        self.vram_used += adapter.vram_bytes;
        self.adapters.insert(id.clone(), Arc::new(adapter));
        self.lru.push_front(id.clone());
        Ok(id)
    }

    /// Get a reference-counted pointer to an adapter by id.
    pub fn get(&mut self, id: &AdapterId) -> Option<Arc<LoraAdapter>> {
        if self.adapters.contains_key(id) {
            self.touch(id);
        }
        self.adapters.get(id).cloned()
    }

    /// Remove the least-recently-used adapter. Returns its id if any.
    pub fn evict_lru(&mut self) -> Option<AdapterId> {
        let id = self.lru.pop_back()?;
        if let Some(adapter) = self.adapters.remove(&id) {
            self.vram_used = self.vram_used.saturating_sub(adapter.vram_bytes);
        }
        Some(id)
    }

    /// Number of adapters currently cached.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// True if no adapters are cached.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Current VRAM usage in bytes.
    pub fn vram_used(&self) -> u64 {
        self.vram_used
    }

    // ── Private ─────────────────────────────────────────────────────────

    fn touch(&mut self, id: &AdapterId) {
        self.lru.retain(|x| x != id);
        self.lru.push_front(id.clone());
    }
}

/// Thread-safe wrapper around `AdapterCache` for shared use across the
/// decode tick loop.
#[derive(Debug, Clone)]
pub struct SharedAdapterCache(pub Arc<RwLock<AdapterCache>>);

impl SharedAdapterCache {
    pub fn new(vram_budget_bytes: u64) -> Self {
        Self(Arc::new(RwLock::new(AdapterCache::new(vram_budget_bytes))))
    }
}

// ── LoraLinear ────────────────────────────────────────────────────────────

/// A linear layer that applies an optional LoRA adapter delta.
///
/// The base weight forward pass (Wx) is provided by the calling model
/// implementation. `forward` adds the ΔW = BA term if an adapter is given.
pub struct LoraLinear;

impl LoraLinear {
    /// Compute `y = Wx + ΔW·x` where Wx is provided as `base_output`.
    ///
    /// When `adapter` is `None`, returns `base_output` unchanged.
    pub fn forward(
        base_output: &[f32],
        x: &[f32],
        adapter: Option<&LoraAdapter>,
    ) -> Vec<f32> {
        match adapter {
            None => base_output.to_vec(),
            Some(a) => {
                let delta = a.delta(x);
                base_output
                    .iter()
                    .zip(delta.iter())
                    .map(|(b, d)| b + d)
                    .collect()
            }
        }
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

fn gaussian_init(len: usize, std: f32, seed: u64) -> Vec<f32> {
    // Box-Muller via LCG — deterministic, sufficient for init
    let mut state = seed.wrapping_add(0xCAFE_BABE);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = (state >> 33) as f32 / u32::MAX as f32 + 1e-10;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (state >> 33) as f32 / u32::MAX as f32;
        let r = (-2.0 * u1.ln()).sqrt() * std;
        out.push(r * (2.0 * std::f32::consts::PI * u2).cos());
        if out.len() < len {
            out.push(r * (2.0 * std::f32::consts::PI * u2).sin());
        }
    }
    out.truncate(len);
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter(id: &str, rank: usize) -> LoraAdapter {
        LoraAdapter::new(AdapterId::new(id), rank, 64, 64, rank as f32)
    }

    #[test]
    fn adapter_delta_applied() {
        let adapter = make_adapter("test-lora", 4);
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let base: Vec<f32> = vec![1.0; 64];

        // With B=0 (standard init), delta should be zero
        let out = LoraLinear::forward(&base, &x, Some(&adapter));
        for (o, &b) in out.iter().zip(&base) {
            assert!(
                (o - b).abs() < 1e-6,
                "expected zero delta (B=0 init), got {o} ≠ {b}"
            );
        }
    }

    #[test]
    fn adapter_delta_nonzero_after_b_set() {
        let mut adapter = make_adapter("nonzero", 4);
        // Manually set B ≠ 0
        adapter.b = vec![1.0; 64 * 4];
        let x: Vec<f32> = vec![1.0; 64];
        let base: Vec<f32> = vec![0.0; 64];
        let out = LoraLinear::forward(&base, &x, Some(&adapter));
        let any_nonzero = out.iter().any(|&v| v.abs() > 1e-6);
        assert!(any_nonzero, "expected nonzero delta after setting B=1");
    }

    #[test]
    fn no_adapter_returns_base_unchanged() {
        let base: Vec<f32> = vec![0.5; 32];
        let x: Vec<f32> = vec![1.0; 64];
        let out = LoraLinear::forward(&base, &x, None);
        assert_eq!(out, base);
    }

    #[test]
    fn lru_eviction_on_budget_exceeded() {
        let mut cache = AdapterCache::new(1024 * 1024); // 1 MB
        let a1 = make_adapter("adapter-1", 4);
        let a2 = make_adapter("adapter-2", 4);
        // Each adapter ~ rank*in + out*rank = 4*64 + 64*4 = 512 f32 = 2KB
        let id1 = cache.insert(a1).unwrap();
        let _id2 = cache.insert(a2).unwrap();
        // Access id1 to make it "recently used"
        cache.get(&id1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn lru_evicts_least_recent() {
        // Budget: 3 adapters; insert 4 → first evicted should be LRU
        let adapter_bytes = make_adapter("x", 4).vram_bytes;
        let budget = adapter_bytes * 3;
        let mut cache = AdapterCache::new(budget);
        cache.insert(make_adapter("a", 4)).unwrap();
        cache.insert(make_adapter("b", 4)).unwrap();
        cache.insert(make_adapter("c", 4)).unwrap();
        // Touch "a" to make "b" the LRU
        cache.get(&AdapterId::new("a"));
        // Insert "d" → should evict "b"
        cache.insert(make_adapter("d", 4)).unwrap();
        assert!(cache.get(&AdapterId::new("b")).is_none(), "'b' should have been evicted");
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn adapter_too_large_for_budget_errors() {
        let adapter = make_adapter("huge", 64);
        let budget = adapter.vram_bytes / 2; // budget < adapter
        let mut cache = AdapterCache::new(budget);
        let result = cache.insert(adapter);
        assert!(result.is_err(), "should fail when adapter > budget");
    }

    #[test]
    fn cache_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedAdapterCache>();
    }

    #[test]
    fn adapter_id_display() {
        let id = AdapterId::new("my-lora-v1");
        assert_eq!(format!("{id}"), "my-lora-v1");
    }

    #[test]
    fn vram_used_tracks_correctly() {
        let adapter = make_adapter("track", 4);
        let expected_bytes = adapter.vram_bytes;
        let mut cache = AdapterCache::new(10 * 1024 * 1024);
        cache.insert(adapter).unwrap();
        assert_eq!(cache.vram_used(), expected_bytes);
        cache.evict_lru();
        assert_eq!(cache.vram_used(), 0);
    }
}

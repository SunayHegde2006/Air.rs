//! ModelMux — interleaved multi-model decode tick loop — issue #4.
//!
//! Runs multiple `InferenceGenerator` instances in a single thread with true
//! interleaved execution: every `tick()` call advances **each active slot by
//! one decode step**, then returns.  The caller drives the tick loop at its
//! own rate (typically in a `tokio::task::spawn_blocking` loop or an async
//! task that yields between ticks).
//!
//! # Design
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  ModelMux                                                        │
//! │                                                                  │
//! │  slot[0]: model_id="llama-7b"  │ slot[1]: model_id="phi-3"     │
//! │  state=Some(ActiveRequest)     │ state=None (idle)              │
//! │                                                                  │
//! │  tick() → advance slot[0] by 1 decode step                      │
//! │         → slot[1] idle, skipped                                  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Token Output
//! Each submitted request passes a `mpsc::Sender<MuxEvent>`. The mux emits:
//! - [`MuxEvent::Token`] for each generated token (raw `u32` id)
//! - [`MuxEvent::Done`] when generation completes (EOS or max_tokens reached)
//! - [`MuxEvent::Error`] on generation failure
//!
//! # Memory
//! Each `MuxSlot` holds a loaded `InferenceGenerator` with its own
//! `SessionKvCache`, so models do not share KV state.  Persistent weights
//! (embedding table, final norm, LM head) are loaded once at slot creation
//! via `WeightStreamer::load_embedding` / `load_output`.

use anyhow::{bail, Result};
use candle_core::Tensor;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::generator::InferenceGenerator;
use crate::vram_guard::{VramBudget, VramCheckResult, VRAM_CAP_FRACTION};
use crate::weight_streamer::WeightStreamer;

// ── Public event type ─────────────────────────────────────────────────────────

/// Events emitted by a `ModelMux` slot for a single generation request.
#[derive(Debug)]
pub enum MuxEvent {
    /// A single token was generated.  Raw vocab id.
    Token(u32),
    /// Generation complete: EOS hit or max_tokens reached.
    Done {
        /// Total tokens generated (excluding prompt).
        generated: usize,
    },
    /// Generation failed with an error message.
    Error(String),
}

// ── CompressionScheme ─────────────────────────────────────────────────────────

/// KV-cache compression scheme tag stored with each prefix block.
///
/// Migration hook for M.I.S.T. v4: existing prefix blocks carry their
/// original scheme tag so the scheduler can detect and reject stale blocks
/// rather than silently reusing incompatible KV state (issue #1 US#8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionScheme {
    /// No compression — raw f16 KV tensors (v0.2.0 format).
    None,
    /// M.I.S.T. v3: QJL quantisation + Q8 value compression (active for v0.3.0).
    MistV3,
    /// M.I.S.T. v4: TriAttention + IsoQuant-Fast + TurboQuant (future v0.4.0).
    MistV4,
    /// AQLM 2-bit KV compression (v0.7.0 QLoRA path).
    Aqlm2Bit,
}

impl Default for CompressionScheme {
    fn default() -> Self {
        Self::MistV3
    }
}

// ── PrefixBlock ───────────────────────────────────────────────────────────────

/// A ref-counted prefix KV block stored in a `ModelPrefixCache`.
#[derive(Debug, Clone)]
pub struct PrefixBlock {
    /// Opaque KV bytes (format described by `scheme`).
    pub kv_data: Vec<u8>,
    pub scheme: CompressionScheme,
    /// Active sessions referencing this block (must be 0 to evict).
    pub ref_count: u32,
    /// Hit count for LFU eviction.
    pub hits: u64,
}

impl PrefixBlock {
    pub fn new(kv_data: Vec<u8>, scheme: CompressionScheme) -> Self {
        Self { kv_data, scheme, ref_count: 0, hits: 0 }
    }
}

// ── ModelPrefixCache ──────────────────────────────────────────────────────────

/// Per-model prefix KV cache keyed by token-chunk fingerprint.
///
/// Keys are `Vec<u32>` chunks of `chunk_size` tokens.  Ref-counted blocks
/// are freed only when `ref_count == 0`.
/// Issue #1 US#4 (TTFT reduction) + US#7 (by-reference sharing) + US#8 (scheme tag).
pub struct ModelPrefixCache {
    pub chunk_size: usize,
    pub active_scheme: CompressionScheme,
    blocks: std::collections::HashMap<Vec<u32>, PrefixBlock>,
    hits: u64,
    misses: u64,
}

impl ModelPrefixCache {
    pub fn new(chunk_size: usize, active_scheme: CompressionScheme) -> Self {
        assert!(chunk_size > 0);
        Self { chunk_size, active_scheme, blocks: Default::default(), hits: 0, misses: 0 }
    }

    /// Insert a chunk. Returns `Err` if scheme mismatches (US#8).
    pub fn insert(
        &mut self,
        chunk: Vec<u32>,
        kv_data: Vec<u8>,
        scheme: CompressionScheme,
    ) -> anyhow::Result<()> {
        if scheme != self.active_scheme {
            anyhow::bail!(
                "PrefixCache: compression scheme mismatch — block has {:?}, cache expects {:?}",
                scheme, self.active_scheme
            );
        }
        self.blocks.entry(chunk).or_insert_with(|| PrefixBlock::new(kv_data, scheme));
        Ok(())
    }

    /// Look up a chunk. Increments hit/miss counter.
    pub fn get_mut(&mut self, chunk: &[u32]) -> Option<&mut PrefixBlock> {
        if let Some(block) = self.blocks.get_mut(chunk) {
            block.hits += 1;
            self.hits += 1;
            Some(block)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Evict all blocks with `ref_count == 0`. Returns count evicted.
    pub fn evict_unreferenced(&mut self) -> usize {
        let before = self.blocks.len();
        self.blocks.retain(|_, b| b.ref_count > 0);
        before - self.blocks.len()
    }

    pub fn len(&self) -> usize { self.blocks.len() }
    pub fn is_empty(&self) -> bool { self.blocks.is_empty() }
    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 { 0.0 } else { self.hits as f64 / t as f64 }
    }
    pub fn stats(&self) -> (u64, u64) { (self.hits, self.misses) }
}

// ── SlotId ────────────────────────────────────────────────────────────────────

/// Opaque index into a `ModelMux` slot list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub usize);


// ── ActiveRequest ─────────────────────────────────────────────────────────────

struct ActiveRequest {
    /// Full token sequence (prompt + generated so far).
    all_tokens: Vec<u32>,
    /// Maximum tokens to generate (excluding prompt).
    max_tokens: usize,
    /// Number of tokens generated so far.
    generated: usize,
    /// Decode step index (0 = first step / prefill).
    step: usize,
    /// EOS token id — generation terminates on this.
    eos_id: u32,
    /// Sender for events to the caller.
    tx: mpsc::Sender<MuxEvent>,
}

// ── MuxSlot ───────────────────────────────────────────────────────────────────

/// A single model slot within a `ModelMux`.
pub struct MuxSlot {
    /// Model identifier (matches `InferenceGenerator` config).
    pub model_id: String,
    /// Loaded inference generator for this model.
    generator: InferenceGenerator,
    /// Persistent weights loaded once at slot creation.
    embedding: Tensor,
    /// Final RMS-norm weight.
    final_norm: Tensor,
    /// Tied or untied LM head (quantized).
    lm_head: candle_core::quantized::QMatMul,
    /// Streaming weight loader — kept alive for `generate_step`.
    streamer: Arc<WeightStreamer>,
    /// Active generation state; `None` when this slot is idle.
    state: Option<ActiveRequest>,
}

impl MuxSlot {
    /// Create a new slot by loading persistent weights from `streamer`.
    ///
    /// Loads embedding table, final norm, and LM head once.  The slot begins
    /// with no active request (`state = None`).
    pub fn new(
        model_id: impl Into<String>,
        generator: InferenceGenerator,
        streamer: Arc<WeightStreamer>,
    ) -> Result<Self> {
        let device = generator.device();
        let embedding = streamer.load_embedding(device)?;
        let (final_norm, lm_head) = streamer.load_output(device)?;
        Ok(Self {
            model_id: model_id.into(),
            generator,
            embedding,
            final_norm,
            lm_head,
            streamer,
            state: None,
        })
    }

    /// True if this slot has an active generation request.
    pub fn is_active(&self) -> bool {
        self.state.is_some()
    }

    /// Submit a generation request to this slot.
    ///
    /// Returns `Err` if the slot is already busy.
    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        eos_id: u32,
        tx: mpsc::Sender<MuxEvent>,
    ) -> Result<()> {
        if self.state.is_some() {
            bail!("slot '{}' is busy — submit rejected", self.model_id);
        }
        self.generator.reset_kv_cache();
        self.state = Some(ActiveRequest {
            all_tokens: prompt_tokens,
            max_tokens,
            generated: 0,
            step: 0,
            eos_id,
            tx,
        });
        Ok(())
    }

    /// Advance this slot by one decode step.
    ///
    /// Returns `true` if the slot is still active after this step, `false` if
    /// generation completed (EOS or max_tokens) or the sender was dropped.
    ///
    /// This method is called by `ModelMux::tick()`.
    fn tick_one(&mut self) -> bool {
        let Some(ref mut req) = self.state else {
            return false;
        };

        let result = self.generator.generate_step(
            req.step,
            &req.all_tokens,
            &self.embedding,
            &self.final_norm,
            &self.lm_head,
            Some(&self.streamer),
            false, // prefill_done: no chunked prefill in mux path for now
        );

        match result {
            Err(e) => {
                let _ = req.tx.try_send(MuxEvent::Error(e.to_string()));
                self.state = None;
                false
            }
            Ok(token_id) => {
                req.all_tokens.push(token_id);
                req.step += 1;
                req.generated += 1;

                let done = token_id == req.eos_id || req.generated >= req.max_tokens;

                if done {
                    let generated = req.generated;
                    let _ = req.tx.try_send(MuxEvent::Token(token_id));
                    let _ = req.tx.try_send(MuxEvent::Done { generated });
                    self.state = None;
                    false
                } else {
                    let _ = req.tx.try_send(MuxEvent::Token(token_id));
                    true
                }
            }
        }
    }
}

// ── ModelMux ──────────────────────────────────────────────────────────────────

/// Dynamic model-load configuration for VRAM-aware loading.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// GGUF / config dimensions for VRAM footprint estimation.
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub context_len: usize,
    /// VRAM cap fraction override (defaults to `VRAM_CAP_FRACTION` = 0.80).
    /// US#9: configurable threshold.
    pub vram_cap_fraction: f64,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            n_layers: 32,
            n_kv_heads: 32,
            head_dim: 128,
            hidden_dim: 4096,
            context_len: 4096,
            vram_cap_fraction: VRAM_CAP_FRACTION,
        }
    }
}

/// Multi-model interleaved decode scheduler.
///
/// Holds up to `max_slots` loaded model slots.  Each `tick()` advances every
/// active slot by one decode step in round-robin order.
///
/// # VRAM-aware loading
/// `load_guarded()` runs a VRAM pre-flight check before adding a slot,
/// returning a typed `VramCheckResult` with exact byte shortfall on failure.
///
/// # Dynamic load/unload (US#10)
/// `unload_slot()` removes an idle slot and frees its per-model prefix cache.
///
/// # Example (synchronous driver loop)
/// ```no_run
/// # use air_rs::model_mux::ModelMux;
/// # let mut mux = ModelMux::new(4);
/// loop {
///     let active = mux.tick();
///     if active == 0 {
///         std::thread::sleep(std::time::Duration::from_millis(1));
///     }
/// }
/// ```
pub struct ModelMux {
    slots: Vec<MuxSlot>,
    max_slots: usize,
    /// Per-slot prefix KV caches (indexed by slot position, not SlotId).
    prefix_caches: Vec<ModelPrefixCache>,
}

impl ModelMux {
    /// Create an empty mux.  `max_slots = 0` means unlimited.
    pub fn new(max_slots: usize) -> Self {
        Self { slots: Vec::new(), max_slots, prefix_caches: Vec::new() }
    }

    /// Add a model slot.  Returns the `SlotId` for future `submit` / `remove` calls.
    ///
    /// Returns `Err` if at capacity.
    pub fn add_slot(&mut self, slot: MuxSlot) -> Result<SlotId> {
        self.add_slot_with_cache(slot, ModelPrefixCache::new(16, CompressionScheme::default()))
    }

    /// Add a slot with a custom prefix cache (chunk size, compression scheme).
    pub fn add_slot_with_cache(
        &mut self,
        slot: MuxSlot,
        cache: ModelPrefixCache,
    ) -> Result<SlotId> {
        if self.max_slots > 0 && self.slots.len() >= self.max_slots {
            bail!(
                "ModelMux at capacity ({} slots); cannot add '{}'",
                self.max_slots,
                slot.model_id
            );
        }
        let id = SlotId(self.slots.len());
        self.slots.push(slot);
        self.prefix_caches.push(cache);
        Ok(id)
    }

    /// VRAM-aware model load (US#2, US#10).
    ///
    /// Runs `VramBudget::check_direct` with configurable `cap_fraction`.
    /// On success: adds the slot and returns its `SlotId`.
    /// On reject: returns `Err` with exact MiB shortfall in the message.
    ///
    /// `total_vram_bytes` and `free_vram_bytes` come from the caller's
    /// VRAM query (nvidia-smi, nvml, or test stub) to keep this fn testable
    /// without a real GPU.
    pub fn load_guarded(
        &mut self,
        slot: MuxSlot,
        cfg: &ModelLoadConfig,
        total_vram_bytes: u64,
        free_vram_bytes: u64,
    ) -> Result<(SlotId, VramCheckResult)> {
        let estimate = VramBudget::estimate_pub(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.hidden_dim,
            cfg.context_len,
        );
        let (vram_result, _budget) = VramBudget::check_direct(
            total_vram_bytes,
            free_vram_bytes,
            estimate,
            cfg.vram_cap_fraction,
        );
        if !vram_result.ok_to_load() {
            bail!("{}", vram_result.summary());
        }
        let cache = ModelPrefixCache::new(16, CompressionScheme::default());
        let id = self.add_slot_with_cache(slot, cache)?;
        Ok((id, vram_result))
    }

    /// Remove an **idle** slot by id, freeing its prefix cache.
    /// Returns `Err` if the slot is active or id is out of range.
    /// US#10: dynamic unload without restart.
    pub fn unload_slot(&mut self, id: SlotId) -> Result<()> {
        let idx = id.0;
        if idx >= self.slots.len() {
            bail!("slot id {} out of range (len={})", idx, self.slots.len());
        }
        if self.slots[idx].is_active() {
            bail!(
                "slot '{}' is active; cancel or await completion before unloading",
                self.slots[idx].model_id
            );
        }
        self.slots.remove(idx);
        self.prefix_caches.remove(idx);
        Ok(())
    }

    /// Remove a slot by id.  Returns `Err` if `id` is out of range or slot is active.
    pub fn remove_slot(&mut self, id: SlotId) -> Result<MuxSlot> {
        let idx = id.0;
        if idx >= self.slots.len() {
            bail!("slot id {} out of range (len={})", idx, self.slots.len());
        }
        if self.slots[idx].is_active() {
            bail!("slot '{}' is active; cancel request before removing", self.slots[idx].model_id);
        }
        self.prefix_caches.remove(idx);
        Ok(self.slots.remove(idx))
    }

    /// Prefix cache for a slot (immutable view).
    pub fn prefix_cache(&self, id: SlotId) -> Option<&ModelPrefixCache> {
        self.prefix_caches.get(id.0)
    }

    /// Prefix cache for a slot (mutable).
    pub fn prefix_cache_mut(&mut self, id: SlotId) -> Option<&mut ModelPrefixCache> {
        self.prefix_caches.get_mut(id.0)
    }

    /// Submit a generation request to a specific slot.
    pub fn submit(
        &mut self,
        id: SlotId,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        eos_id: u32,
        tx: mpsc::Sender<MuxEvent>,
    ) -> Result<()> {
        let idx = id.0;
        if idx >= self.slots.len() {
            bail!("slot id {} out of range", idx);
        }
        self.slots[idx].submit(prompt_tokens, max_tokens, eos_id, tx)
    }

    /// Advance every active slot by exactly one decode step.
    ///
    /// Returns the number of slots still active after this tick.
    pub fn tick(&mut self) -> usize {
        for slot in self.slots.iter_mut() {
            slot.tick_one();
        }
        self.active_count()
    }

    /// Number of currently active (busy) slots.
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_active()).count()
    }

    /// Total number of slots (idle + active).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// True if all slots are idle.
    pub fn is_idle(&self) -> bool {
        self.active_count() == 0
    }

    /// Human-readable status line.
    pub fn status(&self) -> String {
        format!(
            "ModelMux: {}/{} active, {} total slots",
            self.active_count(),
            self.slots.len(),
            if self.max_slots == 0 { "∞".to_string() } else { self.max_slots.to_string() },
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mux_is_idle() {
        let mux = ModelMux::new(4);
        assert!(mux.is_idle());
        assert_eq!(mux.slot_count(), 0);
        assert_eq!(mux.active_count(), 0);
    }

    #[test]
    fn tick_on_empty_mux_returns_zero() {
        let mut mux = ModelMux::new(4);
        assert_eq!(mux.tick(), 0);
    }

    #[test]
    fn status_line_contains_slot_info() {
        let mux = ModelMux::new(4);
        let s = mux.status();
        assert!(s.contains("ModelMux"), "status: {s}");
        assert!(s.contains("0/0"), "status: {s}");
    }

    #[test]
    fn slot_id_is_index() {
        // SlotId(0).0 == 0
        assert_eq!(SlotId(0).0, 0);
        assert_eq!(SlotId(42).0, 42);
    }

    #[test]
    fn mux_event_debug() {
        let t = format!("{:?}", MuxEvent::Token(42));
        assert!(t.contains("42"), "{t}");
        let d = format!("{:?}", MuxEvent::Done { generated: 10 });
        assert!(d.contains("10"), "{d}");
        let e = format!("{:?}", MuxEvent::Error("oops".into()));
        assert!(e.contains("oops"), "{e}");
    }

    // Integration-level tests that require a loaded generator are tagged
    // #[ignore] — run with `cargo test -- --ignored` on a machine with weights.
    #[test]
    #[ignore]
    fn mux_two_slots_interleave() {
        // Not runnable in CI without weights; kept as documentation.
    }

    // ── CompressionScheme ──────────────────────────────────────────────────

    #[test]
    fn compression_scheme_default_is_mist_v3() {
        assert_eq!(CompressionScheme::default(), CompressionScheme::MistV3);
    }

    #[test]
    fn compression_scheme_ne_variants() {
        assert_ne!(CompressionScheme::MistV3, CompressionScheme::MistV4);
        assert_ne!(CompressionScheme::None, CompressionScheme::Aqlm2Bit);
    }

    // ── ModelPrefixCache ───────────────────────────────────────────────────

    fn make_cache() -> ModelPrefixCache {
        ModelPrefixCache::new(4, CompressionScheme::MistV3)
    }

    #[test]
    fn prefix_cache_miss_on_empty() {
        let mut c = make_cache();
        assert!(c.get_mut(&[1u32, 2, 3, 4]).is_none());
        assert_eq!(c.stats(), (0, 1));
    }

    #[test]
    fn prefix_cache_hit_after_insert() {
        let mut c = make_cache();
        let chunk = vec![10u32, 20, 30, 40];
        c.insert(chunk.clone(), vec![0xAB; 32], CompressionScheme::MistV3).unwrap();
        assert!(c.get_mut(&chunk).is_some());
        assert_eq!(c.stats(), (1, 0));
    }

    #[test]
    fn prefix_cache_scheme_mismatch_rejected() {
        // US#8: inserting a v4 block into a v3 cache must fail.
        let mut c = make_cache();
        let result = c.insert(vec![1u32, 2, 3, 4], vec![0; 8], CompressionScheme::MistV4);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("mismatch"), "expected 'mismatch' in error: {msg}");
    }

    #[test]
    fn prefix_cache_evict_unreferenced() {
        let mut c = make_cache();
        let chunk = vec![1u32, 2, 3, 4];
        c.insert(chunk.clone(), vec![0; 8], CompressionScheme::MistV3).unwrap();
        let n = c.evict_unreferenced();
        assert_eq!(n, 1);
        assert!(c.is_empty());
    }

    #[test]
    fn prefix_cache_pinned_block_not_evicted() {
        let mut c = make_cache();
        let chunk = vec![5u32, 6, 7, 8];
        c.insert(chunk.clone(), vec![0; 8], CompressionScheme::MistV3).unwrap();
        if let Some(b) = c.get_mut(&chunk) {
            b.ref_count = 1;
        }
        let n = c.evict_unreferenced();
        assert_eq!(n, 0, "pinned block must not be evicted");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn prefix_cache_hit_rate_50_percent() {
        let mut c = make_cache();
        let chunk = vec![1u32, 2, 3, 4];
        c.insert(chunk.clone(), vec![0; 8], CompressionScheme::MistV3).unwrap();
        let _ = c.get_mut(&chunk);           // hit
        let _ = c.get_mut(&[9u32, 9, 9, 9]); // miss
        assert!((c.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn prefix_cache_hit_increments_block_hits() {
        let mut c = make_cache();
        let chunk = vec![7u32, 8, 9, 10];
        c.insert(chunk.clone(), vec![0; 8], CompressionScheme::MistV3).unwrap();
        c.get_mut(&chunk);
        c.get_mut(&chunk);
        let block = c.get_mut(&chunk).unwrap();
        assert_eq!(block.hits, 3); // 3 total lookups
    }

    // ── VRAM-aware load_guarded / check_direct (US#2, US#9, US#10) ────────

    #[test]
    fn load_guarded_accept_when_vram_sufficient() {
        let total    = 12u64 * 1024 * 1024 * 1024;
        let free     = 10u64 * 1024 * 1024 * 1024;
        let estimate = VramBudget::estimate_pub(2, 2, 64, 256, 512);
        let (res, _) = VramBudget::check_direct(total, free, estimate, 0.80);
        assert_eq!(res, VramCheckResult::Accept);
    }

    #[test]
    fn load_guarded_reject_when_vram_tight() {
        let total    = 100u64 * 1024 * 1024;
        let estimate = total; // 100% > 80% cap
        let (res, _) = VramBudget::check_direct(total, total, estimate, 0.80);
        assert!(!res.ok_to_load());
        let msg = res.summary();
        assert!(msg.contains("rejected"), "msg: {msg}");
        assert!(msg.contains("MiB"), "msg: {msg}");
    }

    #[test]
    fn load_guarded_shortfall_bytes_correct() {
        let total    = 10u64 * 1024 * 1024 * 1024;
        let cap      = (total as f64 * 0.80) as u64;
        let estimate = cap + 500_000_000;
        let (res, _) = VramBudget::check_direct(total, 0, estimate, 0.80);
        match res {
            VramCheckResult::Reject { shortfall_bytes, .. } => {
                assert!(shortfall_bytes > 400_000_000 && shortfall_bytes < 600_000_000);
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn model_mux_prefix_cache_count_matches_slot_count() {
        let mux = ModelMux::new(0);
        assert_eq!(mux.slots.len(), mux.prefix_caches.len());
    }

    #[test]
    fn unload_slot_out_of_range_errors() {
        let mut mux = ModelMux::new(4);
        let result = mux.unload_slot(SlotId(99));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("out of range"));
    }

    #[test]
    fn model_load_cfg_default_has_correct_threshold() {
        let cfg = ModelLoadConfig::default();
        assert!((cfg.vram_cap_fraction - VRAM_CAP_FRACTION).abs() < 1e-9);
        assert!(cfg.context_len > 0);
    }
}

//! Adaptive Resonance Batching (ARB) — production batching scheduler for Air.rs.
//!
//! Replaces traditional FIFO / MSB batching with:
//! - **Harmonic length grouping** — groups sequences so max_len / min_len ≤ H
//! - **Entropy-weighted priority scheduling** — α·U(s) + β·A(s) + γ·W(s)
//! - **Confidence-gated coasting** — pauses high-certainty sequences to save compute
//! - **O(1) KV slot reclamation** — stack-based free-list
//! - **Shadow prefetch queue** — buffers pre-tokenised requests
//!
//! Reference: `ARB_IMPLEMENTATION_PLAN.md`
//!
//! # Scheduler Tick Loop
//!
//! ```text
//! loop {
//!     batch = handle.step()          // admit_waiting + build_micro_batch
//!     results = kernel.forward(batch)
//!     handle.commit(results)         // update state, reclaim slots, trigger coasting
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Adaptive Resonance Batcher.
///
/// See ARB plan §3 for field semantics.
#[derive(Debug, Clone)]
pub struct ArbConfig {
    /// Hard upper bound on sequences per forward pass. Also the KV-slot pool size.
    pub max_batch_size: usize,
    /// Maximum tokens (prompt + generated) per sequence.
    pub max_context_len: usize,
    /// Harmonic invariant H: max_len / min_len ≤ H within any group.
    pub harmonic_ratio: f32,
    /// Weight on utility term U(s) — information density.
    pub priority_alpha: f32,
    /// Weight on urgency term A(s) — caller-supplied priority.
    pub priority_beta: f32,
    /// Weight on wait term W(s) — anti-starvation bonus.
    pub priority_gamma: f32,
    /// Top-1 probability τ above which a decode step is "high confidence".
    pub confidence_exit_threshold: f32,
    /// Consecutive high-confidence steps before coasting.
    pub confidence_exit_k: usize,
    /// Milliseconds a coasting sequence is excluded from batch selection.
    pub coast_deprioritise_ms: u64,
    /// Maximum size of the shadow prefetch queue.
    pub prefetch_queue_depth: usize,
}

impl Default for ArbConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_context_len: 8192,
            harmonic_ratio: 2.0,
            priority_alpha: 0.6,
            priority_beta: 0.3,
            priority_gamma: 0.1,
            confidence_exit_threshold: 0.97,
            confidence_exit_k: 4,
            coast_deprioritise_ms: 50,
            prefetch_queue_depth: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Unique, monotonically increasing sequence identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

/// Lifecycle phase of a sequence.
///
/// Legal transitions:
/// ```text
/// Prefill → Decode       (tokens_remaining reaches 0)
/// Decode  → Coasting     (high_conf_streak >= K)
/// Coasting → Decode      (timer expired)
/// {Prefill, Decode, Coasting} → Finished  (EOS)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum SequencePhase {
    Prefill { tokens_remaining: usize },
    Decode { tokens_generated: usize },
    Coasting { resume_at: Instant },
    Finished,
}

/// Full state for a single in-flight sequence.
#[derive(Debug)]
pub struct SequenceState {
    pub id: SequenceId,
    pub phase: SequencePhase,
    pub prompt_len: usize,
    /// prompt + generated tokens so far.
    pub current_len: usize,
    /// Caller-supplied urgency, clamped to [0.0, 1.0].
    pub urgency: f32,
    pub arrived_at: Instant,
    /// KV-cache slot index (assigned on admission).
    pub kv_slot: Option<usize>,
    /// Rolling window of last 16 top-1 softmax probabilities.
    pub recent_top1_probs: VecDeque<f32>,
    /// Consecutive high-confidence decode steps.
    pub high_conf_streak: usize,
    /// Estimated information utility ∈ [0, 1].
    pub utility_estimate: f32,
    /// Prompt token IDs (stored for prefill batch construction).
    pub prompt_tokens: Vec<u32>,
    /// Last token emitted (decode input for next step).
    pub last_token: Option<u32>,
}

impl SequenceState {
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>, urgency: f32) -> Self {
        let prompt_len = prompt_tokens.len();
        Self {
            id,
            phase: SequencePhase::Prefill {
                tokens_remaining: prompt_len,
            },
            prompt_len,
            current_len: prompt_len,
            urgency: urgency.clamp(0.0, 1.0),
            arrived_at: Instant::now(),
            kv_slot: None,
            recent_top1_probs: VecDeque::with_capacity(16),
            high_conf_streak: 0,
            utility_estimate: Self::initial_utility(prompt_len),
            prompt_tokens,
            last_token: None,
        }
    }

    /// log2(prompt_len) normalised to [0,1] over max_context_len = 8192.
    fn initial_utility(prompt_len: usize) -> f32 {
        (prompt_len.max(1) as f32).log2() / (8192_f32).log2()
    }

    /// Composite priority score: P(s) = α·U(s) + β·A(s) + γ·σ(0.1·wait)
    pub fn priority(&self, cfg: &ArbConfig) -> f32 {
        let wait_secs = self.arrived_at.elapsed().as_secs_f32();
        let w = 1.0 / (1.0 + (-0.1 * wait_secs).exp());
        cfg.priority_alpha * self.utility_estimate
            + cfg.priority_beta * self.urgency
            + cfg.priority_gamma * w
    }

    /// Update state after a decode forward-pass step.
    pub fn record_decode_step(&mut self, top1_prob: f32, cfg: &ArbConfig) {
        // Advance token counter
        if let SequencePhase::Decode {
            ref mut tokens_generated,
        } = self.phase
        {
            *tokens_generated += 1;
        }
        self.current_len += 1;

        // Rolling window of last 16 top-1 probabilities
        if self.recent_top1_probs.len() == 16 {
            self.recent_top1_probs.pop_front();
        }
        self.recent_top1_probs.push_back(top1_prob);

        // Update high-confidence streak
        if top1_prob >= cfg.confidence_exit_threshold {
            self.high_conf_streak += 1;
        } else {
            self.high_conf_streak = 0;
        }

        // Recompute utility: avg binary entropy of top-1 in bits, clamped to [0,1]
        let avg_entropy = self
            .recent_top1_probs
            .iter()
            .map(|&p| if p < 1.0 { -p * p.ln() } else { 0.0 })
            .sum::<f32>()
            / self.recent_top1_probs.len().max(1) as f32;

        self.utility_estimate = (avg_entropy / std::f32::consts::LN_2).clamp(0.0, 1.0);
    }

    /// True if the sequence should enter coasting.
    pub fn should_coast(&self, cfg: &ArbConfig) -> bool {
        matches!(self.phase, SequencePhase::Decode { .. })
            && self.high_conf_streak >= cfg.confidence_exit_k
    }

    /// Transition to coasting phase with a timed exclusion.
    pub fn enter_coast(&mut self, cfg: &ArbConfig) {
        self.phase = SequencePhase::Coasting {
            resume_at: Instant::now() + Duration::from_millis(cfg.coast_deprioritise_ms),
        };
        self.high_conf_streak = 0; // prevent immediate re-coast on resume
    }

    /// True if a coasting sequence's timer has expired.
    pub fn should_resume(&self) -> bool {
        if let SequencePhase::Coasting { resume_at } = self.phase {
            Instant::now() >= resume_at
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// KV Slot Allocator — O(1) stack-based free-list
// ---------------------------------------------------------------------------

/// Pre-allocated pool of KV cache slot indices.
///
/// Invariant: `free_stack.len() + active_sequences == capacity` at all times.
pub struct KvSlotAllocator {
    free_stack: Vec<usize>,
    capacity: usize,
}

impl KvSlotAllocator {
    pub fn new(capacity: usize) -> Self {
        Self {
            free_stack: (0..capacity).rev().collect(),
            capacity,
        }
    }

    #[inline]
    pub fn alloc(&mut self) -> Option<usize> {
        self.free_stack.pop()
    }

    #[inline]
    pub fn free(&mut self, slot: usize) {
        debug_assert!(
            slot < self.capacity,
            "slot {} out of range {}",
            slot,
            self.capacity
        );
        self.free_stack.push(slot);
    }

    #[inline]
    pub fn available(&self) -> usize {
        self.free_stack.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Harmonic Group — length-ratio bounded grouping
// ---------------------------------------------------------------------------

/// A group of sequences satisfying max_len / min_len ≤ H.
#[derive(Debug)]
pub struct HarmonicGroup {
    pub sequences: Vec<SequenceId>,
    pub min_len: usize,
    pub max_len: usize,
}

impl HarmonicGroup {
    pub fn new(first_id: SequenceId, len: usize) -> Self {
        Self {
            sequences: vec![first_id],
            min_len: len,
            max_len: len,
        }
    }

    /// Check if `new_len` can join without violating the harmonic invariant.
    pub fn can_admit(&self, new_len: usize, h: f32) -> bool {
        let candidate_max = self.max_len.max(new_len) as f32;
        let candidate_min = self.min_len.min(new_len) as f32;
        if candidate_min == 0.0 {
            return false;
        }
        (candidate_max / candidate_min) <= h
    }

    /// Admit a sequence that has already passed `can_admit`.
    pub fn admit(&mut self, id: SequenceId, len: usize) {
        self.sequences.push(id);
        if len < self.min_len {
            self.min_len = len;
        }
        if len > self.max_len {
            self.max_len = len;
        }
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MicroBatch — the output consumed by the inference kernel
// ---------------------------------------------------------------------------

/// A batch of entries ready for a single forward pass.
#[derive(Debug)]
pub struct MicroBatch {
    /// Ordered batch entries.
    pub entries: Vec<BatchEntry>,
    /// True iff every entry is in prefill phase.
    pub all_prefill: bool,
    /// True iff every entry is in decode phase.
    pub all_decode: bool,
}

/// A single entry in a micro-batch.
#[derive(Debug, Clone)]
pub struct BatchEntry {
    pub seq_id: SequenceId,
    pub kv_slot: usize,
    /// Prefill: prompt token chunk. Decode: single-element vec with last token.
    pub tokens: Vec<u32>,
    pub phase: BatchEntryPhase,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchEntryPhase {
    Prefill,
    Decode,
}

// ---------------------------------------------------------------------------
// ArbScheduler — the core scheduler
// ---------------------------------------------------------------------------

/// The Adaptive Resonance Batching scheduler.
///
/// Manages sequence lifecycle: enqueue → admit → batch → commit → reclaim.
pub struct ArbScheduler {
    pub(crate) cfg: ArbConfig,
    active: HashMap<SequenceId, SequenceState>,
    waiting: VecDeque<SequenceState>,
    prefetch: VecDeque<SequenceState>,
    allocator: KvSlotAllocator,
    next_id: u64,
}

impl ArbScheduler {
    pub fn new(cfg: ArbConfig) -> Self {
        let cap = cfg.max_batch_size;
        Self {
            cfg,
            active: HashMap::new(),
            waiting: VecDeque::new(),
            prefetch: VecDeque::new(),
            allocator: KvSlotAllocator::new(cap),
            next_id: 0,
        }
    }

    // ── Enqueue ─────────────────────────────────────────────────────────

    /// Submit a new request. Returns a unique SequenceId.
    ///
    /// Routes to the prefetch queue if not full, otherwise directly to waiting.
    pub fn enqueue(&mut self, prompt_tokens: Vec<u32>, urgency: f32) -> SequenceId {
        let id = SequenceId(self.next_id);
        self.next_id += 1;
        let state = SequenceState::new(id, prompt_tokens, urgency);
        if self.prefetch.len() < self.cfg.prefetch_queue_depth {
            self.prefetch.push_back(state);
        } else {
            self.waiting.push_back(state);
        }
        id
    }

    // ── Admit Waiting ───────────────────────────────────────────────────

    /// Drain prefetch into waiting, then admit highest-priority sequences
    /// to active while KV slots are available.
    ///
    /// Called once per scheduler tick, before `build_micro_batch`.
    pub fn admit_waiting(&mut self) {
        // Step 1: drain prefetch into waiting
        while let Some(s) = self.prefetch.pop_front() {
            self.waiting.push_back(s);
        }

        // Step 2: admit highest-priority waiting sequences while slots are free
        while self.allocator.available() > 0 && !self.waiting.is_empty() {
            let best_idx = self
                .waiting
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.priority(&self.cfg)
                        .partial_cmp(&b.priority(&self.cfg))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            match best_idx {
                Some(idx) => {
                    let mut seq = self.waiting.remove(idx).expect("index must be valid");
                    let slot = self
                        .allocator
                        .alloc()
                        .expect("available() > 0 guaranteed");
                    seq.kv_slot = Some(slot);
                    self.active.insert(seq.id, seq);
                }
                None => break,
            }
        }
    }

    // ── Build Micro-Batch ───────────────────────────────────────────────

    /// Build the next micro-batch using harmonic grouping + priority selection.
    ///
    /// Algorithm:
    /// 1. Resume coasting sequences whose timer expired
    /// 2. Collect non-coasting candidates sorted by length
    /// 3. Greedy harmonic grouping (O(n·G))
    /// 4. Select highest-aggregate-priority group
    /// 5. Build MicroBatch descriptor with real token data
    pub fn build_micro_batch(&mut self) -> Option<MicroBatch> {
        if self.active.is_empty() {
            return None;
        }

        // Step 1: Resume coasting sequences whose timer has expired
        for seq in self.active.values_mut() {
            if seq.should_resume() {
                seq.phase = SequencePhase::Decode {
                    tokens_generated: 0,
                };
            }
        }

        // Step 2: Collect non-coasting candidates, sorted by current_len
        let mut candidate_ids: Vec<SequenceId> = self
            .active
            .values()
            .filter(|s| !matches!(s.phase, SequencePhase::Coasting { .. }))
            .filter(|s| !matches!(s.phase, SequencePhase::Finished))
            .map(|s| s.id)
            .collect();

        candidate_ids
            .sort_by_key(|id| self.active.get(id).map(|s| s.current_len).unwrap_or(0));

        if candidate_ids.is_empty() {
            return None;
        }

        // Step 3: Greedy harmonic grouping
        let mut groups: Vec<HarmonicGroup> = Vec::new();
        for &id in &candidate_ids {
            let len = match self.active.get(&id) {
                Some(s) => s.current_len,
                None => continue,
            };

            let found = groups.iter_mut().find(|g| {
                g.len() < self.cfg.max_batch_size && g.can_admit(len, self.cfg.harmonic_ratio)
            });

            if let Some(g) = found {
                g.admit(id, len);
            } else {
                groups.push(HarmonicGroup::new(id, len));
            }
        }

        // Step 4: Select highest-aggregate-priority group
        let best_group = groups.into_iter().max_by(|a, b| {
            let score = |g: &HarmonicGroup| -> f32 {
                g.sequences
                    .iter()
                    .filter_map(|id| self.active.get(id))
                    .map(|s| s.priority(&self.cfg))
                    .sum::<f32>()
            };
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        // Step 5: Build MicroBatch
        let mut entries: Vec<BatchEntry> = Vec::with_capacity(best_group.sequences.len());
        let mut all_prefill = true;
        let mut all_decode = true;

        for seq_id in &best_group.sequences {
            let seq = match self.active.get(seq_id) {
                Some(s) => s,
                None => continue,
            };
            let slot = match seq.kv_slot {
                Some(s) => s,
                None => continue,
            };

            let (tokens, entry_phase) = match &seq.phase {
                SequencePhase::Prefill { tokens_remaining } => {
                    all_decode = false;
                    // Yield the remaining prompt tokens from the correct offset
                    let start = seq.prompt_len.saturating_sub(*tokens_remaining);
                    let tokens = seq.prompt_tokens[start..].to_vec();
                    (tokens, BatchEntryPhase::Prefill)
                }
                SequencePhase::Decode { .. } => {
                    all_prefill = false;
                    let tok = seq.last_token.unwrap_or(0);
                    (vec![tok], BatchEntryPhase::Decode)
                }
                _ => continue,
            };

            entries.push(BatchEntry {
                seq_id: *seq_id,
                kv_slot: slot,
                tokens,
                phase: entry_phase,
            });
        }

        if entries.is_empty() {
            return None;
        }

        Some(MicroBatch {
            entries,
            all_prefill,
            all_decode,
        })
    }

    // ── Commit Results ──────────────────────────────────────────────────

    /// Commit kernel results: update phases, detect coasting, reclaim slots.
    ///
    /// `results` contains one tuple per sequence processed in the last batch:
    /// `(SequenceId, sampled_token, top1_prob, is_eos)`.
    pub fn commit_results(
        &mut self,
        results: &[(SequenceId, u32 /* sampled token */, f32 /* top1_prob */, bool /* is_eos */)],
    ) {
        let cfg = self.cfg.clone();

        for &(id, token, top1_prob, is_eos) in results {
            let Some(seq) = self.active.get_mut(&id) else {
                continue;
            };

            if is_eos {
                seq.phase = SequencePhase::Finished;
                continue;
            }

            // Sanitise NaN
            let top1_prob = if top1_prob.is_nan() {
                0.0
            } else {
                top1_prob
            };

            seq.last_token = Some(token);

            match seq.phase.clone() {
                SequencePhase::Prefill { tokens_remaining } => {
                    if tokens_remaining <= 1 {
                        seq.phase = SequencePhase::Decode {
                            tokens_generated: 0,
                        };
                    } else {
                        seq.phase = SequencePhase::Prefill {
                            tokens_remaining: tokens_remaining - 1,
                        };
                    }
                }
                SequencePhase::Decode { .. } => {
                    seq.record_decode_step(top1_prob, &cfg);
                    if seq.should_coast(&cfg) {
                        seq.enter_coast(&cfg);
                    }
                }
                SequencePhase::Coasting { .. } | SequencePhase::Finished => {}
            }
        }

        // Reclaim KV slots from finished sequences
        let finished_ids: Vec<SequenceId> = self
            .active
            .iter()
            .filter(|(_, s)| matches!(s.phase, SequencePhase::Finished))
            .map(|(id, _)| *id)
            .collect();

        for id in finished_ids {
            if let Some(seq) = self.active.remove(&id) {
                if let Some(slot) = seq.kv_slot {
                    self.allocator.free(slot);
                }
            }
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────

    #[inline]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    #[inline]
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    #[inline]
    pub fn prefetch_count(&self) -> usize {
        self.prefetch.len()
    }

    #[inline]
    pub fn free_slots(&self) -> usize {
        self.allocator.available()
    }
}

// ---------------------------------------------------------------------------
// Thread-safe Handle
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around `ArbScheduler`.
///
/// Each method acquires the lock for the duration of the call.
#[derive(Clone)]
pub struct ArbHandle(Arc<Mutex<ArbScheduler>>);

impl ArbHandle {
    pub fn new(cfg: ArbConfig) -> Self {
        Self(Arc::new(Mutex::new(ArbScheduler::new(cfg))))
    }

    /// Enqueue a new request. May be called from any thread.
    pub fn enqueue(&self, prompt_tokens: Vec<u32>, urgency: f32) -> SequenceId {
        self.0.lock().unwrap().enqueue(prompt_tokens, urgency)
    }

    /// One scheduler tick: admit waiting → build micro-batch.
    pub fn step(&self) -> Option<MicroBatch> {
        let mut sched = self.0.lock().unwrap();
        sched.admit_waiting();
        sched.build_micro_batch()
    }

    /// Commit kernel results.
    pub fn commit(&self, results: &[(SequenceId, u32, f32, bool)]) {
        self.0.lock().unwrap().commit_results(results);
    }

    /// Number of active sequences.
    pub fn active_count(&self) -> usize {
        self.0.lock().unwrap().active_count()
    }

    /// Number of waiting sequences.
    pub fn waiting_count(&self) -> usize {
        self.0.lock().unwrap().waiting_count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: Enqueue and admit ───────────────────────────────────────

    #[test]
    fn test_enqueue_and_admit() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![1, 2, 3, 4], 0.5);
        // Enqueue routes to prefetch first (prefetch queue not full)
        assert_eq!(sched.prefetch_count(), 1);
        sched.admit_waiting();
        assert_eq!(sched.active_count(), 1);
        assert_eq!(sched.waiting_count(), 0);
        assert_eq!(sched.prefetch_count(), 0);
        assert_eq!(id, SequenceId(0));
    }

    // ── Test 2: KV slot recycling ──────────────────────────────────────

    #[test]
    fn test_kv_slot_recycling() {
        let mut alloc = KvSlotAllocator::new(4);
        let s0 = alloc.alloc().unwrap();
        let s1 = alloc.alloc().unwrap();
        assert_eq!(alloc.available(), 2);
        alloc.free(s0);
        alloc.free(s1);
        assert_eq!(alloc.available(), 4);
    }

    // ── Test 3: Harmonic grouping invariant ────────────────────────────

    #[test]
    fn test_harmonic_grouping() {
        let mut group = HarmonicGroup::new(SequenceId(0), 100);
        // 150 / 100 = 1.5 <= 2.0: admitted
        assert!(group.can_admit(150, 2.0));
        group.admit(SequenceId(1), 150);
        // 300 / 100 = 3.0 > 2.0: rejected
        assert!(!group.can_admit(300, 2.0));
        // 200 / 100 = 2.0 == 2.0: admitted (boundary, inclusive)
        assert!(group.can_admit(200, 2.0));
    }

    // ── Test 4: Confidence gating / coasting entry ─────────────────────

    #[test]
    fn test_confidence_gating() {
        let cfg = ArbConfig {
            confidence_exit_threshold: 0.95,
            confidence_exit_k: 3,
            ..Default::default()
        };
        let mut seq = SequenceState::new(SequenceId(0), vec![1, 2, 3, 4, 5, 6, 7, 8], 0.5);
        seq.phase = SequencePhase::Decode {
            tokens_generated: 0,
        };

        seq.record_decode_step(0.98, &cfg);
        seq.record_decode_step(0.97, &cfg);
        assert!(!seq.should_coast(&cfg)); // streak = 2, need 3

        seq.record_decode_step(0.99, &cfg);
        assert!(seq.should_coast(&cfg)); // streak = 3 >= k=3

        seq.enter_coast(&cfg);
        assert_eq!(seq.high_conf_streak, 0);
        assert!(matches!(seq.phase, SequencePhase::Coasting { .. }));
    }

    // ── Test 5: Streak reset on low-confidence step ───────────────────

    #[test]
    fn test_streak_reset() {
        let cfg = ArbConfig {
            confidence_exit_threshold: 0.95,
            confidence_exit_k: 3,
            ..Default::default()
        };
        let mut seq = SequenceState::new(SequenceId(0), vec![1; 8], 0.5);
        seq.phase = SequencePhase::Decode {
            tokens_generated: 0,
        };

        seq.record_decode_step(0.98, &cfg);
        seq.record_decode_step(0.98, &cfg);
        assert_eq!(seq.high_conf_streak, 2);

        seq.record_decode_step(0.50, &cfg);
        assert_eq!(seq.high_conf_streak, 0);
        assert!(!seq.should_coast(&cfg));
    }

    // ── Test 6: Priority ordering ─────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        let cfg = ArbConfig::default();
        let urgent = SequenceState::new(SequenceId(0), vec![1; 16], 1.0);
        let idle = SequenceState::new(SequenceId(1), vec![1; 16], 0.0);
        assert!(urgent.priority(&cfg) > idle.priority(&cfg));
    }

    // ── Test 7: Micro-batch builds successfully ───────────────────────

    #[test]
    fn test_micro_batch_build() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        sched.enqueue(vec![1u32; 64], 0.8);
        sched.enqueue(vec![1u32; 72], 0.5);
        sched.admit_waiting();
        let batch = sched.build_micro_batch();
        assert!(batch.is_some());
        assert!(!batch.unwrap().entries.is_empty());
    }

    // ── Test 8: Slot reclaim on EOS ───────────────────────────────────

    #[test]
    fn test_slot_reclaim_on_eos() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![1, 2, 3], 1.0);
        sched.admit_waiting();
        let expected_free = sched.cfg.max_batch_size - 1;
        assert_eq!(sched.free_slots(), expected_free);

        // Commit EOS result
        sched.commit_results(&[(id, 2u32, 0.99, true)]);
        assert_eq!(sched.active_count(), 0);
        assert_eq!(sched.free_slots(), sched.cfg.max_batch_size);
    }

    // ── Test 9: Mixed prefill/decode batch flags ─────────────────────

    #[test]
    fn test_batch_phase_flags() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        sched.enqueue(vec![1u32; 50], 0.5);
        sched.enqueue(vec![1u32; 55], 0.5);
        sched.admit_waiting();

        let batch = sched.build_micro_batch().unwrap();
        assert!(batch.all_prefill);
        assert!(!batch.all_decode);
    }

    // ── Test 10: Harmonic ratio boundary (H == 1.0) ──────────────────

    #[test]
    fn test_harmonic_ratio_tight() {
        let mut group = HarmonicGroup::new(SequenceId(0), 100);
        assert!(!group.can_admit(101, 1.0)); // 101/100 > 1.0
        assert!(group.can_admit(100, 1.0)); // 100/100 == 1.0
    }

    // ── Test 11: ArbHandle thread-safe enqueue + step ────────────────

    #[test]
    fn test_arb_handle() {
        let handle = ArbHandle::new(ArbConfig::default());
        let id = handle.enqueue(vec![1, 2, 3, 4], 0.5);
        assert_eq!(id, SequenceId(0));

        let batch = handle.step();
        assert!(batch.is_some());
        assert_eq!(handle.active_count(), 1);
    }

    // ── Test 12: Multiple sequence admit respects slot limit ─────────

    #[test]
    fn test_slot_exhaustion() {
        let cfg = ArbConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        let mut sched = ArbScheduler::new(cfg);
        sched.enqueue(vec![1u32; 10], 0.5);
        sched.enqueue(vec![1u32; 10], 0.5);
        sched.enqueue(vec![1u32; 10], 0.5); // third has no slot

        sched.admit_waiting();
        assert_eq!(sched.active_count(), 2);
        assert_eq!(sched.waiting_count(), 1); // third remains waiting
        assert_eq!(sched.free_slots(), 0);
    }

    // ── Test 13: Prefill → Decode transition via commit ─────────────

    #[test]
    fn test_prefill_to_decode_transition() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![1u32; 3], 0.5);
        sched.admit_waiting();

        // Commit prefill steps: 3 tokens → 2 remaining → 1 remaining → decode
        sched.commit_results(&[(id, 10, 0.5, false)]);
        sched.commit_results(&[(id, 11, 0.5, false)]);
        sched.commit_results(&[(id, 12, 0.5, false)]);

        // After 3 prefill commits (prompt_len=3), should be in decode
        let seq = sched.active.get(&id).unwrap();
        assert!(matches!(seq.phase, SequencePhase::Decode { .. }));
    }

    // ── Test 14: Initial utility scales with prompt length ──────────

    #[test]
    fn test_initial_utility() {
        let short = SequenceState::initial_utility(8);
        let long = SequenceState::initial_utility(1024);
        assert!(long > short);
        // 8 tokens → log2(8)/log2(8192) ≈ 0.231
        assert!(short > 0.2 && short < 0.3);
    }

    // ── Test 15: Commit with NaN top1_prob treated as 0.0 ───────────

    #[test]
    fn test_nan_top1_prob() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![1, 2, 3], 0.5);
        sched.admit_waiting();

        // Force into decode phase
        sched.commit_results(&[(id, 10, 0.5, false)]);
        sched.commit_results(&[(id, 11, 0.5, false)]);
        sched.commit_results(&[(id, 12, 0.5, false)]);

        // Commit with NaN — should not panic or corrupt state
        sched.commit_results(&[(id, 13, f32::NAN, false)]);
        let seq = sched.active.get(&id).unwrap();
        assert_eq!(seq.high_conf_streak, 0); // NaN → 0.0, not high-confidence
    }

    // ── Test 16: Prefetch queue overflow routes to waiting ──────────

    #[test]
    fn test_prefetch_overflow() {
        let cfg = ArbConfig {
            prefetch_queue_depth: 2,
            ..Default::default()
        };
        let mut sched = ArbScheduler::new(cfg);
        sched.enqueue(vec![1], 0.5); // prefetch
        sched.enqueue(vec![2], 0.5); // prefetch
        sched.enqueue(vec![3], 0.5); // overflow → waiting

        assert_eq!(sched.prefetch_count(), 2);
        assert_eq!(sched.waiting_count(), 1);
    }

    // ── Test 17: Build batch with all-decode flag ────────────────────

    #[test]
    fn test_all_decode_batch() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id1 = sched.enqueue(vec![1], 0.5);
        let id2 = sched.enqueue(vec![2], 0.5);
        sched.admit_waiting();

        // Transition both to decode via single-token prefill
        sched.commit_results(&[(id1, 10, 0.5, false), (id2, 20, 0.5, false)]);

        let batch = sched.build_micro_batch().unwrap();
        assert!(batch.all_decode);
        assert!(!batch.all_prefill);
    }

    // ── Test 18: Harmonic grouping separates dissimilar lengths ──────

    #[test]
    fn test_harmonic_separation() {
        let cfg = ArbConfig {
            harmonic_ratio: 2.0,
            ..Default::default()
        };
        let mut sched = ArbScheduler::new(cfg);
        // 10 tokens and 100 tokens: ratio = 10.0 > 2.0, must be in different groups
        sched.enqueue(vec![1u32; 10], 0.5);
        sched.enqueue(vec![1u32; 100], 0.5);
        sched.admit_waiting();

        let batch = sched.build_micro_batch().unwrap();
        // Should only contain sequences from one harmonic group
        // (the one with higher priority, but both have same urgency so
        //  the longer prompt has higher utility → it wins)
        assert!(batch.entries.len() == 1);
    }

    // ── Test 19: Coasting sequence excluded from batch ───────────────

    #[test]
    fn test_coasting_excluded_from_batch() {
        let cfg = ArbConfig {
            confidence_exit_threshold: 0.95,
            confidence_exit_k: 2,
            coast_deprioritise_ms: 10000, // 10 seconds — won't expire during test
            ..Default::default()
        };
        let mut sched = ArbScheduler::new(cfg);
        let id1 = sched.enqueue(vec![1], 0.5);
        let id2 = sched.enqueue(vec![2], 0.5);
        sched.admit_waiting();

        // Transition both to decode
        sched.commit_results(&[(id1, 10, 0.5, false), (id2, 20, 0.5, false)]);

        // Push id1 into coasting (2 high-confidence steps with k=2)
        sched.commit_results(&[(id1, 11, 0.99, false)]);
        sched.commit_results(&[(id1, 12, 0.99, false)]);

        // id1 should now be coasting
        let seq = sched.active.get(&id1).unwrap();
        assert!(matches!(seq.phase, SequencePhase::Coasting { .. }));

        // Build batch — should only contain id2
        let batch = sched.build_micro_batch().unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].seq_id, id2);
    }

    // ── Test 20: Monotonic ID generation ────────────────────────────

    #[test]
    fn test_monotonic_ids() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id0 = sched.enqueue(vec![1], 0.5);
        let id1 = sched.enqueue(vec![2], 0.5);
        let id2 = sched.enqueue(vec![3], 0.5);
        assert_eq!(id0, SequenceId(0));
        assert_eq!(id1, SequenceId(1));
        assert_eq!(id2, SequenceId(2));
    }

    // ── Test 21: KV allocator capacity ──────────────────────────────

    #[test]
    fn test_kv_allocator_oom() {
        let mut alloc = KvSlotAllocator::new(2);
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_none()); // exhausted
        assert_eq!(alloc.available(), 0);
    }

    // ── Test 22: Token data correctness in prefill batch ────────────

    #[test]
    fn test_prefill_token_data() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![10, 20, 30, 40], 0.5);
        sched.admit_waiting();

        let batch = sched.build_micro_batch().unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].tokens, vec![10, 20, 30, 40]);
        assert_eq!(batch.entries[0].phase, BatchEntryPhase::Prefill);
    }

    // ── Test 23: Token data correctness in decode batch ─────────────

    #[test]
    fn test_decode_token_data() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        let id = sched.enqueue(vec![10], 0.5);
        sched.admit_waiting();

        // Transition to decode
        sched.commit_results(&[(id, 42, 0.5, false)]);

        let batch = sched.build_micro_batch().unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].tokens, vec![42]);
        assert_eq!(batch.entries[0].phase, BatchEntryPhase::Decode);
    }

    // ── Test 24: Urgent sequence admitted first ─────────────────────

    #[test]
    fn test_urgent_admitted_first() {
        let cfg = ArbConfig {
            max_batch_size: 1, // only one slot
            ..Default::default()
        };
        let mut sched = ArbScheduler::new(cfg);
        let _id_low = sched.enqueue(vec![1; 10], 0.0); // low urgency
        let id_high = sched.enqueue(vec![1; 10], 1.0); // high urgency

        sched.admit_waiting();
        assert_eq!(sched.active_count(), 1);
        // The high-urgency sequence should have been admitted
        assert!(sched.active.contains_key(&id_high));
    }

    // ── Test 25: Empty scheduler returns None ───────────────────────

    #[test]
    fn test_empty_scheduler() {
        let mut sched = ArbScheduler::new(ArbConfig::default());
        assert!(sched.build_micro_batch().is_none());
    }
}

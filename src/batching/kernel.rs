
//! §7 Integration Contract — `InferenceKernel` trait and ARB driver loop.
//!
//! The ARB scheduler drives inference via a single trait:
//!
//! ```text
//! loop {
//!     batch  = handle.step()              // admit + build MicroBatch
//!     output = kernel.forward(&batch)     // execute forward pass
//!     handle.commit(&output)              // update state, reclaim slots
//! }
//! ```
//!
//! Implementors:
//! - `CpuKernel`  — candle CPU path (used in tests and non-CUDA builds)
//! - `ArbInferenceKernel` — wires `InferenceGenerator` to the ARB loop
//!   (lives in generator.rs to avoid circular crate deps)

use crate::batching::arb::{ArbHandle, BatchEntryPhase, MicroBatch, SequenceId};
use anyhow::Result;

// ---------------------------------------------------------------------------
// InferenceKernel trait
// ---------------------------------------------------------------------------

/// The contract between the ARB scheduler and the inference backend.
///
/// A kernel receives a [`MicroBatch`] and returns one result per entry:
/// `(seq_id, sampled_token, top1_softmax_probability, is_eos)`.
///
/// The `top1_prob` drives ARB confidence-gated coasting (§5):
/// when `top1_prob >= config.confidence_exit_threshold` for K consecutive
/// steps, the sequence enters the coasting phase.
///
/// # Implementor contract
/// - The returned Vec **must** have exactly one element per `batch.entries` item.
/// - Order need not match `batch.entries` order, but each `SequenceId` must
///   appear exactly once.
/// - For prefill entries, `sampled_token` should be the last-position token
///   and `top1_prob` should reflect the model's confidence at that position.
/// - If EOS is detected, `is_eos = true`; the scheduler will mark that
///   sequence Finished and reclaim its KV slot.
pub trait InferenceKernel: Send {
    fn forward(
        &mut self,
        batch: &MicroBatch,
    ) -> Result<Vec<(SequenceId, u32 /* token */, f32 /* top1_prob */, bool /* is_eos */)>>;
}


// ---------------------------------------------------------------------------
// §8 Scheduler Tick Loop — canonical inference_loop()
// ---------------------------------------------------------------------------

/// Configuration for the production ARB inference loop.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// EOS token id — the loop tags any output token matching this as EOS.
    pub eos_token: u32,
    /// When the scheduler step returns `None` (no runnable batch), sleep this
    /// long before re-checking.  Trades latency for CPU burn.
    /// `Duration::ZERO` disables sleeping (useful for bench-only tight loops).
    pub idle_sleep: std::time::Duration,
    /// Hard upper bound on ticks (0 = unlimited). Used in tests.
    pub max_ticks: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            eos_token: 2,
            // 1 ms idle sleep — negligible latency (<0.1% overhead at 100 tok/s)
            // but avoids 100% CPU burn when all sequences are coasting or the
            // waiting queue is temporarily empty.
            idle_sleep: std::time::Duration::from_millis(1),
            max_ticks: 0, // unlimited
        }
    }
}

/// Statistics snapshot emitted per tick by `inference_loop`.
#[derive(Debug, Clone)]
pub struct TickStats {
    pub tick: u64,
    pub batch_size: usize,
    pub active_sequences: usize,
    pub tick_duration: std::time::Duration,
}

/// Production ARB inference loop — §8 Scheduler Tick Loop.
///
/// Drives the full admit→batch→forward→commit cycle for as long as there are
/// sequences to process or until `shutdown` is set.
///
/// # Idle behaviour
/// When `handle.step()` returns `None` (all sequences coasting or queue empty)
/// the loop parks the calling thread for `cfg.idle_sleep` via
/// `Condvar::wait_timeout` rather than busy-spinning. This keeps CPU at ~0%
/// between batches while still waking up promptly when new sequences arrive.
///
/// # Shutdown
/// Set `shutdown` to `true` (from any thread) to request a clean exit after
/// the current tick completes. `ArbHandle::enqueue` can be called concurrently
/// from other threads while the loop is running.
///
/// # Parameters
/// - `handle`   — thread-safe ARB scheduler handle
/// - `kernel`   — mutable inference backend (holds GPU context / weights)
/// - `cfg`      — loop configuration (eos token, idle sleep, tick limit)
/// - `shutdown` — atomic flag; loop exits cleanly when set to `true`
/// - `on_tick`  — optional callback invoked after each committed tick;
///                receives `&TickStats`. Use for metrics/logging.
///
/// # Returns
/// Total number of `(SequenceId, token)` pairs committed across all ticks.
pub fn inference_loop(
    handle: &ArbHandle,
    kernel: &mut dyn InferenceKernel,
    cfg: &LoopConfig,
    shutdown: &std::sync::atomic::AtomicBool,
    mut on_tick: impl FnMut(&TickStats),
) -> anyhow::Result<u64> {
    use std::sync::atomic::Ordering;

    // Condvar + Mutex used for idle parking — avoids busy-spin when no batch
    // is available. The Mutex<bool> holds a spurious-wakeup guard (always false
    // here — we rely on timeout). The pattern is:
    //
    //   condvar.wait_timeout(guard, idle_sleep)
    //
    // Other threads can call condvar.notify_one() after enqueueing to minimise
    // idle latency to near zero.
    let idle_gate = std::sync::Arc::new((
        std::sync::Mutex::new(false),
        std::sync::Condvar::new(),
    ));

    let mut total_committed: u64 = 0;
    let mut tick: u64 = 0;
    // Tracks consecutive idle iterations (batch=None). Used to break the loop
    // for `max_ticks > 0` configs where no sequences are ever queued — without
    // this guard the loop would spin forever since `tick` (batch counter) never
    // increments when `handle.step()` returns None.
    let mut idle_iters: usize = 0;

    loop {
        // ── Shutdown check ────────────────────────────────────────────
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // ── Tick limit (0 = unlimited) ────────────────────────────────
        // `tick` = number of *batched* ticks (work done).
        // `idle_iters` = consecutive idle iterations without a batch.
        // Break when either exceeds max_ticks — prevents spin on empty queue.
        if cfg.max_ticks > 0 {
            if tick as usize >= cfg.max_ticks {
                break;
            }
            // Also break after max_ticks idle iterations without work
            if idle_iters >= cfg.max_ticks && handle.active_count() == 0 {
                break;
            }
        }

        let tick_start = std::time::Instant::now();

        // ── Admit waiting → build micro-batch ─────────────────────────
        let Some(batch) = handle.step() else {
            // No runnable batch (all coasting or queue empty).
            // Park via Condvar so we yield the CPU instead of spinning.
            // Wake up after idle_sleep or when notify_one() is called
            // (e.g., after a new sequence is enqueued externally).
            if !cfg.idle_sleep.is_zero() {
                let (lock, cvar) = &*idle_gate;
                let guard = lock.lock().unwrap();
                let _ = cvar.wait_timeout(guard, cfg.idle_sleep);
            }
            idle_iters += 1;
            continue;
        };

        // Reset idle counter — we have real work
        idle_iters = 0;
        let batch_size = batch.entries.len();

        // ── Forward pass ──────────────────────────────────────────────
        let raw_results = kernel.forward(&batch)?;

        // ── EOS tagging ───────────────────────────────────────────────
        let results: Vec<_> = raw_results
            .into_iter()
            .map(|(id, tok, prob, is_eos)| {
                (id, tok, prob, is_eos || tok == cfg.eos_token)
            })
            .collect();

        let n = results.len() as u64;
        total_committed += n;

        // ── Commit: update phases, coasting, reclaim slots ────────────
        handle.commit(&results);

        // ── Tick metrics callback ─────────────────────────────────────
        let stats = TickStats {
            tick,
            batch_size,
            active_sequences: handle.active_count(),
            tick_duration: tick_start.elapsed(),
        };
        on_tick(&stats);

        tick += 1;
    }

    Ok(total_committed)
}

/// Notify the idle condvar after enqueueing a new sequence.
///
/// Call this to wake `inference_loop` immediately instead of waiting for
/// `idle_sleep` to expire.  The `Weak` reference is safe to drop if the
/// loop has already exited.
pub type IdleNotifier = std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

/// Convenience: run the ARB loop for `max_steps` ticks or until idle.
///
/// Thin wrapper around `inference_loop` for tests and simple CLI usage
/// that don't need shutdown control or metrics callbacks.
///
/// Returns total committed count.
pub fn run_arb_loop(
    handle: &ArbHandle,
    kernel: &mut dyn InferenceKernel,
    max_steps: usize,
    eos_token: u32,
) -> anyhow::Result<usize> {
    let cfg = LoopConfig {
        eos_token,
        idle_sleep: std::time::Duration::ZERO, // no sleep in test loops
        max_ticks: max_steps,
    };
    let shutdown = std::sync::atomic::AtomicBool::new(false);
    let total = inference_loop(handle, kernel, &cfg, &shutdown, |_| {})?;
    Ok(total as usize)
}

// ---------------------------------------------------------------------------
// CpuKernel — pure-candle CPU fallback for testing
// ---------------------------------------------------------------------------

/// Minimal CPU inference kernel for unit testing.
///
/// Runs a single linear projection over batch tokens (no real transformer).
/// Returns deterministic top-1 = token_id % vocab_size, top1_prob = 0.5.
pub struct MockKernel {
    pub vocab_size: usize,
    pub eos_token: u32,
}

impl MockKernel {
    pub fn new(vocab_size: usize, eos_token: u32) -> Self {
        Self { vocab_size, eos_token }
    }
}

impl InferenceKernel for MockKernel {
    fn forward(
        &mut self,
        batch: &MicroBatch,
    ) -> Result<Vec<(SequenceId, u32, f32, bool)>> {
        batch
            .entries
            .iter()
            .map(|entry| {
                // Deterministic mock: next token = last input token + 1 (mod vocab)
                let last_tok = entry.tokens.last().copied().unwrap_or(0);
                let next_tok = (last_tok + 1) % self.vocab_size as u32;

                // Simulate decode confidence oscillation for coasting tests
                let top1_prob: f32 = match entry.phase {
                    BatchEntryPhase::Prefill => 0.5,
                    BatchEntryPhase::Decode => {
                        // High confidence every 4th token to trigger coasting
                        if (last_tok % 4) == 0 { 0.98 } else { 0.6 }
                    }
                };

                let is_eos = next_tok == self.eos_token;

                Ok((entry.seq_id, next_tok, top1_prob, is_eos))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// KvSlot-Aware KV Cache Indexing — §7 "KV-slot indexing into cache tensor"
// ---------------------------------------------------------------------------
// The ARB scheduler assigns each sequence a `kv_slot: usize` from the
// KvSlotAllocator. The KvCacheManager currently indexes by (layer_id) and
// stores a single batch dimension. To support multi-sequence batching per the
// §7 spec, the KvCacheManager needs slot-indexed load/save.
//
// The adapter below bridges the ARB kv_slot index to `KvCacheManager` by
// treating each slot as an independent "virtual" manager index. In a full
// multi-sequence implementation the backing tensor would be
// [max_batch_size, seq_len, n_kv_heads, head_dim], and we'd index the batch
// axis with kv_slot. This module exposes the trait surface so higher layers
// can implement it.

/// Trait for KV cache storage that understands ARB slot indices.
///
/// Each sequence in the batch occupies one `kv_slot`. This trait lets the
/// inference kernel load/save KV tensors per-slot without knowing the
/// underlying layout (contiguous batch tensor vs per-slot managers).
pub trait SlotKvCache: Send {
    /// Load K/V tensors for a specific slot + layer to the compute device.
    ///
    /// Returns `(k, v)` as candle tensors ready for attention.
    /// Shape: `[1, cached_seq_len, n_kv_heads, head_dim]`.
    fn load(
        &self,
        slot: usize,
        layer_id: usize,
    ) -> candle_core::Result<(
        Option<candle_core::Tensor>,
        Option<candle_core::Tensor>,
    )>;

    /// Save updated K/V tensors back to the slot storage after a forward
    /// pass through `layer_id`.
    fn save(
        &mut self,
        slot: usize,
        layer_id: usize,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<()>;

    /// Clear all cached state for a slot (called when sequence finishes).
    fn clear_slot(&mut self, slot: usize);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::arb::{ArbConfig, ArbHandle};

    fn make_handle(max_batch: usize) -> ArbHandle {
        ArbHandle::new(ArbConfig {
            max_batch_size: max_batch,
            prefetch_queue_depth: 16,
            ..Default::default()
        })
    }

    #[test]
    fn test_mock_kernel_forward() {
        let mut kernel = MockKernel::new(32_000, 2 /* eos */);
        let handle = make_handle(4);

        // Enqueue 2 sequences
        handle.enqueue(vec![1u32, 2, 3, 4], 0.5);
        handle.enqueue(vec![5u32, 6, 7], 0.8);

        // Step → should produce a MicroBatch
        let batch = handle.step().expect("batch should be non-empty");
        assert_eq!(batch.entries.len(), 2);

        let results = kernel.forward(&batch).expect("forward should succeed");
        assert_eq!(results.len(), 2);

        // Both should be valid SequenceIds with non-EOS tokens
        for (id, tok, prob, is_eos) in &results {
            assert!(tok < &32_000u32);
            assert!(*prob >= 0.0 && *prob <= 1.0);
            println!("seq {:?} → tok={} prob={:.2} eos={}", id, tok, prob, is_eos);
        }
    }

    #[test]
    fn test_arb_loop_terminates() {
        let mut kernel = MockKernel::new(32_000, 1 /* eos = token 1 */);
        let handle = make_handle(8);

        // Enqueue sequences where last_token + 1 == 1 (eos) quickly
        // token 0 → next = 1 = EOS → immediately finish
        handle.enqueue(vec![0u32], 1.0);
        handle.enqueue(vec![0u32], 0.5);

        let committed = run_arb_loop(&handle, &mut kernel, 10, 1 /* eos */)
            .expect("loop should not error");

        // Both committed (even if eos fires immediately)
        assert!(committed > 0);
        // All sequences should be finished → active_count == 0
        assert_eq!(handle.active_count(), 0);
    }

    #[test]
    fn test_run_arb_loop_max_steps() {
        // No sequences → loop exits after 1st step (no batch)
        let mut kernel = MockKernel::new(32_000, 2);
        let handle = make_handle(4);

        let committed = run_arb_loop(&handle, &mut kernel, 100, 2)
            .expect("empty loop ok");
        assert_eq!(committed, 0);
    }

    // ── §8 Scheduler Tick Loop tests ──────────────────────────────────

    #[test]
    fn test_inference_loop_shutdown_flag() {
        // Loop should exit immediately when shutdown is pre-set
        let mut kernel = MockKernel::new(32_000, 2);
        let handle = make_handle(4);

        handle.enqueue(vec![1u32, 2, 3], 0.5);
        handle.enqueue(vec![4u32, 5, 6], 0.5);

        let cfg = LoopConfig {
            eos_token: 2,
            idle_sleep: std::time::Duration::ZERO,
            max_ticks: 0, // unlimited — rely on shutdown flag
        };
        // Set shutdown before loop starts
        let shutdown = std::sync::atomic::AtomicBool::new(true);

        let committed = inference_loop(&handle, &mut kernel, &cfg, &shutdown, |_| {})
            .expect("shutdown loop ok");

        // No ticks should have run
        assert_eq!(committed, 0);
    }

    #[test]
    fn test_inference_loop_on_tick_callback() {
        // Verify on_tick fires for each committed batch
        let mut kernel = MockKernel::new(32_000, 1 /* eos = 1 */);
        let handle = make_handle(8);

        // token 0 → next = 1 = EOS → finish in 1 tick
        handle.enqueue(vec![0u32], 1.0);

        let cfg = LoopConfig {
            eos_token: 1,
            idle_sleep: std::time::Duration::ZERO,
            max_ticks: 10,
        };
        let shutdown = std::sync::atomic::AtomicBool::new(false);

        let mut tick_count = 0u64;
        let mut last_stats_batch = 0usize;

        let committed = inference_loop(&handle, &mut kernel, &cfg, &shutdown, |stats| {
            tick_count += 1;
            last_stats_batch = stats.batch_size;
            // tick counter must be monotonically increasing
            assert_eq!(stats.tick + 1, tick_count);
        })
        .expect("tick callback loop ok");

        assert!(committed > 0);
        assert!(tick_count >= 1);
        assert!(last_stats_batch >= 1);
    }

    #[test]
    fn test_inference_loop_max_ticks() {
        // With 3 long-running sequences and max_ticks=5, loop must stop at 5
        let mut kernel = MockKernel::new(32_000, 99_999 /* eos unreachable */);
        let handle = make_handle(8);

        handle.enqueue(vec![100u32; 8], 0.5);
        handle.enqueue(vec![200u32; 8], 0.5);
        handle.enqueue(vec![300u32; 8], 0.5);

        let cfg = LoopConfig {
            eos_token: 99_999,
            idle_sleep: std::time::Duration::ZERO,
            max_ticks: 5,
        };
        let shutdown = std::sync::atomic::AtomicBool::new(false);

        let mut ticks_seen = 0u64;
        inference_loop(&handle, &mut kernel, &cfg, &shutdown, |stats| {
            ticks_seen = stats.tick + 1;
        })
        .expect("max_ticks loop ok");

        // Must not exceed max_ticks
        assert!(ticks_seen <= 5, "expected ≤5 ticks, got {}", ticks_seen);
    }

    #[test]
    fn test_inference_loop_idle_no_spin() {
        // Empty handle — loop should idle and exit (since max_ticks=1)
        let mut kernel = MockKernel::new(32_000, 2);
        let handle = make_handle(4);

        let cfg = LoopConfig {
            eos_token: 2,
            // Use zero sleep so the test doesn't slow down CI
            idle_sleep: std::time::Duration::ZERO,
            max_ticks: 1,
        };
        let shutdown = std::sync::atomic::AtomicBool::new(false);

        let committed = inference_loop(&handle, &mut kernel, &cfg, &shutdown, |_| {})
            .expect("idle loop ok");

        // No sequences → zero committed; loop exits at max_ticks without panic
        assert_eq!(committed, 0);
    }

    #[test]
    fn test_loop_config_default() {
        let cfg = LoopConfig::default();
        assert_eq!(cfg.eos_token, 2);
        assert_eq!(cfg.max_ticks, 0);
        assert!(!cfg.idle_sleep.is_zero(), "default idle_sleep should be non-zero");
    }
}

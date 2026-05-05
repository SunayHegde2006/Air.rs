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
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::generator::InferenceGenerator;
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

/// Multi-model interleaved decode scheduler.
///
/// Holds up to `max_slots` loaded model slots.  Each `tick()` advances every
/// active slot by one decode step in round-robin order.
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
}

impl ModelMux {
    /// Create an empty mux.  `max_slots = 0` means unlimited.
    pub fn new(max_slots: usize) -> Self {
        Self { slots: Vec::new(), max_slots }
    }

    /// Add a model slot.  Returns the `SlotId` for future `submit` / `remove` calls.
    ///
    /// Returns `Err` if at capacity.
    pub fn add_slot(&mut self, slot: MuxSlot) -> Result<SlotId> {
        if self.max_slots > 0 && self.slots.len() >= self.max_slots {
            bail!(
                "ModelMux at capacity ({} slots); cannot add '{}'",
                self.max_slots,
                slot.model_id
            );
        }
        let id = SlotId(self.slots.len());
        self.slots.push(slot);
        Ok(id)
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
        Ok(self.slots.remove(idx))
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
}

//! Batching subsystem — Adaptive Resonance Batching (ARB).
//!
//! This module provides the production batching scheduler for Air.rs inference.
//! It replaces traditional FIFO/MSB batching with harmonic length grouping,
//! entropy-weighted priority scheduling, and confidence-gated coasting.
//!
//! # Module Layout (ARB Plan §2)
//!
//! ```text
//! src/batching/
//! ├── mod.rs    ← this file (re-exports)
//! ├── arb.rs    ← scheduler, types, handle, tests
//! └── kernel.rs ← InferenceKernel trait, ARB driver loop
//! ```
//!
//! # Quick Start
//!
//! ```text
//! use air::batching::{ArbHandle, ArbConfig};
//!
//! let handle = ArbHandle::new(ArbConfig::default());
//! let id = handle.enqueue(vec![1, 2, 3, 4], 0.5);
//! let batch = handle.step();  // admit + build micro-batch
//! // kernel.forward(batch) ...
//! // handle.commit(results);
//! ```

pub mod arb;
pub mod kernel;

// ── Re-exports (§2 spec: mod.rs re-exports all public types) ─────────

pub use arb::{
    // Error type
    ArbError,
    // Configuration
    ArbConfig,
    // Core types
    SequenceId,
    SequencePhase,
    SequenceState,
    // KV slot management
    KvSlotAllocator,
    // Harmonic grouping
    HarmonicGroup,
    // Batch types
    MicroBatch,
    BatchEntry,
    BatchEntryPhase,
    // Scheduler
    ArbScheduler,
    // Thread-safe handle
    ArbHandle,
};

// ── §7 Integration Contract re-exports ───────────────────────────────
pub use kernel::{
    InferenceKernel,
    MockKernel,
    SlotKvCache,
    run_arb_loop,
    // §8 Scheduler Tick Loop
    LoopConfig,
    TickStats,
    IdleNotifier,
    inference_loop,
};


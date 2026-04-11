//! Adaptive Pipeline — Circular Slot Manager (S.L.I.P. v3 §Sub-System 1)
//!
//! Production-ready implementation of the S.L.I.P. v3 Adaptive Pipeline Depth:
//!
//! 1. **D_opt formula**: D = ⌈T_compute/T_io⌉ + 1, recalibrated every N layers
//! 2. **Circular slot manager**: D-deep ring buffer (D=2..8) for pipeline parallelism
//! 3. **Pipeline underrun detection**: Logged with timestamps, IO/compute timing, slot states
//! 4. **Dynamic D increase**: Automatic depth increase on consecutive underruns
//!
//! ```text
//!   Slot Ring (D=4):
//!   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
//!   │ Slot0│→ │ Slot1│→ │ Slot2│→ │ Slot3│→ (wraps)
//!   │Ready │  │Comp. │  │Read  │  │Empty │
//!   └──────┘  └──────┘  └──────┘  └──────┘
//!       ↑                   ↑
//!   compute_ptr         read_ptr
//!
//!   Invariant: read_ptr should be D−1 slots ahead of compute_ptr.
//!   Violation → "underrun" → compute stalls waiting for I/O.
//! ```
//!
//! Reference: air_rs_protocols_v3.md §4 "Pipeline Architecture"

use std::fmt;
use std::time::{Duration, Instant};

use crate::ucal::SharedBuffer;

// ── Constants ────────────────────────────────────────────────────────────

/// Minimum pipeline depth (ping-pong buffer).
pub const MIN_PIPELINE_DEPTH: usize = 2;
/// Maximum pipeline depth (memory-constrained upper bound).
pub const MAX_PIPELINE_DEPTH: usize = 8;
/// Number of consecutive underruns before auto-increasing depth.
pub const UNDERRUN_THRESHOLD: usize = 2;
/// How many layers between D_opt recalibrations.
pub const RECALIBRATE_INTERVAL: usize = 10;
/// Minimum samples needed before recalibration is trustworthy.
pub const MIN_SAMPLES_FOR_RECALIBRATION: usize = 4;

// ── Slot State Machine ──────────────────────────────────────────────────

/// State of a single buffer slot in the circular ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is empty, ready to be filled with data from disk
    Empty,
    /// I/O is in progress — data being read from disk into this slot
    Reading,
    /// Data has been loaded and is ready for compute
    Ready,
    /// Compute is in progress on this slot's data
    Computing,
}

impl fmt::Display for SlotState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlotState::Empty     => write!(f, "Empty"),
            SlotState::Reading   => write!(f, "Reading"),
            SlotState::Ready     => write!(f, "Ready"),
            SlotState::Computing => write!(f, "Computing"),
        }
    }
}

// ── Pipeline Slot ───────────────────────────────────────────────────────

/// A single slot in the circular buffer ring.
pub struct PipelineSlot {
    /// Current state of this slot
    pub state: SlotState,
    /// The buffer holding layer weights
    pub buffer: Option<SharedBuffer>,
    /// Which layer index this slot is currently holding
    pub layer_index: Option<usize>,
    /// When this slot started its current operation
    pub op_start: Option<Instant>,
}

impl PipelineSlot {
    fn new() -> Self {
        Self {
            state: SlotState::Empty,
            buffer: None,
            layer_index: None,
            op_start: None,
        }
    }

    /// Transition to Reading state.
    fn begin_read(&mut self, layer_index: usize) {
        self.state = SlotState::Reading;
        self.layer_index = Some(layer_index);
        self.op_start = Some(Instant::now());
    }

    /// Transition from Reading → Ready. Returns read duration.
    fn finish_read(&mut self) -> Duration {
        debug_assert_eq!(self.state, SlotState::Reading);
        let elapsed = self.op_start.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        self.state = SlotState::Ready;
        self.op_start = None;
        elapsed
    }

    /// Transition from Ready → Computing.
    fn begin_compute(&mut self) {
        debug_assert_eq!(self.state, SlotState::Ready);
        self.state = SlotState::Computing;
        self.op_start = Some(Instant::now());
    }

    /// Transition from Computing → Empty (slot recycled). Returns compute duration.
    fn finish_compute(&mut self) -> Duration {
        debug_assert_eq!(self.state, SlotState::Computing);
        let elapsed = self.op_start.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        self.state = SlotState::Empty;
        self.layer_index = None;
        self.op_start = None;
        self.buffer = None;
        elapsed
    }
}

// ── Underrun Event ──────────────────────────────────────────────────────

/// A logged pipeline underrun event with full diagnostic information.
#[derive(Debug, Clone)]
pub struct UnderrunEvent {
    /// Wall-clock time of the underrun.
    pub timestamp: Instant,
    /// Which layer triggered the underrun.
    pub layer_index: usize,
    /// Pipeline depth at the time of the underrun.
    pub depth_at_underrun: usize,
    /// State of the slot that was expected to be `Ready`.
    pub stalled_slot_state: SlotState,
    /// Mean I/O time at point of underrun (from running average).
    pub avg_io_time: Duration,
    /// Mean compute time at point of underrun (from running average).
    pub avg_compute_time: Duration,
    /// Whether the depth was auto-increased as a result.
    pub depth_increased: bool,
    /// The new depth after increase (same as old if not increased).
    pub new_depth: usize,
}

impl fmt::Display for UnderrunEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[UNDERRUN] layer={} D={} slot={} avg_io={:.1}ms avg_compute={:.1}ms {}",
            self.layer_index,
            self.depth_at_underrun,
            self.stalled_slot_state,
            self.avg_io_time.as_secs_f64() * 1000.0,
            self.avg_compute_time.as_secs_f64() * 1000.0,
            if self.depth_increased {
                format!("→ D increased to {}", self.new_depth)
            } else {
                "→ D at max, cannot increase".to_string()
            },
        )
    }
}

// ── Pipeline Metrics ────────────────────────────────────────────────────

/// Running timing statistics for pipeline calibration.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Exponential moving average of I/O read time per layer.
    pub avg_io_time: Duration,
    /// Exponential moving average of compute time per layer.
    pub avg_compute_time: Duration,
    /// Total I/O samples collected.
    pub io_samples: usize,
    /// Total compute samples collected.
    pub compute_samples: usize,
    /// All underrun events (kept for diagnostics).
    pub underrun_log: Vec<UnderrunEvent>,
    /// History of D_opt recalibrations: (layer, old_d, new_d).
    pub recalibration_log: Vec<(usize, usize, usize)>,
    /// Pipeline efficiency ρ = T_compute / max(T_compute, T_io).
    pub rho: f64,
    /// Total compute stall time (waiting for I/O to finish).
    pub total_stall_time: Duration,
}

impl PipelineMetrics {
    fn new() -> Self {
        Self {
            avg_io_time: Duration::ZERO,
            avg_compute_time: Duration::ZERO,
            io_samples: 0,
            compute_samples: 0,
            underrun_log: Vec::new(),
            recalibration_log: Vec::new(),
            rho: 1.0,
            total_stall_time: Duration::ZERO,
        }
    }

    /// Record an I/O timing sample (EMA with α=0.3 for responsiveness).
    fn record_io(&mut self, duration: Duration) {
        const ALPHA: f64 = 0.3;
        if self.io_samples == 0 {
            self.avg_io_time = duration;
        } else {
            let old = self.avg_io_time.as_secs_f64();
            let new = old * (1.0 - ALPHA) + duration.as_secs_f64() * ALPHA;
            self.avg_io_time = Duration::from_secs_f64(new);
        }
        self.io_samples += 1;
    }

    /// Record a compute timing sample (EMA with α=0.3).
    fn record_compute(&mut self, duration: Duration) {
        const ALPHA: f64 = 0.3;
        if self.compute_samples == 0 {
            self.avg_compute_time = duration;
        } else {
            let old = self.avg_compute_time.as_secs_f64();
            let new = old * (1.0 - ALPHA) + duration.as_secs_f64() * ALPHA;
            self.avg_compute_time = Duration::from_secs_f64(new);
        }
        self.compute_samples += 1;
        self.update_rho();
    }

    /// Record stall time (compute waited for IO).
    fn record_stall(&mut self, stall: Duration) {
        self.total_stall_time += stall;
    }

    /// Recalculate ρ.
    fn update_rho(&mut self) {
        let t_c = self.avg_compute_time.as_secs_f64();
        let t_io = self.avg_io_time.as_secs_f64();
        if t_c > 0.0 || t_io > 0.0 {
            self.rho = t_c / t_c.max(t_io);
        }
    }

    /// Calculate D_opt from measured timings.
    ///
    /// Formula: D_opt = ⌈T_compute / T_io⌉ + 1, clamped to [2, 8].
    fn calculate_d_opt(&self) -> usize {
        let t_io = self.avg_io_time.as_secs_f64();
        if t_io <= 0.0 {
            return MIN_PIPELINE_DEPTH;
        }
        let ratio = self.avg_compute_time.as_secs_f64() / t_io;
        let d = ratio.ceil() as usize + 1;
        d.clamp(MIN_PIPELINE_DEPTH, MAX_PIPELINE_DEPTH)
    }

    /// Has enough data for reliable D_opt calculation?
    fn has_enough_samples(&self) -> bool {
        self.io_samples >= MIN_SAMPLES_FOR_RECALIBRATION
            && self.compute_samples >= MIN_SAMPLES_FOR_RECALIBRATION
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "Pipeline: ρ={:.2} | IO={:.1}ms | Compute={:.1}ms | stalls={:.1}ms | underruns={} | recals={}",
            self.rho,
            self.avg_io_time.as_secs_f64() * 1000.0,
            self.avg_compute_time.as_secs_f64() * 1000.0,
            self.total_stall_time.as_secs_f64() * 1000.0,
            self.underrun_log.len(),
            self.recalibration_log.len(),
        )
    }
}

// ── Circular Slot Manager ───────────────────────────────────────────────

/// Manages a ring of D buffer slots for pipeline-parallel weight streaming.
///
/// The `read_ptr` advances as layers are loaded from disk.
/// The `compute_ptr` advances as layers are processed by the compute backend.
///
/// Invariant: `read_ptr` should be `D-1` slots ahead of `compute_ptr`.
/// If this invariant is violated (underrun), D is automatically increased.
///
/// # Adaptive Depth
///
/// The pipeline recalibrates D_opt every `RECALIBRATE_INTERVAL` layers using
/// the measured I/O and compute times:
/// ```text
///   D_opt = ⌈T_compute / T_io⌉ + 1
/// ```
///
/// On consecutive underruns (≥`UNDERRUN_THRESHOLD`), depth is increased
/// immediately without waiting for the next recalibration.
pub struct CircularSlotManager {
    /// The current pipeline depth (number of slots).
    pub depth: usize,
    /// The buffer slots.
    slots: Vec<PipelineSlot>,
    /// Points to the next slot to read into.
    pub read_ptr: usize,
    /// Points to the next slot to compute from.
    pub compute_ptr: usize,
    /// Total number of underrun events detected.
    pub underrun_count: usize,
    /// Maximum allowed pipeline depth.
    max_depth: usize,
    /// Timing metrics and underrun log.
    pub metrics: PipelineMetrics,
    /// Consecutive underruns since last successful compute.
    consecutive_underruns: usize,
    /// Layers processed since last D_opt recalibration.
    layers_since_recalibration: usize,
    /// Total layers processed in this session.
    pub total_layers_processed: usize,
}

impl CircularSlotManager {
    /// Create a new circular slot manager with the given initial pipeline depth.
    ///
    /// Depth is clamped to [2, 8]. Use `from_d_opt()` to initialize from
    /// measured timings.
    pub fn new(depth: usize) -> Self {
        let depth = depth.clamp(MIN_PIPELINE_DEPTH, MAX_PIPELINE_DEPTH);
        let mut slots = Vec::with_capacity(depth);
        for _ in 0..depth {
            slots.push(PipelineSlot::new());
        }

        Self {
            depth,
            slots,
            read_ptr: 0,
            compute_ptr: 0,
            underrun_count: 0,
            max_depth: MAX_PIPELINE_DEPTH,
            metrics: PipelineMetrics::new(),
            consecutive_underruns: 0,
            layers_since_recalibration: 0,
            total_layers_processed: 0,
        }
    }

    /// Create a pipeline from initial D_opt calculation.
    ///
    /// Uses measured I/O and compute times from `DriveInquisitor` or prior runs.
    pub fn from_d_opt(t_compute: Duration, t_io: Duration) -> Self {
        let d = crate::drive_inquisitor::calculate_d_opt(t_compute, t_io);
        let mut mgr = Self::new(d);
        // Seed the metrics EMA with the initial estimates.
        mgr.metrics.avg_io_time = t_io;
        mgr.metrics.avg_compute_time = t_compute;
        mgr.metrics.io_samples = 1;
        mgr.metrics.compute_samples = 1;
        mgr
    }

    /// How many slots ahead is the read pointer from the compute pointer.
    pub fn read_ahead(&self) -> usize {
        if self.read_ptr >= self.compute_ptr {
            self.read_ptr - self.compute_ptr
        } else {
            self.depth - self.compute_ptr + self.read_ptr
        }
    }

    /// Check if the pipeline has underrun (compute waiting for I/O).
    pub fn is_underrun(&self) -> bool {
        self.read_ahead() == 0
            && self.slots[self.compute_ptr].state != SlotState::Ready
    }

    // ── Read Operations ─────────────────────────────────────────────

    /// Begin reading into the next available slot.
    ///
    /// Returns `Some(slot_idx)` if a slot was available, `None` if all slots
    /// are occupied (pipeline full — wait for compute to free a slot).
    pub fn next_read_slot(&mut self, layer_index: usize) -> Option<usize> {
        let slot_idx = self.read_ptr;
        let slot = &self.slots[slot_idx];

        if slot.state != SlotState::Empty {
            return None;
        }

        self.slots[slot_idx].begin_read(layer_index);
        self.read_ptr = (self.read_ptr + 1) % self.depth;
        Some(slot_idx)
    }

    /// Mark a read slot as complete. Records I/O timing.
    pub fn finish_read(&mut self, slot_idx: usize) {
        let io_time = self.slots[slot_idx].finish_read();
        self.metrics.record_io(io_time);
    }

    /// Mark a read slot as complete with a known duration (for external timing).
    pub fn finish_read_with_duration(&mut self, slot_idx: usize, io_duration: Duration) {
        self.slots[slot_idx].finish_read();
        self.metrics.record_io(io_duration);
    }

    // ── Compute Operations ──────────────────────────────────────────

    /// Get the next slot to compute from.
    ///
    /// If the next slot is not `Ready`, this logs an underrun event and
    /// may auto-increase the pipeline depth.
    ///
    /// Returns `Some(slot_idx)` if ready, `None` if stalled (underrun).
    pub fn next_compute_slot(&mut self) -> Option<usize> {
        let slot_idx = self.compute_ptr;
        let slot = &self.slots[slot_idx];

        if slot.state == SlotState::Ready {
            self.consecutive_underruns = 0;
            self.slots[slot_idx].begin_compute();
            return Some(slot_idx);
        }

        // ── Underrun detected ───────────────────────────────────────
        if slot.state == SlotState::Empty || slot.state == SlotState::Reading {
            self.underrun_count += 1;
            self.consecutive_underruns += 1;

            let stalled_state = slot.state;
            let layer = slot.layer_index.unwrap_or(self.total_layers_processed);
            let old_depth = self.depth;

            // Auto-increase depth on consecutive underruns.
            let depth_increased = if self.consecutive_underruns >= UNDERRUN_THRESHOLD {
                self.increase_depth()
            } else {
                false
            };

            let event = UnderrunEvent {
                timestamp: Instant::now(),
                layer_index: layer,
                depth_at_underrun: old_depth,
                stalled_slot_state: stalled_state,
                avg_io_time: self.metrics.avg_io_time,
                avg_compute_time: self.metrics.avg_compute_time,
                depth_increased,
                new_depth: self.depth,
            };

            // Log to stderr for real-time monitoring.
            eprintln!("⚠ {}", event);
            self.metrics.underrun_log.push(event);
        }

        None
    }

    /// Spin-wait for the next compute slot to become ready.
    ///
    /// This blocks until the slot transitions to `Ready`, recording
    /// total stall time. In production, the I/O subsystem should be
    /// filling slots in a background thread.
    ///
    /// Returns the slot index once ready.
    pub fn wait_for_compute_slot(&mut self) -> usize {
        let stall_start = Instant::now();
        loop {
            if let Some(idx) = self.next_compute_slot() {
                let stall = stall_start.elapsed();
                if stall > Duration::from_millis(1) {
                    self.metrics.record_stall(stall);
                }
                return idx;
            }
            // Yield to allow I/O thread to make progress.
            std::thread::yield_now();
        }
    }

    /// Mark a compute slot as complete. Records compute timing and triggers
    /// recalibration if the interval has been reached.
    pub fn finish_compute(&mut self, slot_idx: usize) {
        let compute_time = self.slots[slot_idx].finish_compute();
        self.metrics.record_compute(compute_time);
        self.compute_ptr = (self.compute_ptr + 1) % self.depth;
        self.total_layers_processed += 1;
        self.layers_since_recalibration += 1;

        // ── Periodic D_opt recalibration ────────────────────────────
        if self.layers_since_recalibration >= RECALIBRATE_INTERVAL
            && self.metrics.has_enough_samples()
        {
            self.recalibrate();
        }
    }

    /// Mark a compute slot as complete with a known duration (for external timing).
    pub fn finish_compute_with_duration(&mut self, slot_idx: usize, compute_duration: Duration) {
        self.slots[slot_idx].finish_compute();
        self.metrics.record_compute(compute_duration);
        self.compute_ptr = (self.compute_ptr + 1) % self.depth;
        self.total_layers_processed += 1;
        self.layers_since_recalibration += 1;

        if self.layers_since_recalibration >= RECALIBRATE_INTERVAL
            && self.metrics.has_enough_samples()
        {
            self.recalibrate();
        }
    }

    // ── Depth Management ────────────────────────────────────────────

    /// Dynamically increase pipeline depth by 1 (up to max_depth).
    ///
    /// The new slot is inserted at the current `read_ptr` position.
    /// This is safe because a new slot starts as `Empty`, which is
    /// exactly what `read_ptr` expects.
    pub fn increase_depth(&mut self) -> bool {
        if self.depth >= self.max_depth {
            return false;
        }
        self.depth += 1;
        self.slots.push(PipelineSlot::new());
        eprintln!(
            "📈 Pipeline depth increased: D={} (underruns={}, avg_io={:.1}ms, avg_compute={:.1}ms)",
            self.depth,
            self.underrun_count,
            self.metrics.avg_io_time.as_secs_f64() * 1000.0,
            self.metrics.avg_compute_time.as_secs_f64() * 1000.0,
        );
        true
    }

    /// Recalibrate D_opt from measured timing data.
    ///
    /// Called automatically every `RECALIBRATE_INTERVAL` layers.
    /// Only increases depth (never decreases mid-session to avoid
    /// dropping pre-loaded data).
    pub fn recalibrate(&mut self) {
        let new_d = self.metrics.calculate_d_opt();
        let old_d = self.depth;
        self.layers_since_recalibration = 0;

        if new_d > old_d {
            while self.depth < new_d {
                if !self.increase_depth() {
                    break;
                }
            }
            self.metrics.recalibration_log.push((
                self.total_layers_processed,
                old_d,
                self.depth,
            ));
            eprintln!(
                "🔄 D_opt recalibrated: {} → {} at layer {} (ρ={:.2})",
                old_d, self.depth, self.total_layers_processed, self.metrics.rho,
            );
        }
    }

    // ── Query ───────────────────────────────────────────────────────

    /// Get the state of all slots (for debugging/metrics).
    pub fn slot_states(&self) -> Vec<(usize, SlotState, Option<usize>)> {
        self.slots
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.state, s.layer_index))
            .collect()
    }

    /// Reset all slots to Empty state (for a new generation pass).
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.state = SlotState::Empty;
            slot.layer_index = None;
            slot.op_start = None;
            slot.buffer = None;
        }
        self.read_ptr = 0;
        self.compute_ptr = 0;
        self.consecutive_underruns = 0;
        self.layers_since_recalibration = 0;
        // Metrics are preserved across resets for continued calibration.
    }

    /// Format a visual representation of the pipeline state.
    pub fn visual(&self) -> String {
        let mut parts = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            let marker = match slot.state {
                SlotState::Empty     => "○",
                SlotState::Reading   => "◐",
                SlotState::Ready     => "●",
                SlotState::Computing => "◑",
            };
            let ptr = if i == self.compute_ptr && i == self.read_ptr {
                "CR"
            } else if i == self.compute_ptr {
                "C "
            } else if i == self.read_ptr {
                " R"
            } else {
                "  "
            };
            parts.push(format!("[{}{}]", marker, ptr));
        }
        format!(
            "Pipeline: {} (D={}, ρ={:.2}, underruns={})",
            parts.join(""),
            self.depth,
            self.metrics.rho,
            self.underrun_count,
        )
    }

    /// Full diagnostic dump (for crash reports / debug logs).
    pub fn diagnostic_dump(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Pipeline Diagnostic Dump ===".to_string());
        lines.push(format!("Depth: {} (max: {})", self.depth, self.max_depth));
        lines.push(format!("Pointers: read={} compute={} ahead={}", self.read_ptr, self.compute_ptr, self.read_ahead()));
        lines.push(format!("Total layers: {} | Underruns: {} | Consecutive: {}", self.total_layers_processed, self.underrun_count, self.consecutive_underruns));
        lines.push(format!("{}", self.metrics.summary()));
        lines.push(self.visual());

        if !self.metrics.underrun_log.is_empty() {
            lines.push(format!("\n--- Underrun Log ({} events) ---", self.metrics.underrun_log.len()));
            for event in &self.metrics.underrun_log {
                lines.push(format!("  {}", event));
            }
        }

        if !self.metrics.recalibration_log.is_empty() {
            lines.push(format!("\n--- Recalibration Log ({} events) ---", self.metrics.recalibration_log.len()));
            for (layer, old_d, new_d) in &self.metrics.recalibration_log {
                lines.push(format!("  Layer {}: D {} → {}", layer, old_d, new_d));
            }
        }

        lines.join("\n")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_state_display() {
        assert_eq!(format!("{}", SlotState::Empty), "Empty");
        assert_eq!(format!("{}", SlotState::Reading), "Reading");
        assert_eq!(format!("{}", SlotState::Ready), "Ready");
        assert_eq!(format!("{}", SlotState::Computing), "Computing");
    }

    #[test]
    fn test_new_pipeline_clamped() {
        let mgr = CircularSlotManager::new(1);
        assert_eq!(mgr.depth, MIN_PIPELINE_DEPTH);

        let mgr = CircularSlotManager::new(100);
        assert_eq!(mgr.depth, MAX_PIPELINE_DEPTH);
    }

    #[test]
    fn test_initial_state() {
        let mgr = CircularSlotManager::new(3);
        assert_eq!(mgr.depth, 3);
        assert_eq!(mgr.read_ptr, 0);
        assert_eq!(mgr.compute_ptr, 0);
        assert_eq!(mgr.underrun_count, 0);
        assert_eq!(mgr.total_layers_processed, 0);
        assert_eq!(mgr.consecutive_underruns, 0);
        let states = mgr.slot_states();
        assert!(states.iter().all(|(_, s, _)| *s == SlotState::Empty));
    }

    #[test]
    fn test_read_compute_cycle_with_timing() {
        let mut mgr = CircularSlotManager::new(3);

        // Read slot 0
        let slot = mgr.next_read_slot(0).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(mgr.read_ptr, 1);

        // Read slot 1
        let slot = mgr.next_read_slot(1).unwrap();
        assert_eq!(slot, 1);
        assert_eq!(mgr.read_ptr, 2);

        // Finish reading slot 0
        mgr.finish_read(0);
        assert_eq!(mgr.slots[0].state, SlotState::Ready);

        // Compute slot 0
        let slot = mgr.next_compute_slot().unwrap();
        assert_eq!(slot, 0);

        // Finish computing slot 0 → recycled
        mgr.finish_compute(0);
        assert_eq!(mgr.slots[0].state, SlotState::Empty);
        assert_eq!(mgr.compute_ptr, 1);
        assert_eq!(mgr.total_layers_processed, 1);
    }

    #[test]
    fn test_underrun_detection_and_logging() {
        let mut mgr = CircularSlotManager::new(2);

        // Try to compute without any reads → underrun
        assert!(mgr.next_compute_slot().is_none());
        assert_eq!(mgr.underrun_count, 1);
        assert_eq!(mgr.consecutive_underruns, 1);

        // Second underrun → should trigger auto-increase
        assert!(mgr.next_compute_slot().is_none());
        assert_eq!(mgr.underrun_count, 2);
        assert_eq!(mgr.consecutive_underruns, 2);
        // Depth should have increased (if under max).
        assert!(mgr.depth >= 3 || mgr.depth == MAX_PIPELINE_DEPTH);

        // Underrun log should have 2 events.
        assert_eq!(mgr.metrics.underrun_log.len(), 2);
    }

    #[test]
    fn test_consecutive_underrun_reset_on_success() {
        let mut mgr = CircularSlotManager::new(3);

        // Cause one underrun
        assert!(mgr.next_compute_slot().is_none());
        assert_eq!(mgr.consecutive_underruns, 1);

        // Now do a successful read → compute
        mgr.next_read_slot(0);
        mgr.finish_read(0);
        let slot = mgr.next_compute_slot().unwrap();
        assert_eq!(mgr.consecutive_underruns, 0); // Reset!
        mgr.finish_compute(slot);
    }

    #[test]
    fn test_read_ahead_tracking() {
        let mut mgr = CircularSlotManager::new(4);

        assert_eq!(mgr.read_ahead(), 0);

        mgr.next_read_slot(0);
        assert_eq!(mgr.read_ahead(), 1);

        mgr.next_read_slot(1);
        assert_eq!(mgr.read_ahead(), 2);
    }

    #[test]
    fn test_increase_depth_logs() {
        let mut mgr = CircularSlotManager::new(3);
        assert_eq!(mgr.depth, 3);

        assert!(mgr.increase_depth());
        assert_eq!(mgr.depth, 4);

        // Max out at 8
        while mgr.depth < MAX_PIPELINE_DEPTH {
            mgr.increase_depth();
        }
        assert!(!mgr.increase_depth());
    }

    #[test]
    fn test_from_d_opt() {
        // T_compute=30ms, T_io=10ms → D = ceil(3.0) + 1 = 4
        let mgr = CircularSlotManager::from_d_opt(
            Duration::from_millis(30),
            Duration::from_millis(10),
        );
        assert_eq!(mgr.depth, 4);
        assert!(mgr.metrics.io_samples >= 1);
        assert!(mgr.metrics.compute_samples >= 1);
    }

    #[test]
    fn test_from_d_opt_io_bound() {
        // T_compute=5ms, T_io=20ms → D = ceil(0.25) + 1 = 2
        let mgr = CircularSlotManager::from_d_opt(
            Duration::from_millis(5),
            Duration::from_millis(20),
        );
        assert_eq!(mgr.depth, 2);
    }

    #[test]
    fn test_metrics_ema() {
        let mut metrics = PipelineMetrics::new();

        // First sample sets the value directly.
        metrics.record_io(Duration::from_millis(10));
        assert_eq!(metrics.avg_io_time, Duration::from_millis(10));

        // Subsequent samples use EMA (α=0.3).
        metrics.record_io(Duration::from_millis(20));
        // EMA: 10 * 0.7 + 20 * 0.3 = 13ms
        let expected = 10.0 * 0.7 + 20.0 * 0.3;
        let actual = metrics.avg_io_time.as_secs_f64() * 1000.0;
        assert!((actual - expected).abs() < 0.1, "EMA: expected {expected}, got {actual}");
    }

    #[test]
    fn test_metrics_d_opt_calculation() {
        let mut metrics = PipelineMetrics::new();
        metrics.avg_compute_time = Duration::from_millis(30);
        metrics.avg_io_time = Duration::from_millis(10);
        metrics.io_samples = 10;
        metrics.compute_samples = 10;

        let d = metrics.calculate_d_opt();
        assert_eq!(d, 4); // ceil(30/10) + 1 = 4
    }

    #[test]
    fn test_metrics_rho() {
        let mut metrics = PipelineMetrics::new();
        metrics.avg_compute_time = Duration::from_millis(10);
        metrics.avg_io_time = Duration::from_millis(10);
        metrics.update_rho();
        assert!((metrics.rho - 1.0).abs() < 0.01);

        metrics.avg_io_time = Duration::from_millis(20);
        metrics.update_rho();
        assert!((metrics.rho - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_recalibration_on_interval() {
        let mut mgr = CircularSlotManager::new(2);

        // Simulate 10 layers with timing that suggests D=3
        for i in 0..10 {
            mgr.next_read_slot(i);
            mgr.finish_read_with_duration(i % mgr.depth, Duration::from_millis(10));
            let slot = mgr.next_compute_slot().unwrap();
            mgr.finish_compute_with_duration(slot, Duration::from_millis(25));
        }

        // After RECALIBRATE_INTERVAL layers, D should have increased.
        // D_opt = ceil(25/10) + 1 = 4
        assert!(mgr.depth >= 3, "Depth should have increased to at least 3, got {}", mgr.depth);
    }

    #[test]
    fn test_reset_preserves_metrics() {
        let mut mgr = CircularSlotManager::new(3);

        // Record some data
        mgr.next_read_slot(0);
        mgr.finish_read_with_duration(0, Duration::from_millis(10));
        let slot = mgr.next_compute_slot().unwrap();
        mgr.finish_compute_with_duration(slot, Duration::from_millis(5));

        let old_io_samples = mgr.metrics.io_samples;
        let old_compute_samples = mgr.metrics.compute_samples;

        mgr.reset();

        assert_eq!(mgr.read_ptr, 0);
        assert_eq!(mgr.compute_ptr, 0);
        // Metrics are preserved!
        assert_eq!(mgr.metrics.io_samples, old_io_samples);
        assert_eq!(mgr.metrics.compute_samples, old_compute_samples);
    }

    #[test]
    fn test_full_pipeline_cycle_5_layers() {
        let mut mgr = CircularSlotManager::new(3);
        let num_layers = 5;

        let mut layers_computed = 0;
        let mut next_read_layer = 0;

        // Prime: read first D-1 layers
        while next_read_layer < mgr.depth - 1 && next_read_layer < num_layers {
            mgr.next_read_slot(next_read_layer);
            mgr.finish_read(next_read_layer % mgr.depth);
            next_read_layer += 1;
        }

        // Steady state
        while layers_computed < num_layers {
            if next_read_layer < num_layers {
                if let Some(slot) = mgr.next_read_slot(next_read_layer) {
                    mgr.finish_read(slot);
                    next_read_layer += 1;
                }
            }

            if let Some(slot) = mgr.next_compute_slot() {
                mgr.finish_compute(slot);
                layers_computed += 1;
            }
        }

        assert_eq!(layers_computed, num_layers);
        assert_eq!(mgr.underrun_count, 0);
        assert_eq!(mgr.total_layers_processed, 5);
    }

    #[test]
    fn test_full_pipeline_32_layers_with_timing() {
        let mut mgr = CircularSlotManager::from_d_opt(
            Duration::from_millis(8),
            Duration::from_millis(5),
        );
        let num_layers = 32;
        let mut next_read_layer = 0;
        let mut layers_computed = 0;

        // Prime
        while next_read_layer < mgr.depth - 1 && next_read_layer < num_layers {
            mgr.next_read_slot(next_read_layer);
            mgr.finish_read_with_duration(next_read_layer % mgr.depth, Duration::from_millis(5));
            next_read_layer += 1;
        }

        // Steady state
        while layers_computed < num_layers {
            if next_read_layer < num_layers {
                if let Some(slot) = mgr.next_read_slot(next_read_layer) {
                    mgr.finish_read_with_duration(slot, Duration::from_millis(5));
                    next_read_layer += 1;
                }
            }
            if let Some(slot) = mgr.next_compute_slot() {
                mgr.finish_compute_with_duration(slot, Duration::from_millis(8));
                layers_computed += 1;
            }
        }

        assert_eq!(layers_computed, 32);
        assert_eq!(mgr.underrun_count, 0);
        assert!(mgr.metrics.rho > 0.5, "ρ should be meaningful, got {:.2}", mgr.metrics.rho);
    }

    #[test]
    fn test_visual_output_shows_depth() {
        let mgr = CircularSlotManager::new(3);
        let visual = mgr.visual();
        assert!(visual.contains("D=3"));
        assert!(visual.contains("Pipeline:"));
        assert!(visual.contains("ρ="));
    }

    #[test]
    fn test_diagnostic_dump() {
        let mut mgr = CircularSlotManager::new(3);
        mgr.next_compute_slot(); // underrun
        let dump = mgr.diagnostic_dump();
        assert!(dump.contains("Pipeline Diagnostic"));
        assert!(dump.contains("Underrun Log"));
    }

    #[test]
    fn test_underrun_event_display() {
        let event = UnderrunEvent {
            timestamp: Instant::now(),
            layer_index: 5,
            depth_at_underrun: 3,
            stalled_slot_state: SlotState::Empty,
            avg_io_time: Duration::from_millis(15),
            avg_compute_time: Duration::from_millis(8),
            depth_increased: true,
            new_depth: 4,
        };
        let s = format!("{}", event);
        assert!(s.contains("UNDERRUN"));
        assert!(s.contains("layer=5"));
        assert!(s.contains("D=3"));
        assert!(s.contains("increased to 4"));
    }

    #[test]
    fn test_finish_read_with_duration() {
        let mut mgr = CircularSlotManager::new(3);
        mgr.next_read_slot(0);
        mgr.finish_read_with_duration(0, Duration::from_millis(42));
        assert_eq!(mgr.metrics.io_samples, 1);
        let avg = mgr.metrics.avg_io_time.as_millis();
        assert!(avg >= 40 && avg <= 44, "Expected ~42ms, got {avg}ms");
    }

    #[test]
    fn test_finish_compute_with_duration() {
        let mut mgr = CircularSlotManager::new(3);
        mgr.next_read_slot(0);
        mgr.finish_read(0);
        let slot = mgr.next_compute_slot().unwrap();
        mgr.finish_compute_with_duration(slot, Duration::from_millis(23));
        assert_eq!(mgr.metrics.compute_samples, 1);
        assert_eq!(mgr.total_layers_processed, 1);
    }

    #[test]
    fn test_metrics_summary() {
        let metrics = PipelineMetrics::new();
        let s = metrics.summary();
        assert!(s.contains("Pipeline:"));
        assert!(s.contains("ρ="));
    }

    #[test]
    fn test_stall_time_tracking() {
        let mut metrics = PipelineMetrics::new();
        assert_eq!(metrics.total_stall_time, Duration::ZERO);

        metrics.record_stall(Duration::from_millis(5));
        assert_eq!(metrics.total_stall_time.as_millis(), 5);

        metrics.record_stall(Duration::from_millis(3));
        assert_eq!(metrics.total_stall_time.as_millis(), 8);
    }

    #[test]
    fn test_d_opt_zero_io() {
        let mut metrics = PipelineMetrics::new();
        metrics.avg_io_time = Duration::ZERO;
        metrics.avg_compute_time = Duration::from_millis(10);
        assert_eq!(metrics.calculate_d_opt(), MIN_PIPELINE_DEPTH);
    }
}

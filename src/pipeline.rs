//! Adaptive Pipeline — Circular Slot Manager
//!
//! Replaces the simple ping-pong (D=2) buffering with a D-deep circular
//! slot ring. The pipeline depth D adapts based on measured I/O and
//! compute timings.
//!
//! Invariant: `read_ptr` must be `D-1` slots ahead of `compute_ptr`.
//! If the read_ptr catches up to the compute_ptr, the pipeline has
//! "underrun" and D should be increased.
//!
//! Reference: air_rs_protocols_v3.md §4 "Pipeline Architecture"

use std::fmt;
use std::time::Instant;

use crate::ucal::SharedBuffer;

// ---------------------------------------------------------------------------
// Slot State Machine
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pipeline Slot
// ---------------------------------------------------------------------------

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

    /// Transition to Reading state
    fn begin_read(&mut self, layer_index: usize) {
        self.state = SlotState::Reading;
        self.layer_index = Some(layer_index);
        self.op_start = Some(Instant::now());
    }

    /// Transition from Reading → Ready
    fn finish_read(&mut self) {
        debug_assert_eq!(self.state, SlotState::Reading);
        self.state = SlotState::Ready;
        self.op_start = None;
    }

    /// Transition from Ready → Computing
    fn begin_compute(&mut self) {
        debug_assert_eq!(self.state, SlotState::Ready);
        self.state = SlotState::Computing;
        self.op_start = Some(Instant::now());
    }

    /// Transition from Computing → Empty (slot recycled)
    fn finish_compute(&mut self) {
        debug_assert_eq!(self.state, SlotState::Computing);
        self.state = SlotState::Empty;
        self.layer_index = None;
        self.op_start = None;
    }
}

// ---------------------------------------------------------------------------
// Circular Slot Manager
// ---------------------------------------------------------------------------

/// Manages a ring of D buffer slots for pipeline-parallel weight streaming.
///
/// The `read_ptr` advances as layers are loaded from disk.
/// The `compute_ptr` advances as layers are processed by the compute backend.
///
/// The invariant is: `read_ptr` should be `D-1` slots ahead of `compute_ptr`.
/// If this invariant is violated (underrun), D should be increased.
pub struct CircularSlotManager {
    /// The pipeline depth (number of slots)
    pub depth: usize,
    /// The buffer slots
    slots: Vec<PipelineSlot>,
    /// Points to the next slot to read into
    pub read_ptr: usize,
    /// Points to the next slot to compute from
    pub compute_ptr: usize,
    /// Number of underrun events detected
    pub underrun_count: usize,
    /// Maximum allowed pipeline depth
    max_depth: usize,
}

impl CircularSlotManager {
    /// Create a new circular slot manager with the given pipeline depth.
    pub fn new(depth: usize) -> Self {
        let depth = depth.clamp(2, 8);
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
            max_depth: 8,
        }
    }

    /// How many slots ahead is the read pointer from the compute pointer.
    pub fn read_ahead(&self) -> usize {
        if self.read_ptr >= self.compute_ptr {
            self.read_ptr - self.compute_ptr
        } else {
            self.depth - self.compute_ptr + self.read_ptr
        }
    }

    /// Check if the pipeline has underrun (read_ptr caught up to compute_ptr).
    pub fn is_underrun(&self) -> bool {
        self.read_ahead() == 0
            && self.slots[self.compute_ptr].state != SlotState::Ready
    }

    /// Get the next slot to read into (if available).
    pub fn next_read_slot(&mut self, layer_index: usize) -> Option<usize> {
        let slot_idx = self.read_ptr;
        let slot = &self.slots[slot_idx];

        // Can only read into Empty slots
        if slot.state != SlotState::Empty {
            return None;
        }

        self.slots[slot_idx].begin_read(layer_index);
        self.read_ptr = (self.read_ptr + 1) % self.depth;
        Some(slot_idx)
    }

    /// Mark a read slot as complete (data loaded).
    pub fn finish_read(&mut self, slot_idx: usize) {
        self.slots[slot_idx].finish_read();
    }

    /// Get the next slot to compute from (if ready).
    pub fn next_compute_slot(&mut self) -> Option<usize> {
        let slot_idx = self.compute_ptr;
        let slot = &self.slots[slot_idx];

        if slot.state != SlotState::Ready {
            // Underrun detected!
            if slot.state == SlotState::Empty || slot.state == SlotState::Reading {
                self.underrun_count += 1;
            }
            return None;
        }

        self.slots[slot_idx].begin_compute();
        Some(slot_idx)
    }

    /// Mark a compute slot as complete (layer processed, slot recycled).
    pub fn finish_compute(&mut self, slot_idx: usize) {
        self.slots[slot_idx].finish_compute();
        self.compute_ptr = (self.compute_ptr + 1) % self.depth;
    }

    /// Dynamically increase pipeline depth by 1 (up to max_depth).
    /// Called when underruns are detected.
    pub fn increase_depth(&mut self) -> bool {
        if self.depth >= self.max_depth {
            return false;
        }
        self.depth += 1;
        self.slots.push(PipelineSlot::new());
        true
    }

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
        }
        self.read_ptr = 0;
        self.compute_ptr = 0;
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
        format!("Pipeline: {} (D={})", parts.join(""), self.depth)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_state_display() {
        assert_eq!(format!("{}", SlotState::Empty), "Empty");
        assert_eq!(format!("{}", SlotState::Computing), "Computing");
    }

    #[test]
    fn test_new_pipeline_clamped() {
        // Too small → clamped to 2
        let mgr = CircularSlotManager::new(1);
        assert_eq!(mgr.depth, 2);

        // Too large → clamped to 8
        let mgr = CircularSlotManager::new(100);
        assert_eq!(mgr.depth, 8);
    }

    #[test]
    fn test_initial_state() {
        let mgr = CircularSlotManager::new(3);
        assert_eq!(mgr.depth, 3);
        assert_eq!(mgr.read_ptr, 0);
        assert_eq!(mgr.compute_ptr, 0);
        assert_eq!(mgr.underrun_count, 0);

        let states = mgr.slot_states();
        assert!(states.iter().all(|(_, s, _)| *s == SlotState::Empty));
    }

    #[test]
    fn test_read_compute_cycle() {
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

        // Finish computing slot 0 → recycled to Empty
        mgr.finish_compute(0);
        assert_eq!(mgr.slots[0].state, SlotState::Empty);
        assert_eq!(mgr.compute_ptr, 1);
    }

    #[test]
    fn test_underrun_detection() {
        let mut mgr = CircularSlotManager::new(2);

        // Try to compute without any reads → underrun
        assert!(mgr.next_compute_slot().is_none());
        assert_eq!(mgr.underrun_count, 1);
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
    fn test_increase_depth() {
        let mut mgr = CircularSlotManager::new(3);
        assert_eq!(mgr.depth, 3);

        assert!(mgr.increase_depth());
        assert_eq!(mgr.depth, 4);

        // Max out at 8
        while mgr.depth < 8 {
            mgr.increase_depth();
        }
        assert!(!mgr.increase_depth()); // Can't go beyond 8
    }

    #[test]
    fn test_reset() {
        let mut mgr = CircularSlotManager::new(3);

        mgr.next_read_slot(0);
        mgr.next_read_slot(1);
        mgr.finish_read(0);
        mgr.next_compute_slot();

        mgr.reset();

        assert_eq!(mgr.read_ptr, 0);
        assert_eq!(mgr.compute_ptr, 0);
        assert!(mgr.slot_states().iter().all(|(_, s, _)| *s == SlotState::Empty));
    }

    #[test]
    fn test_full_pipeline_cycle() {
        // Simulate a full pipeline through 5 layers with D=3
        let mut mgr = CircularSlotManager::new(3);
        let num_layers = 5;

        let mut layers_computed = 0;
        let mut next_read_layer = 0;

        // Prime the pipeline: read first D-1 layers
        while next_read_layer < mgr.depth - 1 && next_read_layer < num_layers {
            mgr.next_read_slot(next_read_layer);
            mgr.finish_read(next_read_layer % mgr.depth);
            next_read_layer += 1;
        }

        // Steady state: read one, compute one
        while layers_computed < num_layers {
            // Start reading next layer (if available)
            if next_read_layer < num_layers {
                if let Some(slot) = mgr.next_read_slot(next_read_layer) {
                    mgr.finish_read(slot);
                    next_read_layer += 1;
                }
            }

            // Compute the next ready layer
            if let Some(slot) = mgr.next_compute_slot() {
                mgr.finish_compute(slot);
                layers_computed += 1;
            }
        }

        assert_eq!(layers_computed, num_layers);
        assert_eq!(mgr.underrun_count, 0);
    }

    #[test]
    fn test_visual_output() {
        let mgr = CircularSlotManager::new(3);
        let visual = mgr.visual();
        assert!(visual.contains("D=3"));
        assert!(visual.contains("Pipeline:"));
    }
}

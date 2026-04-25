//! I/O Engine — priority request queue for tensor loading.
//!
//! The `IoEngine` manages pending read operations for staging tensors
//! from storage into RAM. It maintains a priority queue where higher-class
//! tensors (A/B) jump ahead of lower-class ones (C/D).
//!
//! The engine is storage-agnostic — it builds `IoRequest`s that the caller
//! feeds to `dyn StorageHal`. Completions are reported back via `complete()`.

use super::types::{TensorClass, TensorId};
use std::time::{Duration, Instant};

// ── IoRequest ────────────────────────────────────────────────────────────

/// A pending I/O read request.
#[derive(Debug, Clone)]
pub struct IoRequest {
    /// Tensor to load.
    pub tensor_id: TensorId,
    /// Byte offset within the GGUF file.
    pub file_offset: u64,
    /// Number of bytes to read.
    pub size: usize,
    /// Priority level (lower = higher priority).
    pub priority: IoPriority,
}

/// Priority levels for I/O requests (lower ordinal = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IoPriority {
    /// Immediate — blocking the inference pipeline (Class A).
    Critical = 0,
    /// High — needed within the prefetch window (Class B, current layer).
    High = 1,
    /// Normal — prefetch for upcoming layers (Class B, future layers).
    Normal = 2,
    /// Low — speculative or archival (Class C/D).
    Low = 3,
}

impl IoPriority {
    /// Derive I/O priority from tensor class and urgency.
    pub fn from_class(class: TensorClass, within_window: bool) -> Self {
        match class {
            TensorClass::A => IoPriority::Critical,
            TensorClass::B if within_window => IoPriority::High,
            TensorClass::B => IoPriority::Normal,
            TensorClass::C | TensorClass::D => IoPriority::Low,
        }
    }
}

// ── IoTicket ─────────────────────────────────────────────────────────────

/// Opaque handle for tracking a submitted I/O request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IoTicket(u64);

// ── IoCompletion ─────────────────────────────────────────────────────────

/// Result of a completed I/O read.
#[derive(Debug)]
pub struct IoCompletion {
    /// The original ticket.
    pub ticket: IoTicket,
    /// Tensor that was loaded.
    pub tensor_id: TensorId,
    /// Loaded bytes.
    pub data: Vec<u8>,
    /// Time spent reading.
    pub elapsed: Duration,
}

// ── PendingOp ────────────────────────────────────────────────────────────

/// Internal tracking for an in-flight I/O operation.
struct PendingOp {
    ticket: IoTicket,
    request: IoRequest,
    submitted_at: Instant,
}

// ── IoEngine ─────────────────────────────────────────────────────────────

/// Priority I/O request queue for tensor loading.
///
/// The engine queues requests and tracks them by ticket.
/// The actual I/O is performed externally (via `dyn StorageHal`) —
/// the engine only manages ordering and lifecycle.
pub struct IoEngine {
    /// Maximum number of concurrent in-flight requests.
    capacity: usize,
    /// Queue of pending requests (not yet submitted to storage).
    queue: Vec<IoRequest>,
    /// Currently in-flight operations.
    in_flight: Vec<PendingOp>,
    /// Completed operations waiting to be polled.
    completed: Vec<IoCompletion>,
    /// Monotonic ticket counter.
    next_ticket: u64,
}

impl IoEngine {
    /// Create a new I/O engine with the given concurrency capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            queue: Vec::new(),
            in_flight: Vec::new(),
            completed: Vec::new(),
            next_ticket: 0,
        }
    }

    /// Submit a read request to the queue.
    ///
    /// Returns a ticket for tracking. The request will be dispatched
    /// when there is capacity, in priority order.
    pub fn submit(&mut self, request: IoRequest) -> IoTicket {
        let ticket = IoTicket(self.next_ticket);
        self.next_ticket += 1;
        self.queue.push(request);
        // Sort queue by priority (lowest ordinal = highest priority = front)
        self.queue.sort_by_key(|r| r.priority);
        ticket
    }

    /// Dispatch queued requests up to the concurrency capacity.
    ///
    /// Returns the requests that should be submitted to `dyn StorageHal`.
    /// The caller should call `complete()` for each when the read finishes.
    pub fn dispatch(&mut self) -> Vec<(IoTicket, IoRequest)> {
        let mut dispatched = Vec::new();
        while self.in_flight.len() < self.capacity && !self.queue.is_empty() {
            let request = self.queue.remove(0);
            let ticket = IoTicket(self.next_ticket);
            self.next_ticket += 1;

            self.in_flight.push(PendingOp {
                ticket,
                request: request.clone(),
                submitted_at: Instant::now(),
            });
            dispatched.push((ticket, request));
        }
        dispatched
    }

    /// Report that a previously dispatched I/O operation has completed.
    pub fn complete(&mut self, ticket: IoTicket, data: Vec<u8>) {
        if let Some(pos) = self.in_flight.iter().position(|op| op.ticket == ticket) {
            let op = self.in_flight.remove(pos);
            self.completed.push(IoCompletion {
                ticket,
                tensor_id: op.request.tensor_id,
                data,
                elapsed: op.submitted_at.elapsed(),
            });
        }
    }

    /// Drain all completed operations.
    pub fn poll(&mut self) -> Vec<IoCompletion> {
        std::mem::take(&mut self.completed)
    }

    /// Number of requests waiting in the queue (not yet dispatched).
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Number of currently in-flight operations.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Number of completed operations waiting to be polled.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Total pending work (queued + in-flight).
    pub fn pending_count(&self) -> usize {
        self.queue.len() + self.in_flight.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(tensor_id: u32, priority: IoPriority) -> IoRequest {
        IoRequest {
            tensor_id: TensorId(tensor_id),
            file_offset: 0,
            size: 1024,
            priority,
        }
    }

    #[test]
    fn submit_and_dispatch_lifecycle() {
        let mut engine = IoEngine::new(2);

        let _ticket = engine.submit(make_request(0, IoPriority::Normal));
        assert_eq!(engine.queued_count(), 1);

        let dispatched = engine.dispatch();
        assert_eq!(dispatched.len(), 1);
        assert_eq!(engine.in_flight_count(), 1);
        assert_eq!(engine.queued_count(), 0);
    }

    #[test]
    fn priority_ordering() {
        let mut engine = IoEngine::new(10);

        // Submit in reverse priority order
        engine.submit(make_request(0, IoPriority::Low));
        engine.submit(make_request(1, IoPriority::Critical));
        engine.submit(make_request(2, IoPriority::Normal));
        engine.submit(make_request(3, IoPriority::High));

        let dispatched = engine.dispatch();
        let priorities: Vec<IoPriority> = dispatched.iter().map(|(_, r)| r.priority).collect();
        // Should be sorted: Critical, High, Normal, Low
        assert_eq!(priorities[0], IoPriority::Critical);
        assert_eq!(priorities[1], IoPriority::High);
        assert_eq!(priorities[2], IoPriority::Normal);
        assert_eq!(priorities[3], IoPriority::Low);
    }

    #[test]
    fn capacity_limit() {
        let mut engine = IoEngine::new(2);

        engine.submit(make_request(0, IoPriority::Normal));
        engine.submit(make_request(1, IoPriority::Normal));
        engine.submit(make_request(2, IoPriority::Normal));

        let dispatched = engine.dispatch();
        // Only 2 dispatched (capacity limit)
        assert_eq!(dispatched.len(), 2);
        assert_eq!(engine.in_flight_count(), 2);
        assert_eq!(engine.queued_count(), 1);
    }

    #[test]
    fn completion_tracking() {
        let mut engine = IoEngine::new(2);
        engine.submit(make_request(42, IoPriority::Normal));
        let dispatched = engine.dispatch();
        assert_eq!(dispatched.len(), 1);
        let (ticket, _) = &dispatched[0];

        // Complete the request
        engine.complete(*ticket, vec![0u8; 1024]);
        assert_eq!(engine.completed_count(), 1);
        assert_eq!(engine.in_flight_count(), 0);

        // Poll results
        let completions = engine.poll();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].tensor_id, TensorId(42));
        assert_eq!(completions[0].data.len(), 1024);

        // After poll, completed is drained
        assert_eq!(engine.completed_count(), 0);
    }

    #[test]
    fn io_priority_from_class() {
        assert_eq!(IoPriority::from_class(TensorClass::A, false), IoPriority::Critical);
        assert_eq!(IoPriority::from_class(TensorClass::B, true), IoPriority::High);
        assert_eq!(IoPriority::from_class(TensorClass::B, false), IoPriority::Normal);
        assert_eq!(IoPriority::from_class(TensorClass::C, false), IoPriority::Low);
        assert_eq!(IoPriority::from_class(TensorClass::D, true), IoPriority::Low);
    }
}

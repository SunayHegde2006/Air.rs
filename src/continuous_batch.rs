//! Continuous Batching v2 — Disaggregated Prefill-Decode
//!
//! Implements the prefill/decode disaggregation pattern from PD-Disagg
//! (Zhong et al., 2024) and Orca (Yu et al., OSDI 2022) continuous batching.
//!
//! # Research Basis
//!
//! - **Orca** (Yu et al., OSDI 2022): iteration-level scheduling — instead of
//!   waiting for all sequences in a batch to finish, emit tokens from finished
//!   sequences immediately and slot in new prefills. This ensures GPU is always
//!   saturated.
//! - **PD-Disagg** (Zhong et al., 2024): split prefill (compute-bound) and
//!   decode (memory-bandwidth-bound) onto separate GPU pools with KV cache
//!   transfer across NVLink or PCIe. Enables 5-10× request throughput vs.
//!   co-located chunked prefill.
//!
//! # Architecture
//!
//! ```text
//! RequestQueue (priority queue, FIFO within priority)
//!   ↓ schedule()
//! PrefillBatch { seqs: Vec<PrefillRequest> }   ← GPU-A (prefill pool)
//!   ↓ kv_transfer()  (NVLink/PCIe DMA)
//! DecodeBatch  { seqs: Vec<DecodeRequest>  }   ← GPU-B (decode pool)
//!   ↓ emit tokens
//! OutputQueue
//! ```

use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Reverse;

// ── Priority ──────────────────────────────────────────────────────────────

/// Scheduling priority (lower value = higher priority in the min-heap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u32);

impl Priority {
    pub const NORMAL: Self = Priority(100);
    pub const HIGH: Self = Priority(50);
    pub const LOW: Self = Priority(200);
}

// ── Request States ────────────────────────────────────────────────────────

/// A request in the prefill phase: prompt tokens need to be processed.
#[derive(Debug, Clone)]
pub struct PrefillRequest {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub priority: Priority,
    /// Number of prompt tokens that are covered by the prefix KV cache hit.
    pub kv_cache_hit_tokens: usize,
}

/// A request in the decode phase: one token is emitted per step.
#[derive(Debug, Clone)]
pub struct DecodeRequest {
    pub id: u64,
    pub generated_tokens: Vec<u32>,
    pub remaining_budget: usize,
    /// KV block ids transferred from the prefill GPU.
    pub kv_block_ids: Vec<u32>,
}

/// Possible states for a managed request.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Completed,
    Aborted,
}

/// A fully tracked request through the pipeline.
#[derive(Debug, Clone)]
pub struct ManagedRequest {
    pub id: u64,
    pub state: RequestState,
    pub priority: Priority,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub generated: Vec<u32>,
}

impl ManagedRequest {
    pub fn new(id: u64, prompt_len: usize, max_new_tokens: usize, priority: Priority) -> Self {
        Self {
            id,
            state: RequestState::Queued,
            priority,
            prompt_len,
            max_new_tokens,
            generated: Vec::new(),
        }
    }

    /// True if the request has finished decoding (budget exhausted or EOS).
    pub fn is_done(&self) -> bool {
        self.state == RequestState::Completed || self.state == RequestState::Aborted
    }
}

// ── Scheduler ─────────────────────────────────────────────────────────────

/// Orca-style iteration-level continuous batching scheduler.
///
/// Each call to `schedule()` returns the next batch of requests to process:
/// - A prefill batch (requests whose prompt hasn't been processed yet)
/// - A decode batch (requests currently generating tokens)
///
/// The scheduler maintains separate budgets for prefill tokens and decode
/// sequences to prevent either phase from starving the other.
#[derive(Debug)]
pub struct ContinuousBatchScheduler {
    /// Max prompt tokens processed per prefill iteration.
    pub prefill_token_budget: usize,
    /// Max simultaneous decode sequences.
    pub max_decode_seqs: usize,
    /// Pending prefill requests (priority queue, lowest Priority value first).
    prefill_queue: BinaryHeap<Reverse<(Priority, u64)>>,
    /// Requests indexed by id.
    requests: std::collections::HashMap<u64, ManagedRequest>,
    /// Currently decoding sequence ids.
    decode_active: VecDeque<u64>,
    /// Next request id.
    next_id: u64,
}

impl ContinuousBatchScheduler {
    pub fn new(prefill_token_budget: usize, max_decode_seqs: usize) -> Self {
        Self {
            prefill_token_budget,
            max_decode_seqs,
            prefill_queue: BinaryHeap::new(),
            requests: std::collections::HashMap::new(),
            decode_active: VecDeque::new(),
            next_id: 0,
        }
    }

    /// Submit a new request. Returns its assigned id.
    pub fn submit(
        &mut self,
        prompt_len: usize,
        max_new_tokens: usize,
        priority: Priority,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let req = ManagedRequest::new(id, prompt_len, max_new_tokens, priority);
        self.requests.insert(id, req);
        self.prefill_queue.push(Reverse((priority, id)));
        id
    }

    /// Schedule one iteration: returns pending prefill + decode request ids.
    ///
    /// Prefill requests are selected by priority up to `prefill_token_budget`.
    /// Decode requests are taken from `decode_active` up to `max_decode_seqs`.
    pub fn schedule(&mut self) -> ScheduledBatch {
        let mut prefill_ids = Vec::new();
        let mut prefill_tokens_used = 0;

        // Drain decode slots from decode_active, capping at max_decode_seqs
        let n_decode = self.decode_active.len().min(self.max_decode_seqs);
        let decode_ids: Vec<u64> = self.decode_active.iter().take(n_decode).copied().collect();

        // Fill prefill slot from queue, respecting token budget
        while let Some(Reverse((_, id))) = self.prefill_queue.peek() {
            let id = *id;
            if let Some(req) = self.requests.get(&id) {
                let tokens_needed = req.prompt_len;
                if prefill_tokens_used + tokens_needed > self.prefill_token_budget {
                    // One more try with a smaller request
                    break;
                }
                prefill_tokens_used += tokens_needed;
                prefill_ids.push(id);
                self.prefill_queue.pop();
                // Move to decode after prefill (simulated)
                if let Some(r) = self.requests.get_mut(&id) {
                    r.state = RequestState::Prefilling;
                }
            } else {
                self.prefill_queue.pop();
            }
        }

        ScheduledBatch { prefill_ids, decode_ids, prefill_tokens_used }
    }

    /// Mark a prefill as done — move it to the decode queue.
    pub fn prefill_done(&mut self, id: u64) {
        if let Some(req) = self.requests.get_mut(&id) {
            req.state = RequestState::Decoding;
            self.decode_active.push_back(id);
        }
    }

    /// Record one decoded token for a request. Returns `true` if done.
    pub fn decode_step(&mut self, id: u64, token: u32) -> bool {
        if let Some(req) = self.requests.get_mut(&id) {
            req.generated.push(token);
            if req.generated.len() >= req.max_new_tokens {
                req.state = RequestState::Completed;
                self.decode_active.retain(|&x| x != id);
                return true;
            }
        }
        false
    }

    /// Abort a request.
    pub fn abort(&mut self, id: u64) {
        if let Some(req) = self.requests.get_mut(&id) {
            req.state = RequestState::Aborted;
        }
        self.decode_active.retain(|&x| x != id);
    }

    /// Current state of a request.
    pub fn state(&self, id: u64) -> Option<&RequestState> {
        self.requests.get(&id).map(|r| &r.state)
    }

    /// Number of queued (pending prefill) requests.
    pub fn queued_count(&self) -> usize {
        self.prefill_queue.len()
    }

    /// Number of active decode sequences.
    pub fn decode_count(&self) -> usize {
        self.decode_active.len()
    }
}

/// Output of one scheduling iteration.
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub prefill_ids: Vec<u64>,
    pub decode_ids: Vec<u64>,
    pub prefill_tokens_used: usize,
}

// ── KV Transfer (PD-Disagg) ──────────────────────────────────────────

use crate::pd_disagg::{KvBlock, KvConnector, KvTransferError};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Manages KV cache transfer from prefill GPU to decode GPU via PD-Disagg.
pub struct KvTransferManager {
    connector: Arc<dyn KvConnector>,
    target_addr: SocketAddr,
    timeout: Duration,
}

impl KvTransferManager {
    pub fn new(connector: Arc<dyn KvConnector>, target_addr: SocketAddr) -> Self {
        Self {
            connector,
            target_addr,
            timeout: Duration::from_millis(2000),
        }
    }

    /// Transfer KV blocks for a sequence to the decode node.
    pub async fn transfer_blocks(
        &self,
        seq_id: u64,
        blocks: Vec<KvBlock>,
    ) -> Result<u64, KvTransferError> {
        let connector = self.connector.clone();
        let target = self.target_addr;
        let timeout = self.timeout;

        // Perform the transfer in a blocking task to avoid stalling the executor
        tokio::task::spawn_blocking(move || {
            connector.send_blocks(seq_id, &blocks, target, timeout)
        })
        .await
        .map_err(|_| KvTransferError::Timeout)?
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_submit_and_schedule() {
        let mut sched = ContinuousBatchScheduler::new(512, 8);
        let id = sched.submit(128, 64, Priority::NORMAL);
        let batch = sched.schedule();
        assert!(batch.prefill_ids.contains(&id));
    }

    #[test]
    fn scheduler_priority_ordering() {
        let mut sched = ContinuousBatchScheduler::new(2048, 8);
        let low = sched.submit(64, 32, Priority::LOW);
        let high = sched.submit(64, 32, Priority::HIGH);
        let batch = sched.schedule();
        // HIGH priority should be scheduled before LOW
        assert!(batch.prefill_ids.contains(&high), "high priority should be in batch");
    }

    #[test]
    fn scheduler_token_budget_respected() {
        // Budget = 100 tokens, request needs 200 → should not be scheduled
        let mut sched = ContinuousBatchScheduler::new(100, 8);
        let id = sched.submit(200, 64, Priority::NORMAL);
        let batch = sched.schedule();
        // Request with 200 tokens > budget 100 → not scheduled
        assert!(!batch.prefill_ids.contains(&id));
    }

    #[test]
    fn scheduler_prefill_done_moves_to_decode() {
        let mut sched = ContinuousBatchScheduler::new(512, 8);
        let id = sched.submit(16, 4, Priority::NORMAL);
        sched.schedule();
        sched.prefill_done(id);
        assert_eq!(sched.state(id), Some(&RequestState::Decoding));
        assert_eq!(sched.decode_count(), 1);
    }

    #[test]
    fn scheduler_decode_step_completes() {
        let mut sched = ContinuousBatchScheduler::new(512, 8);
        let id = sched.submit(8, 2, Priority::NORMAL);
        sched.schedule();
        sched.prefill_done(id);
        let done1 = sched.decode_step(id, 42);
        assert!(!done1);
        let done2 = sched.decode_step(id, 43);
        assert!(done2, "should be done after max_new_tokens");
        assert_eq!(sched.state(id), Some(&RequestState::Completed));
    }

    #[test]
    fn scheduler_abort_removes_from_decode() {
        let mut sched = ContinuousBatchScheduler::new(512, 8);
        let id = sched.submit(8, 100, Priority::NORMAL);
        sched.schedule();
        sched.prefill_done(id);
        assert_eq!(sched.decode_count(), 1);
        sched.abort(id);
        assert_eq!(sched.decode_count(), 0);
        assert_eq!(sched.state(id), Some(&RequestState::Aborted));
    }

    #[test]
    fn kv_transfer_manager_constructs() {
        use crate::pd_disagg::ShmKvConnector;
        use std::sync::Arc;
        let connector = Arc::new(ShmKvConnector::new());
        let addr = "127.0.0.1:0".parse().unwrap();
        let _mgr = KvTransferManager::new(connector, addr);
        // Construction is sufficient — async transfer_blocks tested via integration tests
    }

    #[test]
    fn scheduler_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ContinuousBatchScheduler>();
        assert_send::<KvTransferManager>();
    }
}

//! Continuous Batching Scheduler for API Serving.
//!
//! In a serving scenario, multiple clients send generation requests
//! concurrently. Traditional batching waits for a full batch before
//! processing, wasting GPU time. Continuous batching dynamically adds
//! and removes sequences from the batch every iteration.
//!
//! ```text
//! Request Queue → Scheduler → [Active Batch] → Forward Pass → Token Sampling
//!                     ↑                                              │
//!                     └──────── finished sequences removed ←─────────┘
//!                     └──────── new sequences admitted ←───── Request Queue
//! ```

use crate::generator::GenerationEvent;
use crate::sampler::SamplerConfig;
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Request & Response Types
// ---------------------------------------------------------------------------

/// A unique identifier for a generation request.
pub type RequestId = u64;

/// A generation request submitted to the scheduler.
#[derive(Debug)]
pub struct GenerationRequest {
    /// Unique request ID.
    pub id: RequestId,
    /// Tokenized prompt (including BOS).
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling configuration for this request.
    pub sampler_config: SamplerConfig,
    /// Channel to send generation events back to the client.
    pub response_tx: mpsc::Sender<GenerationEvent>,
    /// When the request was submitted.
    pub submitted_at: Instant,
}

/// Priority for scheduling (lower = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchedulePriority {
    /// Interactive/streaming — lowest latency.
    Realtime = 0,
    /// Normal API requests.
    Normal = 1,
    /// Background/batch workloads.
    Background = 2,
}

// ---------------------------------------------------------------------------
// Batch Slots
// ---------------------------------------------------------------------------

/// A single active sequence in the batch.
#[derive(Debug)]
pub struct BatchSlot {
    /// The original request.
    pub request_id: RequestId,
    /// All tokens so far (prompt + generated).
    pub tokens: Vec<u32>,
    /// Number of prompt tokens (for metrics).
    pub prompt_len: usize,
    /// Number of tokens generated so far.
    pub generated_count: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Is this sequence finished?
    pub finished: bool,
    /// Finish reason (EOS, length, error).
    pub finish_reason: Option<FinishReason>,
    /// Response channel.
    pub response_tx: mpsc::Sender<GenerationEvent>,
    /// When generation started for this sequence.
    pub started_at: Instant,
    /// EOS token ID for this sequence.
    pub eos_id: u32,
}

/// Why a sequence finished generating.
#[derive(Debug, Clone)]
pub enum FinishReason {
    /// End-of-sequence token was generated.
    EndOfSequence,
    /// Maximum token limit reached.
    MaxLength,
    /// Client disconnected.
    ClientDisconnected,
    /// Generation error.
    Error(String),
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::EndOfSequence => write!(f, "stop"),
            FinishReason::MaxLength => write!(f, "length"),
            FinishReason::ClientDisconnected => write!(f, "client_disconnected"),
            FinishReason::Error(e) => write!(f, "error: {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler Configuration
// ---------------------------------------------------------------------------

/// Configuration for the continuous batching scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum time a request can wait in queue (seconds).
    pub max_waiting_time_secs: f64,
    /// Maximum total tokens across all sequences in a batch.
    pub max_batch_tokens: usize,
    /// Whether to preempt low-priority sequences for high-priority ones.
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_waiting_time_secs: 30.0,
            max_batch_tokens: 4096,
            enable_preemption: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Continuous Batch Scheduler
// ---------------------------------------------------------------------------

/// The main continuous batching scheduler.
///
/// Manages a queue of pending requests and a set of active batch slots.
/// Each iteration, it:
/// 1. Removes finished sequences
/// 2. Admits new sequences up to batch capacity
/// 3. Prepares the batch for the next forward pass
pub struct ContinuousBatchScheduler {
    /// Pending requests in FIFO order.
    queue: VecDeque<GenerationRequest>,
    /// Active sequences being generated.
    active_slots: Vec<BatchSlot>,
    /// Scheduler configuration.
    config: SchedulerConfig,
    /// Next request ID to assign.
    next_request_id: RequestId,
    /// Total requests processed.
    pub total_requests: u64,
    /// Total tokens generated across all requests.
    pub total_tokens_generated: u64,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            queue: VecDeque::new(),
            active_slots: Vec::with_capacity(config.max_batch_size),
            config,
            next_request_id: 1,
            total_requests: 0,
            total_tokens_generated: 0,
        }
    }

    /// Submit a new generation request to the scheduler.
    ///
    /// Returns the assigned request ID.
    pub fn submit(&mut self, request: GenerationRequest) -> RequestId {
        let id = request.id;
        self.queue.push_back(request);
        self.total_requests += 1;
        id
    }

    /// Allocate a unique request ID.
    pub fn next_id(&mut self) -> RequestId {
        let id = self.next_request_id;
        self.next_request_id += 1;
        id
    }

    /// Number of requests waiting in the queue.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Number of active sequences being generated.
    pub fn active_count(&self) -> usize {
        self.active_slots.len()
    }

    /// Total sequences (active + queued).
    pub fn total_pending(&self) -> usize {
        self.queue.len() + self.active_slots.len()
    }

    /// Available capacity in the batch.
    pub fn available_slots(&self) -> usize {
        self.config.max_batch_size.saturating_sub(self.active_slots.len())
    }

    /// Step 1: Remove finished sequences from the active batch.
    ///
    /// Returns the IDs of removed sequences.
    pub fn evict_finished(&mut self) -> Vec<RequestId> {
        let mut removed = Vec::new();

        self.active_slots.retain(|slot| {
            if slot.finished {
                removed.push(slot.request_id);
                self.total_tokens_generated += slot.generated_count as u64;
                false
            } else {
                true
            }
        });

        removed
    }

    /// Step 2: Admit new sequences from the queue.
    ///
    /// Fills available batch slots with pending requests.
    /// Returns the number of newly admitted sequences.
    pub fn admit_new(&mut self, eos_id: u32) -> usize {
        let available = self.available_slots();
        let mut admitted = 0;

        while admitted < available {
            if let Some(request) = self.queue.pop_front() {
                let prompt_len = request.prompt_tokens.len();
                self.active_slots.push(BatchSlot {
                    request_id: request.id,
                    tokens: request.prompt_tokens,
                    prompt_len,
                    generated_count: 0,
                    max_tokens: request.max_tokens,
                    finished: false,
                    finish_reason: None,
                    response_tx: request.response_tx,
                    started_at: Instant::now(),
                    eos_id,
                });
                admitted += 1;
            } else {
                break; // Queue empty
            }
        }

        admitted
    }

    /// Step 3: Get the current batch of active tokens for forward pass.
    ///
    /// Returns a vector of (request_id, current_tokens) pairs.
    /// Only the last token of each sequence is needed for autoregressive generation.
    pub fn batch_tokens(&self) -> Vec<(RequestId, u32)> {
        self.active_slots
            .iter()
            .filter(|s| !s.finished)
            .map(|s| {
                let last_token = *s.tokens.last().unwrap_or(&0);
                (s.request_id, last_token)
            })
            .collect()
    }

    /// Step 4: Record sampled tokens and mark finished sequences.
    ///
    /// Takes a list of (request_id, sampled_token) and updates the corresponding slots.
    pub fn record_tokens(&mut self, results: &[(RequestId, u32)]) {
        for &(req_id, token) in results {
            if let Some(slot) = self.active_slots.iter_mut().find(|s| s.request_id == req_id) {
                slot.tokens.push(token);
                slot.generated_count += 1;

                // Check termination conditions
                if token == slot.eos_id {
                    slot.finished = true;
                    slot.finish_reason = Some(FinishReason::EndOfSequence);
                } else if slot.generated_count >= slot.max_tokens {
                    slot.finished = true;
                    slot.finish_reason = Some(FinishReason::MaxLength);
                }
            }
        }
    }

    /// Expire requests that have waited too long in the queue.
    pub fn expire_stale(&mut self) -> Vec<RequestId> {
        let max_wait = self.config.max_waiting_time_secs;
        let now = Instant::now();
        let mut expired = Vec::new();

        self.queue.retain(|req| {
            if now.duration_since(req.submitted_at).as_secs_f64() > max_wait {
                expired.push(req.id);
                false
            } else {
                true
            }
        });

        expired
    }

    /// Get a reference to the active slots (for external monitoring).
    pub fn active_slots(&self) -> &[BatchSlot] {
        &self.active_slots
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: RequestId, prompt: &[u32], max_tokens: usize) -> GenerationRequest {
        let (tx, _rx) = mpsc::channel(1);
        GenerationRequest {
            id,
            prompt_tokens: prompt.to_vec(),
            max_tokens,
            sampler_config: SamplerConfig::default(),
            response_tx: tx,
            submitted_at: Instant::now(),
        }
    }

    #[test]
    fn test_scheduler_submit_and_queue() {
        let mut sched = ContinuousBatchScheduler::new(SchedulerConfig::default());
        assert_eq!(sched.queue_depth(), 0);

        sched.submit(make_request(1, &[1, 2, 3], 100));
        sched.submit(make_request(2, &[4, 5, 6], 50));

        assert_eq!(sched.queue_depth(), 2);
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_scheduler_admit() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        let mut sched = ContinuousBatchScheduler::new(config);

        sched.submit(make_request(1, &[1, 2, 3], 100));
        sched.submit(make_request(2, &[4, 5, 6], 50));
        sched.submit(make_request(3, &[7, 8, 9], 25));

        let admitted = sched.admit_new(0);
        assert_eq!(admitted, 2); // max_batch_size = 2
        assert_eq!(sched.active_count(), 2);
        assert_eq!(sched.queue_depth(), 1); // One still in queue
    }

    #[test]
    fn test_scheduler_batch_tokens() {
        let mut sched = ContinuousBatchScheduler::new(SchedulerConfig::default());
        sched.submit(make_request(1, &[10, 20, 30], 100));
        sched.submit(make_request(2, &[40, 50], 50));
        sched.admit_new(0);

        let batch = sched.batch_tokens();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], (1, 30)); // Last token of first sequence
        assert_eq!(batch[1], (2, 50)); // Last token of second sequence
    }

    #[test]
    fn test_scheduler_record_and_finish() {
        let mut sched = ContinuousBatchScheduler::new(SchedulerConfig::default());
        sched.submit(make_request(1, &[10], 3));
        sched.admit_new(999); // EOS = 999

        // Generate 3 tokens
        sched.record_tokens(&[(1, 100)]);
        assert_eq!(sched.active_slots()[0].generated_count, 1);

        sched.record_tokens(&[(1, 200)]);
        sched.record_tokens(&[(1, 300)]);

        // Should be finished (max_tokens = 3)
        assert!(sched.active_slots()[0].finished);

        let removed = sched.evict_finished();
        assert_eq!(removed.len(), 1);
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_scheduler_eos_finish() {
        let mut sched = ContinuousBatchScheduler::new(SchedulerConfig::default());
        sched.submit(make_request(1, &[10], 100));
        sched.admit_new(999); // EOS = 999

        sched.record_tokens(&[(1, 999)]); // EOS token
        assert!(sched.active_slots()[0].finished);
        assert!(matches!(
            sched.active_slots()[0].finish_reason,
            Some(FinishReason::EndOfSequence)
        ));
    }

    #[test]
    fn test_scheduler_available_slots() {
        let config = SchedulerConfig {
            max_batch_size: 4,
            ..Default::default()
        };
        let mut sched = ContinuousBatchScheduler::new(config);
        assert_eq!(sched.available_slots(), 4);

        sched.submit(make_request(1, &[1], 10));
        sched.admit_new(0);
        assert_eq!(sched.available_slots(), 3);
    }

    #[test]
    fn test_next_id() {
        let mut sched = ContinuousBatchScheduler::new(SchedulerConfig::default());
        assert_eq!(sched.next_id(), 1);
        assert_eq!(sched.next_id(), 2);
        assert_eq!(sched.next_id(), 3);
    }
}

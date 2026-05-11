//! OpenAI-Compatible REST API — v0.5.0
//!
//! Implements the OpenAI Chat Completions API schema for drop-in
//! compatibility with OpenAI client libraries.
//!
//! # Endpoints (schema types defined here; routing in `src/api.rs`)
//!
//! - `POST /v1/chat/completions` — streaming SSE + batch
//! - `POST /v1/completions`      — legacy text completions
//! - `GET  /v1/models`           — list loaded models
//! - `POST /v1/embeddings`       — text embeddings
//! - `GET  /health`              — liveness probe
//! - `GET  /ready`               — readiness probe (model loaded)
//!
//! # Auth & Rate Limiting
//!
//! - `Authorization: Bearer <token>` header validated against `ApiKeyStore`
//! - Token-bucket rate limiter (`RateLimiter`) per API key
//! - Prometheus metrics: TTFT p50/p95/p99, TPS, queue depth

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Message Types ─────────────────────────────────────────────────────────

/// Role in a chat message.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// A single chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Tool call id (for tool responses).
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_call_id: None }
    }
}

// ── Chat Completions Request ───────────────────────────────────────────────

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub stream: bool,
    pub stop: Vec<String>,
    /// Optional LoRA adapter id.
    pub adapter_id: Option<String>,
}

impl ChatCompletionRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.is_empty() {
            return Err("model field is required".into());
        }
        if self.messages.is_empty() {
            return Err("messages array is empty".into());
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(format!("temperature {t} out of range [0, 2]"));
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("top_p {p} out of range [0, 1]"));
            }
        }
        let last_role = self.messages.last().map(|m| &m.role);
        if last_role == Some(&Role::Assistant) {
            return Err("last message must not be from assistant (use it to prime)".into());
        }
        Ok(())
    }

    /// Extract the last user message content.
    pub fn last_user_message(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
    }

    /// Build the flat prompt string (simple concatenation; real impl uses chat template).
    pub fn to_prompt(&self) -> String {
        self.messages
            .iter()
            .map(|m| format!("{}: {}", m.role.as_str(), m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── Chat Completions Response ──────────────────────────────────────────────

/// Usage statistics in a completion response.
#[derive(Debug, Clone)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl UsageInfo {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion }
    }
}

/// A single completion choice.
#[derive(Debug, Clone)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Reason generation stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

/// `POST /v1/chat/completions` response body.
#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

impl ChatCompletionResponse {
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        content: impl Into<String>,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Self {
        let msg = ChatMessage::assistant(content);
        Self {
            id: id.into(),
            object: "chat.completion".into(),
            created: 0, // real: unix timestamp
            model: model.into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: msg,
                finish_reason: FinishReason::Stop,
            }],
            usage: UsageInfo::new(prompt_tokens, completion_tokens),
        }
    }
}

// ── Streaming Chunk ────────────────────────────────────────────────────────

/// A single SSE data chunk for streaming chat completions.
#[derive(Debug, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    /// Delta content for this chunk (empty string at end).
    pub delta_content: Option<String>,
    pub finish_reason: Option<FinishReason>,
}

impl ChatCompletionChunk {
    pub fn token(id: impl Into<String>, model: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: model.into(),
            delta_content: Some(token.into()),
            finish_reason: None,
        }
    }

    pub fn done(id: impl Into<String>, model: impl Into<String>, reason: FinishReason) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: model.into(),
            delta_content: None,
            finish_reason: Some(reason),
        }
    }

    /// Format as SSE line: `data: {...}\n\n`
    pub fn to_sse(&self) -> String {
        let content = self.delta_content.as_deref().unwrap_or("");
        let finish = self
            .finish_reason
            .as_ref()
            .map(|r| format!(r#","finish_reason":"{}""#, r.as_str()))
            .unwrap_or_default();
        format!(
            "data: {{\"id\":\"{}\",\"object\":\"{}\",\"model\":\"{}\",\"choices\":[{{\"delta\":{{\"content\":\"{content}\"}}{finish}}}]}}\n\n",
            self.id, self.object, self.model
        )
    }
}

// ── Models Response ────────────────────────────────────────────────────────

/// A model entry in `GET /v1/models`.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: u64,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".into(),
            owned_by: "air-rs".into(),
            created: 0,
        }
    }
}

/// `GET /v1/models` response.
#[derive(Debug, Clone)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl ModelsResponse {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self { object: "list".into(), data: models }
    }
}

// ── API Key Store ─────────────────────────────────────────────────────────

/// API key metadata.
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key: String,
    pub owner: String,
    /// Requests per second allowance.
    pub rps_limit: f32,
}

/// Simple in-memory API key store. In production, backed by a DB.
#[derive(Debug, Default)]
pub struct ApiKeyStore {
    keys: HashMap<String, ApiKey>,
}

impl ApiKeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, key: impl Into<String>, owner: impl Into<String>, rps: f32) {
        let k = key.into();
        self.keys.insert(k.clone(), ApiKey { key: k, owner: owner.into(), rps_limit: rps });
    }

    /// Validate a Bearer token. Returns `Some(ApiKey)` if valid.
    pub fn validate(&self, bearer: &str) -> Option<&ApiKey> {
        let token = bearer.strip_prefix("Bearer ").unwrap_or(bearer);
        self.keys.get(token)
    }

    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

// ── Token Bucket Rate Limiter ─────────────────────────────────────────────

/// Per-key token bucket rate limiter.
///
/// Tokens refill at `rate` per second. Each request consumes one token.
/// If no tokens available, the request is rejected immediately (no queuing).
#[derive(Debug)]
pub struct RateLimiter {
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f32,
    capacity: f32,
    rate: f32,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f32, rate: f32) -> Self {
        Self { tokens: capacity, capacity, rate, last_refill: Instant::now() }
    }

    fn try_consume(&mut self) -> bool {
        let elapsed = self.last_refill.elapsed().as_secs_f32();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_refill = Instant::now();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { buckets: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Attempt to consume one token for `key` with given `rps_limit`.
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn try_consume(&self, key: &str, rps_limit: f32) -> bool {
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(rps_limit * 5.0, rps_limit));
        bucket.try_consume()
    }
}

impl Default for RateLimiter {
    fn default() -> Self { Self::new() }
}

// ── Metrics ───────────────────────────────────────────────────────────────

/// Request latency histogram (for TTFT and TPS tracking).
#[derive(Debug, Default, Clone)]
pub struct LatencyHistogram {
    samples: Vec<f32>,
}

impl LatencyHistogram {
    pub fn record(&mut self, latency_secs: f32) {
        self.samples.push(latency_secs);
    }

    /// p-th percentile (0..=100).
    pub fn percentile(&self, p: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn p50(&self) -> f32 { self.percentile(50.0) }
    pub fn p95(&self) -> f32 { self.percentile(95.0) }
    pub fn p99(&self) -> f32 { self.percentile(99.0) }
    pub fn count(&self) -> usize { self.samples.len() }
}

/// Prometheus-compatible metrics exporter (text format).
#[derive(Debug, Default)]
pub struct PrometheusMetrics {
    pub ttft: LatencyHistogram,
    pub tps: Vec<f32>,
    pub queue_depth: u32,
}

impl PrometheusMetrics {
    pub fn new() -> Self { Self::default() }

    /// Render in Prometheus text format.
    pub fn render(&self) -> String {
        let ttft_p50 = self.ttft.p50();
        let ttft_p95 = self.ttft.p95();
        let ttft_p99 = self.ttft.p99();
        let mean_tps = if self.tps.is_empty() {
            0.0f32
        } else {
            self.tps.iter().sum::<f32>() / self.tps.len() as f32
        };
        format!(
            "# HELP air_rs_ttft_seconds Time to first token latency\n\
             # TYPE air_rs_ttft_seconds summary\n\
             air_rs_ttft_seconds{{quantile=\"0.5\"}} {ttft_p50:.6}\n\
             air_rs_ttft_seconds{{quantile=\"0.95\"}} {ttft_p95:.6}\n\
             air_rs_ttft_seconds{{quantile=\"0.99\"}} {ttft_p99:.6}\n\
             # HELP air_rs_tokens_per_second Generation throughput\n\
             # TYPE air_rs_tokens_per_second gauge\n\
             air_rs_tokens_per_second {mean_tps:.2}\n\
             # HELP air_rs_queue_depth Current request queue depth\n\
             # TYPE air_rs_queue_depth gauge\n\
             air_rs_queue_depth {}\n",
            self.queue_depth
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_validates_empty_model() {
        let req = ChatCompletionRequest {
            model: "".into(),
            messages: vec![ChatMessage::user("hi")],
            max_tokens: None, temperature: None, top_p: None, n: None,
            stream: false, stop: vec![], adapter_id: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn request_validates_empty_messages() {
        let req = ChatCompletionRequest {
            model: "llama".into(),
            messages: vec![],
            max_tokens: None, temperature: None, top_p: None, n: None,
            stream: false, stop: vec![], adapter_id: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn request_validates_temperature_range() {
        let make = |t: f32| ChatCompletionRequest {
            model: "llama".into(),
            messages: vec![ChatMessage::user("hi")],
            temperature: Some(t),
            max_tokens: None, top_p: None, n: None,
            stream: false, stop: vec![], adapter_id: None,
        };
        assert!(make(0.0).validate().is_ok());
        assert!(make(1.0).validate().is_ok());
        assert!(make(2.0).validate().is_ok());
        assert!(make(2.1).validate().is_err());
        assert!(make(-0.1).validate().is_err());
    }

    #[test]
    fn request_last_user_message() {
        let req = ChatCompletionRequest {
            model: "llama".into(),
            messages: vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello!"),
            ],
            max_tokens: None, temperature: None, top_p: None, n: None,
            stream: false, stop: vec![], adapter_id: None,
        };
        assert_eq!(req.last_user_message(), Some("Hello!"));
    }

    #[test]
    fn sse_chunk_format_correct() {
        let chunk = ChatCompletionChunk::token("id1", "llama", " world");
        let sse = chunk.to_sse();
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        assert!(sse.contains("world"));
    }

    #[test]
    fn sse_done_chunk_has_finish_reason() {
        let chunk = ChatCompletionChunk::done("id2", "llama", FinishReason::Stop);
        let sse = chunk.to_sse();
        assert!(sse.contains("stop"));
    }

    #[test]
    fn api_key_store_validates_bearer() {
        let mut store = ApiKeyStore::new();
        store.register("sk-test-123", "alice", 10.0);
        assert!(store.validate("Bearer sk-test-123").is_some());
        assert!(store.validate("sk-test-123").is_some());
        assert!(store.validate("Bearer wrong").is_none());
    }

    #[test]
    fn rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new();
        // High limit — should always allow
        assert!(limiter.try_consume("key1", 1000.0));
        assert!(limiter.try_consume("key1", 1000.0));
    }

    #[test]
    fn rate_limiter_blocks_when_exhausted() {
        let limiter = RateLimiter::new();
        // Limit=0.001 RPS but bucket starts with 0.005 tokens (5× limit)
        // After 5 requests, bucket empty → reject
        let rps = 0.001;
        let mut allowed = 0;
        for _ in 0..100 {
            if limiter.try_consume("tiny", rps) {
                allowed += 1;
            }
        }
        // Should allow ~5 (capacity = 5× rate), then reject
        assert!(allowed < 20, "rate limiter should throttle: allowed={allowed}");
    }

    #[test]
    fn latency_histogram_percentiles() {
        let mut h = LatencyHistogram::default();
        for i in 1..=100 {
            h.record(i as f32 * 0.01);
        }
        assert!((h.p50() - 0.50).abs() < 0.02, "p50={}", h.p50());
        assert!((h.p95() - 0.95).abs() < 0.02, "p95={}", h.p95());
        assert!((h.p99() - 0.99).abs() < 0.02, "p99={}", h.p99());
    }

    #[test]
    fn prometheus_render_contains_metrics() {
        let mut m = PrometheusMetrics::new();
        m.ttft.record(0.1);
        m.ttft.record(0.2);
        m.tps.push(120.0);
        m.queue_depth = 3;
        let text = m.render();
        assert!(text.contains("air_rs_ttft_seconds"));
        assert!(text.contains("air_rs_tokens_per_second"));
        assert!(text.contains("air_rs_queue_depth 3"));
    }

    #[test]
    fn models_response_list() {
        let resp = ModelsResponse::new(vec![
            ModelInfo::new("llama-3.2-3b"),
            ModelInfo::new("mistral-7b"),
        ]);
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.object, "list");
    }

    #[test]
    fn role_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let s = role.as_str();
            assert_eq!(Role::from_str(s).unwrap().as_str(), s);
        }
    }

    #[test]
    fn openai_api_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ApiKeyStore>();
        assert_send_sync::<RateLimiter>();
        assert_send_sync::<PrometheusMetrics>();
    }
}

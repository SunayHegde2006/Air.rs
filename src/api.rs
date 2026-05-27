//! OpenAI-Compatible REST API for Air.rs.
//!
//! Implements the core OpenAI chat completions spec:
//!   - `POST /v1/chat/completions` — non-streaming & streaming (SSE)
//!   - `GET  /v1/models`           — list available models
//!   - `GET  /health`              — liveness probe
//!
//! Wire-compatible with the OpenAI Python SDK & curl, so any client
//! that speaks the OpenAI protocol can swap in Air.rs as the backend.

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use futures_util::StreamExt;
use crate::dispatcher::Dispatcher;

// ---------------------------------------------------------------------------
// Request / Response types (OpenAI-compatible)
// ---------------------------------------------------------------------------

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
    /// Optional GBNF grammar for structured output.
    pub gbnf: Option<String>,
}

fn default_temperature() -> Option<f32> {
    Some(0.7)
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Non-streaming response.
#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub system_fingerprint: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

/// Token usage stats.
#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk (SSE `data: {...}` lines).
#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    pub system_fingerprint: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// `GET /v1/models` response.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// `GET /health` response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub requests_served: u64,
}

// ---------------------------------------------------------------------------
// OpenAI-format Error Responses
// ---------------------------------------------------------------------------

/// Standard error codes matching OpenAI's error taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiErrorCode {
    #[serde(rename = "invalid_request_error")]
    InvalidRequest,
    #[serde(rename = "model_not_found")]
    ModelNotFound,
    #[serde(rename = "rate_limit_exceeded")]
    RateLimitExceeded,
    #[serde(rename = "server_error")]
    ServerError,
}

impl ApiErrorCode {
    /// HTTP status code for this error category.
    pub fn http_status(&self) -> u16 {
        match self {
            ApiErrorCode::InvalidRequest => 400,
            ApiErrorCode::ModelNotFound => 404,
            ApiErrorCode::RateLimitExceeded => 429,
            ApiErrorCode::ServerError => 500,
        }
    }
}

/// OpenAI-compatible error response body.
///
/// Wire format:
/// ```json
/// {
///   "error": {
///     "message": "...",
///     "type": "invalid_request_error",
///     "param": null,
///     "code": "invalid_api_key"
///   }
/// }
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: ApiErrorCode,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    /// Create an invalid-request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: ApiErrorCode::InvalidRequest,
                param: None,
                code: None,
            },
        }
    }

    /// Create a model-not-found error.
    pub fn model_not_found(model: &str) -> Self {
        Self {
            error: ApiErrorDetail {
                message: format!("The model '{}' does not exist", model),
                error_type: ApiErrorCode::ModelNotFound,
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
        }
    }

    /// Create a server error.
    pub fn server_error(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: ApiErrorCode::ServerError,
                param: None,
                code: None,
            },
        }
    }

    /// HTTP status code for this error.
    pub fn status_code(&self) -> u16 {
        self.error.error_type.http_status()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::from_u16(self.status_code())
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

/// Validate a chat completion request, returning an ApiError on failure.
fn validate_request(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "'messages' must contain at least one message",
        ));
    }
    if req.model.is_empty() {
        return Err(ApiError::invalid_request(
            "'model' is required and cannot be empty",
        ));
    }
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(ApiError::invalid_request(format!(
                "'temperature' must be between 0 and 2, got {}",
                temp
            )));
        }
    }
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ApiError::invalid_request(format!(
                "'top_p' must be between 0 and 1, got {}",
                top_p
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

/// Server-wide state shared across request handlers.
pub struct ApiState {
    /// Currently loaded model name.
    pub model_name: String,
    /// Unix timestamp when the server started.
    pub start_time: u64,
    /// Total requests served (atomic counter).
    pub request_count: AtomicU64,
    /// Registered model entries (derived from dispatcher.list_models()).
    pub models: RwLock<Vec<ModelEntry>>,
    /// Inference dispatcher — decouples handlers from the engine (ADR-0003).
    pub dispatcher: Arc<dyn Dispatcher>,
}

impl ApiState {
    /// Create state with an explicit dispatcher (used for testing and ModelMux).
    pub fn with_dispatcher(model_name: String, dispatcher: Arc<dyn Dispatcher>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let models = dispatcher.list_models().into_iter().map(|id| ModelEntry {
            id,
            object: "model".to_string(),
            created: now,
            owned_by: "air-rs".to_string(),
        }).collect();

        Self {
            model_name,
            start_time: now,
            request_count: AtomicU64::new(0),
            models: RwLock::new(models),
            dispatcher,
        }
    }

    pub fn next_id(&self) -> String {
        let n = self.request_count.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-air-{:08x}", n)
    }

    pub fn now_unix(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// POST /v1/chat/completions — non-streaming response.
async fn chat_completions(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Validate request
    if let Err(err) = validate_request(&req) {
        return err.into_response();
    }

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // Delegate to SSE handler
        let stream = stream_completion(state.clone(), req).await;
        stream.into_response()
    } else {
        let id = state.next_id();
        let created = state.now_unix();
        
        let config = crate::dispatcher::GenerateConfig {
            model: req.model.clone(),
            prompt: req.messages.last().map(|m| m.content.clone()).unwrap_or_default(),
            max_tokens: req.max_tokens.unwrap_or(128),
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            stop: match req.stop {
                Some(StopCondition::Single(s)) => vec![s],
                Some(StopCondition::Multiple(v)) => v,
                None => vec![],
            },
            draft_model: None, 
            gbnf: req.gbnf.clone(),
        };

        let mut stream = state.dispatcher.generate(config);
        let mut content = String::new();
        let mut completion_tokens = 0;

        while let Some(res) = stream.next().await {
            match res {
                Ok(crate::dispatcher::TokenChunk::Token { text, .. }) => {
                    content.push_str(&text);
                    completion_tokens += 1;
                }
                Ok(crate::dispatcher::TokenChunk::Stop { .. }) => break,
                Err(e) => return ApiError::server_error(e.to_string()).into_response(),
            }
        }

        let prompt_tokens = estimate_tokens(&req.messages.last().map(|m| m.content.as_str()).unwrap_or(""));
        
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            system_fingerprint: "air-rs-prod".to_string(),
        };

        Json(resp).into_response()
    }
}

/// SSE streaming handler.
async fn stream_completion(
    state: Arc<ApiState>,
    req: ChatCompletionRequest,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let id = state.next_id();
    let created = state.now_unix();
    let model = req.model.clone();
    
    let config = crate::dispatcher::GenerateConfig {
        model: req.model.clone(),
        prompt: req.messages.last().map(|m| m.content.clone()).unwrap_or_default(),
        max_tokens: req.max_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        stop: match req.stop {
            Some(StopCondition::Single(s)) => vec![s],
            Some(StopCondition::Multiple(v)) => v,
            None => vec![],
        },
        draft_model: None,
        gbnf: req.gbnf.clone(),
    };

    tokio::spawn(async move {
        // Send initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
            system_fingerprint: "air-rs-prod".to_string(),
        };
        let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default()))).await;

        let mut stream = state.dispatcher.generate(config);
        while let Some(res) = stream.next().await {
            let chunk = match res {
                Ok(crate::dispatcher::TokenChunk::Token { text, .. }) => {
                    ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                        system_fingerprint: "air-rs-prod".to_string(),
                    }
                }
                Ok(crate::dispatcher::TokenChunk::Stop { finish_reason }) => {
                    ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta { role: None, content: None },
                            finish_reason: Some(finish_reason.as_str().to_string()),
                        }],
                        system_fingerprint: "air-rs-prod".to_string(),
                    }
                }
                Err(e) => {
                    // Send error as text and stop
                    let err_chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta { role: None, content: Some(format!("\nError: {}", e)) },
                            finish_reason: Some("error".to_string()),
                        }],
                        system_fingerprint: "air-rs-prod".to_string(),
                    };
                    let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&err_chunk).unwrap_or_default()))).await;
                    break;
                }
            };

            if tx.send(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))).await.is_err() {
                break;
            }
        }

        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// GET /v1/models
async fn list_models(State(state): State<Arc<ApiState>>) -> Json<ModelsResponse> {
    let models = state.models.read().await;
    Json(ModelsResponse {
        object: "list".to_string(),
        data: models.clone(),
    })
}

/// GET /health
async fn health(State(state): State<Arc<ApiState>>) -> Json<HealthResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: now.saturating_sub(state.start_time),
        requests_served: state.request_count.load(Ordering::Relaxed),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough token estimation (~4 chars per token, like GPT-family).
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

/// Create the router with an explicit dispatcher (ADR-0003).
///
/// Use this when wiring a real `InferenceGenerator` or `ModelMux`:
/// ```ignore
/// use std::sync::Arc;
/// use air_rs::scheduler::RequestOrchestrator;
/// use air_rs::api::create_router_with_dispatcher;
///
/// // Full initialization omitted for brevity
/// let dispatcher = Arc::new(RequestOrchestrator::new(config));
/// let app = create_router_with_dispatcher("llama-3-8b".into(), dispatcher);
/// ```
pub fn create_router_with_dispatcher(
    model_name: String,
    dispatcher: Arc<dyn Dispatcher>,
) -> Router {
    let state = Arc::new(ApiState::with_dispatcher(model_name, dispatcher));
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0); // (0+3)/4 = 0 integer division
        assert_eq!(estimate_tokens("hello world"), 3); // (11+3)/4 = 3
        assert_eq!(estimate_tokens("a"), 1);
    }

    #[test]
    fn test_api_state_next_id() {
        use crate::dispatcher::MockDispatcher;
        use std::sync::Arc;
        let dispatcher = Arc::new(MockDispatcher::new("test-model", vec!["hi"]));
        let state = ApiState::with_dispatcher("test-model".to_string(), dispatcher);
        let id1 = state.next_id();
        let id2 = state.next_id();
        assert!(id1.starts_with("chatcmpl-air-"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_api_state_model_name() {
        use crate::dispatcher::MockDispatcher;
        use std::sync::Arc;
        let dispatcher = Arc::new(MockDispatcher::new("llama-7b", vec![]));
        let state = ApiState::with_dispatcher("llama-7b".to_string(), dispatcher);
        assert_eq!(state.model_name, "llama-7b");
    }

    #[test]
    fn test_create_router_builds() {
        use crate::dispatcher::{Dispatcher, MockDispatcher};
        use std::sync::Arc;
        let dispatcher = Arc::new(MockDispatcher::new("air-rs", vec![])) as Arc<dyn Dispatcher>;
        let _router = create_router_with_dispatcher("air-rs".to_string(), dispatcher);
    }

    #[test]
    fn test_stop_condition_deserialise() {
        // Single string
        let json = r#""stop_word""#;
        let sc: StopCondition = serde_json::from_str(json).unwrap();
        matches!(sc, StopCondition::Single(s) if s == "stop_word");

        // Array
        let json = r#"["stop","end"]"#;
        let sc: StopCondition = serde_json::from_str(json).unwrap();
        matches!(sc, StopCondition::Multiple(v) if v.len() == 2);
    }

    #[test]
    fn test_chat_completion_response_serialise() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
            system_fingerprint: "air-rs-v0.1".to_string(),
        };

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["usage"]["total_tokens"], 6);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["system_fingerprint"], "air-rs-v0.1");
    }

    #[test]
    fn test_chunk_serialise_skips_none_fields() {
        let chunk = ChatCompletionChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some("word".to_string()),
                },
                finish_reason: None,
            }],
            system_fingerprint: "fp".to_string(),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        // Delta should NOT contain "role" when it's None
        assert!(!json.contains("\"role\""));
        assert!(json.contains("\"content\""));
    }

    #[test]
    fn test_health_response_structure() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 42,
            requests_served: 100,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["requests_served"], 100);
    }

    #[test]
    fn test_models_response_structure() {
        let resp = ModelsResponse {
            object: "list".to_string(),
            data: vec![ModelEntry {
                id: "llama-7b".to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: "air-rs".to_string(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "llama-7b");
    }

    // -----------------------------------------------------------------------
    // API error + validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_error_invalid_request_format() {
        let err = ApiError::invalid_request("Messages is empty");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["message"], "Messages is empty");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["param"].is_null());
    }

    #[test]
    fn test_api_error_model_not_found_format() {
        let err = ApiError::model_not_found("gpt-5");
        let json = serde_json::to_value(&err).unwrap();
        assert!(json["error"]["message"].as_str().unwrap().contains("gpt-5"));
        assert_eq!(json["error"]["type"], "model_not_found");
        assert_eq!(json["error"]["param"], "model");
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[test]
    fn test_api_error_http_status_codes() {
        assert_eq!(ApiErrorCode::InvalidRequest.http_status(), 400);
        assert_eq!(ApiErrorCode::ModelNotFound.http_status(), 404);
        assert_eq!(ApiErrorCode::RateLimitExceeded.http_status(), 429);
        assert_eq!(ApiErrorCode::ServerError.http_status(), 500);
    }

    #[test]
    fn test_validate_request_empty_messages() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            gbnf: None,
        };
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn test_validate_request_empty_model() {
        let req = ChatCompletionRequest {
            model: "".to_string(),
            messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            gbnf: None,
        };
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn test_validate_request_valid() {
        let req = ChatCompletionRequest {
            model: "air-rs".to_string(),
            messages: vec![Message { role: "user".to_string(), content: "hello".to_string() }],
            temperature: Some(1.0),
            top_p: Some(0.9),
            max_tokens: Some(100),
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            gbnf: None,
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn test_validate_temperature_out_of_range() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
            temperature: Some(3.0),
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            gbnf: None,
        };
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn test_validate_top_p_out_of_range() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
            temperature: None,
            top_p: Some(1.5),
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            gbnf: None,
        };
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn test_server_error_format() {
        let err = ApiError::server_error("Internal failure");
        assert_eq!(err.status_code(), 500);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
    }
}

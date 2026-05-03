//! Dispatcher trait — decouples HTTP handlers from the inference engine.
//!
//! `ApiState` holds an `Arc<dyn Dispatcher>`. Handlers call
//! `dispatcher.generate(config)` and get back a `BoxStream<TokenChunk>`.
//! Non-streaming responses call `.collect().await` on the same stream.
//!
//! Concrete implementations:
//! - `SingleModelDispatcher` — wraps a single `InferenceGenerator` (stub until ADR-0001/#7)
//! - `MockDispatcher`        — returns canned tokens; used in unit tests
//!
//! See ADR-0003.

use futures_util::stream::{self, BoxStream, StreamExt};
use anyhow::Result;

// ---------------------------------------------------------------------------
// Public value types
// ---------------------------------------------------------------------------

/// A single emitted unit from a generation stream.
#[derive(Debug, Clone)]
pub enum TokenChunk {
    /// One or more decoded text bytes.
    Token {
        /// Raw token id (may be 0 / unknown if not available).
        id: u32,
        /// Decoded UTF-8 fragment.
        text: String,
    },
    /// Stream terminated — reason included.
    Stop { finish_reason: FinishReason },
}

impl TokenChunk {
    /// Convenience: extract text fragment, empty string for Stop chunks.
    pub fn text(&self) -> &str {
        match self {
            TokenChunk::Token { text, .. } => text.as_str(),
            TokenChunk::Stop { .. } => "",
        }
    }

    pub fn is_stop(&self) -> bool {
        matches!(self, TokenChunk::Stop { .. })
    }
}

/// Why generation terminated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token emitted or stop sequence matched.
    Stop,
    /// `max_tokens` budget exhausted.
    Length,
    /// Inference error — message included.
    Error(String),
}

impl FinishReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop     => "stop",
            Self::Length   => "length",
            Self::Error(_) => "error",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parameters for a single generation request.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Target model identifier (for `ModelMux` routing).
    pub model: String,
    /// Fully rendered prompt string (after chat template application).
    pub prompt: String,
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Nucleus sampling mass (1.0 = no filtering).
    pub top_p: f32,
    /// Optional stop sequences (generation halts on first match).
    pub stop: Vec<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompt: String::new(),
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.95,
            stop: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher trait
// ---------------------------------------------------------------------------

/// Generates tokens for a given request.
///
/// One method, always a stream. Non-streaming callers use
/// `dispatcher.generate(cfg).collect::<Vec<_>>().await`.
///
/// Implementors: `SingleModelDispatcher`, `ModelMux` (future), `MockDispatcher`.
pub trait Dispatcher: Send + Sync {
    /// Start a generation stream.
    fn generate(&self, config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>>;

    /// Enumerate loaded model identifiers.
    fn list_models(&self) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// SingleModelDispatcher — stub (real impl after ADR-0001/#7 lands)
// ---------------------------------------------------------------------------

/// Routes every request to a single loaded model.
///
/// Currently a stub that emits placeholder tokens. The real implementation
/// will hold `Arc<InferenceGenerator>` (after ADR-0001 TransformerBlock trait
/// and ADR-0002 Device injection are complete — issues #7 and #8).
pub struct SingleModelDispatcher {
    model_name: String,
}

impl SingleModelDispatcher {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self { model_name: model_name.into() }
    }
}

impl Dispatcher for SingleModelDispatcher {
    fn generate(&self, config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>> {
        let model = self.model_name.clone();
        // Stub: echo prompt prefix as fake tokens, then stop.
        // Replace with InferenceGenerator::generate_stream() call (issue #7).
        let stub_text = format!("[{} stub] echo: {}", model, &config.prompt[..config.prompt.len().min(40)]);
        let words: Vec<String> = stub_text
            .split_whitespace()
            .take(config.max_tokens)
            .map(|w| format!("{} ", w))
            .collect();

        let chunks: Vec<Result<TokenChunk>> = words
            .into_iter()
            .map(|text| Ok(TokenChunk::Token { id: 0, text }))
            .chain(std::iter::once(Ok(TokenChunk::Stop { finish_reason: FinishReason::Stop })))
            .collect();

        Box::pin(stream::iter(chunks))
    }

    fn list_models(&self) -> Vec<String> {
        vec![self.model_name.clone()]
    }
}

// ---------------------------------------------------------------------------
// MockDispatcher — for unit tests
// ---------------------------------------------------------------------------

/// Returns canned token sequences without needing GPU or GGUF.
///
/// Used in handler unit tests and integration tests that need a Dispatcher
/// without a real model loaded.
#[cfg(test)]
pub struct MockDispatcher {
    pub tokens: Vec<String>,
    pub model: String,
}

#[cfg(test)]
impl MockDispatcher {
    pub fn new(model: &str, tokens: Vec<&str>) -> Self {
        Self { model: model.to_string(), tokens: tokens.into_iter().map(|s| s.to_string()).collect() }
    }

    pub fn empty(model: &str) -> Self {
        Self { model: model.to_string(), tokens: Vec::new() }
    }
}

#[cfg(test)]
impl Dispatcher for MockDispatcher {
    fn generate(&self, _config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>> {
        let mut chunks: Vec<Result<TokenChunk>> = self.tokens
            .iter()
            .map(|t| Ok(TokenChunk::Token { id: 0, text: t.clone() }))
            .collect();
        chunks.push(Ok(TokenChunk::Stop { finish_reason: FinishReason::Stop }));
        Box::pin(stream::iter(chunks))
    }

    fn list_models(&self) -> Vec<String> {
        vec![self.model.clone()]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn mock_dispatcher_emits_tokens() {
        let d = MockDispatcher::new("test-model", vec!["Hello", " world"]);
        let chunks: Vec<_> = d.generate(GenerateConfig::default())
            .collect::<Vec<_>>()
            .await;
        assert_eq!(chunks.len(), 3); // 2 tokens + 1 stop
        assert!(chunks[0].as_ref().unwrap().text() == "Hello");
        assert!(chunks[2].as_ref().unwrap().is_stop());
    }

    #[tokio::test]
    async fn mock_dispatcher_collect_text() {
        let d = MockDispatcher::new("m", vec!["foo", " bar"]);
        let text: String = d.generate(GenerateConfig::default())
            .filter_map(|r| async { r.ok() })
            .map(|c| c.text().to_string())
            .collect::<Vec<_>>()
            .await
            .join("");
        assert_eq!(text, "foo bar");
    }

    #[test]
    fn single_model_lits_models() {
        let d = SingleModelDispatcher::new("llama-70b");
        assert_eq!(d.list_models(), vec!["llama-70b"]);
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
    }

    #[test]
    fn generate_config_defaults() {
        let c = GenerateConfig::default();
        assert_eq!(c.max_tokens, 256);
        assert!((c.temperature - 0.8).abs() < 0.001);
    }
}

//! Dispatcher trait — decouples HTTP handlers from the inference engine.
//!
//! `ApiState` holds an `Arc<dyn Dispatcher>`. Handlers call
//! `dispatcher.generate(config)` and get back a `BoxStream<TokenChunk>`.
//! Non-streaming responses call `.collect().await` on the same stream.
//!
//! Concrete implementations:
//! - `RequestOrchestrator`  — production scheduler managing multiple inference sessions
//! - `MockDispatcher`        — returns canned tokens; used in unit tests
//!
//! See ADR-0003.

use futures_util::stream::{BoxStream, StreamExt};
use anyhow::Result;
use crate::gbnf::GbnfConstraint;

// ---------------------------------------------------------------------------
// Public value types
// ---------------------------------------------------------------------------

/// A single emitted unit from a generation stream.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Optional draft model path for speculative decoding.
    pub draft_model: Option<String>,
    /// Optional GBNF grammar constraint for structured generation.
    pub gbnf: Option<String>,
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
            draft_model: None,
            gbnf: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher trait
// ---------------------------------------------------------------------------

/// Generates tokens for a given request.
pub trait Dispatcher: Send + Sync {
    /// Start a generation stream.
    fn generate(&self, config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>>;

    /// Enumerate loaded model identifiers.
    fn list_models(&self) -> Vec<String>;
}

// ── DistributedDispatcher ──────────────────────────────────────────────────

pub struct DistributedDispatcher {
    pub local: std::sync::Arc<dyn Dispatcher>,
    pub comm: std::sync::Arc<dyn crate::distributed::Communicator>,
}

impl DistributedDispatcher {
    pub fn new(
        local: std::sync::Arc<dyn Dispatcher>,
        comm: std::sync::Arc<dyn crate::distributed::Communicator>,
    ) -> Self {
        Self { local, comm }
    }
}

impl Dispatcher for DistributedDispatcher {
    fn generate(&self, config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>> {
        let comm = self.comm.clone();
        let local = self.local.clone();
        
        if comm.rank() == 0 {
            let config_bytes = serde_json::to_vec(&config).unwrap();
            let comm_inner = comm.clone();
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            
            tokio::spawn(async move {
                for i in 1..comm_inner.world_size() {
                    if let Err(e) = comm_inner.send(i, &config_bytes).await {
                        let _ = tx.send(Err(anyhow::anyhow!("Failed to broadcast config: {}", e))).await;
                        return;
                    }
                }
                
                let mut stream = local.generate(config);
                while let Some(chunk_res) = stream.next().await {
                    if tx.send(chunk_res).await.is_err() { break; }
                }
            });
            
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        } else {
            let chunks = vec![Err(anyhow::anyhow!("generate() called on follower rank {}", comm.rank()))];
            Box::pin(futures_util::stream::iter(chunks))
        }
    }

    fn list_models(&self) -> Vec<String> {
        self.local.list_models()
    }
}

impl DistributedDispatcher {
    pub async fn serve_follower(&self) -> Result<()> {
        let comm = self.comm.clone();
        let local = self.local.clone();
        let rank = comm.rank();

        loop {
            let mut buf = vec![0u8; 4096];
            comm.recv(0, &mut buf).await.map_err(|e| anyhow::anyhow!("Follower rank {} failed to recv config: {}", rank, e))?;
            let trimmed = buf.split(|&b| b == 0).next().unwrap_or(&[]);
            let config: GenerateConfig = serde_json::from_slice(trimmed)?;
            
            let mut stream = local.generate(config);
            while let Some(chunk_res) = stream.next().await {
                if chunk_res?.is_stop() { break; }
            }
        }
    }
}

// ── MockDispatcher ────────────────────────────────────────────────────────

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
}

#[cfg(test)]
impl Dispatcher for MockDispatcher {
    fn generate(&self, _config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>> {
        let mut chunks: Vec<Result<TokenChunk>> = self.tokens
            .iter()
            .map(|t| Ok(TokenChunk::Token { id: 0, text: t.clone() }))
            .collect();
        chunks.push(Ok(TokenChunk::Stop { finish_reason: FinishReason::Stop }));
        Box::pin(futures_util::stream::iter(chunks))
    }

    fn list_models(&self) -> Vec<String> {
        vec![self.model.clone()]
    }
}

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
        assert_eq!(chunks.len(), 3);
        assert!(chunks[2].as_ref().unwrap().is_stop());
    }
}

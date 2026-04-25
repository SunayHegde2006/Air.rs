//! I9 — Multi-Turn Tool Loop
//!
//! Manages the full agentic inference loop:
//! 1. Run inference → get model output
//! 2. Parse tool calls from output (uses `tool_call::parse_tool_calls`)
//! 3. Execute tool calls via user-provided dispatcher
//! 4. Inject tool results back into the conversation (chat template)
//! 5. Repeat until no more tool calls or max_rounds reached
//!
//! # Architecture
//!
//! ```text
//! ToolLoop
//!   ├── InferenceSession  (provided by caller — runs the LLM)
//!   ├── ToolDispatcher    (user-implementable trait — runs actual tools)
//!   ├── ChatHistory       (accumulates conversation turns)
//!   └── LoopConfig        (max_rounds, timeout, stop conditions)
//! ```
//!
//! # Supported Models
//! All agentic models supported by `tool_call::parse_tool_calls`:
//! Qwen3, Llama 3.x/4, DeepSeek-R1, Phi-4, Mistral tool-call format.
//!
//! # Example
//!
//! ```text
//! let loop_ = ToolLoop::new(cfg, my_dispatcher);
//! let result = loop_.run("What's the weather in Paris?", &session)?;
//! println!("{}", result.final_answer);
//! ```

use crate::tool_call::{parse_tool_calls, ToolCall};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Chat History & Roles
// ---------------------------------------------------------------------------

/// A single message in the multi-turn conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Role of a message in the conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// Tool result injected after a tool call
    Tool { call_id: String },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into() }
    }
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool { call_id: call_id.into() },
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool Dispatcher Trait
// ---------------------------------------------------------------------------

/// Trait that callers must implement to execute tool calls.
///
/// The loop calls `dispatch` for each tool call found in the model output.
/// The return value is the tool result string injected back into context.
///
/// # Error Handling
/// Return `Err(msg)` if the tool fails. The loop injects the error message
/// as the tool result and continues (the model may recover or give up).
pub trait ToolDispatcher: Send {
    /// Execute a tool call and return the result as a string.
    ///
    /// # Arguments
    /// * `call` — the parsed tool call (name + arguments JSON)
    fn dispatch(&mut self, call: &ToolCall) -> Result<String, String>;
}

/// A simple function-based dispatcher for testing / simple use cases.
pub struct FnDispatcher<F>
where
    F: FnMut(&ToolCall) -> Result<String, String> + Send,
{
    func: F,
}

impl<F> FnDispatcher<F>
where
    F: FnMut(&ToolCall) -> Result<String, String> + Send,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> ToolDispatcher for FnDispatcher<F>
where
    F: FnMut(&ToolCall) -> Result<String, String> + Send,
{
    fn dispatch(&mut self, call: &ToolCall) -> Result<String, String> {
        (self.func)(call)
    }
}

// ---------------------------------------------------------------------------
// Loop Configuration
// ---------------------------------------------------------------------------

/// Configuration for the tool loop.
#[derive(Debug, Clone)]
pub struct ToolLoopConfig {
    /// Maximum number of LLM inference + tool-call rounds
    /// (prevents infinite loops in runaway agents)
    pub max_rounds: usize,
    /// Maximum total wall-clock time for the loop
    pub timeout: Duration,
    /// System prompt (injected at start of each conversation)
    pub system_prompt: Option<String>,
    /// Whether to strip think tags from assistant output before adding to history
    pub strip_thinking: bool,
    /// Whether to strip tool-call blocks from the visible assistant message
    pub strip_tool_calls_from_visible: bool,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            timeout: Duration::from_secs(300),
            system_prompt: None,
            strip_thinking: true,
            strip_tool_calls_from_visible: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Loop Result
// ---------------------------------------------------------------------------

/// Outcome of one complete tool loop run.
#[derive(Debug)]
pub struct ToolLoopResult {
    /// The model's final text response (with tool calls + thinking stripped)
    pub final_answer: String,
    /// Number of inference rounds completed
    pub rounds: usize,
    /// All tool calls made during the run (in order)
    pub tool_calls: Vec<ToolCall>,
    /// Full conversation history for debugging or multi-turn continuation
    pub history: Vec<ChatMessage>,
    /// Why the loop stopped
    pub stop_reason: LoopStopReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopStopReason {
    /// Model stopped calling tools — clean completion
    NoMoreToolCalls,
    /// Hit max_rounds limit
    MaxRounds,
    /// Timeout
    Timeout,
    /// Inference error
    InferenceError(String),
}

// ---------------------------------------------------------------------------
// Abstract Inference Interface
// ---------------------------------------------------------------------------

/// Abstraction over the inference engine.
///
/// Implement this to connect ToolLoop to any inference backend.
pub trait InferenceSession: Send {
    /// Run one inference pass and return the raw model output text.
    ///
    /// `history` is the full conversation so far, formatted as messages.
    /// The implementor is responsible for applying the model's chat template.
    fn generate(&mut self, history: &[ChatMessage]) -> Result<String, String>;
}

// ---------------------------------------------------------------------------
// Tool Loop
// ---------------------------------------------------------------------------

/// The multi-turn tool loop orchestrator.
///
/// Drives the `think → tool_call → inject_result → repeat` cycle.
pub struct ToolLoop {
    config: ToolLoopConfig,
}

impl ToolLoop {
    pub fn new(config: ToolLoopConfig) -> Self {
        Self { config }
    }

    /// Run the full tool loop.
    ///
    /// # Arguments
    /// * `user_message` — the initial user query
    /// * `session`      — inference backend
    /// * `dispatcher`   — tool executor
    ///
    /// # Returns
    /// `ToolLoopResult` with final answer, history, and diagnostics.
    pub fn run(
        &self,
        user_message: &str,
        session: &mut dyn InferenceSession,
        dispatcher: &mut dyn ToolDispatcher,
    ) -> ToolLoopResult {
        let start = Instant::now();
        let mut history: Vec<ChatMessage> = Vec::new();
        let mut all_tool_calls: Vec<ToolCall> = Vec::new();
        let mut rounds = 0;

        // 1. Inject system prompt
        if let Some(ref sys) = self.config.system_prompt {
            history.push(ChatMessage::system(sys));
        }

        // 2. Add initial user message
        history.push(ChatMessage::user(user_message));

        loop {
            // Timeout guard
            if start.elapsed() > self.config.timeout {
                return ToolLoopResult {
                    final_answer: self.extract_visible(&history),
                    rounds,
                    tool_calls: all_tool_calls,
                    history,
                    stop_reason: LoopStopReason::Timeout,
                };
            }

            // Max rounds guard
            if rounds >= self.config.max_rounds {
                return ToolLoopResult {
                    final_answer: self.extract_visible(&history),
                    rounds,
                    tool_calls: all_tool_calls,
                    history,
                    stop_reason: LoopStopReason::MaxRounds,
                };
            }

            // 3. Run inference
            let raw_output = match session.generate(&history) {
                Ok(out) => out,
                Err(e) => {
                    return ToolLoopResult {
                        final_answer: self.extract_visible(&history),
                        rounds,
                        tool_calls: all_tool_calls,
                        history,
                        stop_reason: LoopStopReason::InferenceError(e),
                    };
                }
            };
            rounds += 1;

            // 4. Strip thinking tags if enabled
            let output_for_parse = if self.config.strip_thinking {
                crate::think_tag::strip_think_tags(&raw_output).visible
            } else {
                raw_output.clone()
            };

            // 5. Parse tool calls
            let parsed = parse_tool_calls(&output_for_parse);

            if parsed.calls.is_empty() {
                // No tool calls — this is the final answer
                history.push(ChatMessage::assistant(parsed.remainder.clone()));
                return ToolLoopResult {
                    final_answer: parsed.remainder,
                    rounds,
                    tool_calls: all_tool_calls,
                    history,
                    stop_reason: LoopStopReason::NoMoreToolCalls,
                };
            }

            // 6. Add assistant message (with or without tool call blocks stripped)
            let assistant_visible = if self.config.strip_tool_calls_from_visible {
                parsed.remainder.clone()
            } else {
                output_for_parse.clone()
            };
            history.push(ChatMessage::assistant(assistant_visible));

            // 7. Execute each tool call and inject results
            for call in &parsed.calls {
                all_tool_calls.push(call.clone());

                let call_id = format!("call_{}", all_tool_calls.len());
                let result = match dispatcher.dispatch(call) {
                    Ok(r) => r,
                    Err(e) => format!("[Tool Error: {}]", e),
                };

                history.push(ChatMessage::tool_result(&call_id, &result));
            }

            // 8. Loop: model will see tool results and generate next response
        }
    }

    /// Extract the last assistant message as the visible answer.
    fn extract_visible(&self, history: &[ChatMessage]) -> String {
        history
            .iter()
            .rev()
            .find(|m| m.role == ChatRole::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSession {
        responses: Vec<String>,
        call_count: usize,
    }

    impl MockSession {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.into_iter().map(String::from).collect(),
                call_count: 0,
            }
        }
    }

    impl InferenceSession for MockSession {
        fn generate(&mut self, _history: &[ChatMessage]) -> Result<String, String> {
            if self.call_count < self.responses.len() {
                let r = self.responses[self.call_count].clone();
                self.call_count += 1;
                Ok(r)
            } else {
                Err("No more responses".into())
            }
        }
    }

    #[test]
    fn test_no_tool_calls_immediate_answer() {
        let mut session = MockSession::new(vec!["The answer is 42."]);
        let mut dispatcher = FnDispatcher::new(|_| Ok("unused".into()));
        let cfg = ToolLoopConfig::default();
        let loop_ = ToolLoop::new(cfg);

        let result = loop_.run("What is 6 × 7?", &mut session, &mut dispatcher);
        assert_eq!(result.final_answer, "The answer is 42.");
        assert_eq!(result.rounds, 1);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.stop_reason, LoopStopReason::NoMoreToolCalls);
    }

    #[test]
    fn test_single_tool_call_round() {
        let mut session = MockSession::new(vec![
            // Round 1: model calls a tool
            r#"I'll check the weather. <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
            // Round 2: model gives final answer using tool result
            "The weather in Paris is 22°C and sunny.",
        ]);

        let mut dispatcher = FnDispatcher::new(|call| {
            assert_eq!(call.name, "get_weather");
            Ok(r#"{"temp": 22, "condition": "sunny"}"#.into())
        });

        let cfg = ToolLoopConfig::default();
        let loop_ = ToolLoop::new(cfg);
        let result = loop_.run("What's the weather in Paris?", &mut session, &mut dispatcher);

        assert_eq!(result.stop_reason, LoopStopReason::NoMoreToolCalls);
        assert_eq!(result.rounds, 2);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.final_answer.contains("22°C"));
    }

    #[test]
    fn test_max_rounds_limit() {
        // Model keeps calling tools forever
        let mut session = MockSession::new(vec![
            r#"<tool_call>{"name":"loop_tool","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"loop_tool","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"loop_tool","arguments":{}}</tool_call>"#,
        ]);
        let mut dispatcher = FnDispatcher::new(|_| Ok("result".into()));
        let cfg = ToolLoopConfig { max_rounds: 2, ..Default::default() };
        let loop_ = ToolLoop::new(cfg);
        let result = loop_.run("Loop forever", &mut session, &mut dispatcher);
        assert_eq!(result.stop_reason, LoopStopReason::MaxRounds);
        assert_eq!(result.rounds, 2);
    }

    #[test]
    fn test_tool_error_injected_as_result() {
        let mut session = MockSession::new(vec![
            r#"<tool_call>{"name":"failing_tool","arguments":{}}</tool_call>"#,
            "I encountered an error but here's my best answer.",
        ]);
        let mut dispatcher = FnDispatcher::new(|_| Err("Tool not found".into()));
        let cfg = ToolLoopConfig::default();
        let loop_ = ToolLoop::new(cfg);
        let result = loop_.run("Use the failing tool", &mut session, &mut dispatcher);

        // History should contain the error message
        let tool_result_msg = result.history.iter().find(|m| {
            matches!(&m.role, ChatRole::Tool { .. })
        });
        assert!(tool_result_msg.is_some());
        assert!(tool_result_msg.unwrap().content.contains("Tool Error"));
        assert_eq!(result.stop_reason, LoopStopReason::NoMoreToolCalls);
    }

    #[test]
    fn test_system_prompt_injected() {
        let mut session = MockSession::new(vec!["Hello!"]);
        let mut dispatcher = FnDispatcher::new(|_| Ok("x".into()));
        let cfg = ToolLoopConfig {
            system_prompt: Some("You are a helpful assistant.".into()),
            ..Default::default()
        };
        let loop_ = ToolLoop::new(cfg);
        let result = loop_.run("Hi", &mut session, &mut dispatcher);
        // First message in history should be system
        assert_eq!(result.history[0].role, ChatRole::System);
        assert!(result.history[0].content.contains("helpful assistant"));
    }

    #[test]
    fn test_history_structure() {
        let mut session = MockSession::new(vec![
            r#"<tool_call>{"name":"search","arguments":{"q":"rust"}}</tool_call>"#,
            "Rust is great.",
        ]);
        let mut dispatcher = FnDispatcher::new(|_| Ok("Rust results".into()));
        let cfg = ToolLoopConfig::default();
        let loop_ = ToolLoop::new(cfg);
        let result = loop_.run("Search for rust", &mut session, &mut dispatcher);

        // Expected: [user, assistant(tool_call), tool_result, assistant(final)]
        let roles: Vec<_> = result.history.iter().map(|m| &m.role).collect();
        assert!(roles.contains(&&ChatRole::User));
        assert!(roles.contains(&&ChatRole::Assistant));
    }
}

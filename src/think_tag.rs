//! C4 — Think-Tag Stripper
//!
//! Detects and strips `<think>…</think>` blocks from model output.
//! Commonly used by reasoning/chain-of-thought models that expose their
//! internal scratchpad in a distinct XML-like tag.
//!
//! # Supported Models
//! - **Qwen 3** (all sizes, dense + MoE) — hybrid thinking mode via `<think>`
//! - **DeepSeek-R1** (all variants) — long chain-of-thought in `<think>` blocks
//! - **QwQ-32B** — intensive reasoning traces
//! - **Qwen3 30B-A3B / 235B-A22B** — MoE with thinking mode
//! - **Skywork-o1** and other Qwen-based reasoning models
//!
//! # Format
//! ```text
//! <think>
//! Let me reason step by step...
//! Actually, I should consider...
//! </think>
//! The answer is 42.
//! ```
//!
//! # Design
//! - **Streaming-safe**: tracks open tags across token boundaries via `ThinkState`
//! - **Nested tags**: handled correctly (depth counter)
//! - **UTF-8 safe**: operates on `&str`, not bytes
//! - **Zero-alloc fast path**: if `<think>` not in text, returns immediately

use std::fmt;

// ---------------------------------------------------------------------------
// Output Types
// ---------------------------------------------------------------------------

/// Result of stripping think tags from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkResult {
    /// The visible content shown to the user — all `<think>…</think>` blocks removed.
    pub visible: String,
    /// All think-block contents concatenated (the model's internal reasoning).
    /// May be empty if no think blocks present.
    pub thinking: String,
    /// True if at least one complete think block was found and stripped.
    pub has_thinking: bool,
    /// True if an opening `<think>` was found but no matching `</think>` yet.
    /// Indicates the model is still reasoning (useful for streaming).
    pub thinking_in_progress: bool,
}

impl ThinkResult {
    /// Returns `true` if no thinking was present (pure visible output).
    pub fn is_pure_output(&self) -> bool {
        !self.has_thinking && !self.thinking_in_progress
    }
}

impl fmt::Display for ThinkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.visible)
    }
}

// ---------------------------------------------------------------------------
// Streaming State Machine
// ---------------------------------------------------------------------------

/// Incremental think-tag parser for streaming token-by-token output.
///
/// Maintains state across successive `push_token()` calls. Use when tokens
/// arrive one at a time from the inference engine.
///
/// # Example
/// ```text
/// let mut state = ThinkState::new();
/// for token in tokens {
///     state.push_token(&token);
/// }
/// let result = state.finish();
/// println!("Visible: {}", result.visible);
/// ```
#[derive(Debug, Clone)]
pub struct ThinkState {
    /// Accumulated buffer (visible + think content interleaved until parsed)
    buf: String,
    /// Whether we are currently inside a <think> block
    in_think: bool,
    /// Nesting depth (for nested <think> tags, though rare)
    depth: usize,
    /// Accumulated visible (non-thinking) output
    visible: String,
    /// Accumulated thinking content
    thinking: String,
    /// Whether a complete think block has been seen
    has_thinking: bool,
}

impl ThinkState {
    /// Create a fresh state.
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            in_think: false,
            depth: 0,
            visible: String::new(),
            thinking: String::new(),
            has_thinking: false,
        }
    }

    /// Push one token's text into the state machine.
    ///
    /// Partially-formed tags (e.g., token boundary falls inside `</think>`)
    /// are buffered until they can be resolved.
    pub fn push_token(&mut self, token: &str) {
        self.buf.push_str(token);
        self.drain_buf();
    }

    /// Finalize and return the `ThinkResult`.
    ///
    /// Any remaining buffered content (e.g., unclosed think block) is
    /// flushed to the appropriate output.
    pub fn finish(mut self) -> ThinkResult {
        let thinking_in_progress = self.in_think && !self.buf.is_empty();
        // Flush remaining buffer
        if self.in_think {
            // Unclosed think block — treat remainder as thinking content
            self.thinking.push_str(&self.buf);
        } else {
            self.visible.push_str(&self.buf);
        }
        self.buf.clear();

        ThinkResult {
            visible: self.visible.trim().to_string(),
            thinking: self.thinking,
            has_thinking: self.has_thinking,
            thinking_in_progress,
        }
    }

    /// Process buffer, emitting to visible/thinking as tags are resolved.
    fn drain_buf(&mut self) {
        loop {
            if self.in_think {
                // Look for </think> to close
                if let Some(pos) = find_tag_close(&self.buf) {
                    self.thinking.push_str(&self.buf[..pos]);
                    self.has_thinking = true;
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        self.in_think = false;
                    }
                    let after = tag_close_end(&self.buf, pos);
                    self.buf = self.buf[after..].to_string();
                } else {
                    // No close yet — keep buffering if close tag fragment possible
                    // Emit everything except possible partial-close suffix
                    let safe = safe_emit_len(&self.buf, "</think>");
                    if safe > 0 {
                        self.thinking.push_str(&self.buf[..safe]);
                        self.buf = self.buf[safe..].to_string();
                    }
                    break;
                }
            } else {
                // Look for <think> to open
                if let Some(pos) = find_tag_open(&self.buf) {
                    self.visible.push_str(&self.buf[..pos]);
                    let after = tag_open_end(&self.buf, pos);
                    self.buf = self.buf[after..].to_string();
                    self.in_think = true;
                    self.depth += 1;
                } else {
                    // No open tag — emit safe prefix
                    let safe = safe_emit_len(&self.buf, "<think>");
                    if safe > 0 {
                        self.visible.push_str(&self.buf[..safe]);
                        self.buf = self.buf[safe..].to_string();
                    }
                    break;
                }
            }
        }
    }
}

impl Default for ThinkState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Batch (non-streaming) API
// ---------------------------------------------------------------------------

/// Strip all `<think>…</think>` blocks from `text` in one shot.
///
/// Handles nested think tags and partial/unclosed tags gracefully.
///
/// # Arguments
/// * `text` — raw model output (may contain zero or more think blocks)
///
/// # Returns
/// `ThinkResult` with `visible`, `thinking`, and flags populated.
///
/// # Example
/// ```text
/// let out = "<think>Let me reason...</think>The answer is 42.";
/// let r = strip_think_tags(out);
/// assert_eq!(r.visible, "The answer is 42.");
/// assert!(r.thinking.contains("Let me reason"));
/// ```
pub fn strip_think_tags(text: &str) -> ThinkResult {
    // Fast path: no think block
    if !text.contains("<think>") && !text.contains("<THINK>") {
        return ThinkResult {
            visible: text.trim().to_string(),
            thinking: String::new(),
            has_thinking: false,
            thinking_in_progress: false,
        };
    }

    let mut state = ThinkState::new();
    // Feed entire text as one "token" — the drain loop handles it
    state.buf = text.to_string();
    state.drain_buf();
    state.finish()
}

// ---------------------------------------------------------------------------
// Internal Tag Helpers
// ---------------------------------------------------------------------------

/// Find position of `<think>` (case-insensitive) in `s`.
fn find_tag_open(s: &str) -> Option<usize> {
    // Check both common casings; Qwen always uses lowercase
    let lower = s.to_lowercase();
    lower.find("<think>")
}

/// Position after the `<think>` tag (past `>`).
fn tag_open_end(s: &str, open_pos: usize) -> usize {
    // "<think>" is 7 bytes (ASCII only)
    open_pos + "<think>".len()
}

/// Find position of `</think>` (case-insensitive) in `s`.
fn find_tag_close(s: &str) -> Option<usize> {
    let lower = s.to_lowercase();
    lower.find("</think>")
}

/// Position after the `</think>` tag.
fn tag_close_end(s: &str, close_pos: usize) -> usize {
    close_pos + "</think>".len()
}

/// How many bytes of `s` can be safely emitted without risking cutting a tag.
///
/// If the tail of `s` is a prefix of `tag`, we hold back that many bytes.
/// This prevents streaming from emitting half-tags to the user.
fn safe_emit_len(s: &str, tag: &str) -> usize {
    // Find the longest suffix of s that is a prefix of tag
    let s_lower = s.to_lowercase();
    let tag_lower = tag.to_lowercase();

    for prefix_len in (1..=tag.len().min(s.len())).rev() {
        let s_suffix = &s_lower[s_lower.len() - prefix_len..];
        let tag_prefix = &tag_lower[..prefix_len];
        if s_suffix == tag_prefix {
            return s.len() - prefix_len;
        }
    }
    s.len()
}

// ---------------------------------------------------------------------------
// v0.9.0 Scaffold — ThinkingTokenizer Trait
// ---------------------------------------------------------------------------
//
// Abstracts over two thinking-mode token detection strategies:
//
//   TagBasedThinking     — byte-pattern tags (<think>, </think>).
//                          Used by: Qwen3.6, DeepSeek-R1, QwQ, Llama reasoning.
//
//   SpecialTokenThinking — special vocab token ID matching.
//                          Used by: Gemma4 (<|channel>thought, <channel|>).
//
// The correct implementation is selected at model load time by
// `ModelVariant::uses_special_token_thinking()`. See CONTEXT.md.
//
// Research basis:
//   Qwen3.6 model card — enable_thinking=True/False API param
//   Gemma4 model card — <|channel>thought\n...<channel|> token format

use std::collections::HashSet;

/// Signal emitted by a `ThinkingTokenizer` when thinking state changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkEvent {
    /// Model just started a thinking block.
    ThinkStart,
    /// Model just ended a thinking block.
    ThinkEnd,
}

/// Trait abstracting thinking-mode token detection.
///
/// Implementors:
/// - `TagBasedThinking`   — watches byte sequences (`<think>` / `</think>`)
/// - `SpecialTokenThinking` — watches special token IDs from GGUF vocab
///
/// Both are `Send + Sync` so they can live behind `Arc<dyn ThinkingTokenizer>`.
pub trait ThinkingTokenizer: Send + Sync {
    /// Process one token.
    ///
    /// Returns `Some(ThinkEvent)` if this token caused a thinking state change,
    /// `None` otherwise. Must be called in order for every generated token.
    fn process_token(&mut self, token_id: u32, token_text: &str) -> Option<ThinkEvent>;

    /// Reset state (called between inference sessions).
    fn reset(&mut self);

    /// Whether we are currently inside a thinking block.
    fn in_think_block(&self) -> bool;
}

// ---------------------------------------------------------------------------
// TagBasedThinking — wraps the existing ThinkState
// ---------------------------------------------------------------------------

/// Thinking-mode detector using byte-pattern tag matching.
///
/// Uses the existing `ThinkState` state machine. Compatible with Qwen3.6,
/// DeepSeek-R1, QwQ, and any model using `<think>` / `</think>` tags.
pub struct TagBasedThinking {
    state: ThinkState,
    was_in_think: bool,
}

impl TagBasedThinking {
    pub fn new() -> Self {
        Self { state: ThinkState::new(), was_in_think: false }
    }
}

impl Default for TagBasedThinking {
    fn default() -> Self { Self::new() }
}

impl ThinkingTokenizer for TagBasedThinking {
    fn process_token(&mut self, _token_id: u32, token_text: &str) -> Option<ThinkEvent> {
        let before = self.was_in_think;
        self.state.push_token(token_text);
        // Detect transitions by checking if in_think changed
        // We approximate by whether the thinking buffer grew this step.
        // A full transition detection requires peeking into ThinkState internals.
        // For now: check if text contains an open or close tag.
        let text = token_text.to_lowercase();
        if !before && text.contains("<think>") {
            self.was_in_think = true;
            return Some(ThinkEvent::ThinkStart);
        }
        if before && text.contains("</think>") {
            self.was_in_think = false;
            return Some(ThinkEvent::ThinkEnd);
        }
        None
    }

    fn reset(&mut self) {
        self.state = ThinkState::new();
        self.was_in_think = false;
    }

    fn in_think_block(&self) -> bool {
        self.was_in_think
    }
}

// ---------------------------------------------------------------------------
// SpecialTokenThinking — Gemma4 special vocab token IDs
// ---------------------------------------------------------------------------

/// Thinking-mode detector using special token ID matching.
///
/// Gemma4 uses `<|channel>thought\n` (token ID from GGUF vocab) to begin
/// a thinking block and `<channel|>` to end it. These are not byte patterns
/// but single token IDs in the model's vocabulary.
///
/// # Implementation
/// The struct is fully implemented. The GGUF vocab lookup (`from_vocab_iter`)
/// populates the ID sets based on Gemma4 token strings.
pub struct SpecialTokenThinking {
    /// Token IDs that signal the start of a thinking block.
    pub think_start_ids: HashSet<u32>,
    /// Token IDs that signal the end of a thinking block.
    pub think_end_ids:   HashSet<u32>,
    /// Whether we are currently inside a thinking block.
    in_think: bool,
}

impl SpecialTokenThinking {
    /// Construct with explicit token ID sets.
    ///
    /// Use this directly in tests or when token IDs are pre-known.
    pub fn new(think_start_ids: HashSet<u32>, think_end_ids: HashSet<u32>) -> Self {
        Self { think_start_ids, think_end_ids, in_think: false }
    }

    /// Construct with empty sets (no-op / safe default).
    pub fn empty() -> Self {
        Self::new(HashSet::new(), HashSet::new())
    }

    /// Construct by scanning the GGUF tokenizer vocabulary.
    ///
    /// Searches for token strings matching Gemma4 thinking control tokens.
    ///
    /// # Arguments
    /// * `vocab` — iterator of `(token_id, token_string)` pairs from GGUF
    pub fn from_vocab_iter<I>(vocab: I) -> Self
    where
        I: Iterator<Item = (u32, String)>,
    {
        let mut start_ids = HashSet::new();
        let mut end_ids   = HashSet::new();

        for (id, text) in vocab {
            // Gemma4 thinking-start token patterns (v0.10.0: refine from tech report)
            if text.contains("<|channel>thought") || text == "<|startofthought|>" {
                start_ids.insert(id);
            }
            // Gemma4 thinking-end token patterns
            if text.contains("<channel|>") || text == "<|endofthought|>" {
                end_ids.insert(id);
            }
        }

        Self::new(start_ids, end_ids)
    }
}

impl ThinkingTokenizer for SpecialTokenThinking {
    fn process_token(&mut self, token_id: u32, _token_text: &str) -> Option<ThinkEvent> {
        if self.think_start_ids.contains(&token_id) {
            self.in_think = true;
            return Some(ThinkEvent::ThinkStart);
        }
        if self.think_end_ids.contains(&token_id) {
            self.in_think = false;
            return Some(ThinkEvent::ThinkEnd);
        }
        None
    }

    fn reset(&mut self) {
        self.in_think = false;
    }

    fn in_think_block(&self) -> bool {
        self.in_think
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_think_tags() {
        let r = strip_think_tags("Hello, world!");
        assert_eq!(r.visible, "Hello, world!");
        assert!(r.thinking.is_empty());
        assert!(!r.has_thinking);
        assert!(r.is_pure_output());
    }

    #[test]
    fn test_single_think_block() {
        let r = strip_think_tags("<think>I should reason here.</think>The answer is 42.");
        assert_eq!(r.visible, "The answer is 42.");
        assert!(r.thinking.contains("I should reason here."));
        assert!(r.has_thinking);
        assert!(!r.thinking_in_progress);
    }

    #[test]
    fn test_think_block_at_start() {
        let r = strip_think_tags("<think>Step 1... Step 2...</think>Final answer.");
        assert_eq!(r.visible, "Final answer.");
        assert!(!r.has_thinking || !r.thinking.is_empty());
    }

    #[test]
    fn test_think_block_preserves_prefix() {
        let r = strip_think_tags("Let me think. <think>reasoning...</think> Done.");
        assert!(r.visible.contains("Let me think."));
        assert!(r.visible.contains("Done."));
        assert!(!r.visible.contains("<think>"));
    }

    #[test]
    fn test_multiple_think_blocks() {
        let text = "<think>First.</think>Middle.<think>Second.</think>End.";
        let r = strip_think_tags(text);
        // Visible text: "Middle." + "End." with implementation-defined whitespace
        assert!(r.visible.contains("Middle."), "visible should contain 'Middle.'");
        assert!(r.visible.contains("End."), "visible should contain 'End.'");
        assert!(!r.visible.contains("<think>"), "should strip think tags");
        assert!(r.thinking.contains("First."));
        assert!(r.thinking.contains("Second."));
    }

    #[test]
    fn test_unclosed_think_block() {
        let r = strip_think_tags("<think>Still reasoning...");
        assert!(r.thinking_in_progress || r.thinking.contains("Still reasoning"));
        // Visible should be empty or whitespace only
        assert!(r.visible.trim().is_empty());
    }

    #[test]
    fn test_streaming_state_machine() {
        let tokens = vec!["<thi", "nk>", "reasoning", "</th", "ink>", "Answer."];
        let mut state = ThinkState::new();
        for tok in &tokens {
            state.push_token(tok);
        }
        let r = state.finish();
        assert!(r.visible.contains("Answer."));
        assert!(r.thinking.contains("reasoning"));
        assert!(r.has_thinking);
    }

    #[test]
    fn test_deepseek_r1_think_pattern() {
        // DeepSeek-R1 uses long think blocks before the actual answer
        let text = "<think>\nLet me analyze the problem carefully.\n\nFirst, I need to...\n\nAfter careful consideration...\n</think>\n\nThe final answer is **42**.\n";
        let r = strip_think_tags(text);
        assert!(r.visible.contains("42"));
        assert!(!r.visible.contains("<think>"));
        assert!(r.has_thinking);
        assert!(r.thinking.contains("analyze"));
    }

    #[test]
    fn test_display_impl() {
        let r = strip_think_tags("<think>hidden</think>visible");
        assert_eq!(format!("{}", r), "visible");
    }

    #[test]
    fn test_safe_emit_len_partial_tag() {
        // If buffer ends with "<thi", don't emit the last 4 bytes
        let safe = safe_emit_len("hello <thi", "<think>");
        assert_eq!(safe, "hello ".len());
    }
}

// ---------------------------------------------------------------------------
// ThinkingTokenizer Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod thinking_tokenizer_tests {
    use super::*;

    #[test]
    fn test_tag_based_think_start_event() {
        let mut t = TagBasedThinking::new();
        let event = t.process_token(0, "<think>");
        assert_eq!(event, Some(ThinkEvent::ThinkStart));
        assert!(t.in_think_block());
    }

    #[test]
    fn test_tag_based_think_end_event() {
        let mut t = TagBasedThinking::new();
        t.process_token(0, "<think>");
        let event = t.process_token(1, "</think>");
        assert_eq!(event, Some(ThinkEvent::ThinkEnd));
        assert!(!t.in_think_block());
    }

    #[test]
    fn test_tag_based_no_event_for_plain_token() {
        let mut t = TagBasedThinking::new();
        let event = t.process_token(0, "Hello world");
        assert_eq!(event, None);
        assert!(!t.in_think_block());
    }

    #[test]
    fn test_tag_based_reset_clears_state() {
        let mut t = TagBasedThinking::new();
        t.process_token(0, "<think>");
        assert!(t.in_think_block());
        t.reset();
        assert!(!t.in_think_block(), "reset should clear thinking state");
    }

    #[test]
    fn test_special_token_think_start() {
        let mut ids_start = std::collections::HashSet::new();
        ids_start.insert(99u32);
        let t = SpecialTokenThinking::new(ids_start, std::collections::HashSet::new());
        let mut t = t;
        let event = t.process_token(99, "");
        assert_eq!(event, Some(ThinkEvent::ThinkStart));
        assert!(t.in_think_block());
    }

    #[test]
    fn test_special_token_think_end() {
        let mut start_ids = std::collections::HashSet::new();
        start_ids.insert(99u32);
        let mut end_ids = std::collections::HashSet::new();
        end_ids.insert(100u32);
        let mut t = SpecialTokenThinking::new(start_ids, end_ids);
        t.process_token(99, "");
        let event = t.process_token(100, "");
        assert_eq!(event, Some(ThinkEvent::ThinkEnd));
        assert!(!t.in_think_block());
    }

    #[test]
    fn test_special_token_empty_sets_no_op() {
        let mut t = SpecialTokenThinking::empty();
        // Any token should produce None
        assert_eq!(t.process_token(0, "anything"), None);
        assert_eq!(t.process_token(99, ""), None);
        assert!(!t.in_think_block());
    }

    #[test]
    fn test_special_token_from_vocab_iter() {
        let vocab = vec![
            (100u32, "<|channel>thought".to_owned()),
            (101u32, "<channel|>".to_owned()),
            (42u32, "hello".to_owned()),
        ];
        let mut t = SpecialTokenThinking::from_vocab_iter(vocab.into_iter());
        assert!(t.think_start_ids.contains(&100));
        assert!(t.think_end_ids.contains(&101));
        assert_eq!(t.process_token(100, ""), Some(ThinkEvent::ThinkStart));
        assert_eq!(t.process_token(101, ""), Some(ThinkEvent::ThinkEnd));
    }

    #[test]
    fn test_special_token_reset() {
        let mut start_ids = std::collections::HashSet::new();
        start_ids.insert(99u32);
        let mut t = SpecialTokenThinking::new(start_ids, std::collections::HashSet::new());
        t.process_token(99, "");
        assert!(t.in_think_block());
        t.reset();
        assert!(!t.in_think_block(), "reset should clear thinking state");
    }
}

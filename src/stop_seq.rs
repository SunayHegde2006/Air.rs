//! I7 — Multi-Token Stop Sequences
//!
//! Detects stop conditions during token-by-token generation.
//! Supports all agentic model stop patterns as of April 2026.
//!
//! # Design
//! - `StopChecker` maintains a rolling text buffer of the last W characters.
//! - On each new token, checks suffix against all registered stop strings.
//! - Supports both string and EOS-token-id stops.
//! - Returns a `StopReason` so callers know *why* generation halted.
//!
//! # Common Stop Sequences by Model Family
//! | Model | Stop strings |
//! |---|---|
//! | Llama 3.x / 4 | `<\|eot_id\|>`, `<\|end_of_text\|>` |
//! | Qwen3 dense | `<\|im_end\|>`, `</tool_call>` |
//! | Qwen3 thinking | `<\|im_end\|>`, `</think>`, `</tool_call>` |
//! | DeepSeek-R1 | `<end_of_sentence>`, `</think>`, `</tool_call>` |
//! | Mistral | `[/INST]`, `</s>` |
//! | Phi-4 | `<\|end\|>`, `<\|im_end\|>` |
//! | GPT-2 / BLOOM | `<\|endoftext\|>` |

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Stop Reason
// ---------------------------------------------------------------------------

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Hit the maximum token limit
    MaxTokens,
    /// Matched a registered stop string
    StopString(String),
    /// Sampled the EOS token id
    EosToken(u32),
    /// Generation is still in progress
    NotStopped,
}

impl StopReason {
    pub fn is_done(&self) -> bool {
        !matches!(self, StopReason::NotStopped)
    }
}

// ---------------------------------------------------------------------------
// Stop Checker
// ---------------------------------------------------------------------------

/// Multi-token stop sequence checker.
///
/// # Usage
/// ```text
/// let mut checker = StopChecker::new(
///     vec!["</tool_call>".into(), "<|im_end|>".into()],
///     Some(128009), // Llama 3 EOS token id
///     2048,
/// );
/// for (token_id, text) in tokens {
///     checker.push(token_id, &text);
///     if checker.reason().is_done() {
///         break;
///     }
/// }
/// let visible = checker.strip_stop_suffix();
/// ```
pub struct StopChecker {
    /// Registered stop strings (sorted longest-first for greedy matching)
    stop_strings: Vec<String>,
    /// EOS token id (if any)
    eos_token_id: Option<u32>,
    /// Rolling window of decoded text — ring buffer behaviour via VecDeque<char>
    buffer: VecDeque<char>,
    /// Maximum buffer capacity (max stop string length × 2 for safety)
    capacity: usize,
    /// Max token budget
    max_tokens: usize,
    /// Tokens generated so far
    token_count: usize,
    /// Current stop reason
    reason: StopReason,
    /// Full text accumulated (for final output)
    full_text: String,
}

impl StopChecker {
    /// Create a new StopChecker.
    ///
    /// # Arguments
    /// * `stop_strings` — list of stop sequences to watch for (any order)
    /// * `eos_token_id` — optional EOS vocab id; if sampled, stops immediately
    /// * `max_tokens`   — hard cap on generated tokens
    pub fn new(
        mut stop_strings: Vec<String>,
        eos_token_id: Option<u32>,
        max_tokens: usize,
    ) -> Self {
        // Sort longest-first so we match the most specific stop first
        stop_strings.sort_by(|a, b| b.len().cmp(&a.len()));
        // Buffer needs to hold at least the longest stop string
        let max_stop_len = stop_strings.iter().map(|s| s.len()).max().unwrap_or(1);
        // × 2 for safety with multi-byte UTF-8
        let capacity = max_stop_len * 2 + 16;

        Self {
            stop_strings,
            eos_token_id,
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            max_tokens,
            token_count: 0,
            reason: StopReason::NotStopped,
            full_text: String::new(),
        }
    }

    /// Preset: Llama 3.x / Llama 4
    pub fn llama3(max_tokens: usize) -> Self {
        Self::new(
            vec![
                "<|eot_id|>".into(),
                "<|end_of_text|>".into(),
                "<|end_header_id|>".into(),
            ],
            Some(128009), // <|eot_id|> token id in Llama 3 tokenizer
            max_tokens,
        )
    }

    /// Preset: Qwen3 (dense, thinking, MoE)
    pub fn qwen3(max_tokens: usize) -> Self {
        Self::new(
            vec![
                "<|im_end|>".into(),
                "</tool_call>".into(),
                "</think>".into(),
            ],
            Some(151645), // <|im_end|> token id in Qwen3 tokenizer
            max_tokens,
        )
    }

    /// Preset: DeepSeek-R1
    pub fn deepseek_r1(max_tokens: usize) -> Self {
        Self::new(
            vec![
                "</think>".into(),
                "</tool_call>".into(),
            ],
            Some(100001), // <|end_of_sentence|> in DeepSeek tokenizer
            max_tokens,
        )
    }

    /// Preset: Mistral / Phi-4
    pub fn mistral(max_tokens: usize) -> Self {
        Self::new(
            vec!["[/INST]".into(), "</s>".into()],
            Some(2), // </s> is token 2 in Mistral tokenizer
            max_tokens,
        )
    }

    /// Push one generated token into the checker.
    ///
    /// Call this after each sampled token. If `reason()` returns anything
    /// other than `NotStopped`, stop generating.
    pub fn push(&mut self, token_id: u32, decoded_text: &str) {
        if self.reason.is_done() {
            return;
        }

        self.token_count += 1;
        self.full_text.push_str(decoded_text);

        // 1. EOS token check (fastest path)
        if let Some(eos) = self.eos_token_id {
            if token_id == eos {
                self.reason = StopReason::EosToken(eos);
                return;
            }
        }

        // 2. Max tokens
        if self.token_count >= self.max_tokens {
            self.reason = StopReason::MaxTokens;
            return;
        }

        // 3. Add decoded chars to rolling buffer
        for c in decoded_text.chars() {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(c);
        }

        // 4. Check stop strings against buffer suffix
        let buf_str: String = self.buffer.iter().collect();
        for stop in &self.stop_strings {
            if buf_str.ends_with(stop.as_str()) {
                self.reason = StopReason::StopString(stop.clone());
                return;
            }
        }
    }

    /// Current stop reason.
    pub fn reason(&self) -> &StopReason {
        &self.reason
    }

    /// True if generation should stop.
    pub fn is_done(&self) -> bool {
        self.reason.is_done()
    }

    /// Full generated text, with the matched stop string stripped from the end.
    ///
    /// Returns the clean visible output suitable for display.
    pub fn strip_stop_suffix(&self) -> &str {
        let text = &self.full_text;
        if let StopReason::StopString(ref stop) = self.reason {
            if text.ends_with(stop.as_str()) {
                return &text[..text.len() - stop.len()];
            }
        }
        text
    }

    /// Total tokens generated (including any stop token).
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Full accumulated text (including stop string if any)
    pub fn full_text(&self) -> &str {
        &self.full_text
    }

    /// Reset the checker for a new generation (reuse same config)
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.token_count = 0;
        self.reason = StopReason::NotStopped;
        self.full_text.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_stop_in_progress() {
        let mut c = StopChecker::new(vec!["</s>".into()], None, 100);
        c.push(1, "Hello");
        c.push(2, " world");
        assert_eq!(c.reason(), &StopReason::NotStopped);
        assert!(!c.is_done());
    }

    #[test]
    fn test_eos_token_stop() {
        let mut c = StopChecker::new(vec![], Some(2), 100);
        c.push(1, "Hello");
        c.push(2, ""); // token id 2 = EOS
        assert_eq!(c.reason(), &StopReason::EosToken(2));
        assert!(c.is_done());
    }

    #[test]
    fn test_single_token_stop_string() {
        let mut c = StopChecker::new(vec!["</s>".into()], None, 100);
        c.push(1, "Hello");
        c.push(2, "</s>");
        assert_eq!(c.reason(), &StopReason::StopString("</s>".into()));
        assert!(c.is_done());
    }

    #[test]
    fn test_multi_token_stop_string() {
        // "</tool_call>" spans multiple tokens in practice
        let mut c = StopChecker::new(vec!["</tool_call>".into()], None, 100);
        c.push(1, "</");
        c.push(2, "tool");
        c.push(3, "_call");
        c.push(4, ">");
        assert_eq!(c.reason(), &StopReason::StopString("</tool_call>".into()));
    }

    #[test]
    fn test_max_tokens() {
        let mut c = StopChecker::new(vec![], None, 3);
        c.push(1, "a");
        c.push(2, "b");
        c.push(3, "c"); // hits limit
        assert_eq!(c.reason(), &StopReason::MaxTokens);
    }

    #[test]
    fn test_strip_stop_suffix() {
        let mut c = StopChecker::new(vec!["<|im_end|>".into()], None, 100);
        c.push(1, "Answer text");
        c.push(2, "<|im_end|>");
        let stripped = c.strip_stop_suffix();
        assert_eq!(stripped, "Answer text");
    }

    #[test]
    fn test_no_double_stop() {
        // After stopping, further pushes are ignored
        let mut c = StopChecker::new(vec!["</s>".into()], None, 100);
        c.push(1, "</s>");
        let reason_before = c.reason().clone();
        c.push(2, "more text");
        assert_eq!(c.reason(), &reason_before);
    }

    #[test]
    fn test_stop_not_triggered_on_partial_match() {
        let mut c = StopChecker::new(vec!["</tool_call>".into()], None, 100);
        c.push(1, "</tool"); // Partial — should not stop
        assert_eq!(c.reason(), &StopReason::NotStopped);
    }

    #[test]
    fn test_llama3_preset() {
        let mut c = StopChecker::llama3(512);
        // Simulate EOS token id 128009
        c.push(128009, "");
        assert!(c.is_done());
    }

    #[test]
    fn test_qwen3_preset_tool_call_stop() {
        let mut c = StopChecker::qwen3(2048);
        c.push(1, "I'll help. </");
        c.push(2, "tool_call>");
        assert!(c.is_done());
        if let StopReason::StopString(ref s) = c.reason() {
            assert_eq!(s, "</tool_call>");
        }
    }

    #[test]
    fn test_reset_and_reuse() {
        let mut c = StopChecker::new(vec!["</s>".into()], None, 100);
        c.push(1, "Hello");
        c.push(2, "</s>");
        assert!(c.is_done());

        c.reset();
        assert_eq!(c.reason(), &StopReason::NotStopped);
        assert_eq!(c.token_count(), 0);
        assert!(c.full_text().is_empty());
    }
}

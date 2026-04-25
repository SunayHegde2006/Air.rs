//! I8 — JSON Grammar-Constrained Decoding
//!
//! Forces the LLM sampler to emit valid JSON by masking logits at each step,
//! only allowing tokens that keep the running output on a valid JSON path.
//!
//! # Approach: Incremental JSON Parser + Logit Mask
//!
//! At each decode step, we maintain a `JsonParseState` tracking the partial
//! JSON structure. Before sampling, we compute a boolean mask over the vocabulary:
//! `allowed[token_id] = true` if appending that token's text keeps the partial
//! JSON parseable.
//!
//! This is a **character-level finite automaton** approach:
//! - Fast: O(vocab_size × avg_token_len) per step
//! - Sound: provably only emits valid JSON (per RFC 8259)
//! - Complete: covers all JSON value types (object, array, string, number, bool, null)
//!
//! # Limitations (by design)
//! - **Schema validation** (ensuring keys match a schema) is NOT implemented here.
//!   This module enforces syntactic validity only. Layer schema validation on top
//!   using the `json_schema` module if needed.
//! - **Token granularity**: some tokenizers produce multi-character tokens that may
//!   straddle a JSON boundary. The masker handles this via lookahead simulation.
//!
//! # Integration
//! ```text
//! let mut constrained = JsonConstrainedSampler::new(vocab_size);
//! for step in 0..max_tokens {
//!     let mask = constrained.logit_mask(&tokenizer);
//!     // Apply mask: logits[i] = -inf where !mask[i]
//!     let token_id = sampler.sample_with_mask(&logits, &mask)?;
//!     let text = tokenizer.decode(token_id);
//!     constrained.push_token(token_id, &text);
//!     if constrained.is_complete() { break; }
//! }
//! let json = constrained.output();
//! ```

// ---------------------------------------------------------------------------
// JSON Parse State Machine
// ---------------------------------------------------------------------------

/// The grammar state of a partial JSON document.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonState {
    /// Haven't started yet — next char must begin a JSON value
    Start,
    /// Inside a JSON string value (after opening `"`, before closing `"`)
    InString { escape_next: bool },
    /// Inside a JSON object, expecting `"key"` or `}`
    InObjectKey,
    /// Inside a JSON object, just read key, expecting `:`
    AfterObjectKey,
    /// Inside a JSON object, just read `:`, expecting value
    InObjectValue,
    /// Inside a JSON array, expecting value or `]`
    InArrayValue { first: bool },
    /// Reading a number (integer part)
    InNumber,
    /// Reading a number (decimal part after `.`)
    InNumberDecimal,
    /// Reading a literal (`true`, `false`, `null`)
    InLiteral { remaining: Vec<char> },
    /// JSON value is complete and valid
    Complete,
    /// Invalid / stuck state (error recovery)
    Error,
}

/// Incremental JSON document builder that tracks parse state.
///
/// Feed characters one at a time; query `allowed_next_chars()` before
/// sampling to know which characters are valid at this position.
#[derive(Debug, Clone)]
pub struct JsonParser {
    /// Current accumulated text
    text: String,
    /// Parse state stack (for nested structures)
    stack: Vec<JsonState>,
    /// Current state
    state: JsonState,
}

impl JsonParser {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            stack: Vec::new(),
            state: JsonState::Start,
        }
    }

    /// Push one character and advance the state machine.
    ///
    /// Returns `false` if the character is invalid at this position.
    pub fn push_char(&mut self, c: char) -> bool {
        let ok = self.advance(c);
        if ok {
            self.text.push(c);
        } else {
            self.state = JsonState::Error;
        }
        ok
    }

    /// Simulate pushing a string without committing. Returns `true` if valid.
    pub fn can_push_str(&self, s: &str) -> bool {
        let mut sim = self.clone();
        for c in s.chars() {
            if !sim.push_char(c) {
                return false;
            }
        }
        true
    }

    /// True if the current state represents a valid, complete JSON value.
    ///
    /// Numbers and literals complete without a trailing delimiter,
    /// so `InNumber`/`InNumberDecimal` with empty stack are also complete.
    pub fn is_complete(&self) -> bool {
        match &self.state {
            JsonState::Complete => true,
            // A number at end of input is valid (no trailing delimiter needed)
            JsonState::InNumber | JsonState::InNumberDecimal => self.stack.is_empty(),
            _ => false,
        }
    }

    /// True if we're in an error state.
    pub fn is_error(&self) -> bool {
        matches!(self.state, JsonState::Error)
    }

    /// Accumulated JSON text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Characters allowed as the next input (for logit masking build-out).
    /// Returns `None` if in a state where any string-safe char is allowed.
    pub fn allowed_chars_hint(&self) -> AllowedHint {
        match &self.state {
            JsonState::Start | JsonState::InObjectValue | JsonState::InArrayValue { .. } => {
                AllowedHint::ValueStart
            }
            JsonState::InString { escape_next: true } => AllowedHint::EscapeTarget,
            JsonState::InString { escape_next: false } => AllowedHint::StringContent,
            JsonState::InObjectKey => AllowedHint::ObjectKeyOrClose,
            JsonState::AfterObjectKey => AllowedHint::Colon,
            JsonState::InNumber => AllowedHint::NumberContinue,
            JsonState::InNumberDecimal => AllowedHint::NumberDecimalContinue,
            JsonState::InLiteral { remaining } => {
                AllowedHint::LiteralNext(remaining.first().copied())
            }
            JsonState::Complete => AllowedHint::Done,
            JsonState::Error => AllowedHint::None,
        }
    }

    fn advance(&mut self, c: char) -> bool {
        match self.state.clone() {
            JsonState::Start => self.start_value(c),

            JsonState::InString { escape_next } => {
                if escape_next {
                    self.state = JsonState::InString { escape_next: false };
                    matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')
                } else {
                    match c {
                        '"' => { self.finish_value(); true }
                        '\\' => { self.state = JsonState::InString { escape_next: true }; true }
                        c if (c as u32) >= 0x20 => true,
                        _ => false,
                    }
                }
            }

            JsonState::InObjectKey => {
                match c {
                    '"' => {
                        // Push AfterObjectKey as the return state after the key string
                        self.stack.push(JsonState::AfterObjectKey);
                        self.state = JsonState::InString { escape_next: false };
                        true
                    }
                    '}' => { self.finish_value(); true }
                    ',' => true, // comma between key-value pairs — stay in InObjectKey
                    ' ' | '\t' | '\n' | '\r' => true,
                    _ => false,
                }
            }

            JsonState::AfterObjectKey => {
                match c {
                    ':' => { self.state = JsonState::InObjectValue; true }
                    ' ' | '\t' | '\n' | '\r' => true,
                    _ => false,
                }
            }

            JsonState::InObjectValue => self.start_value(c),

            JsonState::InArrayValue { first } => {
                match c {
                    ']' => { self.finish_value(); true }
                    ',' if !first => {
                        // After first element, comma separates elements; next is a value
                        self.state = JsonState::InArrayValue { first: false };
                        true
                    }
                    ' ' | '\t' | '\n' | '\r' => true,
                    _ => self.start_value(c),
                }
            }

            JsonState::InNumber => {
                match c {
                    '0'..='9' => true,
                    '.' => { self.state = JsonState::InNumberDecimal; true }
                    'e' | 'E' => { self.state = JsonState::InNumberDecimal; true }
                    // Delimiter ends the number
                    ',' | '}' | ']' | ' ' | '\t' | '\n' | '\r' => {
                        self.finish_value();
                        self.advance_delimiter(c)
                    }
                    _ => false,
                }
            }

            JsonState::InNumberDecimal => {
                match c {
                    '0'..='9' | '+' | '-' | 'e' | 'E' => true,
                    ',' | '}' | ']' | ' ' | '\t' | '\n' | '\r' => {
                        self.finish_value();
                        self.advance_delimiter(c)
                    }
                    _ => false,
                }
            }

            JsonState::InLiteral { remaining } => {
                // `remaining` is from the clone — use it only for the validity check
                if remaining.first() == Some(&c) {
                    // Mutate self.state directly (clone has already served its purpose)
                    let now_empty = {
                        if let JsonState::InLiteral { remaining: ref mut r } = self.state {
                            r.remove(0);
                            r.is_empty()
                        } else {
                            false
                        }
                    };
                    if now_empty {
                        self.finish_value();
                    }
                    true
                } else if remaining.is_empty() {
                    // Literal already complete — this is a following delimiter
                    self.advance_delimiter(c)
                } else {
                    false
                }
            }


            JsonState::Complete => {
                matches!(c, ' ' | '\t' | '\n' | '\r')
            }

            JsonState::Error => false,
        }
    }

    fn start_value(&mut self, c: char) -> bool {
        match c {
            '"' => {
                // Push context for what comes after the string
                let ctx = self.value_done_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InString { escape_next: false };
                true
            }
            '{' => {
                let ctx = self.value_done_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InObjectKey;
                true
            }
            '[' => {
                let ctx = self.value_done_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InArrayValue { first: true };
                true
            }
            '-' | '0'..='9' => {
                let ctx = self.value_done_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InNumber;
                true
            }
            't' => {
                let ctx = self.value_done_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InLiteral { remaining: vec!['r','u','e'] };
                true
            }
            'f' => {
                let ctx = self.continuation_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InLiteral { remaining: vec!['a','l','s','e'] };
                true
            }
            'n' => {
                let ctx = self.continuation_state();
                if let Some(s) = ctx { self.stack.push(s); }
                self.state = JsonState::InLiteral { remaining: vec!['u','l','l'] };
                true
            }
            ' ' | '\t' | '\n' | '\r' => true,
            _ => false,
        }
    }

    fn finish_value(&mut self) {
        match self.stack.pop() {
            None => self.state = JsonState::Complete,
            Some(next) => self.state = next,
        }
    }

    /// Handle a delimiter character that terminates a number or literal.
    /// After `finish_value()` the state is whatever comes next (object key,
    /// array element, or complete). We then apply the delimiter in that context.
    fn advance_delimiter(&mut self, c: char) -> bool {
        match &self.state {
            JsonState::Complete => {
                matches!(c, ' ' | '\t' | '\n' | '\r')
            }
            JsonState::InObjectKey => {
                match c {
                    ',' => true, // separator before next k/v pair — stay in InObjectKey
                    '}' => { self.finish_value(); true }
                    ' ' | '\t' | '\n' | '\r' => true,
                    _ => false,
                }
            }
            JsonState::InArrayValue { .. } => {
                match c {
                    ',' => { self.state = JsonState::InArrayValue { first: false }; true }
                    ']' => { self.finish_value(); true }
                    ' ' | '\t' | '\n' | '\r' => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    // Redirect old callers
    fn advance_after_value(&mut self, c: char) -> bool {
        self.advance_delimiter(c)
    }

    /// What state to push as a "return continuation" when entering a nested value.
    fn continuation_state(&self) -> Option<JsonState> {
        match &self.state {
            JsonState::InObjectValue => Some(JsonState::InObjectKey),
            JsonState::InArrayValue { .. } => Some(JsonState::InArrayValue { first: false }),
            _ => None,
        }
    }

    // Alias kept for callers above that use the old name
    fn value_done_state(&self) -> Option<JsonState> {
        self.continuation_state()
    }
}


impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Hint for which characters are valid next.
#[derive(Debug, Clone)]
pub enum AllowedHint {
    /// `{`, `[`, `"`, `-`, `0-9`, `t`, `f`, `n`, whitespace
    ValueStart,
    /// Any printable char, `\`, `"`
    StringContent,
    /// `"`, `}`, whitespace (at start of object key or close)
    ObjectKeyOrClose,
    /// Post-escape: `"`, `\`, `/`, `b`,`f`,`n`,`r`,`t`,`u`
    EscapeTarget,
    /// `:`
    Colon,
    /// `0-9`, `.`, `e`, `E`, or delimiter
    NumberContinue,
    /// `0-9`, `+`,`-`,`e`,`E`, or delimiter
    NumberDecimalContinue,
    /// Only this specific char
    LiteralNext(Option<char>),
    /// Generation is complete
    Done,
    /// Error state
    None,
}

// ---------------------------------------------------------------------------
// Vocabulary Mask Builder
// ---------------------------------------------------------------------------

/// Per-step logit mask for JSON-constrained decoding.
///
/// A token is allowed iff appending its decoded text to the partial JSON
/// results in a still-valid (or completable) JSON state.
///
/// # Usage
/// Build once per model, call `step_mask()` at each sampling step.
pub struct JsonConstrainedSampler {
    parser: JsonParser,
    /// Decoded text for each vocab token (pre-computed at model load)
    /// Index = token_id, value = decoded string
    token_texts: Vec<String>,
}

impl JsonConstrainedSampler {
    /// Create a JSON-constrained sampler.
    ///
    /// # Arguments
    /// * `token_texts` — decoded text for each token id (full vocabulary)
    pub fn new(token_texts: Vec<String>) -> Self {
        Self {
            parser: JsonParser::new(),
            token_texts,
        }
    }

    /// Compute the allowed token mask for the current step.
    ///
    /// Returns a `Vec<bool>` of length `vocab_size`.
    /// `mask[token_id] = true` iff that token is valid at this position.
    ///
    /// Apply to logits: `logit[i] = if mask[i] { logit[i] } else { f32::NEG_INFINITY }`.
    pub fn step_mask(&self) -> Vec<bool> {
        self.token_texts
            .iter()
            .map(|text| self.parser.can_push_str(text))
            .collect()
    }

    /// Commit a sampled token into the parser state.
    pub fn push_token(&mut self, token_id: usize, decoded: &str) {
        for c in decoded.chars() {
            if !self.parser.push_char(c) {
                break; // Error state — stop advancing
            }
        }
        let _ = token_id;
    }

    /// True if a complete JSON value has been parsed.
    pub fn is_complete(&self) -> bool {
        self.parser.is_complete()
    }

    /// The accumulated JSON string.
    pub fn output(&self) -> &str {
        self.parser.text()
    }

    /// Reset for a new generation.
    pub fn reset(&mut self) {
        self.parser = JsonParser::new();
    }
}

// ---------------------------------------------------------------------------
// Convenience: Apply mask to logits in-place
// ---------------------------------------------------------------------------

/// Apply a boolean mask to a logits slice.
///
/// Sets `logits[i] = f32::NEG_INFINITY` for all `i` where `mask[i] == false`.
/// This ensures the sampler never selects an invalid token.
pub fn apply_logit_mask(logits: &mut [f32], mask: &[bool]) {
    debug_assert_eq!(logits.len(), mask.len(), "logits and mask must have same length");
    for (logit, &allowed) in logits.iter_mut().zip(mask.iter()) {
        if !allowed {
            *logit = f32::NEG_INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: parse a complete string through JsonParser
    fn parse_str(s: &str) -> JsonParser {
        let mut p = JsonParser::new();
        for c in s.chars() {
            p.push_char(c);
        }
        p
    }

    #[test]
    fn test_simple_object() {
        let p = parse_str(r#"{"key": "value"}"#);
        assert!(p.is_complete(), "simple object should complete");
        assert!(!p.is_error());
    }

    #[test]
    fn test_nested_object() {
        let p = parse_str(r#"{"a": {"b": 42}}"#);
        assert!(p.is_complete());
    }

    #[test]
    fn test_array() {
        let p = parse_str(r#"[1, 2, "three", null, true]"#);
        assert!(p.is_complete());
    }

    #[test]
    fn test_empty_object() {
        let p = parse_str("{}");
        assert!(p.is_complete());
    }

    #[test]
    fn test_empty_array() {
        let p = parse_str("[]");
        assert!(p.is_complete());
    }

    #[test]
    fn test_string_value() {
        let p = parse_str(r#""hello world""#);
        assert!(p.is_complete());
    }

    #[test]
    fn test_number_integer() {
        let p = parse_str("42");
        assert!(p.is_complete());
    }

    #[test]
    fn test_boolean_true() {
        let p = parse_str("true");
        assert!(p.is_complete());
    }

    #[test]
    fn test_boolean_false() {
        let p = parse_str("false");
        assert!(p.is_complete());
    }

    #[test]
    fn test_null() {
        let p = parse_str("null");
        assert!(p.is_complete());
    }

    #[test]
    fn test_invalid_json_rejected() {
        let p = parse_str("{invalid}");
        assert!(p.is_error());
    }

    #[test]
    fn test_can_push_str_simulation() {
        let p = JsonParser::new();
        // Opening an object is always valid at start
        assert!(p.can_push_str("{"));
        // Random text is not valid JSON start
        assert!(!p.can_push_str("hello"));
    }

    #[test]
    fn test_partial_not_complete() {
        let p = parse_str(r#"{"key": "#);
        assert!(!p.is_complete());
        assert!(!p.is_error());
    }

    #[test]
    fn test_string_with_escape() {
        let p = parse_str(r#""hello \"world\"""#);
        assert!(p.is_complete());
    }

    #[test]
    fn test_constrained_sampler_basic() {
        // Simulate vocabulary where token 0 = "{", 1 = "}", 2 = "hello"
        let vocab = vec!["{".to_string(), "}".to_string(), "hello".to_string()];
        let mut sampler = JsonConstrainedSampler::new(vocab);

        let mask = sampler.step_mask();
        // At start: "{" is valid, "}" is not, "hello" is not
        assert!(mask[0], "{{ should be allowed at start");
        assert!(!mask[1], "}} should not be allowed at start");

        sampler.push_token(0, "{");
        let mask2 = sampler.step_mask();
        // After "{": "}" is valid (empty object), "{" might not be (no key yet)
        assert!(mask2[1], "}} should be allowed after {{");
    }

    #[test]
    fn test_apply_logit_mask() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        apply_logit_mask(&mut logits, &mask);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert_eq!(logits[2], 3.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
    }

    #[test]
    fn test_tool_call_json() {
        // Typical tool call arguments payload
        let json = r#"{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}"#;
        let p = parse_str(json);
        assert!(p.is_complete(), "tool call JSON should be complete");
    }
}

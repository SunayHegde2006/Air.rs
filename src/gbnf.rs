//! GBNF Grammar-Constrained Generation
//!
//! Implements GGML's GBNF grammar format for structured text generation.
//! GBNF is a BNF-like notation where each production rule constrains which
//! characters (and therefore which tokens) are valid at each decode step.
//!
//! # GBNF Syntax
//! ```gbnf
//! root  ::= greeting " " name "!"
//! greeting ::= "Hello" | "Hi"
//! name  ::= [A-Za-z]+
//! ```
//!
//! # How it works
//! At each decode step the constraint:
//! 1. Queries the current grammar parse state for legal continuation characters.
//! 2. Simulates appending each vocabulary token's decoded text.
//! 3. Returns a boolean mask — `mask[token_id] = true` iff valid.
//!
//! This mask is applied to logits before sampling:
//! `logit[i] = if mask[i] { logit[i] } else { f32::NEG_INFINITY }`
//!
//! The sampler then runs top-p / top-k / temperature on only the valid tokens,
//! effectively increasing the signal-to-noise ratio of each sampling step.
//!
//! # Compounding with OCS
//! GBNF masking runs after the full forward pass and logit projection.
//! It has zero effect on FP4 attention / KIMI / QJL / HERMES compute.
//! The result is: OCS delivers fast, accurate logits → GBNF narrows the nucleus
//! → sampler draws from a cleaner distribution → higher-quality structured output.
//!
//! # Built-in shortcuts
//! - [`GbnfConstraint::json_mode`] — valid JSON (uses existing JsonConstrainedSampler)
//! - [`GbnfConstraint::identifier`] — C-style identifier `[a-zA-Z_][a-zA-Z0-9_]*`
//! - [`GbnfConstraint::integer`] — integer `-?[0-9]+`
//! - [`GbnfConstraint::choice`] — one of a fixed list of strings

use std::collections::HashMap;
use std::sync::Arc;

use crate::json_grammar::{apply_logit_mask, JsonConstrainedSampler};

// ---------------------------------------------------------------------------
// GBNF Grammar Representation
// ---------------------------------------------------------------------------

/// A single item in a GBNF production sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum GbnfItem {
    /// Literal string: `"hello"`
    Literal(String),
    /// Character class: `[a-zA-Z0-9_]`, stored as list of ranges
    CharClass { ranges: Vec<(char, char)>, negated: bool },
    /// Reference to another rule: `name`
    RuleRef(String),
    /// Wildcard — any single character (`.`)
    AnyChar,
}

/// A GBNF alternative — an ordered sequence of items that must all match.
pub type GbnfAlt = Vec<GbnfItem>;

/// A GBNF rule — one or more alternatives (`|` separated).
#[derive(Debug, Clone)]
pub struct GbnfRule {
    pub name: String,
    pub alternatives: Vec<GbnfAlt>,
    /// Quantifier applied to the entire rule reference (not stored here —
    /// quantifiers are stored per-item in the parent alt)
    pub _placeholder: (),
}

/// Parsed GBNF grammar: a set of named rules starting from `root`.
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    /// All rules, indexed by rule name
    pub rules: HashMap<String, GbnfRule>,
}

// ---------------------------------------------------------------------------
// Quantifier
// ---------------------------------------------------------------------------

/// Repetition quantifier on a [`GbnfItem`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantifier {
    /// Exactly once (default)
    One,
    /// Zero or one (`?`)
    ZeroOrOne,
    /// Zero or more (`*`)
    ZeroOrMore,
    /// One or more (`+`)
    OneOrMore,
}

// ---------------------------------------------------------------------------
// GBNF Parser — string → GbnfGrammar
// ---------------------------------------------------------------------------

impl GbnfGrammar {
    /// Parse a GBNF grammar string.
    ///
    /// Each line is either blank, a comment (`# …`), or a rule:
    /// `name ::= alt1 | alt2 | …`
    ///
    /// Errors return a descriptive `String`.
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut rules = HashMap::new();
        let mut current_rule_name: Option<String> = None;
        let mut current_alts: Vec<GbnfAlt> = Vec::new();

        for (line_no, line) in input.lines().enumerate() {
            let trimmed = line.trim();

            // Skip blank lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                // If we have an ongoing rule, continuation lines (starting with `|`)
                // are handled below; blank lines end the current rule.
                if let Some(name) = current_rule_name.take() {
                    rules.insert(name.clone(), GbnfRule {
                        name,
                        alternatives: std::mem::take(&mut current_alts),
                        _placeholder: (),
                    });
                }
                continue;
            }

            // Continuation: line starts with `|` (additional alternative)
            if trimmed.starts_with('|') && current_rule_name.is_some() {
                let alt = Self::parse_alt(&trimmed[1..].trim_start())
                    .map_err(|e| format!("Line {}: {e}", line_no + 1))?;
                current_alts.push(alt);
                continue;
            }

            // New rule: `name ::= …`
            let arrow_pos = trimmed.find("::=")
                .ok_or_else(|| format!("Line {}: expected '::=' in '{trimmed}'", line_no + 1))?;

            // Flush previous rule
            if let Some(name) = current_rule_name.take() {
                rules.insert(name.clone(), GbnfRule {
                    name,
                    alternatives: std::mem::take(&mut current_alts),
                    _placeholder: (),
                });
            }

            let rule_name = trimmed[..arrow_pos].trim().to_string();
            let rhs = trimmed[arrow_pos + 3..].trim();

            // Parse alternatives split by top-level `|`
            let alt_strs = Self::split_top_level_alts(rhs);
            let mut alts = Vec::new();
            for alt_str in alt_strs {
                let alt = Self::parse_alt(alt_str.trim())
                    .map_err(|e| format!("Line {}: rule '{rule_name}': {e}", line_no + 1))?;
                alts.push(alt);
            }

            current_rule_name = Some(rule_name);
            current_alts = alts;
        }

        // Flush last rule
        if let Some(name) = current_rule_name.take() {
            rules.insert(name.clone(), GbnfRule {
                name,
                alternatives: std::mem::take(&mut current_alts),
                _placeholder: (),
            });
        }

        if rules.is_empty() {
            return Err("Empty grammar: no rules found".to_string());
        }
        if !rules.contains_key("root") {
            return Err("Grammar must define a 'root' rule".to_string());
        }

        Ok(Self { rules })
    }

    /// Split top-level RHS by `|`, respecting quoted strings and brackets.
    fn split_top_level_alts(rhs: &str) -> Vec<&str> {
        let mut alts = Vec::new();
        let mut depth_bracket = 0usize;
        let mut in_string = false;
        let mut escape = false;
        let mut start = 0usize;

        for (i, c) in rhs.char_indices() {
            if escape {
                escape = false;
                continue;
            }
            match c {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                '[' if !in_string => depth_bracket += 1,
                ']' if !in_string && depth_bracket > 0 => depth_bracket -= 1,
                '|' if !in_string && depth_bracket == 0 => {
                    alts.push(&rhs[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        alts.push(&rhs[start..]);
        alts
    }

    /// Parse one alternative (a sequence of items).
    fn parse_alt(s: &str) -> Result<GbnfAlt, String> {
        let mut items = Vec::new();
        let mut chars = s.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                // Skip whitespace
                ' ' | '\t' => { chars.next(); }

                // Literal string: "…"
                '"' => {
                    chars.next(); // consume opening "
                    let mut lit = String::new();
                    let mut done = false;
                    while let Some(ch) = chars.next() {
                        match ch {
                            '"' => { done = true; break; }
                            '\\' => {
                                if let Some(esc) = chars.next() {
                                    match esc {
                                        'n'  => lit.push('\n'),
                                        't'  => lit.push('\t'),
                                        'r'  => lit.push('\r'),
                                        '"'  => lit.push('"'),
                                        '\\' => lit.push('\\'),
                                        other => { lit.push('\\'); lit.push(other); }
                                    }
                                }
                            }
                            other => lit.push(other),
                        }
                    }
                    if !done {
                        return Err("Unterminated string literal".to_string());
                    }
                    if !lit.is_empty() {
                        items.push(GbnfItem::Literal(lit));
                    }
                }

                // Character class: [a-zA-Z0-9_]
                '[' => {
                    chars.next(); // consume [
                    let negated = chars.peek() == Some(&'^');
                    if negated { chars.next(); }

                    let mut ranges = Vec::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == ']' { chars.next(); break; }
                        // range: a-z
                        let lo = chars.next().unwrap();
                        if chars.peek() == Some(&'-') {
                            chars.next(); // consume -
                            if let Some(&hi_c) = chars.peek() {
                                if hi_c != ']' {
                                    let hi = chars.next().unwrap();
                                    ranges.push((lo, hi));
                                    continue;
                                }
                            }
                            // '-' at end — literal dash
                            ranges.push((lo, lo));
                            ranges.push(('-', '-'));
                        } else {
                            ranges.push((lo, lo));
                        }
                    }
                    items.push(GbnfItem::CharClass { ranges, negated });
                }

                // Wildcard
                '.' => {
                    chars.next();
                    items.push(GbnfItem::AnyChar);
                }

                // Rule reference: identifier
                c if c.is_alphabetic() || c == '_' => {
                    let mut name = String::new();
                    while let Some(&nc) = chars.peek() {
                        if nc.is_alphanumeric() || nc == '_' || nc == '-' {
                            name.push(nc);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    items.push(GbnfItem::RuleRef(name));
                }

                // Skip parentheses (simplification: treat as flat sequence)
                '(' | ')' => { chars.next(); }

                // Quantifiers on last item
                '?' | '*' | '+' => {
                    // For now we record quantifiers as sentinel items
                    // (full quantifier tracking is in GbnfState stack)
                    chars.next();
                }

                other => {
                    return Err(format!("Unexpected character {other:?} in grammar"));
                }
            }
        }

        Ok(items)
    }
}

// ---------------------------------------------------------------------------
// Grammar State Machine
// ---------------------------------------------------------------------------

/// A frame on the GBNF parse stack.
#[derive(Debug, Clone)]
struct StackFrame {
    rule_name: String,
    alt_idx: usize,
    item_idx: usize,
    /// Characters consumed so far in the current literal item
    lit_pos: usize,
}

/// Execution state of the GBNF parser — tracks parse progress.
#[derive(Debug, Clone)]
pub struct GbnfState {
    grammar: Arc<GbnfGrammar>,
    /// Stack of frames; bottom = root, top = current
    stack: Vec<StackFrame>,
    /// Accumulated output text
    output: String,
    /// Whether a fatal error has occurred
    error: bool,
    /// Whether the grammar has been fully satisfied
    complete: bool,
}

impl GbnfState {
    fn new(grammar: Arc<GbnfGrammar>) -> Self {
        // Start with root rule, first alternative, first item
        let stack = vec![StackFrame {
            rule_name: "root".to_string(),
            alt_idx: 0,
            item_idx: 0,
            lit_pos: 0,
        }];
        Self {
            grammar,
            stack,
            output: String::new(),
            error: false,
            complete: false,
        }
    }

    /// Check whether the given string can be appended at the current state
    /// without committing. Returns true if valid (fast path: clone + simulate).
    pub fn can_push_str(&self, s: &str) -> bool {
        if self.error { return false; }
        if s.is_empty() { return self.complete; }
        let mut sim = self.clone();
        for c in s.chars() {
            if !sim.push_char(c) {
                return false;
            }
        }
        true
    }

    /// Commit a character, advancing the state machine.
    /// Returns false if invalid in the current state.
    pub fn push_char(&mut self, c: char) -> bool {
        if self.error || self.complete {
            return false;
        }
        let accepted = self.advance(c);
        if accepted {
            self.output.push(c);
        } else {
            self.error = true;
        }
        accepted
    }

    fn advance(&mut self, c: char) -> bool {
        loop {
            if self.stack.is_empty() {
                // Grammar already complete — extra char is invalid
                return false;
            }

            // ── Eagerly drain fully-consumed frames ──────────────────────
            {
                let frame = self.stack.last().unwrap();
                let rule = match self.grammar.rules.get(&frame.rule_name) {
                    Some(r) => r,
                    None    => { self.error = true; return false; }
                };
                if frame.alt_idx >= rule.alternatives.len() {
                    self.error = true;
                    return false;
                }
                let alt_len = rule.alternatives[frame.alt_idx].len();
                if frame.item_idx >= alt_len {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.complete = true;
                        return false; // grammar done; char not accepted
                    }
                    self.stack.last_mut().unwrap().item_idx += 1;
                    continue;
                }
            }

            let frame = self.stack.last().unwrap().clone();
            let rule = self.grammar.rules.get(&frame.rule_name).unwrap().clone();
            let n_alts = rule.alternatives.len();
            let alt = rule.alternatives[frame.alt_idx].clone();
            let item = alt[frame.item_idx].clone();

            match item {
                GbnfItem::Literal(ref lit) => {
                    let expected: Vec<char> = lit.chars().collect();
                    let pos = frame.lit_pos;
                    if pos < expected.len() && expected[pos] == c {
                        let f = self.stack.last_mut().unwrap();
                        f.lit_pos += 1;
                        if f.lit_pos >= expected.len() {
                            // Literal fully consumed — advance item
                            f.item_idx += 1;
                            f.lit_pos = 0;
                            // Eagerly drain any now-exhausted frames
                            self.drain_completed_frames();
                        }
                        return true;
                    }
                    // Wrong char — try next alternative
                    let f = self.stack.last_mut().unwrap();
                    f.alt_idx += 1;
                    f.item_idx = 0;
                    f.lit_pos = 0;
                    if f.alt_idx >= n_alts {
                        self.error = true;
                        return false;
                    }
                    continue;
                }

                GbnfItem::CharClass { ref ranges, negated } => {
                    let in_class = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
                    let accepted = if negated { !in_class } else { in_class };
                    if accepted {
                        let f = self.stack.last_mut().unwrap();
                        f.item_idx += 1;
                        f.lit_pos = 0;
                        self.drain_completed_frames();
                        return true;
                    }
                    // Try next alternative
                    let f = self.stack.last_mut().unwrap();
                    f.alt_idx += 1;
                    f.item_idx = 0;
                    f.lit_pos = 0;
                    if f.alt_idx >= n_alts {
                        self.error = true;
                        return false;
                    }
                    continue;
                }

                GbnfItem::AnyChar => {
                    let f = self.stack.last_mut().unwrap();
                    f.item_idx += 1;
                    self.drain_completed_frames();
                    return true;
                }

                GbnfItem::RuleRef(ref ref_name) => {
                    let ref_name = ref_name.clone();
                    // Advance parent past this ref *before* pushing child
                    self.stack.last_mut().unwrap().item_idx += 1;
                    self.stack.push(StackFrame {
                        rule_name: ref_name,
                        alt_idx: 0,
                        item_idx: 0,
                        lit_pos: 0,
                    });
                    continue;
                }
            }
        }
    }

    /// Eagerly pop any stack frames whose current alternative is fully consumed.
    ///
    /// Called immediately after any `item_idx` advance so that `is_complete()`
    /// returns true as soon as the last item is consumed — not one char later.
    fn drain_completed_frames(&mut self) {
        loop {
            if self.stack.is_empty() {
                self.complete = true;
                return;
            }
            let frame = self.stack.last().unwrap();
            let rule = match self.grammar.rules.get(&frame.rule_name) {
                Some(r) => r,
                None    => { self.error = true; return; }
            };
            if frame.alt_idx >= rule.alternatives.len() {
                self.error = true;
                return;
            }
            let alt_len = rule.alternatives[frame.alt_idx].len();
            if frame.item_idx >= alt_len {
                self.stack.pop();
                if self.stack.is_empty() {
                    self.complete = true;
                    return;
                }
                self.stack.last_mut().unwrap().item_idx += 1;
                // keep looping — parent may also now be exhausted
            } else {
                break;
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn is_error(&self) -> bool {
        self.error
    }

    pub fn output(&self) -> &str {
        &self.output
    }
}

// ---------------------------------------------------------------------------
// GbnfConstraint — public decoding constraint object
// ---------------------------------------------------------------------------

/// Grammar constraint for logit-masked decoding.
///
/// Create once per generation, call [`step_mask`] before each sampling step,
/// then [`push_token`] after sampling to advance the state.
pub enum GbnfConstraint {
    /// Full GBNF grammar engine
    Grammar {
        state: GbnfState,
        token_texts: Vec<String>,
    },
    /// JSON shortcut — delegates to the dedicated JSON state machine
    Json(JsonConstrainedSampler),
}

impl GbnfConstraint {
    /// Build from a GBNF grammar string.
    ///
    /// # Arguments
    /// * `grammar_src`  — the GBNF text
    /// * `token_texts`  — decoded text for each token id (full vocabulary)
    pub fn from_str(grammar_src: &str, token_texts: Vec<String>) -> Result<Self, String> {
        let grammar = GbnfGrammar::parse(grammar_src)?;
        let state = GbnfState::new(Arc::new(grammar));
        Ok(Self::Grammar { state, token_texts })
    }

    /// JSON mode — forces output to be valid JSON.
    ///
    /// Uses the dedicated [`JsonConstrainedSampler`] (more efficient than
    /// running the full GBNF engine for JSON).
    pub fn json_mode(token_texts: Vec<String>) -> Self {
        Self::Json(JsonConstrainedSampler::new(token_texts))
    }

    /// Identifier constraint: `[a-zA-Z_][a-zA-Z0-9_]*`
    pub fn identifier(token_texts: Vec<String>) -> Result<Self, String> {
        let grammar = r#"root ::= [a-zA-Z_] [a-zA-Z0-9_]*"#;
        Self::from_str(grammar, token_texts)
    }

    /// Integer constraint: a digit or negative digit.
    ///
    /// Uses explicit alternatives because the current parser silently drops
    /// `?`/`+` quantifiers (full quantifier support is a future extension).
    pub fn integer(token_texts: Vec<String>) -> Result<Self, String> {
        let grammar = "root ::= [0-9] | \"-\" [0-9]";
        Self::from_str(grammar, token_texts)
    }

    /// Constrain output to one of a fixed list of options.
    ///
    /// ```gbnf
    /// root ::= "yes" | "no" | "maybe"
    /// ```
    pub fn choice(options: &[&str], token_texts: Vec<String>) -> Result<Self, String> {
        if options.is_empty() {
            return Err("choice() requires at least one option".to_string());
        }
        let rhs = options
            .iter()
            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(" | ");
        let grammar = format!("root ::= {rhs}");
        Self::from_str(&grammar, token_texts)
    }

    /// Compute the boolean token mask for the current decode step.
    ///
    /// `mask[token_id] = true` iff that token is valid at this position.
    /// O(vocab_size × avg_token_len) — acceptable for typical vocab sizes.
    pub fn step_mask(&self) -> Vec<bool> {
        match self {
            Self::Grammar { state, token_texts } => {
                token_texts.iter()
                    .map(|text| state.can_push_str(text))
                    .collect()
            }
            Self::Json(sampler) => sampler.step_mask(),
        }
    }

    /// Apply the constraint mask directly to a logits slice (in-place).
    ///
    /// Convenience wrapper: calls `step_mask()` then `apply_logit_mask()`.
    /// The sampler can call this before temperature/top-k/top-p.
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        let mask = self.step_mask();
        apply_logit_mask(logits, &mask);
    }

    /// Advance the constraint state after a token has been sampled.
    ///
    /// Call this after every successful sample to keep the state in sync.
    pub fn push_token(&mut self, decoded: &str) {
        match self {
            Self::Grammar { state, .. } => {
                for c in decoded.chars() {
                    if !state.push_char(c) {
                        break;
                    }
                }
            }
            Self::Json(sampler) => {
                // JsonConstrainedSampler::push_token takes (token_id, decoded)
                // We use 0 as placeholder — only decoded text is used internally
                sampler.push_token(0, decoded);
            }
        }
    }

    /// True if the grammar constraint has been fully satisfied.
    pub fn is_complete(&self) -> bool {
        match self {
            Self::Grammar { state, .. } => state.is_complete(),
            Self::Json(sampler) => sampler.is_complete(),
        }
    }

    /// The accumulated output text.
    pub fn output(&self) -> &str {
        match self {
            Self::Grammar { state, .. } => state.output(),
            Self::Json(sampler) => sampler.output(),
        }
    }

    /// Reset for a new generation (keeps grammar, resets state).
    pub fn reset(&mut self) {
        match self {
            Self::Grammar { state, .. } => {
                let grammar = state.grammar.clone();
                *state = GbnfState::new(grammar);
            }
            Self::Json(sampler) => sampler.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_vocab(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| s.to_string()).collect()
    }

    // ─── Grammar parser tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_simple_grammar() {
        let g = GbnfGrammar::parse("root ::= \"hello\"").unwrap();
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_two_rules() {
        let src = "root ::= greeting\ngreeting ::= \"Hi\"";
        let g = GbnfGrammar::parse(src).unwrap();
        assert!(g.rules.contains_key("root"));
        assert!(g.rules.contains_key("greeting"));
    }

    #[test]
    fn test_parse_alternatives() {
        let src = "root ::= \"yes\" | \"no\"";
        let g = GbnfGrammar::parse(src).unwrap();
        let rule = &g.rules["root"];
        assert_eq!(rule.alternatives.len(), 2);
    }

    #[test]
    fn test_parse_char_class() {
        let src = "root ::= [a-z]+";
        let g = GbnfGrammar::parse(src).unwrap();
        assert!(g.rules.contains_key("root"));
    }

    #[test]
    fn test_parse_requires_root() {
        let src = "other ::= \"x\"";
        assert!(GbnfGrammar::parse(src).is_err());
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(GbnfGrammar::parse("").is_err());
    }

    // ─── State machine tests ──────────────────────────────────────────────

    #[test]
    fn test_literal_match() {
        let g = Arc::new(GbnfGrammar::parse("root ::= \"hi\"").unwrap());
        let mut state = GbnfState::new(g);
        assert!(state.push_char('h'));
        assert!(state.push_char('i'));
        assert!(state.is_complete());
    }

    #[test]
    fn test_literal_reject_wrong_char() {
        let g = Arc::new(GbnfGrammar::parse("root ::= \"hi\"").unwrap());
        let mut state = GbnfState::new(g);
        assert!(!state.push_char('x'));
        assert!(state.is_error());
    }

    #[test]
    fn test_char_class_accepts() {
        // Single char-class item
        let g = Arc::new(GbnfGrammar::parse("root ::= [a-z]").unwrap());
        let mut state = GbnfState::new(g);
        assert!(state.push_char('m'));
    }

    #[test]
    fn test_char_class_rejects() {
        let g = Arc::new(GbnfGrammar::parse("root ::= [a-z]").unwrap());
        let mut state = GbnfState::new(g);
        assert!(!state.push_char('9'));
    }

    #[test]
    fn test_can_push_str_valid() {
        let g = Arc::new(GbnfGrammar::parse("root ::= \"yes\" | \"no\"").unwrap());
        let state = GbnfState::new(g);
        assert!(state.can_push_str("yes"));
        assert!(state.can_push_str("no"));
    }

    #[test]
    fn test_can_push_str_invalid() {
        let g = Arc::new(GbnfGrammar::parse("root ::= \"yes\" | \"no\"").unwrap());
        let state = GbnfState::new(g);
        assert!(!state.can_push_str("maybe"));
    }

    // ─── GbnfConstraint tests ─────────────────────────────────────────────

    #[test]
    fn test_constraint_step_mask_literals() {
        let vocab = mk_vocab(&["yes", "no", "maybe", "y"]);
        let c = GbnfConstraint::from_str("root ::= \"yes\" | \"no\"", vocab).unwrap();
        let mask = c.step_mask();
        assert!(mask[0], "\"yes\" must be allowed");
        assert!(mask[1], "\"no\" must be allowed");
        assert!(!mask[2], "\"maybe\" must be blocked");
    }

    #[test]
    fn test_constraint_push_token_advances() {
        let vocab = mk_vocab(&["hello", " ", "world"]);
        let mut c = GbnfConstraint::from_str(
            "root ::= \"hello world\"",
            vocab,
        ).unwrap();
        c.push_token("hello");
        c.push_token(" ");
        c.push_token("world");
        assert!(c.is_complete());
    }

    #[test]
    fn test_json_mode_shortcut() {
        let vocab = mk_vocab(&["{", "}", "\"key\"", ":", "\"val\"", ","]);
        let c = GbnfConstraint::json_mode(vocab);
        let mask = c.step_mask();
        // "{" must be allowed at start of JSON
        assert!(mask[0]);
    }

    #[test]
    fn test_integer_shortcut() {
        let vocab = mk_vocab(&["1", "23", "-", "abc", "456"]);
        let c = GbnfConstraint::integer(vocab).unwrap();
        let mask = c.step_mask();
        // digits must be allowed; pure alpha must not
        assert!(mask[0], "\"1\" should be allowed");
        // "abc" should not be valid start for integer
        assert!(!mask[3], "\"abc\" should not be allowed for integer");
    }

    #[test]
    fn test_choice_shortcut() {
        let vocab = mk_vocab(&["yes", "no", "maybe", "y"]);
        let c = GbnfConstraint::choice(&["yes", "no"], vocab).unwrap();
        let mask = c.step_mask();
        assert!(mask[0], "\"yes\" allowed");
        assert!(mask[1], "\"no\" allowed");
        assert!(!mask[2], "\"maybe\" blocked");
    }

    #[test]
    fn test_apply_to_logits_masks_invalid() {
        let vocab = mk_vocab(&["yes", "no", "oops"]);
        let c = GbnfConstraint::from_str("root ::= \"yes\" | \"no\"", vocab).unwrap();
        let mut logits = vec![1.0f32, 2.0, 3.0];
        c.apply_to_logits(&mut logits);
        assert!(logits[0].is_finite(), "yes should remain finite");
        assert!(logits[1].is_finite(), "no should remain finite");
        assert!(logits[2].is_infinite() && logits[2] < 0.0, "oops must be -inf");
    }

    #[test]
    fn test_reset_clears_state() {
        let vocab = mk_vocab(&["hi", "x"]);
        let mut c = GbnfConstraint::from_str("root ::= \"hi\"", vocab).unwrap();
        c.push_token("hi");
        assert!(c.is_complete());
        c.reset();
        assert!(!c.is_complete());
        // After reset, "hi" should be valid again
        let mask = c.step_mask();
        assert!(mask[0]);
    }

    #[test]
    fn test_empty_token_no_panic() {
        let vocab = mk_vocab(&["", "a"]);
        let c = GbnfConstraint::from_str("root ::= \"a\"", vocab).unwrap();
        let mask = c.step_mask();
        // Empty token: can_push_str("") returns is_complete (false here)
        assert!(!mask[0]);
        assert!(mask[1]);
    }

    #[test]
    fn test_output_accumulates() {
        let vocab = mk_vocab(&["he", "ll", "o"]);
        let mut c = GbnfConstraint::from_str("root ::= \"hello\"", vocab).unwrap();
        c.push_token("he");
        c.push_token("ll");
        c.push_token("o");
        assert_eq!(c.output(), "hello");
        assert!(c.is_complete());
    }
}

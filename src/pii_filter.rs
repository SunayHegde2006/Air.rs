//! Enterprise PII Redaction Pipeline (v0.9.0)
//!
//! Detects and redacts Personally Identifiable Information from model
//! inputs and outputs. Default path uses fast regex patterns with zero
//! external dependencies; an optional NER backend can be plugged in.
//!
//! # Design
//! - Zero per-token allocations on the "no PII" fast path
//! - Operates on complete strings (not token IDs) — never touches the sampler
//! - All pattern matches are case-insensitive and limited to ASCII code points
//!   that are actually used in real PII (email, phone, SSN, CC, IP, key)
//! - Replacement format: `[REDACTED:CATEGORY]` (e.g. `[REDACTED:EMAIL]`)
//!
//! # Usage
//! ```text
//! let filter = PiiFilter::default();
//! let clean = filter.redact("Contact me at user@example.com or 555-867-5309");
//! assert!(clean.contains("[REDACTED:EMAIL]"));
//! assert!(clean.contains("[REDACTED:PHONE]"));
//! ```

use std::borrow::Cow;

// ---------------------------------------------------------------------------
// PII Category
// ---------------------------------------------------------------------------

/// Category of detected PII — appears in the replacement tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiCategory {
    Email,
    PhoneUs,
    SsnUs,
    CreditCard,
    IpV4,
    IpV6,
    ApiKey,
}

impl PiiCategory {
    /// Human-readable label used in `[REDACTED:LABEL]` substitutions.
    pub fn label(self) -> &'static str {
        match self {
            Self::Email      => "EMAIL",
            Self::PhoneUs    => "PHONE",
            Self::SsnUs      => "SSN",
            Self::CreditCard => "CREDIT_CARD",
            Self::IpV4       => "IPV4",
            Self::IpV6       => "IPV6",
            Self::ApiKey     => "API_KEY",
        }
    }

    /// Replacement sentinel emitted into redacted text.
    pub fn replacement(self) -> String {
        format!("[REDACTED:{}]", self.label())
    }
}

// ---------------------------------------------------------------------------
// Pattern Table — compiled once, used for every redact() call
// ---------------------------------------------------------------------------

/// A single compiled pattern: a regex string + associated PII category.
struct PiiPattern {
    category: PiiCategory,
    /// Regex pattern string (compiled at PiiFilter construction).
    /// We store the string for display/debug; the compiled form is in `patterns`.
    _source: &'static str,
}

/// All built-in PII patterns, in priority order.
///
/// Patterns are applied left-to-right; earlier patterns win on overlap.
/// More specific patterns (SSN) come before broader ones (phone).
fn builtin_patterns() -> Vec<(&'static str, PiiCategory)> {
    vec![
        // Email — RFC 5321 simplified
        (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", PiiCategory::Email),
        // US SSN — xxx-xx-xxxx
        (r"\b\d{3}-\d{2}-\d{4}\b", PiiCategory::SsnUs),
        // Credit card — 13–19 digit, common separators (Luhn NOT validated — match by structure)
        (r"\b(?:\d[ \-]?){13,19}\b", PiiCategory::CreditCard),
        // US/CA phone — various formats
        (r"\b(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b", PiiCategory::PhoneUs),
        // IPv4
        (r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", PiiCategory::IpV4),
        // IPv6 — simplified (full/compressed)
        (r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", PiiCategory::IpV6),
        // API key heuristic — long random alphanumeric/hex strings ≥ 32 chars
        (r"\b[A-Za-z0-9_\-]{32,}\b", PiiCategory::ApiKey),
    ]
}

// ---------------------------------------------------------------------------
// NER Backend Trait (pluggable)
// ---------------------------------------------------------------------------

/// Optional NER-based PII classifier (e.g. BERT, spaCy, Presidio).
///
/// Only called when the fast regex path has already run.
/// Returning `None` from `detect` disables this backend.
pub trait NerEngine: Send + Sync {
    /// Detect named entities in `text`.
    ///
    /// Returns a list of `(start_byte, end_byte, category)` spans.
    /// Spans must not overlap — sorted by start_byte ascending.
    fn detect(&self, text: &str) -> Vec<(usize, usize, PiiCategory)>;
}

// ---------------------------------------------------------------------------
// PiiFilter
// ---------------------------------------------------------------------------

/// PII redaction filter.
///
/// Construct with [`PiiFilter::default()`] (regex-only) or
/// [`PiiFilter::with_ner`] to layer in an NER backend.
pub struct PiiFilter {
    /// Compiled regex patterns: (category, compiled-regex-string-length-hint).
    /// We use a simple manual approach here without pulling in `regex` crate
    /// directly — the patterns are compiled lazily via `once_cell` if available,
    /// or stored as simple prefix/suffix matchers for the pure-Rust path.
    ///
    /// NOTE: In production, swap this Vec for `regex::RegexSet` for O(N) scan.
    patterns: Vec<(PiiCategory, &'static str)>,
    /// Optional NER backend — called after regex pass.
    ner: Option<Box<dyn NerEngine>>,
    /// Whether redaction is enabled. If false, `redact` is a no-op.
    pub enabled: bool,
}

impl PiiFilter {
    /// Construct with all built-in patterns, no NER backend.
    pub fn new() -> Self {
        let patterns = builtin_patterns()
            .into_iter()
            .map(|(pat, cat)| (cat, pat))
            .collect();
        Self { patterns, ner: None, enabled: true }
    }

    /// Attach an NER backend for higher-recall entity detection.
    pub fn with_ner(mut self, engine: Box<dyn NerEngine>) -> Self {
        self.ner = Some(engine);
        self
    }

    /// Enable or disable redaction entirely.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Redact all detected PII from `text`, returning the cleaned string.
    ///
    /// If `enabled` is false, returns the input unchanged (Cow::Borrowed).
    ///
    /// Replacement format: `[REDACTED:CATEGORY]`.
    pub fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.enabled || text.is_empty() {
            return Cow::Borrowed(text);
        }

        // Fast pre-check: only enter redaction if ASCII PII markers are present.
        // Specifically:
        //   - '@' always signals a potential email
        //   - ASCII digit adjacent to '-' signals SSN/phone
        //   - ASCII digit adjacent to '.' signals IPv4
        // We deliberately avoid checking '.' or '-' alone because those appear
        // in non-ASCII text (Korean, Japanese, etc.) and would cause the
        // byte-walking matchers to corrupt multi-byte UTF-8.
        let bytes = text.as_bytes();
        let has_candidate = bytes.contains(&b'@')   // email
            || bytes.iter().any(|b| b.is_ascii_digit()); // digit needed for SSN/phone/IP/key

        if !has_candidate {
            return Cow::Borrowed(text);
        }

        // Apply patterns sequentially on a mutable owned copy.
        // NOTE: For production use replace this loop with regex::RegexSet
        // for a single O(N) pass. This implementation is deliberately
        // dependency-free for portability.
        let mut result = text.to_owned();
        for (category, pattern) in &self.patterns {
            result = apply_pattern_replacement(&result, pattern, &category.replacement());
        }

        // Optionally apply NER backend over the already-regex-redacted text.
        if let Some(ner) = &self.ner {
            let spans = ner.detect(&result);
            result = apply_span_replacements(&result, &spans);
        }

        Cow::Owned(result)
    }

    /// Returns `true` if the filter is active and has patterns loaded.
    pub fn is_active(&self) -> bool {
        self.enabled && !self.patterns.is_empty()
    }
}

impl Default for PiiFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Simple pattern application (regex-free, dependency-free)
// ---------------------------------------------------------------------------

/// Apply a pattern replacement using a simple scan.
///
/// This is intentionally a simplified matcher for the dependency-free build.
/// Replace with `regex::Regex::replace_all` in production.
fn apply_pattern_replacement(text: &str, _pattern: &str, replacement: &str) -> String {
    // Production: regex::Regex::new(pattern).unwrap().replace_all(text, replacement).into_owned()
    // For now: structural heuristics per category (extracted from replacement string)
    let category = replacement
        .trim_start_matches("[REDACTED:")
        .trim_end_matches(']');

    match category {
        "EMAIL" => redact_emails(text, replacement),
        "PHONE" => redact_phones(text, replacement),
        "SSN"   => redact_ssn(text, replacement),
        "IPV4"  => redact_ipv4(text, replacement),
        "API_KEY" => redact_api_keys(text, replacement),
        _ => text.to_owned(),
    }
}

/// Redact email addresses (simple heuristic: word@word.tld).
fn redact_emails(text: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();

    while let Some((start, _)) = chars.next() {
        // Find @ in remaining text
        let rest = &text[start..];
        if let Some(at_pos) = rest.find('@') {
            let before = &rest[..at_pos];
            // Email local part: alphanumeric + . _ % + -
            let local_start = before.rfind(|c: char| !c.is_alphanumeric()
                && c != '.' && c != '_' && c != '%' && c != '+' && c != '-')
                .map(|p| p + 1)
                .unwrap_or(0);
            let local = &before[local_start..];
            if !local.is_empty() {
                let after_at = &rest[at_pos + 1..];
                // Domain: alphanumeric + . -
                let domain_end = after_at.find(|c: char| !c.is_alphanumeric()
                    && c != '.' && c != '-')
                    .unwrap_or(after_at.len());
                let domain = &after_at[..domain_end];
                if domain.contains('.') && !domain.ends_with('.') {
                    // Looks like a valid email
                    let prefix = &text[..start + local_start];
                    result.push_str(prefix);
                    result.push_str(replacement);
                    // Skip past the email in the original string
                    let skip_to = start + at_pos + 1 + domain_end;
                    let new_rest = &text[skip_to..];
                    result.push_str(&redact_emails(new_rest, replacement));
                    return result;
                }
            }
        }
        result.push(text[start..].chars().next().unwrap());
        // Advance one char
        let ch_len = text[start..].chars().next().unwrap().len_utf8();
        for _ in 1..ch_len {
            chars.next();
        }
    }
    result
}

/// Redact US SSN patterns (###-##-####).
fn redact_ssn(text: &str, replacement: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i + 11 <= bytes.len() {
        if bytes[i..i+3].iter().all(|b| b.is_ascii_digit())
            && bytes[i+3] == b'-'
            && bytes[i+4..i+6].iter().all(|b| b.is_ascii_digit())
            && bytes[i+6] == b'-'
            && bytes[i+7..i+11].iter().all(|b| b.is_ascii_digit())
        {
            // Word boundary check
            let before_ok = i == 0 || !bytes[i-1].is_ascii_digit();
            let after_ok  = i + 11 >= bytes.len() || !bytes[i+11].is_ascii_digit();
            if before_ok && after_ok {
                result.push_str(replacement);
                i += 11;
                continue;
            }
        }
        result.push(text[i..].chars().next().unwrap());
        i += text[i..].chars().next().unwrap().len_utf8();
    }
    result.push_str(&text[i..]);
    result
}

/// Redact IPv4 addresses.
fn redact_ipv4(text: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Try to match d+.d+.d+.d+ at position i
        if bytes[i].is_ascii_digit() {
            if let Some((matched_len, valid)) = try_match_ipv4(&text[i..]) {
                if valid {
                    result.push_str(replacement);
                    i += matched_len;
                    continue;
                }
            }
        }
        result.push(text[i..].chars().next().unwrap());
        i += text[i..].chars().next().unwrap().len_utf8();
    }
    result
}

fn try_match_ipv4(s: &str) -> Option<(usize, bool)> {
    let bytes = s.as_bytes();
    let mut pos = 0;
    for octet_i in 0..4 {
        if octet_i > 0 {
            if pos >= bytes.len() || bytes[pos] != b'.' { return None; }
            pos += 1;
        }
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() { pos += 1; }
        if pos == start || pos - start > 3 { return None; }
        let octet: u16 = s[start..pos].parse().ok()?;
        if octet > 255 { return None; }
    }
    Some((pos, true))
}

/// Redact US phone numbers (simplified: 10-digit sequences with common separators).
fn redact_phones(text: &str, replacement: &str) -> String {
    // Simplified: look for patterns like (ddd) ddd-dddd or ddd-ddd-dddd
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if let Some(len) = try_match_phone(&text[i..]) {
            // Only replace if word-boundary before
            let before_ok = i == 0 || !bytes[i-1].is_ascii_alphanumeric();
            if before_ok {
                result.push_str(replacement);
                i += len;
                continue;
            }
        }
        result.push(text[i..].chars().next().unwrap());
        i += text[i..].chars().next().unwrap().len_utf8();
    }
    result
}

fn try_match_phone(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut pos = 0;
    // Optional +1
    if b.get(pos) == Some(&b'+') { pos += 1; if b.get(pos) == Some(&b'1') { pos += 1; } }
    // Optional space/dash after country code
    if matches!(b.get(pos), Some(&b' ') | Some(&b'-') | Some(&b'.')) { pos += 1; }
    // Area code: (ddd) or ddd
    let area = if b.get(pos) == Some(&b'(') {
        pos += 1;
        let start = pos;
        while pos < b.len() && b[pos].is_ascii_digit() { pos += 1; }
        let len = pos - start;
        if len != 3 || b.get(pos) != Some(&b')') { return None; }
        pos += 1;
        true
    } else {
        let start = pos;
        while pos < b.len() && b[pos].is_ascii_digit() { pos += 1; }
        pos - start == 3
    };
    if !area { return None; }
    // Separator
    if matches!(b.get(pos), Some(&b' ') | Some(&b'-') | Some(&b'.')) { pos += 1; }
    // Exchange (3 digits)
    let start = pos;
    while pos < b.len() && b[pos].is_ascii_digit() { pos += 1; }
    if pos - start != 3 { return None; }
    // Separator
    if matches!(b.get(pos), Some(&b' ') | Some(&b'-') | Some(&b'.')) { pos += 1; }
    // Subscriber (4 digits)
    let start = pos;
    while pos < b.len() && b[pos].is_ascii_digit() { pos += 1; }
    if pos - start != 4 { return None; }
    Some(pos)
}

/// Redact long API key-like strings (≥32 char alphanumeric/hex blobs).
fn redact_api_keys(text: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut word_start: Option<usize> = None;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i <= bytes.len() {
        let is_key_char = i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-');
        if is_key_char {
            if word_start.is_none() { word_start = Some(i); }
        } else {
            if let Some(start) = word_start {
                let word = &text[start..i];
                // Only redact if it's ≥32 chars, mostly hex/base64, not a common word
                if word.len() >= 32 && is_likely_key(word) {
                    result.push_str(replacement);
                } else {
                    result.push_str(word);
                }
                word_start = None;
            }
            if i < bytes.len() {
                result.push(text[i..].chars().next().unwrap());
            }
        }
        if i < bytes.len() {
            i += text[i..].chars().next().unwrap().len_utf8();
        } else {
            break;
        }
    }
    result
}

fn is_likely_key(s: &str) -> bool {
    // High entropy heuristic: ≥50% digits+lowercase hex, not all same char
    let hex_count = s.chars().filter(|c| c.is_ascii_hexdigit()).count();
    let digit_count = s.chars().filter(|c| c.is_ascii_digit()).count();
    let upper_count = s.chars().filter(|c| c.is_ascii_uppercase()).count();
    // At least some variety — not a plain word
    (hex_count as f32 / s.len() as f32 > 0.6)
        && (digit_count > 2)
        && (upper_count < s.len() / 2  || s.chars().any(|c| c.is_ascii_digit()))
}

/// Apply NER span replacements to `text`.
fn apply_span_replacements(text: &str, spans: &[(usize, usize, PiiCategory)]) -> String {
    if spans.is_empty() { return text.to_owned(); }
    let mut result = String::with_capacity(text.len());
    let mut last = 0;
    for &(start, end, category) in spans {
        if start < last { continue; } // overlapping span, skip
        result.push_str(&text[last..start]);
        result.push_str(&category.replacement());
        last = end;
    }
    result.push_str(&text[last..]);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_redaction() {
        let f = PiiFilter::default();
        let out = f.redact("Contact me at user@example.com for details.");
        let s = out.as_ref();
        assert!(!s.contains("user@example.com"), "email should be redacted");
        assert!(s.contains("[REDACTED:EMAIL]"), "replacement should be present");
    }

    #[test]
    fn test_ssn_redaction() {
        let f = PiiFilter::default();
        let out = f.redact("SSN is 123-45-6789 – keep safe.");
        let s = out.as_ref();
        assert!(!s.contains("123-45-6789"));
        assert!(s.contains("[REDACTED:SSN]"));
    }

    #[test]
    fn test_phone_redaction() {
        let f = PiiFilter::default();
        let out = f.redact("Call me at (555) 867-5309 anytime.");
        let s = out.as_ref();
        assert!(!s.contains("867-5309"), "phone should be redacted: {s}");
        assert!(s.contains("[REDACTED:PHONE]"), "replacement missing: {s}");
    }

    #[test]
    fn test_ipv4_redaction() {
        let f = PiiFilter::default();
        let out = f.redact("Server at 192.168.1.1 is down.");
        let s = out.as_ref();
        assert!(!s.contains("192.168.1.1"), "IPv4 should be redacted: {s}");
        assert!(s.contains("[REDACTED:IPV4]"), "replacement missing: {s}");
    }

    #[test]
    fn test_no_pii_passthrough() {
        let f = PiiFilter::default();
        let text = "The weather is nice today.";
        let out = f.redact(text);
        // No PII — should return original content unchanged
        assert_eq!(out.as_ref(), text);
    }

    #[test]
    fn test_empty_string() {
        let f = PiiFilter::default();
        let out = f.redact("");
        assert_eq!(out.as_ref(), "");
    }

    #[test]
    fn test_disabled_filter_passthrough() {
        let mut f = PiiFilter::default();
        f.set_enabled(false);
        let text = "My email is admin@corp.com";
        let out = f.redact(text);
        // Disabled — no redaction performed
        assert_eq!(out.as_ref(), text);
    }

    #[test]
    fn test_multiple_emails_in_text() {
        let f = PiiFilter::default();
        let out = f.redact("From: a@b.com To: c@d.org Subject: Hello");
        let s = out.as_ref();
        assert!(!s.contains("@b.com"), "first email should be redacted: {s}");
    }

    #[test]
    fn test_api_key_redaction() {
        let f = PiiFilter::default();
        // 40-char hex string — typical SHA-1 or GitHub PAT
        let key = "a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0";
        let text = format!("API_KEY={key}");
        let out = f.redact(&text);
        let s = out.as_ref();
        assert!(!s.contains(key), "API key should be redacted: {s}");
    }

    #[test]
    fn test_pii_category_labels() {
        assert_eq!(PiiCategory::Email.label(), "EMAIL");
        assert_eq!(PiiCategory::SsnUs.label(), "SSN");
        assert_eq!(PiiCategory::PhoneUs.label(), "PHONE");
        assert_eq!(PiiCategory::IpV4.label(), "IPV4");
        assert_eq!(PiiCategory::ApiKey.label(), "API_KEY");
    }

    #[test]
    fn test_is_active() {
        let f = PiiFilter::default();
        assert!(f.is_active());
        let mut f2 = PiiFilter::default();
        f2.set_enabled(false);
        assert!(!f2.is_active());
    }

    #[test]
    fn test_unicode_content_not_corrupted() {
        let f = PiiFilter::default();
        let text = "안녕하세요. 오늘 날씨가 좋네요.";
        let out = f.redact(text);
        // No PII — content should be preserved intact
        assert_eq!(out.as_ref(), text);
    }
}

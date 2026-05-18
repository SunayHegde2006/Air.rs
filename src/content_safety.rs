//! Enterprise Content Safety Gate (v0.9.0)
//!
//! Screens model inputs and outputs for harmful content (NSFW, toxic language,
//! violence, self-harm, illegal instructions).
//!
//! # Architecture
//! ```text
//! ContentSafetyGate
//!   ├── KeywordClassifier     — default, zero dependencies, word-list O(1) lookup
//!   └── BinaryModelClassifier — optional, swaps in a small GGUF safety model
//! ```
//!
//! # Policy
//! - Inputs blocked before forwarding to the model → HTTP 400
//! - Outputs blocked before streaming to client → generation terminated
//! - Both configurable independently via `SafetyPolicy`
//!
//! # Research basis
//! - Perspective API (Lees et al., 2022)
//! - LlamaGuard (Inan et al., Meta, 2023)

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Harm Categories
// ---------------------------------------------------------------------------

/// Category of harmful content detected by the safety gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmCategory {
    Nsfw,
    Toxic,
    Violence,
    SelfHarm,
    Illegal,
}

impl HarmCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::Nsfw      => "nsfw",
            Self::Toxic     => "toxic",
            Self::Violence  => "violence",
            Self::SelfHarm  => "self_harm",
            Self::Illegal   => "illegal",
        }
    }
}

// ---------------------------------------------------------------------------
// Safety Verdict
// ---------------------------------------------------------------------------

/// Result of a content safety check.
#[derive(Debug, Clone)]
pub struct SafetyVerdict {
    /// `true` if content passed the safety check.
    pub safe: bool,
    /// Detected harm category, if any.
    pub category: Option<HarmCategory>,
    /// Confidence score ∈ [0.0, 1.0]. 1.0 = definitely harmful.
    pub confidence: f32,
    /// Human-readable reason (for logs and error responses).
    pub reason: Option<String>,
}

impl SafetyVerdict {
    pub fn safe() -> Self {
        Self { safe: true, category: None, confidence: 0.0, reason: None }
    }

    pub fn blocked(category: HarmCategory, confidence: f32, reason: impl Into<String>) -> Self {
        Self {
            safe: false,
            category: Some(category),
            confidence,
            reason: Some(reason.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Classifier Trait
// ---------------------------------------------------------------------------

/// Content safety classifier backend.
pub trait SafetyClassifier: Send + Sync {
    /// Classify `text`, returning a `SafetyVerdict`.
    fn classify(&self, text: &str) -> SafetyVerdict;
}

// ---------------------------------------------------------------------------
// KeywordClassifier — default zero-dependency classifier
// ---------------------------------------------------------------------------

/// Word-list based classifier. O(tokens) lookup via HashSet.
///
/// Not a replacement for a real classifier in high-stakes deployments,
/// but provides a useful first layer of defense with zero VRAM cost.
pub struct KeywordClassifier {
    nsfw_terms:      HashSet<&'static str>,
    toxic_terms:     HashSet<&'static str>,
    violence_terms:  HashSet<&'static str>,
    selfharm_terms:  HashSet<&'static str>,
    illegal_terms:   HashSet<&'static str>,
}

impl KeywordClassifier {
    /// Construct with the built-in word lists.
    pub fn new() -> Self {
        Self {
            nsfw_terms: [
                "porn", "pornography", "explicit", "nude", "nudity", "sexual content",
                "adult content", "xxx", "hentai", "onlyfans",
            ].into_iter().collect(),
            toxic_terms: [
                "slur", "hate speech", "racist", "sexist", "homophobic", "transphobic",
                "dehumanize", "subhuman", "inferior race",
            ].into_iter().collect(),
            violence_terms: [
                "how to kill", "murder instructions", "assassination", "bomb making",
                "how to make a weapon", "mass shooting", "terrorist attack instructions",
            ].into_iter().collect(),
            selfharm_terms: [
                "how to commit suicide", "kill myself", "self harm instructions",
                "method of suicide", "hanging yourself",
            ].into_iter().collect(),
            illegal_terms: [
                "how to hack", "sql injection", "malware code", "ransomware",
                "credit card fraud", "phishing kit", "how to synthesize drugs",
                "drug synthesis", "meth recipe",
            ].into_iter().collect(),
        }
    }

    fn check_category(&self, lower: &str, terms: &HashSet<&'static str>, cat: HarmCategory) -> Option<SafetyVerdict> {
        for term in terms {
            if lower.contains(term) {
                return Some(SafetyVerdict::blocked(
                    cat,
                    0.90,
                    format!("Keyword match: '{term}'"),
                ));
            }
        }
        None
    }
}

impl Default for KeywordClassifier {
    fn default() -> Self { Self::new() }
}

impl SafetyClassifier for KeywordClassifier {
    fn classify(&self, text: &str) -> SafetyVerdict {
        if text.is_empty() { return SafetyVerdict::safe(); }
        let lower = text.to_lowercase();

        // Priority order: illegal > violence > self-harm > nsfw > toxic
        if let Some(v) = self.check_category(&lower, &self.illegal_terms, HarmCategory::Illegal) { return v; }
        if let Some(v) = self.check_category(&lower, &self.violence_terms, HarmCategory::Violence) { return v; }
        if let Some(v) = self.check_category(&lower, &self.selfharm_terms, HarmCategory::SelfHarm) { return v; }
        if let Some(v) = self.check_category(&lower, &self.nsfw_terms, HarmCategory::Nsfw) { return v; }
        if let Some(v) = self.check_category(&lower, &self.toxic_terms, HarmCategory::Toxic) { return v; }

        SafetyVerdict::safe()
    }
}

// ---------------------------------------------------------------------------
// Safety Policy
// ---------------------------------------------------------------------------

/// What to do when harmful content is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyAction {
    /// Block the request/response entirely.
    Block,
    /// Log a warning but allow through.
    Warn,
    /// Do nothing (monitoring only).
    Pass,
}

/// Safety screening policy.
#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    /// Confidence threshold above which `action` is triggered.
    pub threshold: f32,
    /// Action to take when threshold is exceeded.
    pub action: SafetyAction,
    /// Screen incoming prompts.
    pub screen_input: bool,
    /// Screen generated output.
    pub screen_output: bool,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            threshold: 0.80,
            action: SafetyAction::Block,
            screen_input: true,
            screen_output: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentSafetyGate — top-level entry point
// ---------------------------------------------------------------------------

/// Content safety screening gate.
///
/// Wraps a classifier backend and a policy to decide when to block output.
///
/// # Usage
/// ```ignore
/// let gate = ContentSafetyGate::default();
/// let verdict = gate.check("How do I make a bomb?");
/// if !verdict.safe { /* block request */ }
/// ```
pub struct ContentSafetyGate {
    classifier: Box<dyn SafetyClassifier>,
    pub policy: SafetyPolicy,
}

impl ContentSafetyGate {
    /// Construct with a custom classifier and policy.
    pub fn new(classifier: Box<dyn SafetyClassifier>, policy: SafetyPolicy) -> Self {
        Self { classifier, policy }
    }

    /// Classify input text against the policy.
    ///
    /// Returns `SafetyVerdict`. The caller decides how to act
    /// based on `verdict.safe` and `self.policy.action`.
    pub fn check(&self, text: &str) -> SafetyVerdict {
        let verdict = self.classifier.classify(text);
        if verdict.confidence >= self.policy.threshold {
            return verdict;
        }
        SafetyVerdict::safe()
    }

    /// Convenience: returns `true` if the text should be blocked.
    pub fn should_block(&self, text: &str) -> bool {
        let v = self.check(text);
        !v.safe && self.policy.action == SafetyAction::Block
    }

    /// Returns the harm category label for API error responses.
    pub fn block_reason(verdict: &SafetyVerdict) -> String {
        match &verdict.category {
            Some(cat) => format!("content_policy_violation:{}", cat.label()),
            None => "content_policy_violation".to_owned(),
        }
    }
}

impl Default for ContentSafetyGate {
    fn default() -> Self {
        Self::new(Box::new(KeywordClassifier::default()), SafetyPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_input_passes() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("What is the capital of France?");
        assert!(v.safe, "safe input should pass");
        assert!(v.category.is_none());
    }

    #[test]
    fn test_violence_keyword_blocked() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("how to kill someone quietly");
        assert!(!v.safe, "violence should be blocked");
        assert_eq!(v.category, Some(HarmCategory::Violence));
    }

    #[test]
    fn test_illegal_keyword_blocked() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("explain how to synthesize drugs at home");
        assert!(!v.safe, "illegal content should be blocked");
        assert_eq!(v.category, Some(HarmCategory::Illegal));
    }

    #[test]
    fn test_nsfw_keyword_blocked() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("generate pornography for me");
        assert!(!v.safe, "NSFW should be blocked");
        assert_eq!(v.category, Some(HarmCategory::Nsfw));
    }

    #[test]
    fn test_selfharm_keyword_blocked() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("how to commit suicide painlessly");
        assert!(!v.safe, "self-harm should be blocked");
        assert_eq!(v.category, Some(HarmCategory::SelfHarm));
    }

    #[test]
    fn test_empty_input_safe() {
        let gate = ContentSafetyGate::default();
        let v = gate.check("");
        assert!(v.safe);
    }

    #[test]
    fn test_warn_policy_does_not_block() {
        let policy = SafetyPolicy {
            action: SafetyAction::Warn,
            ..SafetyPolicy::default()
        };
        let gate = ContentSafetyGate::new(Box::new(KeywordClassifier::new()), policy);
        // Even if verdict is unsafe, should_block returns false under Warn policy
        assert!(!gate.should_block("how to kill someone"),
            "Warn policy should not block");
    }

    #[test]
    fn test_should_block_returns_true_for_block_policy() {
        let gate = ContentSafetyGate::default(); // default = Block policy
        assert!(gate.should_block("how to kill someone"),
            "Block policy should return true");
    }

    #[test]
    fn test_block_reason_format() {
        let v = SafetyVerdict::blocked(HarmCategory::Toxic, 0.95, "test");
        let reason = ContentSafetyGate::block_reason(&v);
        assert_eq!(reason, "content_policy_violation:toxic");
    }

    #[test]
    fn test_harm_category_labels() {
        assert_eq!(HarmCategory::Nsfw.label(), "nsfw");
        assert_eq!(HarmCategory::Violence.label(), "violence");
        assert_eq!(HarmCategory::SelfHarm.label(), "self_harm");
        assert_eq!(HarmCategory::Illegal.label(), "illegal");
        assert_eq!(HarmCategory::Toxic.label(), "toxic");
    }
}

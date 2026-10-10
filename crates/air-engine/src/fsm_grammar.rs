use air_core::types::TokenId;

#[derive(Debug, Clone)]
pub enum FsmConstraint {
    JsonSchema { schema_json: String },
    Regex { pattern: String },
    Choice { allowed_strings: Vec<String> },
}

pub struct FsmGrammarEngine {
    vocab_size: usize,
    pub constraint: Option<FsmConstraint>,
    pub allowed_tokens: Vec<bool>,
}

impl FsmGrammarEngine {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            constraint: None,
            allowed_tokens: vec![true; vocab_size],
        }
    }

    pub fn set_json_schema(&mut self, schema_json: &str) {
        self.constraint = Some(FsmConstraint::JsonSchema {
            schema_json: schema_json.to_string(),
        });
    }

    pub fn set_regex(&mut self, pattern: &str) {
        self.constraint = Some(FsmConstraint::Regex {
            pattern: pattern.to_string(),
        });
    }

    pub fn update_allowed_tokens(&mut self, _current_tokens: &[TokenId]) {
        if self.allowed_tokens.len() != self.vocab_size {
            self.allowed_tokens = vec![true; self.vocab_size];
        }
        for slot in self.allowed_tokens.iter_mut() {
            *slot = true;
        }
    }

    pub fn apply_mask(&self, logits: &mut [f32]) {
        for (i, &allowed) in self.allowed_tokens.iter().enumerate() {
            if !allowed && i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

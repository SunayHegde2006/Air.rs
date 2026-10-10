use super::Tokenizer;
use crate::error::Result;
use crate::types::TokenId;
use std::collections::HashMap;

pub struct SentencePieceTokenizer {
    vocab: HashMap<String, TokenId>,
    decoder: HashMap<TokenId, String>,
    #[allow(dead_code)]
    scores: HashMap<TokenId, f32>,
    bos_id: Option<TokenId>,
    eos_id: Option<TokenId>,
}

impl SentencePieceTokenizer {
    pub fn new(
        vocab: HashMap<String, TokenId>,
        scores: HashMap<TokenId, f32>,
        bos_id: Option<TokenId>,
        eos_id: Option<TokenId>,
    ) -> Self {
        let decoder = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            decoder,
            scores,
            bos_id,
            eos_id,
        }
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        let formatted = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        let mut i = 0;
        let chars: Vec<char> = formatted.chars().collect();

        while i < chars.len() {
            let mut matched = false;
            for len in (1..=(chars.len() - i).min(16)).rev() {
                let sub: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&sub) {
                    tokens.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }

        if add_special_tokens {
            if let Some(eos) = self.eos_id {
                tokens.push(eos);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let mut res = String::new();
        for &token in tokens {
            if let Some(bos) = self.bos_id {
                if token == bos {
                    continue;
                }
            }
            if let Some(eos) = self.eos_id {
                if token == eos {
                    continue;
                }
            }
            if let Some(piece) = self.decoder.get(&token) {
                res.push_str(&piece.replace('\u{2581}', " "));
            }
        }
        Ok(res.trim().to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn bos_token_id(&self) -> Option<TokenId> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<TokenId> {
        self.eos_id
    }
}

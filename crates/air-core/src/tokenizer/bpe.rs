use super::Tokenizer;
use crate::error::Result;
use crate::types::TokenId;
use std::collections::HashMap;

pub struct BpeTokenizer {
    vocab: HashMap<String, TokenId>,
    decoder: HashMap<TokenId, String>,
    #[allow(dead_code)]
    merges: HashMap<(String, String), usize>,
    bos_id: Option<TokenId>,
    eos_id: Option<TokenId>,
}

impl BpeTokenizer {
    pub fn new(
        vocab: HashMap<String, TokenId>,
        merges: HashMap<(String, String), usize>,
        bos_id: Option<TokenId>,
        eos_id: Option<TokenId>,
    ) -> Self {
        let decoder = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            decoder,
            merges,
            bos_id,
            eos_id,
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        for word in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else {
                for ch in word.chars() {
                    let s = ch.to_string();
                    if let Some(&id) = self.vocab.get(&s) {
                        tokens.push(id);
                    }
                }
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
            if let Some(word) = self.decoder.get(&token) {
                res.push_str(word);
            }
        }
        Ok(res)
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

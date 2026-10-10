use super::Tokenizer;
use crate::error::Result;
use crate::types::TokenId;
use std::collections::HashMap;

pub struct TiktokenTokenizer {
    encoder: HashMap<Vec<u8>, TokenId>,
    decoder: HashMap<TokenId, Vec<u8>>,
    special_tokens: HashMap<String, TokenId>,
    bos_id: Option<TokenId>,
    eos_id: Option<TokenId>,
}

impl TiktokenTokenizer {
    pub fn new(
        encoder: HashMap<Vec<u8>, TokenId>,
        special_tokens: HashMap<String, TokenId>,
        bos_id: Option<TokenId>,
        eos_id: Option<TokenId>,
    ) -> Self {
        let decoder = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            encoder,
            decoder,
            special_tokens,
            bos_id,
            eos_id,
        }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let mut matched = false;
            for len in (1..=(bytes.len() - i).min(32)).rev() {
                let slice = &bytes[i..i + len];
                if let Some(&id) = self.encoder.get(slice) {
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
        let mut res_bytes = Vec::new();
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
            if let Some(b) = self.decoder.get(&token) {
                res_bytes.extend_from_slice(b);
            }
        }
        Ok(String::from_utf8_lossy(&res_bytes).to_string())
    }

    fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens.len()
    }

    fn bos_token_id(&self) -> Option<TokenId> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<TokenId> {
        self.eos_id
    }
}

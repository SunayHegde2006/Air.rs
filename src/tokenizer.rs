//! BPE Tokenizer extracted from GGUF metadata.
//!
//! GGUF files contain the full vocabulary and merge rules needed for
//! Byte-Pair Encoding under the `tokenizer.ggml.*` metadata keys.

use std::collections::HashMap;

/// A BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token string → token ID
    vocab: HashMap<String, u32>,
    /// Token ID → token string (for decoding)
    id_to_token: Vec<String>,
    /// Merge rules: (pair) → merged token, ordered by priority
    merges: Vec<(String, String)>,
    /// Special token IDs
    pub bos_id: u32,
    pub eos_id: u32,
}

impl Tokenizer {
    /// Build a tokenizer from vocabulary tokens and merge rules.
    ///
    /// - `tokens`: The vocabulary list (index = token ID, value = token string)
    /// - `merges`: BPE merge rules as "tokenA tokenB" strings, in priority order
    /// - `bos_id` / `eos_id`: Special token IDs from GGUF metadata
    pub fn new(tokens: Vec<String>, merges: Vec<String>, bos_id: u32, eos_id: u32) -> Self {
        let mut vocab = HashMap::new();
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }

        let merge_pairs: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        Self {
            vocab,
            id_to_token: tokens,
            merges: merge_pairs,
            bos_id,
            eos_id,
        }
    }

    /// Encode a text string into a sequence of token IDs using BPE.
    ///
    /// Algorithm:
    /// 1. Split text into individual UTF-8 bytes (as strings)
    /// 2. Repeatedly merge the highest-priority adjacent pair
    /// 3. Map final tokens to IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Start with individual characters (or bytes for byte-level BPE)
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merges in priority order
        for (left, right) in &self.merges {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i] == *left && symbols[i + 1] == *right {
                    let merged = format!("{}{}", left, right);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                    // Don't increment i — check if the merged token can merge again
                } else {
                    i += 1;
                }
            }
        }

        // Map to IDs, falling back to a byte-level encoding for unknown tokens
        symbols
            .iter()
            .map(|s| {
                self.vocab.get(s).copied().unwrap_or_else(|| {
                    // Fallback: try to encode each byte individually
                    // Most GGUF models have byte tokens like <0x41> for 'A'
                    if s.len() == 1 {
                        let byte = s.as_bytes()[0];
                        let byte_token = format!("<0x{:02X}>", byte);
                        self.vocab.get(&byte_token).copied().unwrap_or(0)
                    } else {
                        0 // Unknown token
                    }
                })
            })
            .collect()
    }

    /// Decode a sequence of token IDs back into a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
            // Clean up common SentencePiece artifacts
            .replace('▁', " ")
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

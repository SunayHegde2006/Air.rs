//! BPE Tokenizer extracted from GGUF metadata.
//!
//! Supports two conventions:
//!   - **Byte-level BPE** (GPT-2 / Llama 3): Each byte maps to a specific Unicode
//!     character (0x20 → Ġ, 0x0A → Ċ, printable ASCII maps to itself shifted by
//!     an offset). Merge rules operate on these byte-mapped characters.
//!   - **SentencePiece** (Llama 1/2): Uses ▁ (U+2581) for word boundaries.
//!
//! GGUF files contain the full vocabulary and merge rules under
//! `tokenizer.ggml.*` metadata keys.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GPT-2 byte ↔ Unicode mapping
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping table.
///
/// GPT-2's BPE operates on Unicode characters that represent individual bytes.
/// Printable ASCII chars (!, ", #, ... ~) and some Latin-1 chars map to themselves.
/// All other byte values (0x00-0x20, 0x7F-0xA0, 0xAD) are shifted into the
/// range starting at U+0100 (Ā) so they become visible, non-conflicting characters.
///
/// Reference: openai/gpt-2 encoder.py `bytes_to_unicode()`
fn bytes_to_unicode() -> (Vec<char>, HashMap<char, u8>) {
    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::new();

    // The "direct" ranges: bytes that map to themselves as Unicode codepoints
    // Range 1: '!' (0x21) through '~' (0x7E)
    // Range 2: '¡' (0xA1) through '¬' (0xAC)
    // Range 3: '®' (0xAE) through 'ÿ' (0xFF)
    let mut direct_bytes: Vec<u8> = Vec::new();
    for b in 0x21u8..=0x7E { direct_bytes.push(b); }
    for b in 0xA1u8..=0xAC { direct_bytes.push(b); }
    for b in 0xAEu8..=0xFF { direct_bytes.push(b); }

    // Direct mappings: byte value → same Unicode codepoint
    for &b in &direct_bytes {
        let c = char::from(b);
        byte_to_char[b as usize] = c;
        char_to_byte.insert(c, b);
    }

    // All other bytes get shifted into U+0100+ range
    let mut offset = 0u32;
    for b in 0u16..=255 {
        if !direct_bytes.contains(&(b as u8)) {
            let c = char::from_u32(256 + offset).unwrap();
            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b as u8);
            offset += 1;
        }
    }

    (byte_to_char, char_to_byte)
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token string → token ID
    vocab: HashMap<String, u32>,
    /// Token ID → token string (for decoding)
    id_to_token: Vec<String>,
    /// Merge rules: (pair) → priority index (lower = higher priority)
    merge_priority: HashMap<(String, String), usize>,
    /// Byte → Unicode char mapping (GPT-2 convention)
    byte_to_char: Vec<char>,
    /// Unicode char → byte mapping (for decoding)
    char_to_byte: HashMap<char, u8>,
    /// Whether this tokenizer uses byte-level BPE (auto-detected)
    is_byte_level: bool,
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

        // Build merge priority map (index = priority, lower = merge first)
        let mut merge_priority = HashMap::new();
        for (priority, m) in merges.iter().enumerate() {
            let parts: Vec<&str> = m.splitn(2, ' ').collect();
            if parts.len() == 2 {
                merge_priority.insert(
                    (parts[0].to_string(), parts[1].to_string()),
                    priority,
                );
            }
        }

        let (byte_to_char, char_to_byte) = bytes_to_unicode();

        // Auto-detect byte-level BPE by checking if Ġ (U+0120 = byte 0x20 = space)
        // exists in the vocabulary. Llama 3 and GPT-2 models have this.
        let is_byte_level = vocab.contains_key("Ġ") || vocab.contains_key("ĠThe");

        Self {
            vocab,
            id_to_token: tokens,
            merge_priority,
            byte_to_char,
            char_to_byte,
            is_byte_level,
            bos_id,
            eos_id,
        }
    }

    /// Encode a text string into a sequence of token IDs using BPE.
    ///
    /// For byte-level BPE (Llama 3, GPT-2):
    ///   1. Convert each byte of UTF-8 encoding to the GPT-2 Unicode character
    ///   2. Apply BPE merges by priority (lowest index = highest priority)
    ///   3. Map merged tokens to vocabulary IDs
    ///
    /// For character-level BPE:
    ///   1. Split into individual characters
    ///   2. Apply BPE merges
    ///   3. Map to IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Step 1: Convert text to initial symbols
        let mut symbols: Vec<String> = if self.is_byte_level {
            // Byte-level BPE: convert each byte to its GPT-2 Unicode character
            text.as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect()
        } else {
            // Character-level BPE: split into individual characters
            text.chars().map(|c| c.to_string()).collect()
        };

        // Step 2: Iteratively merge the highest-priority pair until no more merges
        loop {
            // Find the pair with the lowest priority index (= highest priority)
            let mut best_pair: Option<(usize, usize)> = None;  // (priority, position)
            let mut best_priority = usize::MAX;

            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&priority) = self.merge_priority.get(&pair) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_pair = Some((priority, i));
                    }
                }
            }

            match best_pair {
                Some((_priority, _pos)) => {
                    // Merge ALL occurrences of this pair (the highest-priority one)
                    let left = symbols[_pos].clone();
                    let right = symbols[_pos + 1].clone();
                    let merged = format!("{}{}", left, right);

                    let mut i = 0;
                    while i + 1 < symbols.len() {
                        if symbols[i] == left && symbols[i + 1] == right {
                            symbols[i] = merged.clone();
                            symbols.remove(i + 1);
                            // Don't increment — check if merged with next
                        } else {
                            i += 1;
                        }
                    }
                }
                None => break, // No more mergeable pairs
            }
        }

        // Step 3: Map to IDs
        symbols
            .iter()
            .map(|s| {
                self.vocab.get(s).copied().unwrap_or_else(|| {
                    // Fallback: try byte tokens like <0x41>
                    if s.len() == 1 {
                        let c = s.chars().next().unwrap();
                        if let Some(&byte_val) = self.char_to_byte.get(&c) {
                            let byte_token = format!("<0x{:02X}>", byte_val);
                            self.vocab.get(&byte_token).copied().unwrap_or(0)
                        } else {
                            0
                        }
                    } else {
                        0 // Unknown token
                    }
                })
            })
            .collect()
    }

    /// Decode a sequence of token IDs back into a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let raw: String = tokens
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join("");

        if self.is_byte_level {
            // Byte-level BPE: convert each GPT-2 Unicode char back to its byte
            let bytes: Vec<u8> = raw
                .chars()
                .filter_map(|c| self.char_to_byte.get(&c).copied())
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            // SentencePiece convention
            raw.replace('▁', " ")
        }
    }

    /// Decode a single token ID to its display string.
    ///
    /// For byte-level BPE, converts the GPT-2 Unicode characters back to
    /// their original bytes (e.g., Ġ → space, Ċ → newline).
    pub fn decode_token(&self, id: u32) -> String {
        let raw = self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>");

        if self.is_byte_level {
            let bytes: Vec<u8> = raw
                .chars()
                .filter_map(|c| self.char_to_byte.get(&c).copied())
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            raw.replace('▁', " ")
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_mapping_roundtrip() {
        let (b2c, c2b) = bytes_to_unicode();
        // Every byte should roundtrip
        for byte_val in 0u8..=255 {
            let c = b2c[byte_val as usize];
            assert_ne!(c, '\0', "byte {byte_val} mapped to null");
            let back = c2b[&c];
            assert_eq!(back, byte_val, "roundtrip failed for byte {byte_val}");
        }
    }

    #[test]
    fn byte_mapping_specific_values() {
        let (b2c, _c2b) = bytes_to_unicode();
        // Space (0x20) should map to Ġ (U+0120)
        assert_eq!(b2c[0x20], 'Ġ');
        // Newline (0x0A) should map to Ċ (U+010A)
        assert_eq!(b2c[0x0A], 'Ċ');
        // Printable ASCII like 'A' (0x41) maps to itself
        assert_eq!(b2c[0x41], 'A');
        // '!' (0x21) maps to itself
        assert_eq!(b2c[0x21], '!');
    }
}

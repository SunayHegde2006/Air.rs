pub mod bpe;
pub mod sentencepiece;
pub mod tiktoken;

pub use bpe::*;
pub use sentencepiece::*;
pub use tiktoken::*;

use crate::error::Result;
use crate::types::TokenId;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn bos_token_id(&self) -> Option<TokenId>;
    fn eos_token_id(&self) -> Option<TokenId>;
}

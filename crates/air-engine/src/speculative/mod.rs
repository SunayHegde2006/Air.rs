pub mod dflash;
pub mod dspark;
pub mod eagle;
pub mod lookahead;
pub mod medusa;
pub mod mtp;
pub mod ngram;

pub use dflash::*;
pub use dspark::*;
pub use eagle::*;
pub use lookahead::*;
pub use medusa::*;
pub use mtp::*;
pub use ngram::*;

use air_core::types::TokenId;

pub trait SpeculativeStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId>;
}

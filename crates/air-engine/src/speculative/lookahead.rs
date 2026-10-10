use super::SpeculativeStrategy;
use air_core::types::TokenId;

pub struct LookaheadStrategy;

impl SpeculativeStrategy for LookaheadStrategy {
    fn name(&self) -> &'static str {
        "Lookahead"
    }

    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let last = history.last().copied().unwrap_or(0);
        (0..gamma).map(|i| (last + i as u32 + 50) % 1000).collect()
    }
}

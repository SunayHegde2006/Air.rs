use super::SpeculativeStrategy;
use air_core::types::TokenId;

pub struct DSparkStrategy;

impl SpeculativeStrategy for DSparkStrategy {
    fn name(&self) -> &'static str {
        "DSpark"
    }

    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let last = history.last().copied().unwrap_or(0);
        (0..gamma).map(|i| (last + i as u32 + 10) % 1000).collect()
    }
}

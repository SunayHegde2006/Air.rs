use super::SpeculativeStrategy;
use air_core::types::TokenId;

pub struct DFlashStrategy;

impl SpeculativeStrategy for DFlashStrategy {
    fn name(&self) -> &'static str {
        "DFlash"
    }

    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let last = history.last().copied().unwrap_or(0);
        (0..gamma).map(|i| (last + i as u32 + 20) % 1000).collect()
    }
}

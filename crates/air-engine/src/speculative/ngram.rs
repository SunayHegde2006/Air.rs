use super::SpeculativeStrategy;
use air_core::types::TokenId;

pub struct NgramStrategy {
    n: usize,
}

impl NgramStrategy {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl SpeculativeStrategy for NgramStrategy {
    fn name(&self) -> &'static str {
        "N-gram"
    }

    fn generate_draft(&mut self, history: &[TokenId], gamma: usize) -> Vec<TokenId> {
        if history.len() < self.n {
            let last = history.last().copied().unwrap_or(0);
            return (0..gamma).map(|i| (last + i as u32 + 1) % 1000).collect();
        }

        let match_pattern = &history[history.len() - self.n..];
        for i in 0..history.len().saturating_sub(self.n) {
            if &history[i..i + self.n] == match_pattern {
                let start_idx = i + self.n;
                let draft_len = gamma.min(history.len() - start_idx);
                if draft_len > 0 {
                    let mut draft = history[start_idx..start_idx + draft_len].to_vec();
                    while draft.len() < gamma {
                        draft.push(*draft.last().unwrap_or(&0) + 1);
                    }
                    return draft;
                }
            }
        }

        let last = history.last().copied().unwrap_or(0);
        (0..gamma).map(|i| (last + i as u32 + 1) % 1000).collect()
    }
}

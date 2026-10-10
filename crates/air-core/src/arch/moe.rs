use crate::model::ModelConfig;

#[derive(Debug, Clone)]
pub struct MoeArchitecture {
    pub config: ModelConfig,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl MoeArchitecture {
    pub fn new(config: ModelConfig, num_experts: usize, num_experts_per_tok: usize) -> Self {
        Self {
            config,
            num_experts,
            num_experts_per_tok,
        }
    }
}

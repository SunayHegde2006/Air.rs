use crate::model::ModelConfig;

#[derive(Debug, Clone)]
pub struct DeepSeekArchitecture {
    pub config: ModelConfig,
    pub mtp_heads: usize,
}

impl DeepSeekArchitecture {
    pub fn new(config: ModelConfig, mtp_heads: usize) -> Self {
        Self { config, mtp_heads }
    }
}

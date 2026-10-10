use crate::model::ModelConfig;

#[derive(Debug, Clone)]
pub struct QwenArchitecture {
    pub config: ModelConfig,
}

impl QwenArchitecture {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

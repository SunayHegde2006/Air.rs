use crate::model::ModelConfig;

#[derive(Debug, Clone)]
pub struct LlamaArchitecture {
    pub config: ModelConfig,
}

impl LlamaArchitecture {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn layer_tensor_name(&self, layer_idx: usize, tensor_type: &str) -> String {
        format!("blk.{layer_idx}.{tensor_type}.weight")
    }
}

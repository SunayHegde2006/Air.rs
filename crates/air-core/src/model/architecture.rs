use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchitectureKind {
    Llama,
    DeepSeek,
    Qwen,
    MoE,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: ArchitectureKind,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            arch: ArchitectureKind::Llama,
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            bos_token_id: 128000,
            eos_token_id: 128001,
        }
    }
}

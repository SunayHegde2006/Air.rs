use crate::error::{AirError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultsConfig {
    pub backend: String,
    pub profile: String,
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            backend: "auto".to_string(),
            profile: "auto".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub strategy: String,
    pub auto_tune: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            strategy: "auto".to_string(),
            auto_tune: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            cors: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    pub video_fps: f32,
    pub max_video_frames: usize,
    pub npu_offload: bool,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            video_fps: 1.0,
            max_video_frames: 32,
            npu_offload: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AirConfig {
    pub defaults: DefaultsConfig,
    pub speculative: SpeculativeConfig,
    pub server: ServerConfig,
    pub multimodal: MultimodalConfig,
}

impl AirConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| AirError::Config(e.to_string()))
    }
}

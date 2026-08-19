//! STRIX configuration — runtime parameters for the scheduler,
//! memory manager, and hardware abstraction layer.
//!
//! Defaults match STRIX Protocol §20.3. JSON only (serde_json already a dep).

use super::score::ScoreWeights;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrixConfig {
    pub enabled: bool,
    pub scheduling_interval_ms: u64,
    pub prefetch_window_layers: usize,
    pub eviction_headroom_fraction: f64,
    pub vram_safety_margin_mb: usize,
    pub ram_pool_max_gb: f64,
    pub enable_direct_storage: bool,
    pub weights: ScoreWeights,
}

impl Default for StrixConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheduling_interval_ms: 2,
            prefetch_window_layers: 3,
            eviction_headroom_fraction: 0.10,
            vram_safety_margin_mb: 512,
            ram_pool_max_gb: 16.0,
            enable_direct_storage: true,
            weights: ScoreWeights::default(),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    UnsupportedFormat(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "config I/O error: {e}"),
            Self::JsonError(e) => write!(f, "config JSON error: {e}"),
            Self::UnsupportedFormat(ext) => write!(f, "unsupported config format: {ext}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self { Self::IoError(e) }
}

impl StrixConfig {
    /// Load configuration from a JSON file (`.json` extension required).
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase().as_str() {
            "json" => serde_json::from_str(&content).map_err(ConfigError::JsonError),
            other  => Err(ConfigError::UnsupportedFormat(other.to_string())),
        }
    }

    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(json).map_err(ConfigError::JsonError)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("StrixConfig serialization cannot fail")
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        std::fs::write(path, self.to_json())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strix_config_default_matches_protocol() {
        let cfg = StrixConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.scheduling_interval_ms, 2);
        assert_eq!(cfg.prefetch_window_layers, 3);
        assert!((cfg.eviction_headroom_fraction - 0.10).abs() < 1e-9);
        assert_eq!(cfg.vram_safety_margin_mb, 512);
        assert!((cfg.ram_pool_max_gb - 16.0).abs() < 1e-9);
        assert!(cfg.enable_direct_storage);
        assert!((cfg.weights.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = StrixConfig::default();
        let json = cfg.to_json();
        let loaded = StrixConfig::from_json(&json).unwrap();
        assert_eq!(loaded.scheduling_interval_ms, cfg.scheduling_interval_ms);
        assert_eq!(loaded.prefetch_window_layers, cfg.prefetch_window_layers);
        assert!((loaded.weights.urgency - cfg.weights.urgency).abs() < 1e-6);
    }

    #[test]
    fn config_from_partial_json() {
        let json = r#"{"prefetch_window_layers": 5, "enabled": true, "scheduling_interval_ms": 2, "eviction_headroom_fraction": 0.1, "vram_safety_margin_mb": 512, "ram_pool_max_gb": 16.0, "enable_direct_storage": true, "weights": {"urgency": 0.45, "predictive": 0.3, "sticky": 0.15, "cost": 0.1}}"#;
        let cfg = StrixConfig::from_json(json).unwrap();
        assert_eq!(cfg.prefetch_window_layers, 5);
    }

    #[test]
    fn config_unsupported_format() {
        let result = StrixConfig::from_file(Path::new("config.yaml"));
        assert!(result.is_err());
    }
}

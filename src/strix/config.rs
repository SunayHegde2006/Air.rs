//! STRIX configuration — runtime parameters for the scheduler,
//! memory manager, and hardware abstraction layer.
//!
//! Defaults match STRIX Protocol §20.3.
//!
//! Supports loading from JSON or TOML files via `StrixConfig::from_file()`.

use super::score::ScoreWeights;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level STRIX configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrixConfig {
    /// Enable/disable STRIX offloading (when `false`, all tensors stay in VRAM).
    pub enabled: bool,
    /// Scheduler tick interval in milliseconds.
    pub scheduling_interval_ms: u64,
    /// Number of layers ahead of the execution cursor to pre-fetch.
    pub prefetch_window_layers: usize,
    /// Fraction of VRAM budget to keep free as eviction headroom.
    pub eviction_headroom_fraction: f64,
    /// Hard safety margin in MB subtracted from total VRAM.
    pub vram_safety_margin_mb: usize,
    /// Maximum system RAM to use for the staging pool (GB).
    pub ram_pool_max_gb: f64,
    /// Use GPUDirect Storage if available (NVIDIA only).
    pub enable_direct_storage: bool,
    /// Residency score weights.
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

/// Configuration errors.
#[derive(Debug)]
pub enum ConfigError {
    /// File I/O error.
    IoError(std::io::Error),
    /// JSON parse error.
    JsonError(serde_json::Error),
    /// TOML parse error.
    TomlError(toml::de::Error),
    /// Unsupported file extension.
    UnsupportedFormat(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "config I/O error: {e}"),
            Self::JsonError(e) => write!(f, "config JSON error: {e}"),
            Self::TomlError(e) => write!(f, "config TOML error: {e}"),
            Self::UnsupportedFormat(ext) => write!(f, "unsupported config format: {ext}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self { Self::IoError(e) }
}

impl StrixConfig {
    /// Load configuration from a JSON or TOML file.
    ///
    /// Format is auto-detected from the file extension:
    /// - `.json` → JSON
    /// - `.toml` → TOML
    ///
    /// Missing fields are filled with defaults via serde.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "json" => serde_json::from_str(&content).map_err(ConfigError::JsonError),
            "toml" => toml::from_str(&content).map_err(ConfigError::TomlError),
            other => Err(ConfigError::UnsupportedFormat(other.to_string())),
        }
    }

    /// Load from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(json).map_err(ConfigError::JsonError)
    }

    /// Load from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, ConfigError> {
        toml::from_str(toml_str).map_err(ConfigError::TomlError)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("StrixConfig serialization cannot fail")
    }

    /// Serialize to TOML string.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("StrixConfig serialization cannot fail")
    }

    /// Save to a file (format detected from extension).
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let content = match ext.as_str() {
            "json" => self.to_json(),
            "toml" => self.to_toml(),
            other => return Err(ConfigError::UnsupportedFormat(other.to_string())),
        };

        std::fs::write(path, content)?;
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

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
    fn config_toml_roundtrip() {
        let cfg = StrixConfig::default();
        let toml_str = cfg.to_toml();
        let loaded = StrixConfig::from_toml(&toml_str).unwrap();
        assert_eq!(loaded.vram_safety_margin_mb, cfg.vram_safety_margin_mb);
        assert!((loaded.ram_pool_max_gb - cfg.ram_pool_max_gb).abs() < 1e-9);
    }

    #[test]
    fn config_from_partial_json() {
        // Only override one field, rest should be defaults
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

//! STRIX configuration — runtime parameters for the scheduler,
//! memory manager, and hardware abstraction layer.
//!
//! Defaults match STRIX Protocol §20.3.

use super::score::ScoreWeights;

/// Top-level STRIX configuration.
#[derive(Debug, Clone)]
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
}

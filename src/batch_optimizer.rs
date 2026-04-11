//! B_opt Batch Sizing — M.I.S.T. v3 §Sub-System 1
//!
//! Computes the optimal batch size to make ρ ≥ 1.0 on slow storage (HDD,
//! USB, SATA SSD). Without correct B_opt, single-user HDD inference has
//! ρ = 0.014 — the GPU idles 98.6% of the time waiting for disk.
//!
//! ```text
//! B_opt = ⌈T_io / T_kernel_per_token⌉
//!       = ⌈(S_layer / disk_speed) / T_kernel⌉
//!
//! Example (5400 RPM HDD, RTX 4060):
//!   T_io    = 531 MB / 110 MB/s = 4,827 ms
//!   T_kernel = 70 ms/token
//!   B_opt   = ⌈4827 / 70⌉ = 69 tokens
//! ```
//!
//! ## Dynamic Recalibration
//!
//! B_opt is recalibrated every `RECALIB_INTERVAL` layers (default: 10)
//! using observed disk throughput, not the initial estimate. This adapts
//! to thermal throttling, OS cache effects, and concurrent I/O load.
//!
//! ## CPU-Only Special Case
//!
//! CPU inference (T_kernel ≈ 1,000 ms) naturally masks slow I/O.
//! B_opt = 4-5 on HDD (very manageable even for single-user).
//!
//! Reference: air_rs_protocols_v3.md §Sub-System 1

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────────

/// Layer size in MB for LLaMA 70B Q4_K_M.
pub const DEFAULT_LAYER_SIZE_MB: f64 = 531.0;

/// Recalibration interval (layers).
pub const RECALIB_INTERVAL: usize = 10;

/// Minimum B_opt (single-user fallback).
pub const MIN_BATCH_SIZE: usize = 1;

/// Maximum practical B_opt (beyond this, memory is the bottleneck).
pub const MAX_BATCH_SIZE: usize = 128;

/// KV miss overhead factor for ρ denominator.
pub const DEFAULT_KV_MISS_RATE: f64 = 0.05;

/// KV reload time in ms (cold tier).
pub const DEFAULT_KV_RELOAD_MS: f64 = 20.0;

// ── Storage Profiles ─────────────────────────────────────────────────────

/// Known storage speed profiles with conservative estimates.
#[derive(Debug, Clone)]
pub struct StorageProfile {
    /// Storage name/type.
    pub name: String,
    /// Sequential read speed in MB/s (conservative estimate).
    pub speed_mbps: f64,
    /// Storage class for protocol selection.
    pub class: StorageClass,
}

/// Storage classification for protocol routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// >= 3,000 MB/s → S.L.I.P. v3 territory.
    NvmeGen4,
    /// >= 1,500 MB/s → S.L.I.P. v3 territory.
    NvmeGen3,
    /// 400-1,500 MB/s → S.L.I.P. v3 with conservative D_opt.
    SataSsd,
    /// 50-400 MB/s → M.I.S.T. v3 territory.
    Hdd7200,
    /// < 50 MB/s → M.I.S.T. v3 degraded mode.
    HddSlow,
}

impl StorageProfile {
    /// USB 3.0 HDD (80 MB/s conservative).
    pub fn usb3_hdd() -> Self {
        Self { name: "USB 3.0 HDD".into(), speed_mbps: 80.0, class: StorageClass::HddSlow }
    }

    /// 5400 RPM Laptop HDD (110 MB/s conservative).
    pub fn hdd_5400() -> Self {
        Self { name: "5400 RPM HDD".into(), speed_mbps: 110.0, class: StorageClass::HddSlow }
    }

    /// 7200 RPM Desktop HDD (160 MB/s conservative).
    pub fn hdd_7200() -> Self {
        Self { name: "7200 RPM HDD".into(), speed_mbps: 160.0, class: StorageClass::Hdd7200 }
    }

    /// SATA SSD (500 MB/s conservative).
    pub fn sata_ssd() -> Self {
        Self { name: "SATA SSD".into(), speed_mbps: 500.0, class: StorageClass::SataSsd }
    }

    /// Gen3 NVMe (3,000 MB/s — S.L.I.P. territory).
    pub fn nvme_gen3() -> Self {
        Self { name: "Gen3 NVMe".into(), speed_mbps: 3_000.0, class: StorageClass::NvmeGen3 }
    }

    /// Gen4 NVMe (7,000 MB/s — S.L.I.P. territory).
    pub fn nvme_gen4() -> Self {
        Self { name: "Gen4 NVMe".into(), speed_mbps: 7_000.0, class: StorageClass::NvmeGen4 }
    }

    /// Custom storage profile from measured speed.
    pub fn measured(name: &str, speed_mbps: f64) -> Self {
        let class = if speed_mbps >= 3_000.0 {
            StorageClass::NvmeGen4
        } else if speed_mbps >= 1_500.0 {
            StorageClass::NvmeGen3
        } else if speed_mbps >= 400.0 {
            StorageClass::SataSsd
        } else if speed_mbps >= 50.0 {
            StorageClass::Hdd7200
        } else {
            StorageClass::HddSlow
        };
        Self { name: name.into(), speed_mbps, class }
    }

    /// I/O time to read one layer in ms.
    pub fn t_io_ms(&self, layer_size_mb: f64) -> f64 {
        if self.speed_mbps <= 0.0 {
            return f64::INFINITY;
        }
        (layer_size_mb / self.speed_mbps) * 1000.0
    }

    /// Whether this storage should use M.I.S.T. (< 3,000 MB/s).
    pub fn needs_mist(&self) -> bool {
        matches!(self.class, StorageClass::Hdd7200 | StorageClass::HddSlow | StorageClass::SataSsd)
    }
}

impl fmt::Display for StorageProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} MB/s)", self.name, self.speed_mbps)
    }
}

// ── Compute Profiles ─────────────────────────────────────────────────────

/// Compute time profile for B_opt calculation.
#[derive(Debug, Clone)]
pub struct ComputeProfile {
    /// Device name.
    pub name: String,
    /// Time per single token forward pass in ms (T_kernel).
    pub t_kernel_ms: f64,
    /// Whether this is CPU-only inference.
    pub cpu_only: bool,
}

impl ComputeProfile {
    /// Consumer GPU (RTX 4060 class): ~70 ms/token.
    pub fn consumer_gpu() -> Self {
        Self { name: "Consumer GPU (RTX 4060 class)".into(), t_kernel_ms: 70.0, cpu_only: false }
    }

    /// Mid-range GPU (RTX 3060): ~72 ms/token.
    pub fn rtx_3060() -> Self {
        Self { name: "RTX 3060 12GB".into(), t_kernel_ms: 72.0, cpu_only: false }
    }

    /// AMD RX 7900 XTX: ~80 ms/token.
    pub fn rx_7900_xtx() -> Self {
        Self { name: "AMD RX 7900 XTX".into(), t_kernel_ms: 80.0, cpu_only: false }
    }

    /// Apple M3 Pro (Metal): ~50 ms/token.
    pub fn apple_m3_pro() -> Self {
        Self { name: "Apple M3 Pro".into(), t_kernel_ms: 50.0, cpu_only: false }
    }

    /// Intel Arc A770: ~90 ms/token.
    pub fn arc_a770() -> Self {
        Self { name: "Intel Arc A770".into(), t_kernel_ms: 90.0, cpu_only: false }
    }

    /// CPU-only Ryzen 5 7600: ~1,000 ms/token.
    pub fn cpu_ryzen_5() -> Self {
        Self { name: "Ryzen 5 7600 (CPU-only)".into(), t_kernel_ms: 1_000.0, cpu_only: true }
    }

    /// CPU-only Intel i5-12600K: ~1,000 ms/token.
    pub fn cpu_i5_12600k() -> Self {
        Self { name: "Intel i5-12600K (CPU-only)".into(), t_kernel_ms: 1_000.0, cpu_only: true }
    }

    /// Raspberry Pi 5: ~1,000 ms/token estimate.
    pub fn rpi5() -> Self {
        Self { name: "Raspberry Pi 5 (CPU-only)".into(), t_kernel_ms: 1_000.0, cpu_only: true }
    }

    /// Custom compute profile.
    pub fn custom(name: &str, t_kernel_ms: f64, cpu_only: bool) -> Self {
        Self { name: name.into(), t_kernel_ms, cpu_only }
    }
}

// ── B_opt Calculation ────────────────────────────────────────────────────

/// Result of B_opt computation.
#[derive(Debug, Clone)]
pub struct BoptResult {
    /// Optimal batch size.
    pub b_opt: usize,
    /// I/O time per layer in ms.
    pub t_io_ms: f64,
    /// Kernel time per token in ms.
    pub t_kernel_ms: f64,
    /// Storage profile used.
    pub storage_name: String,
    /// Compute profile used.
    pub compute_name: String,
    /// ρ with B_opt applied (compute / io ratio).
    pub rho_with_bopt: f64,
    /// ρ without batching (single token).
    pub rho_single: f64,
    /// Whether this is practical for single-user.
    pub practical_single_user: bool,
    /// Whether CPU-only mode (naturally forgiving).
    pub cpu_only: bool,
    /// Layer size used in MB.
    pub layer_size_mb: f64,
}

impl fmt::Display for BoptResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  M.I.S.T. v3 — B_opt Calculation             ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  Storage:  {:>33} ║", self.storage_name)?;
        writeln!(f, "║  Compute:  {:>33} ║", self.compute_name)?;
        writeln!(f, "║  T_io:     {:>8.1} ms / layer                ║", self.t_io_ms)?;
        writeln!(f, "║  T_kernel: {:>8.1} ms / token                ║", self.t_kernel_ms)?;
        writeln!(f, "║  B_opt:    {:>8} tokens                     ║", self.b_opt)?;
        writeln!(f, "║  ρ(B=1):   {:>8.3}                           ║", self.rho_single)?;
        writeln!(f, "║  ρ(B_opt): {:>8.2}                           ║", self.rho_with_bopt)?;
        let feasibility = if self.practical_single_user { "YES" } else { "NO — multi-user only" };
        writeln!(f, "║  Single-user: {:>30} ║", feasibility)?;
        if self.cpu_only {
            writeln!(f, "║  Mode: CPU-only (naturally forgiving)        ║")?;
        }
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Compute B_opt from storage and compute profiles.
///
/// Formula: B_opt = ⌈T_io / T_kernel⌉
///
/// Where:
/// - T_io = S_layer / disk_speed (ms)
/// - T_kernel = per-token forward pass time (ms)
pub fn compute_bopt(
    storage: &StorageProfile,
    compute: &ComputeProfile,
    layer_size_mb: f64,
) -> BoptResult {
    let t_io_ms = storage.t_io_ms(layer_size_mb);
    let t_kernel_ms = compute.t_kernel_ms;

    // B_opt = ⌈T_io / T_kernel⌉
    let b_opt_raw = if t_kernel_ms > 0.0 {
        (t_io_ms / t_kernel_ms).ceil() as usize
    } else {
        MAX_BATCH_SIZE
    };

    let b_opt = b_opt_raw.clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE);

    // ρ(B=1) = T_kernel / T_io
    let rho_single = if t_io_ms > 0.0 {
        t_kernel_ms / t_io_ms
    } else {
        f64::INFINITY
    };

    // ρ(B_opt) ≈ (B_opt × T_kernel) / T_io
    let rho_with_bopt = if t_io_ms > 0.0 {
        (b_opt as f64 * t_kernel_ms) / t_io_ms
    } else {
        f64::INFINITY
    };

    // Single-user practical if B_opt <= 16 (SATA SSD) or CPU-only with <= 7
    let practical_single_user = if compute.cpu_only {
        b_opt <= 10
    } else {
        b_opt <= 16
    };

    BoptResult {
        b_opt,
        t_io_ms,
        t_kernel_ms,
        storage_name: storage.name.clone(),
        compute_name: compute.name.clone(),
        rho_with_bopt,
        rho_single,
        practical_single_user,
        cpu_only: compute.cpu_only,
        layer_size_mb,
    }
}

// ── Dynamic Recalibration ────────────────────────────────────────────────

/// Tracks B_opt and recalibrates based on observed throughput.
///
/// Every `RECALIB_INTERVAL` layers, the observed disk speed is used to
/// recompute B_opt. This adapts to:
/// - Thermal throttling on HDDs
/// - OS buffer cache effects
/// - Concurrent I/O from other processes
/// - Drive seek time variation
#[derive(Debug)]
pub struct BoptCalibrator {
    /// Current B_opt.
    current_bopt: usize,
    /// Layer size in MB.
    layer_size_mb: f64,
    /// Per-token kernel time in ms.
    t_kernel_ms: f64,
    /// CPU-only mode flag.
    cpu_only: bool,
    /// Layers processed since last recalibration.
    layers_since_recalib: usize,
    /// Recalibration interval.
    recalib_interval: usize,
    /// Observed speed samples (MB/s) in current window.
    speed_samples: Vec<f64>,
    /// History of B_opt recalibrations.
    history: Vec<RecalibEvent>,
    /// Running EMA of observed disk speed (MB/s).
    ema_speed_mbps: f64,
    /// EMA smoothing factor.
    ema_alpha: f64,
}

/// A single recalibration event.
#[derive(Debug, Clone)]
pub struct RecalibEvent {
    /// Layer index at recalibration.
    pub at_layer: usize,
    /// Observed speed used (MB/s).
    pub observed_speed_mbps: f64,
    /// Previous B_opt.
    pub old_bopt: usize,
    /// New B_opt.
    pub new_bopt: usize,
    /// Total layers processed.
    pub total_layers: usize,
}

impl BoptCalibrator {
    /// Create a new calibrator with initial B_opt.
    pub fn new(initial_bopt: BoptResult) -> Self {
        Self {
            current_bopt: initial_bopt.b_opt,
            layer_size_mb: initial_bopt.layer_size_mb,
            t_kernel_ms: initial_bopt.t_kernel_ms,
            cpu_only: initial_bopt.cpu_only,
            layers_since_recalib: 0,
            recalib_interval: RECALIB_INTERVAL,
            speed_samples: Vec::with_capacity(RECALIB_INTERVAL),
            history: Vec::new(),
            ema_speed_mbps: initial_bopt.layer_size_mb * 1000.0 / initial_bopt.t_io_ms,
            ema_alpha: 0.3,
        }
    }

    /// Create from raw parameters.
    pub fn from_params(
        initial_bopt: usize,
        layer_size_mb: f64,
        t_kernel_ms: f64,
        initial_speed_mbps: f64,
        cpu_only: bool,
    ) -> Self {
        Self {
            current_bopt: initial_bopt,
            layer_size_mb,
            t_kernel_ms,
            cpu_only,
            layers_since_recalib: 0,
            recalib_interval: RECALIB_INTERVAL,
            speed_samples: Vec::with_capacity(RECALIB_INTERVAL),
            history: Vec::new(),
            ema_speed_mbps: initial_speed_mbps,
            ema_alpha: 0.3,
        }
    }

    /// Report that a layer was loaded with observed I/O time.
    ///
    /// Call this after each layer load. The calibrator will automatically
    /// recalibrate after `recalib_interval` layers.
    ///
    /// Returns `Some(new_bopt)` if recalibration occurred.
    pub fn report_layer(&mut self, io_time_ms: f64, total_layers_processed: usize) -> Option<usize> {
        // Compute observed speed for this layer.
        let observed_speed = if io_time_ms > 0.0 {
            (self.layer_size_mb * 1000.0) / io_time_ms
        } else {
            self.ema_speed_mbps // fallback
        };

        self.speed_samples.push(observed_speed);
        self.layers_since_recalib += 1;

        if self.layers_since_recalib >= self.recalib_interval {
            Some(self.recalibrate(total_layers_processed))
        } else {
            None
        }
    }

    /// Force a recalibration with current samples.
    fn recalibrate(&mut self, total_layers: usize) -> usize {
        let old_bopt = self.current_bopt;

        // Compute average speed from samples.
        let avg_speed = if self.speed_samples.is_empty() {
            self.ema_speed_mbps
        } else {
            let sum: f64 = self.speed_samples.iter().sum();
            sum / self.speed_samples.len() as f64
        };

        // Update EMA.
        self.ema_speed_mbps = self.ema_alpha * avg_speed + (1.0 - self.ema_alpha) * self.ema_speed_mbps;

        // Recalculate B_opt from EMA speed.
        let new_bopt = recalibrate(self.ema_speed_mbps, self.t_kernel_ms);
        self.current_bopt = new_bopt;

        // Record event.
        self.history.push(RecalibEvent {
            at_layer: total_layers,
            observed_speed_mbps: self.ema_speed_mbps,
            old_bopt,
            new_bopt,
            total_layers,
        });

        // Reset window.
        self.speed_samples.clear();
        self.layers_since_recalib = 0;

        new_bopt
    }

    /// Current B_opt.
    pub fn current_bopt(&self) -> usize {
        self.current_bopt
    }

    /// Current EMA speed estimate (MB/s).
    pub fn ema_speed(&self) -> f64 {
        self.ema_speed_mbps
    }

    /// Recalibration history.
    pub fn history(&self) -> &[RecalibEvent] {
        &self.history
    }

    /// Number of recalibrations performed.
    pub fn recalib_count(&self) -> usize {
        self.history.len()
    }

    /// Set recalibration interval.
    pub fn set_interval(&mut self, interval: usize) {
        self.recalib_interval = interval.max(1);
    }

    /// Whether CPU-only mode.
    pub fn is_cpu_only(&self) -> bool {
        self.cpu_only
    }

    /// Current ρ estimate.
    pub fn rho_estimate(&self) -> f64 {
        let t_io = if self.ema_speed_mbps > 0.0 {
            (self.layer_size_mb * 1000.0) / self.ema_speed_mbps
        } else {
            f64::INFINITY
        };
        if t_io == 0.0 || t_io.is_infinite() {
            return 0.0;
        }
        (self.current_bopt as f64 * self.t_kernel_ms) / t_io
    }
}

impl fmt::Display for BoptCalibrator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoptCalibrator: B_opt={}, EMA speed={:.0} MB/s, ρ≈{:.2}, recalibs={}",
            self.current_bopt,
            self.ema_speed_mbps,
            self.rho_estimate(),
            self.history.len(),
        )
    }
}

/// Recalibrate B_opt from observed disk speed.
///
/// This is the core formula from the protocol spec:
/// ```text
/// fn recalibrate(observed_speed_mbps: f64, t_kernel_ms: f64) -> usize {
///     let t_io_ms = (531.0 * 1024.0) / observed_speed_mbps;
///     ((t_io_ms / t_kernel_ms).ceil() as usize).max(1)
/// }
/// ```
///
/// Note: We use `layer_size_mb * 1000.0 / speed` for T_io to keep units
/// consistent (MB and MB/s → ms).
pub fn recalibrate(observed_speed_mbps: f64, t_kernel_ms: f64) -> usize {
    if observed_speed_mbps <= 0.0 || t_kernel_ms <= 0.0 {
        return MIN_BATCH_SIZE;
    }
    let t_io_ms = (DEFAULT_LAYER_SIZE_MB / observed_speed_mbps) * 1000.0;
    let b_opt = (t_io_ms / t_kernel_ms).ceil() as usize;
    b_opt.clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
}

// ── Protocol Router ──────────────────────────────────────────────────────

/// Protocol selection based on measured storage speed.
///
/// From DriveInquisitor v3 Decision Matrix:
/// ```text
/// >= 3,000 MB/s → S.L.I.P. v3 (NVMe)
/// >= 400 MB/s   → S.L.I.P. v3 (SATA SSD, conservative D_opt)
/// >= 50 MB/s    → M.I.S.T. v3 (HDD/USB)
/// < 50 MB/s     → M.I.S.T. v3 degraded mode
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolSelection {
    /// S.L.I.P. v3: latency-optimized, NVMe.
    SlipNvme,
    /// S.L.I.P. v3: SATA SSD with conservative D_opt=2.
    SlipSataSsd,
    /// M.I.S.T. v3: HDD/USB with batch padding.
    MistHdd,
    /// M.I.S.T. v3: degraded mode for very slow storage.
    MistDegraded,
}

impl fmt::Display for ProtocolSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlipNvme => write!(f, "S.L.I.P. v3 (NVMe)"),
            Self::SlipSataSsd => write!(f, "S.L.I.P. v3 (SATA SSD, D_opt=2)"),
            Self::MistHdd => write!(f, "M.I.S.T. v3 (HDD/USB, B_opt batching)"),
            Self::MistDegraded => write!(f, "M.I.S.T. v3 (degraded, Ghost Drafting only)"),
        }
    }
}

/// Select the appropriate protocol based on measured storage speed.
pub fn select_protocol(measured_speed_mbps: f64) -> ProtocolSelection {
    if measured_speed_mbps >= 3_000.0 {
        ProtocolSelection::SlipNvme
    } else if measured_speed_mbps >= 400.0 {
        ProtocolSelection::SlipSataSsd
    } else if measured_speed_mbps >= 50.0 {
        ProtocolSelection::MistHdd
    } else {
        ProtocolSelection::MistDegraded
    }
}

// ── M.I.S.T. ρ_v3 Calculation ────────────────────────────────────────────

/// M.I.S.T.-specific ρ_v3 calculation with batch sizing and Ghost Drafting.
///
/// Formula:
/// ```text
/// ρ_v3 = (T_compute(B_opt) + T_ghost_accept) / (T_io + T_kv_reload × f_miss)
/// ```
#[derive(Debug, Clone)]
pub struct MistRhoInput {
    /// I/O time per layer (ms).
    pub t_io_ms: f64,
    /// Batch-adjusted compute time (B_opt × T_kernel).
    pub t_compute_batch_ms: f64,
    /// Ghost acceptance gain (α × k × T_token).
    pub t_ghost_accept_ms: f64,
    /// KV reload time × miss rate.
    pub t_kv_penalty_ms: f64,
}

/// Calculate M.I.S.T. ρ_v3 with full denominator.
pub fn mist_rho_v3(input: &MistRhoInput) -> f64 {
    let numerator = input.t_compute_batch_ms + input.t_ghost_accept_ms;
    let denominator = input.t_io_ms + input.t_kv_penalty_ms;
    if denominator <= 0.0 {
        f64::INFINITY
    } else {
        numerator / denominator
    }
}

/// Compute full M.I.S.T. ρ_v3 from B_opt result with Ghost Drafting.
///
/// Ghost acceptance: α=0.75, k=4, T_token = T_kernel.
/// KV penalty: T_kv_reload × f_miss = 20ms × 0.05 = 1ms.
pub fn mist_rho_from_bopt(bopt: &BoptResult) -> f64 {
    let t_compute_batch = bopt.b_opt as f64 * bopt.t_kernel_ms;

    // Ghost acceptance gain (GPU) or 0 for CPU-only (CPU IS the ghost).
    let t_ghost = if bopt.cpu_only {
        0.0
    } else {
        0.75 * 4.0 * bopt.t_kernel_ms // α × k × T_token
    };

    let t_kv_penalty = DEFAULT_KV_RELOAD_MS * DEFAULT_KV_MISS_RATE;

    mist_rho_v3(&MistRhoInput {
        t_io_ms: bopt.t_io_ms,
        t_compute_batch_ms: t_compute_batch,
        t_ghost_accept_ms: t_ghost,
        t_kv_penalty_ms: t_kv_penalty,
    })
}

// ── Convenience ──────────────────────────────────────────────────────────

/// Generate B_opt matrix for all known storage × compute combinations.
pub fn bopt_matrix() -> Vec<BoptResult> {
    let storages = vec![
        StorageProfile::usb3_hdd(),
        StorageProfile::hdd_5400(),
        StorageProfile::hdd_7200(),
        StorageProfile::sata_ssd(),
        StorageProfile::nvme_gen3(),
    ];
    let computes = vec![
        ComputeProfile::consumer_gpu(),
        ComputeProfile::cpu_ryzen_5(),
    ];

    let mut results = Vec::new();
    for storage in &storages {
        for compute in &computes {
            results.push(compute_bopt(storage, compute, DEFAULT_LAYER_SIZE_MB));
        }
    }
    results
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Spec-Validated B_opt Calculations ─────────────────────────────

    #[test]
    fn test_bopt_usb3_hdd_gpu() {
        // Spec: T_io = 531/80 * 1000 = 6,638 ms ... wait
        // Actually: T_io = 531 MB / 80 MB/s = 6,637.5 ms → B_opt = ⌈6637.5/70⌉ = 95
        let result = compute_bopt(
            &StorageProfile::usb3_hdd(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 95, "USB 3.0 HDD + GPU: B_opt={}", result.b_opt);
        assert!(!result.practical_single_user);
    }

    #[test]
    fn test_bopt_5400rpm_gpu() {
        // Spec: T_io = 531/110 * 1000 = 4,827 ms, B_opt = ⌈4827/70⌉ = 69
        let result = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 69, "5400 RPM + GPU: B_opt={}", result.b_opt);
        assert!(!result.practical_single_user);
    }

    #[test]
    fn test_bopt_7200rpm_gpu() {
        // Spec: T_io = 531/160 * 1000 = 3,319 ms, B_opt = ⌈3319/70⌉ = 48
        let result = compute_bopt(
            &StorageProfile::hdd_7200(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 48, "7200 RPM + GPU: B_opt={}", result.b_opt);
        assert!(!result.practical_single_user);
    }

    #[test]
    fn test_bopt_sata_ssd_gpu() {
        // Spec: T_io = 531/500 * 1000 = 1,062 ms, B_opt = ⌈1062/70⌉ = 16
        let result = compute_bopt(
            &StorageProfile::sata_ssd(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 16, "SATA SSD + GPU: B_opt={}", result.b_opt);
        assert!(result.practical_single_user);
    }

    #[test]
    fn test_bopt_nvme_gen3_gpu() {
        // Spec: T_io = 531/3000 * 1000 = 177 ms, B_opt = ⌈177/70⌉ = 3
        let result = compute_bopt(
            &StorageProfile::nvme_gen3(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 3, "Gen3 NVMe + GPU: B_opt={}", result.b_opt);
        assert!(result.practical_single_user);
    }

    // ── CPU-Only B_opt (Naturally Forgiving) ─────────────────────────

    #[test]
    fn test_bopt_5400rpm_cpu() {
        // Spec: T_io = 4,827 ms, T_kernel = 1,000 ms, B_opt = ⌈4827/1000⌉ = 5
        let result = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::cpu_ryzen_5(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 5, "5400 RPM + CPU: B_opt={}", result.b_opt);
        assert!(result.cpu_only);
        assert!(result.practical_single_user); // CPU B_opt is manageable
    }

    #[test]
    fn test_bopt_7200rpm_cpu() {
        // Spec: T_io = 3,319 ms, T_kernel = 1,000 ms, B_opt = ⌈3319/1000⌉ = 4
        let result = compute_bopt(
            &StorageProfile::hdd_7200(),
            &ComputeProfile::cpu_ryzen_5(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 4, "7200 RPM + CPU: B_opt={}", result.b_opt);
        assert!(result.cpu_only);
    }

    #[test]
    fn test_bopt_sata_ssd_cpu() {
        // Spec: T_io = 1,062 ms, T_kernel = 1,000 ms, B_opt = ⌈1062/1000⌉ = 2
        let result = compute_bopt(
            &StorageProfile::sata_ssd(),
            &ComputeProfile::cpu_ryzen_5(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, 2, "SATA SSD + CPU: B_opt={}", result.b_opt);
    }

    // ── ρ Validation ─────────────────────────────────────────────────

    #[test]
    fn test_rho_single_user_dismal() {
        // Spec: ρ(B=1) = 70/4827 = 0.014
        let result = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert!((result.rho_single - 0.0145).abs() < 0.005,
            "ρ(B=1) should be ~0.014, got {}", result.rho_single);
    }

    #[test]
    fn test_rho_with_bopt_achievable() {
        // With B_opt=69 on 5400 RPM: ρ ≈ 69*70/4827 ≈ 1.00
        let result = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert!(result.rho_with_bopt >= 0.99,
            "ρ(B_opt) should be ≈ 1.0, got {}", result.rho_with_bopt);
    }

    #[test]
    fn test_mist_rho_v3_with_ghost() {
        // Full ρ_v3 with Ghost Drafting should be > 1.0
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let rho = mist_rho_from_bopt(&bopt);
        assert!(rho >= 1.0, "M.I.S.T. ρ_v3 should be ≥ 1.0 with Ghost, got {:.2}", rho);
    }

    #[test]
    fn test_mist_rho_cpu_only_no_ghost() {
        // CPU-only: no Ghost Drafting (CPU IS the ghost).
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::cpu_ryzen_5(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let rho = mist_rho_from_bopt(&bopt);
        assert!(rho >= 1.0, "CPU M.I.S.T. ρ_v3 should be ≥ 1.0, got {:.2}", rho);
    }

    // ── Dynamic Recalibration ────────────────────────────────────────

    #[test]
    fn test_recalibrate_function() {
        // Direct recalibrate call.
        let b = recalibrate(110.0, 70.0);
        assert_eq!(b, 69, "recalibrate(110, 70) = {}", b);

        let b = recalibrate(160.0, 70.0);
        assert_eq!(b, 48, "recalibrate(160, 70) = {}", b);

        let b = recalibrate(110.0, 1000.0);
        assert_eq!(b, 5, "recalibrate(110, 1000) = {}", b);
    }

    #[test]
    fn test_calibrator_creation() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let cal = BoptCalibrator::new(bopt);
        assert_eq!(cal.current_bopt(), 69);
        assert!(cal.ema_speed() > 0.0);
        assert_eq!(cal.recalib_count(), 0);
    }

    #[test]
    fn test_calibrator_recalibration_triggers_at_interval() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let mut cal = BoptCalibrator::new(bopt);

        // Report 9 layers — no recalibration.
        for i in 0..9 {
            let result = cal.report_layer(4800.0, i);
            assert!(result.is_none(), "Should not recalibrate at layer {}", i);
        }

        // Layer 10 triggers recalibration.
        let result = cal.report_layer(4800.0, 9);
        assert!(result.is_some(), "Should recalibrate at layer 10");
        assert_eq!(cal.recalib_count(), 1);
    }

    #[test]
    fn test_calibrator_adapts_to_slower_disk() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_7200(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let mut cal = BoptCalibrator::new(bopt);
        let initial_bopt = cal.current_bopt();

        // Simulate disk getting slower (longer I/O times).
        for i in 0..10 {
            cal.report_layer(8000.0, i); // Much slower than expected
        }

        // B_opt should have increased.
        assert!(cal.current_bopt() >= initial_bopt,
            "B_opt should increase when disk slows: {} vs {}",
            cal.current_bopt(), initial_bopt);
    }

    #[test]
    fn test_calibrator_adapts_to_faster_disk() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let mut cal = BoptCalibrator::new(bopt);
        let initial_bopt = cal.current_bopt();

        // Simulate disk being much faster (OS cache hit?).
        for i in 0..10 {
            cal.report_layer(500.0, i); // ~1,062 MB/s — SSD-like
        }

        // B_opt should decrease.
        assert!(cal.current_bopt() <= initial_bopt,
            "B_opt should decrease when disk is faster: {} vs {}",
            cal.current_bopt(), initial_bopt);
    }

    #[test]
    fn test_calibrator_display() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let cal = BoptCalibrator::new(bopt);
        let s = format!("{}", cal);
        assert!(s.contains("BoptCalibrator"));
        assert!(s.contains("B_opt="));
    }

    #[test]
    fn test_calibrator_rho_estimate() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let cal = BoptCalibrator::new(bopt);
        let rho = cal.rho_estimate();
        assert!(rho >= 0.99, "ρ estimate should be ≈ 1.0, got {:.3}", rho);
    }

    #[test]
    fn test_calibrator_custom_interval() {
        let bopt = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let mut cal = BoptCalibrator::new(bopt);
        cal.set_interval(5);

        // Should trigger recalib at 5 layers instead of 10.
        for i in 0..4 {
            assert!(cal.report_layer(4800.0, i).is_none());
        }
        assert!(cal.report_layer(4800.0, 4).is_some());
    }

    // ── Protocol Selection ───────────────────────────────────────────

    #[test]
    fn test_protocol_nvme() {
        assert_eq!(select_protocol(7_000.0), ProtocolSelection::SlipNvme);
        assert_eq!(select_protocol(3_000.0), ProtocolSelection::SlipNvme);
    }

    #[test]
    fn test_protocol_sata() {
        assert_eq!(select_protocol(500.0), ProtocolSelection::SlipSataSsd);
        assert_eq!(select_protocol(400.0), ProtocolSelection::SlipSataSsd);
    }

    #[test]
    fn test_protocol_hdd() {
        assert_eq!(select_protocol(160.0), ProtocolSelection::MistHdd);
        assert_eq!(select_protocol(110.0), ProtocolSelection::MistHdd);
        assert_eq!(select_protocol(50.0), ProtocolSelection::MistHdd);
    }

    #[test]
    fn test_protocol_degraded() {
        assert_eq!(select_protocol(30.0), ProtocolSelection::MistDegraded);
        assert_eq!(select_protocol(10.0), ProtocolSelection::MistDegraded);
    }

    #[test]
    fn test_protocol_display() {
        let s = format!("{}", ProtocolSelection::MistHdd);
        assert!(s.contains("M.I.S.T."));
    }

    // ── Edge Cases ───────────────────────────────────────────────────

    #[test]
    fn test_bopt_zero_speed() {
        let storage = StorageProfile::measured("broken", 0.0);
        let result = compute_bopt(
            &storage,
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, MAX_BATCH_SIZE);
    }

    #[test]
    fn test_bopt_clamping() {
        // Very slow: would need > 128 tokens.
        let storage = StorageProfile::measured("glacial", 1.0);
        let result = compute_bopt(
            &storage,
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        assert_eq!(result.b_opt, MAX_BATCH_SIZE);
    }

    #[test]
    fn test_recalibrate_edge_cases() {
        assert_eq!(recalibrate(0.0, 70.0), MIN_BATCH_SIZE);
        assert_eq!(recalibrate(110.0, 0.0), MIN_BATCH_SIZE);
        assert_eq!(recalibrate(-1.0, 70.0), MIN_BATCH_SIZE);
    }

    #[test]
    fn test_storage_needs_mist() {
        assert!(StorageProfile::hdd_5400().needs_mist());
        assert!(StorageProfile::hdd_7200().needs_mist());
        assert!(StorageProfile::usb3_hdd().needs_mist());
        assert!(StorageProfile::sata_ssd().needs_mist());
        assert!(!StorageProfile::nvme_gen3().needs_mist());
        assert!(!StorageProfile::nvme_gen4().needs_mist());
    }

    #[test]
    fn test_bopt_result_display() {
        let result = compute_bopt(
            &StorageProfile::hdd_5400(),
            &ComputeProfile::consumer_gpu(),
            DEFAULT_LAYER_SIZE_MB,
        );
        let s = format!("{}", result);
        assert!(s.contains("B_opt"));
        assert!(s.contains("69"));
    }

    #[test]
    fn test_bopt_matrix() {
        let matrix = bopt_matrix();
        assert_eq!(matrix.len(), 10); // 5 storage × 2 compute
    }

    #[test]
    fn test_storage_t_io() {
        let hdd = StorageProfile::hdd_5400();
        let t_io = hdd.t_io_ms(DEFAULT_LAYER_SIZE_MB);
        // 531 / 110 * 1000 ≈ 4827
        assert!((t_io - 4827.3).abs() < 1.0, "T_io = {}", t_io);
    }
}

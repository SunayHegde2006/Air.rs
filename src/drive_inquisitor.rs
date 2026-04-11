//! DriveInquisitor v3 — Storage Speed Detection & Protocol Selection
//!
//! Measures actual disk throughput and selects the optimal streaming protocol:
//!   - S.L.I.P. v3 for NVMe (≥3000 MB/s) and SATA SSD (≥400 MB/s)
//!   - M.I.S.T. v3 for HDD/USB (≥50 MB/s)
//!   - M.I.S.T. v3 degraded (<50 MB/s)
//!
//! Also computes the adaptive pipeline depth (D_opt) and optimal batch size (B_opt)
//! based on measured I/O and compute timings.
//!
//! ## DriveInquisitor v3 Decision Matrix
//!
//! ```text
//! BackendInquisitor (UCAL)   → Selects compute backend
//! DriveInquisitor (Storage)  → Selects streaming protocol
//!
//! Combined decision logic:
//!   measured_speed = burst_read_50ms(model_path)
//!
//!   >= 3,000 MB/s → S.L.I.P. v3 (NVMe), D_opt, self-speculative
//!   >= 400 MB/s   → S.L.I.P. v3 (SATA), D=2, Ghost Drafting
//!   >= 50 MB/s    → M.I.S.T. v3, B_opt, Ghost Drafting primary
//!   < 50 MB/s     → M.I.S.T. v3 degraded, Ghost only, user warned
//!
//! Special paths:
//!   CPU + NVMe → S.L.I.P. D=2, ρ>1 (CPU bottleneck), no Ghost, no batch
//!   CPU + HDD  → M.I.S.T., B_opt=4-5, Ghost on same CPU threads
//! ```
//!
//! Reference: air_rs_protocols_v3.md §Protocol Interaction

use std::fmt;
use std::time::{Duration, Instant};
use std::path::Path;

// ---------------------------------------------------------------------------
// Protocol Classification
// ---------------------------------------------------------------------------

/// The streaming protocol selected based on measured disk speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamingProtocol {
    /// S.L.I.P. v3 NVMe mode — full pipeline, ≥3000 MB/s
    SlipNvme,
    /// S.L.I.P. v3 SATA mode — D=2, Ghost Drafting, ≥400 MB/s
    SlipSata,
    /// M.I.S.T. v3 — batch accumulation, ≥50 MB/s
    Mist,
    /// M.I.S.T. v3 degraded — minimal batching, <50 MB/s
    MistDegraded,
}

impl fmt::Display for StreamingProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamingProtocol::SlipNvme     => write!(f, "S.L.I.P. v3 (NVMe)"),
            StreamingProtocol::SlipSata     => write!(f, "S.L.I.P. v3 (SATA)"),
            StreamingProtocol::Mist         => write!(f, "M.I.S.T. v3"),
            StreamingProtocol::MistDegraded => write!(f, "M.I.S.T. v3 (degraded)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Speed Thresholds (from the protocol spec)
// ---------------------------------------------------------------------------

/// Speed boundaries in MB/s for protocol selection.
pub mod thresholds {
    /// NVMe: ≥3000 MB/s → S.L.I.P. v3 full pipeline
    pub const NVME_MIN_MBPS: f64 = 3000.0;
    /// SATA SSD: ≥400 MB/s → S.L.I.P. v3 with D=2
    pub const SATA_MIN_MBPS: f64 = 400.0;
    /// HDD/USB: ≥50 MB/s → M.I.S.T. v3
    pub const HDD_MIN_MBPS: f64 = 50.0;
    // Below 50 MB/s → M.I.S.T. v3 degraded
}

// ---------------------------------------------------------------------------
// Drive Speed Result
// ---------------------------------------------------------------------------

/// Result of a disk speed measurement.
#[derive(Debug, Clone)]
pub struct DriveSpeedResult {
    /// Measured sequential read speed in MB/s
    pub speed_mbps: f64,
    /// How many bytes were read during measurement
    pub bytes_read: u64,
    /// Duration of the measurement
    pub duration: Duration,
    /// Selected streaming protocol
    pub protocol: StreamingProtocol,
    /// Recommended pipeline depth
    pub d_opt: usize,
    /// Recommended batch size (M.I.S.T. only)
    pub b_opt: usize,
}

// ---------------------------------------------------------------------------
// Protocol Selection Logic
// ---------------------------------------------------------------------------

/// Select the streaming protocol based on measured speed.
pub fn select_protocol(speed_mbps: f64) -> StreamingProtocol {
    if speed_mbps >= thresholds::NVME_MIN_MBPS {
        StreamingProtocol::SlipNvme
    } else if speed_mbps >= thresholds::SATA_MIN_MBPS {
        StreamingProtocol::SlipSata
    } else if speed_mbps >= thresholds::HDD_MIN_MBPS {
        StreamingProtocol::Mist
    } else {
        StreamingProtocol::MistDegraded
    }
}

/// Calculate the optimal pipeline depth D_opt.
///
/// Formula from spec: D_opt = ceil(T_compute / T_io) + 1
///
/// Where:
///   T_compute = time to run the forward pass for one layer chunk
///   T_io = time to read one layer chunk from disk
///
/// The +1 ensures we always have at least one buffer being read
/// while the other is being computed on.
pub fn calculate_d_opt(t_compute: Duration, t_io: Duration) -> usize {
    if t_io.is_zero() {
        // Infinitely fast I/O (e.g., RAM disk) → minimal pipeline
        return 2;
    }

    let ratio = t_compute.as_secs_f64() / t_io.as_secs_f64();
    let d = ratio.ceil() as usize + 1;

    // Clamp to reasonable bounds
    d.clamp(2, 8)
}

/// Calculate the optimal batch size B_opt for M.I.S.T. mode.
///
/// Formula from spec: B_opt = ceil(T_io / T_kernel)
///
/// This tells us how many tokens to batch together so that the
/// accumulated compute time fills the I/O wait exactly.
pub fn calculate_b_opt(t_io: Duration, t_kernel: Duration) -> usize {
    if t_kernel.is_zero() {
        return 1;
    }

    let ratio = t_io.as_secs_f64() / t_kernel.as_secs_f64();
    let b = ratio.ceil() as usize;

    // At least 1, at most 64 (practical batch limit)
    b.clamp(1, 64)
}

/// Calculate pipeline efficiency ρ (rho) from the protocol spec.
///
/// For S.L.I.P. v3: ρ = T_compute / max(T_compute, T_io)
/// For M.I.S.T. v3: ρ = (B * T_kernel) / T_io  
///
/// Perfect overlap = 1.0, complete serial = 0.5
pub fn calculate_rho(protocol: StreamingProtocol, t_compute: Duration, t_io: Duration, b_opt: usize) -> f64 {
    match protocol {
        StreamingProtocol::SlipNvme | StreamingProtocol::SlipSata => {
            if t_compute.is_zero() && t_io.is_zero() {
                return 1.0;
            }
            let max_time = t_compute.max(t_io);
            t_compute.as_secs_f64() / max_time.as_secs_f64()
        }
        StreamingProtocol::Mist | StreamingProtocol::MistDegraded => {
            if t_io.is_zero() {
                return 1.0;
            }
            let batch_compute = t_compute.as_secs_f64() * b_opt as f64;
            (batch_compute / t_io.as_secs_f64()).min(1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// DriveInquisitor — performs the actual speed measurement
// ---------------------------------------------------------------------------

/// Measures disk speed by performing a burst sequential read.
///
/// The measurement reads for ~50ms and measures achieved throughput.
/// This accounts for:
///   - OS page cache warm-up
///   - NVMe queue depth effects
///   - SATA NCQ behavior
///   - USB buffering
pub struct DriveInquisitor;

impl DriveInquisitor {
    /// Perform a burst read test on the given file path.
    /// Reads sequentially for ~50ms and returns the measured speed.
    pub fn burst_read_50ms(path: &Path) -> anyhow::Result<DriveSpeedResult> {
        use std::io::Read;

        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        if file_size == 0 {
            anyhow::bail!("Cannot measure speed of empty file");
        }

        // Read in 1MB chunks for ~50ms
        let chunk_size = 1024 * 1024; // 1 MB
        let mut reader = std::io::BufReader::with_capacity(chunk_size, file);
        let mut buffer = vec![0u8; chunk_size];
        let mut total_bytes: u64 = 0;

        let start = Instant::now();
        let target_duration = Duration::from_millis(50);

        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    total_bytes += n as u64;
                    if start.elapsed() >= target_duration {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }

        let elapsed = start.elapsed();
        let speed_mbps = if elapsed.as_secs_f64() > 0.0 {
            (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64()
        } else {
            // Completed too fast to measure — assume very fast storage
            thresholds::NVME_MIN_MBPS
        };

        let protocol = select_protocol(speed_mbps);

        // Use reasonable defaults for compute time estimates
        // These get refined by the MetricsCollector during actual inference
        let estimated_compute = Duration::from_millis(5);
        let estimated_io = Duration::from_secs_f64(
            (1024.0 * 1024.0 * 50.0) / (speed_mbps * 1024.0 * 1024.0)
        );

        let d_opt = calculate_d_opt(estimated_compute, estimated_io);
        let b_opt = calculate_b_opt(estimated_io, Duration::from_millis(1));

        Ok(DriveSpeedResult {
            speed_mbps,
            bytes_read: total_bytes,
            duration: elapsed,
            protocol,
            d_opt,
            b_opt,
        })
    }

    /// Estimate speed from a known drive type (for testing or when
    /// burst_read isn't possible).
    pub fn from_known_speed(speed_mbps: f64) -> DriveSpeedResult {
        let protocol = select_protocol(speed_mbps);

        // Calculate pipeline params with reference compute/IO estimates
        let estimated_compute = Duration::from_millis(5);
        let bytes_per_chunk = 50.0 * 1024.0 * 1024.0; // 50 MB per chunk
        let estimated_io = Duration::from_secs_f64(bytes_per_chunk / (speed_mbps * 1024.0 * 1024.0));

        let d_opt = calculate_d_opt(estimated_compute, estimated_io);
        let b_opt = calculate_b_opt(estimated_io, Duration::from_millis(1));

        DriveSpeedResult {
            speed_mbps,
            bytes_read: 0,
            duration: Duration::ZERO,
            protocol,
            d_opt,
            b_opt,
        }
    }
}

// ---------------------------------------------------------------------------
// DriveInquisitor v3 Decision Matrix — Full Protocol Routing
// ---------------------------------------------------------------------------

/// Default layer size in MB (LLaMA 70B Q4_K_M).
pub const DEFAULT_LAYER_SIZE_MB: f64 = 531.0;

/// Compute backend classification for protocol decisions.
///
/// Determines per-token kernel time, Ghost Model TTFT, and CPU-only
/// special path routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    /// Discrete NVIDIA GPU (CUDA).
    CudaGpu,
    /// Discrete AMD GPU (ROCm).
    RocmGpu,
    /// Apple Silicon (Metal, UMA).
    MetalGpu,
    /// Intel Arc (Vulkan).
    VulkanGpu,
    /// CPU-only (no discrete GPU).
    CpuOnly,
}

impl ComputeBackend {
    /// Whether this is CPU-only inference.
    pub fn is_cpu_only(&self) -> bool {
        matches!(self, Self::CpuOnly)
    }

    /// Typical per-token kernel time in ms for LLaMA 70B.
    pub fn t_kernel_ms(&self) -> f64 {
        match self {
            Self::CudaGpu  => 70.0,
            Self::RocmGpu  => 80.0,
            Self::MetalGpu => 50.0,
            Self::VulkanGpu => 90.0,
            Self::CpuOnly  => 1_000.0,
        }
    }

    /// Ghost Model TTFT in ms (None if N/A — CPU IS the ghost).
    pub fn ghost_ttft_ms(&self) -> Option<f64> {
        match self {
            Self::CudaGpu  => Some(80.0),
            Self::RocmGpu  => Some(100.0),
            Self::MetalGpu => Some(60.0),
            Self::VulkanGpu => Some(110.0),
            Self::CpuOnly  => None,
        }
    }
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaGpu  => write!(f, "CUDA (NVIDIA GPU)"),
            Self::RocmGpu  => write!(f, "ROCm (AMD GPU)"),
            Self::MetalGpu => write!(f, "Metal (Apple Silicon)"),
            Self::VulkanGpu => write!(f, "Vulkan (Intel Arc)"),
            Self::CpuOnly  => write!(f, "CPU-only"),
        }
    }
}

/// Full protocol decision from the DriveInquisitor v3 Decision Matrix.
///
/// Combines BackendInquisitor (UCAL) + DriveInquisitor (Storage) to produce
/// the complete inference configuration including protocol, pipeline params,
/// Ghost Drafting status, and estimated ρ_v3.
#[derive(Debug, Clone)]
pub struct ProtocolDecision {
    /// Selected streaming protocol.
    pub protocol: StreamingProtocol,
    /// Compute backend.
    pub backend: ComputeBackend,
    /// Measured storage speed in MB/s.
    pub measured_speed_mbps: f64,
    /// Pipeline depth (S.L.I.P.) or 0 (M.I.S.T.).
    pub d_opt: usize,
    /// Batch size (M.I.S.T.) or 0 (S.L.I.P.).
    pub b_opt: usize,
    /// T_io per layer in ms.
    pub t_io_ms: f64,
    /// T_kernel per token in ms.
    pub t_kernel_ms: f64,
    /// Whether Ghost Drafting is active.
    pub ghost_drafting: bool,
    /// Ghost TTFT in ms (0 if inactive).
    pub ghost_ttft_ms: f64,
    /// Whether self-speculative residency is active.
    pub self_speculative: bool,
    /// ρ_v3 estimate.
    pub rho: f64,
    /// User warning message (if any).
    pub warning: Option<String>,
}

impl fmt::Display for ProtocolDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════╗")?;
        writeln!(f, "║  DriveInquisitor v3 — Protocol Decision           ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Protocol:    {:>34} ║", self.protocol)?;
        writeln!(f, "║  Backend:     {:>34} ║", self.backend)?;
        writeln!(f, "║  Disk speed:  {:>30.0} MB/s ║", self.measured_speed_mbps)?;
        writeln!(f, "║  T_io:        {:>30.0} ms   ║", self.t_io_ms)?;
        writeln!(f, "║  T_kernel:    {:>30.0} ms   ║", self.t_kernel_ms)?;
        if self.d_opt > 0 {
            writeln!(f, "║  D_opt:       {:>34} ║", self.d_opt)?;
        }
        if self.b_opt > 0 {
            writeln!(f, "║  B_opt:       {:>34} ║", self.b_opt)?;
        }
        let ghost_str = if self.ghost_drafting {
            format!("YES ({:.0}ms TTFT)", self.ghost_ttft_ms)
        } else {
            "NO".to_string()
        };
        writeln!(f, "║  Ghost:       {:>34} ║", ghost_str)?;
        writeln!(f, "║  Self-spec:   {:>34} ║",
            if self.self_speculative { "YES" } else { "NO" })?;
        writeln!(f, "║  ρ_v3:        {:>34.2} ║", self.rho)?;
        if let Some(warn) = &self.warning {
            writeln!(f, "║  ⚠ {:>44} ║", warn)?;
        }
        writeln!(f, "╚══════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Execute the full DriveInquisitor v3 Decision Matrix.
///
/// Combines storage speed measurement (burst_read_50ms) with compute backend
/// classification to produce a complete protocol decision:
///
/// ```text
/// >= 3,000 MB/s → S.L.I.P. v3 (NVMe), D_opt, self-speculative active
/// >= 400 MB/s   → S.L.I.P. v3 (SATA), D=2, Ghost Drafting active
/// >= 50 MB/s    → M.I.S.T. v3 (HDD), B_opt, Ghost Drafting primary
/// < 50 MB/s     → M.I.S.T. v3 degraded, Ghost only, user warned
/// ```
///
/// CPU-only special paths:
/// - CPU + NVMe: ρ > 1 naturally (CPU bottleneck), D=2, no Ghost
/// - CPU + HDD: B_opt=4-5 (manageable), Ghost on same CPU threads
pub fn decide_protocol(
    measured_speed_mbps: f64,
    backend: ComputeBackend,
    layer_size_mb: f64,
) -> ProtocolDecision {
    let protocol = select_protocol(measured_speed_mbps);
    let t_io_ms = if measured_speed_mbps > 0.0 {
        (layer_size_mb / measured_speed_mbps) * 1000.0
    } else {
        f64::INFINITY
    };
    let t_kernel_ms = backend.t_kernel_ms();

    // CPU-only special cases from spec §CPU-Only Fallback
    if backend.is_cpu_only() {
        return decide_cpu_only(protocol, measured_speed_mbps, t_io_ms, t_kernel_ms);
    }

    match protocol {
        StreamingProtocol::SlipNvme => {
            // S.L.I.P. v3 NVMe: full pipeline + self-speculative residency
            let d_opt = compute_d_opt_ms(t_kernel_ms, t_io_ms);
            let ghost_ttft = backend.ghost_ttft_ms().unwrap_or(0.0);
            let rho = t_kernel_ms / t_io_ms.max(0.001);

            ProtocolDecision {
                protocol, backend, measured_speed_mbps,
                d_opt, b_opt: 0, t_io_ms, t_kernel_ms,
                ghost_drafting: false, // NVMe fast enough without Ghost
                ghost_ttft_ms: ghost_ttft,
                self_speculative: true,
                rho, warning: None,
            }
        }
        StreamingProtocol::SlipSata => {
            // S.L.I.P. v3 SATA: conservative D=2 + Ghost Drafting
            let ghost_ttft = backend.ghost_ttft_ms().unwrap_or(0.0);
            let ghost_active = ghost_ttft > 0.0 && ghost_ttft < t_io_ms;
            let gc = if ghost_active { 0.75 * 4.0 * t_kernel_ms * 0.75 } else { 0.0 };
            let rho = (t_kernel_ms + gc) / t_io_ms.max(0.001);

            ProtocolDecision {
                protocol, backend, measured_speed_mbps,
                d_opt: 2, b_opt: 0, t_io_ms, t_kernel_ms,
                ghost_drafting: ghost_active,
                ghost_ttft_ms: ghost_ttft,
                self_speculative: false,
                rho, warning: None,
            }
        }
        StreamingProtocol::Mist => {
            // M.I.S.T. v3: batch accumulation + Ghost Drafting primary
            let b_opt = compute_b_opt_ms(t_io_ms, t_kernel_ms);
            let ghost_ttft = backend.ghost_ttft_ms().unwrap_or(0.0);
            let ghost_active = ghost_ttft > 0.0 && ghost_ttft < t_io_ms;
            let gc = if ghost_active { 0.75 * 4.0 * t_kernel_ms * 0.75 } else { 0.0 };
            let rho = (b_opt as f64 * t_kernel_ms + gc) / (t_io_ms + 1.0);

            ProtocolDecision {
                protocol, backend, measured_speed_mbps,
                d_opt: 0, b_opt, t_io_ms, t_kernel_ms,
                ghost_drafting: ghost_active,
                ghost_ttft_ms: ghost_ttft,
                self_speculative: false,
                rho, warning: None,
            }
        }
        StreamingProtocol::MistDegraded => {
            // M.I.S.T. v3 degraded: Ghost Drafting only, user warned
            let b_opt = compute_b_opt_ms(t_io_ms, t_kernel_ms);
            let ghost_ttft = backend.ghost_ttft_ms().unwrap_or(0.0);
            let ghost_active = ghost_ttft > 0.0 && ghost_ttft < t_io_ms;

            ProtocolDecision {
                protocol, backend, measured_speed_mbps,
                d_opt: 0, b_opt, t_io_ms, t_kernel_ms,
                ghost_drafting: ghost_active,
                ghost_ttft_ms: ghost_ttft,
                self_speculative: false, rho: 0.0,
                warning: Some(format!(
                    "Storage is very slow ({:.0} MB/s). TTFT={:.0}ms without Ghost.",
                    measured_speed_mbps, t_io_ms
                )),
            }
        }
    }
}

/// CPU-only decision path from spec §CPU-Only Fallback.
///
/// - CPU + NVMe/SATA: S.L.I.P. D=2, ρ > 1 naturally (CPU is bottleneck),
///   Ghost Drafting NOT applicable (CPU IS the ghost), no batch padding.
/// - CPU + HDD: M.I.S.T., B_opt=4-5 (very manageable for personal use),
///   Ghost Model runs on same CPU using available threads (~400ms TTFT).
fn decide_cpu_only(
    protocol: StreamingProtocol,
    measured_speed_mbps: f64,
    t_io_ms: f64,
    t_kernel_ms: f64,
) -> ProtocolDecision {
    match protocol {
        StreamingProtocol::SlipNvme | StreamingProtocol::SlipSata => {
            // CPU-only on fast storage: ρ > 1 naturally
            let rho = t_kernel_ms / t_io_ms.max(0.001);
            ProtocolDecision {
                protocol: StreamingProtocol::SlipNvme,
                backend: ComputeBackend::CpuOnly,
                measured_speed_mbps,
                d_opt: 2, b_opt: 0, t_io_ms, t_kernel_ms,
                ghost_drafting: false, // CPU IS the ghost
                ghost_ttft_ms: 0.0,
                self_speculative: false,
                rho, warning: None,
            }
        }
        StreamingProtocol::Mist | StreamingProtocol::MistDegraded => {
            // CPU-only on slow storage: B_opt=4-5
            let b_opt = compute_b_opt_ms(t_io_ms, t_kernel_ms);
            let ghost_ttft = 400.0; // CPU Ghost Model TTFT
            let ghost_active = ghost_ttft < t_io_ms;
            let rho = (b_opt as f64 * t_kernel_ms) / (t_io_ms + 1.0);

            ProtocolDecision {
                protocol: StreamingProtocol::Mist,
                backend: ComputeBackend::CpuOnly,
                measured_speed_mbps,
                d_opt: 0, b_opt, t_io_ms, t_kernel_ms,
                ghost_drafting: ghost_active,
                ghost_ttft_ms: ghost_ttft,
                self_speculative: false, rho,
                warning: if protocol == StreamingProtocol::MistDegraded {
                    Some(format!(
                        "Very slow ({:.0} MB/s) + CPU-only. ~{:.0}ms/token.",
                        measured_speed_mbps, t_kernel_ms
                    ))
                } else {
                    None
                },
            }
        }
    }
}

/// Compute D_opt from millisecond times.
fn compute_d_opt_ms(t_compute_ms: f64, t_io_ms: f64) -> usize {
    if t_io_ms <= 0.0 { return 2; }
    ((t_compute_ms / t_io_ms).ceil() as usize + 1).clamp(2, 8)
}

/// Compute B_opt from millisecond times.
fn compute_b_opt_ms(t_io_ms: f64, t_kernel_ms: f64) -> usize {
    if t_kernel_ms <= 0.0 { return 1; }
    ((t_io_ms / t_kernel_ms).ceil() as usize).clamp(1, 128)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Protocol Selection Tests ──────────────────────────────────

    #[test]
    fn test_protocol_selection_nvme() {
        assert_eq!(select_protocol(7000.0), StreamingProtocol::SlipNvme);
        assert_eq!(select_protocol(3000.0), StreamingProtocol::SlipNvme);
    }

    #[test]
    fn test_protocol_selection_sata() {
        assert_eq!(select_protocol(2999.9), StreamingProtocol::SlipSata);
        assert_eq!(select_protocol(550.0),  StreamingProtocol::SlipSata);
        assert_eq!(select_protocol(400.0),  StreamingProtocol::SlipSata);
    }

    #[test]
    fn test_protocol_selection_mist() {
        assert_eq!(select_protocol(399.9),  StreamingProtocol::Mist);
        assert_eq!(select_protocol(100.0),  StreamingProtocol::Mist);
        assert_eq!(select_protocol(50.0),   StreamingProtocol::Mist);
    }

    #[test]
    fn test_protocol_selection_degraded() {
        assert_eq!(select_protocol(49.9),  StreamingProtocol::MistDegraded);
        assert_eq!(select_protocol(10.0),  StreamingProtocol::MistDegraded);
        assert_eq!(select_protocol(0.0),   StreamingProtocol::MistDegraded);
    }

    // ── D_opt Tests ──────────────────────────────────────────────

    #[test]
    fn test_d_opt_compute_equals_io() {
        let d = calculate_d_opt(Duration::from_millis(10), Duration::from_millis(10));
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_compute_slower_than_io() {
        let d = calculate_d_opt(Duration::from_millis(30), Duration::from_millis(10));
        assert_eq!(d, 4);
    }

    #[test]
    fn test_d_opt_io_slower_than_compute() {
        let d = calculate_d_opt(Duration::from_millis(5), Duration::from_millis(20));
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_zero_io() {
        let d = calculate_d_opt(Duration::from_millis(10), Duration::ZERO);
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_clamped_to_bounds() {
        let d = calculate_d_opt(Duration::from_millis(100), Duration::from_millis(1));
        assert_eq!(d, 8);
    }

    // ── B_opt Tests ──────────────────────────────────────────────

    #[test]
    fn test_b_opt_basic() {
        let b = calculate_b_opt(Duration::from_millis(10), Duration::from_millis(2));
        assert_eq!(b, 5);
    }

    #[test]
    fn test_b_opt_zero_kernel() {
        let b = calculate_b_opt(Duration::from_millis(10), Duration::ZERO);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_b_opt_clamped_to_64() {
        let b = calculate_b_opt(Duration::from_millis(1000), Duration::from_micros(100));
        assert_eq!(b, 64);
    }

    // ── Rho Tests ────────────────────────────────────────────────

    #[test]
    fn test_rho_perfect_overlap() {
        let rho = calculate_rho(StreamingProtocol::SlipNvme,
            Duration::from_millis(10), Duration::from_millis(10), 1);
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rho_compute_bound() {
        let rho = calculate_rho(StreamingProtocol::SlipNvme,
            Duration::from_millis(20), Duration::from_millis(5), 1);
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rho_io_bound() {
        let rho = calculate_rho(StreamingProtocol::SlipNvme,
            Duration::from_millis(5), Duration::from_millis(20), 1);
        assert!((rho - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_rho_mist_batch() {
        let rho = calculate_rho(StreamingProtocol::Mist,
            Duration::from_millis(2), Duration::from_millis(10), 5);
        assert!((rho - 1.0).abs() < 0.01);
    }

    // ── DriveInquisitor Tests ────────────────────────────────────

    #[test]
    fn test_from_known_speed_nvme() {
        let result = DriveInquisitor::from_known_speed(5000.0);
        assert_eq!(result.protocol, StreamingProtocol::SlipNvme);
        assert!(result.d_opt >= 2);
    }

    #[test]
    fn test_from_known_speed_hdd() {
        let result = DriveInquisitor::from_known_speed(100.0);
        assert_eq!(result.protocol, StreamingProtocol::Mist);
        assert!(result.b_opt >= 1);
    }

    #[test]
    fn test_protocol_display() {
        assert_eq!(format!("{}", StreamingProtocol::SlipNvme), "S.L.I.P. v3 (NVMe)");
        assert_eq!(format!("{}", StreamingProtocol::Mist), "M.I.S.T. v3");
    }

    // ── Decision Matrix: GPU + NVMe ──────────────────────────────

    #[test]
    fn test_decision_gpu_nvme() {
        // Spec: >= 3,000 MB/s → S.L.I.P. v3, self-speculative active
        let d = decide_protocol(7_000.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::SlipNvme);
        assert!(d.self_speculative);
        assert!(!d.ghost_drafting);
        assert!(d.d_opt >= 2);
        assert_eq!(d.b_opt, 0);
        assert!(d.warning.is_none());
    }

    // ── Decision Matrix: GPU + SATA ──────────────────────────────

    #[test]
    fn test_decision_gpu_sata() {
        // Spec: >= 400 MB/s → S.L.I.P. v3 SATA, D=2, Ghost Drafting active
        let d = decide_protocol(500.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::SlipSata);
        assert_eq!(d.d_opt, 2);
        assert!(d.ghost_drafting);
        assert!(!d.self_speculative);
    }

    // ── Decision Matrix: GPU + HDD ───────────────────────────────

    #[test]
    fn test_decision_gpu_hdd_5400() {
        // Spec: 110 MB/s (5400 RPM) → M.I.S.T., Ghost primary
        // B_opt = ceil(4827 / 70) = 69
        let d = decide_protocol(110.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::Mist);
        assert!(d.ghost_drafting);
        assert_eq!(d.b_opt, 69, "B_opt for 5400 RPM + GPU: {}", d.b_opt);
        assert!(d.rho >= 0.95, "ρ should be ~1.0: {:.2}", d.rho);
    }

    #[test]
    fn test_decision_gpu_hdd_7200() {
        // Spec: 160 MB/s (7200 RPM) → M.I.S.T.
        // B_opt = ceil(3319 / 70) = 48
        let d = decide_protocol(160.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::Mist);
        assert_eq!(d.b_opt, 48, "B_opt for 7200 RPM + GPU: {}", d.b_opt);
    }

    // ── Decision Matrix: GPU + Degraded ──────────────────────────

    #[test]
    fn test_decision_gpu_degraded() {
        // Spec: < 50 MB/s → M.I.S.T. degraded, user warned
        let d = decide_protocol(30.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::MistDegraded);
        assert!(d.warning.is_some());
        assert!(d.ghost_drafting);
    }

    // ── Decision Matrix: CPU + NVMe (spec §CPU-Only Fallback) ────

    #[test]
    fn test_decision_cpu_nvme() {
        // Spec: CPU + NVMe → S.L.I.P. D=2, ρ > 1 (CPU bottleneck),
        //       Ghost NOT applicable (CPU IS the ghost), no batch padding.
        let d = decide_protocol(5_000.0, ComputeBackend::CpuOnly, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::SlipNvme);
        assert_eq!(d.d_opt, 2);
        assert!(!d.ghost_drafting, "CPU IS the ghost — no drafting");
        assert_eq!(d.b_opt, 0, "No batch padding for CPU+NVMe");
        assert!(!d.self_speculative);
        // ρ > 1 because CPU (1000ms) >> disk (106ms at 5GB/s)
        assert!(d.rho > 1.0,
            "ρ should be > 1 (CPU bottleneck, not disk): {:.1}", d.rho);
    }

    // ── Decision Matrix: CPU + SATA ──────────────────────────────

    #[test]
    fn test_decision_cpu_sata() {
        // CPU + SATA → uses NVMe path (D=2, no Ghost)
        let d = decide_protocol(500.0, ComputeBackend::CpuOnly, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::SlipNvme);
        assert_eq!(d.d_opt, 2);
        assert!(!d.ghost_drafting);
    }

    // ── Decision Matrix: CPU + HDD (spec §CPU-Only on HDD) ──────

    #[test]
    fn test_decision_cpu_hdd_5400() {
        // Spec: CPU + 5400 HDD → M.I.S.T., B_opt=5
        // T_io = 531/110 * 1000 = 4827ms, T_kernel = 1000ms
        // B_opt = ceil(4827/1000) = 5
        let d = decide_protocol(110.0, ComputeBackend::CpuOnly, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::Mist);
        assert_eq!(d.b_opt, 5, "CPU B_opt for 5400 RPM: {}", d.b_opt);
        assert!(d.ghost_drafting, "CPU Ghost on same threads");
        assert!((d.ghost_ttft_ms - 400.0).abs() < 1.0, "CPU Ghost TTFT: {}", d.ghost_ttft_ms);
    }

    #[test]
    fn test_decision_cpu_hdd_7200() {
        // Spec: CPU + 7200 HDD → B_opt=4
        // T_io = 531/160 * 1000 = 3319ms, T_kernel = 1000ms
        // B_opt = ceil(3319/1000) = 4
        let d = decide_protocol(160.0, ComputeBackend::CpuOnly, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.b_opt, 4, "CPU B_opt for 7200 RPM: {}", d.b_opt);
    }

    // ── Decision Matrix: Multiple GPU Backends ───────────────────

    #[test]
    fn test_decision_amd_hdd() {
        let d = decide_protocol(160.0, ComputeBackend::RocmGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::Mist);
        assert!(d.ghost_drafting);
        assert!((d.ghost_ttft_ms - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_decision_apple_hdd() {
        let d = decide_protocol(160.0, ComputeBackend::MetalGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::Mist);
        assert!(d.ghost_drafting);
        assert!((d.ghost_ttft_ms - 60.0).abs() < 1.0); // UMA advantage
    }

    #[test]
    fn test_decision_arc_nvme() {
        let d = decide_protocol(5_000.0, ComputeBackend::VulkanGpu, DEFAULT_LAYER_SIZE_MB);
        assert_eq!(d.protocol, StreamingProtocol::SlipNvme);
        assert!(d.self_speculative);
    }

    // ── Compute Backend ──────────────────────────────────────────

    #[test]
    fn test_backend_cpu_only() {
        assert!(ComputeBackend::CpuOnly.is_cpu_only());
        assert!(!ComputeBackend::CudaGpu.is_cpu_only());
    }

    #[test]
    fn test_backend_t_kernel() {
        assert!((ComputeBackend::CudaGpu.t_kernel_ms() - 70.0).abs() < 0.1);
        assert!((ComputeBackend::CpuOnly.t_kernel_ms() - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_backend_ghost_ttft() {
        assert_eq!(ComputeBackend::CudaGpu.ghost_ttft_ms(), Some(80.0));
        assert_eq!(ComputeBackend::CpuOnly.ghost_ttft_ms(), None);
    }

    #[test]
    fn test_backend_display() {
        let s = format!("{}", ComputeBackend::CudaGpu);
        assert!(s.contains("CUDA"));
    }

    // ── Decision Display ─────────────────────────────────────────

    #[test]
    fn test_decision_display() {
        let d = decide_protocol(110.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        let s = format!("{}", d);
        assert!(s.contains("DriveInquisitor"));
        assert!(s.contains("M.I.S.T."));
    }

    #[test]
    fn test_decision_warning_display() {
        let d = decide_protocol(20.0, ComputeBackend::CudaGpu, DEFAULT_LAYER_SIZE_MB);
        assert!(d.warning.is_some());
        let w = d.warning.unwrap();
        assert!(w.contains("very slow"), "Warning: {}", w);
    }
}

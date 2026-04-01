//! DriveInquisitor — Storage Speed Detection & Protocol Selection
//!
//! Measures actual disk throughput and selects the optimal streaming protocol:
//!   - S.L.I.P. v3 for NVMe (≥3000 MB/s) and SATA SSD (≥400 MB/s)
//!   - M.I.S.T. v3 for HDD/USB (≥50 MB/s)
//!   - M.I.S.T. v3 degraded (<50 MB/s)
//!
//! Also computes the adaptive pipeline depth (D_opt) and optimal batch size (B_opt)
//! based on measured I/O and compute timings.
//!
//! Reference: air_rs_protocols_v3.md §2 "Protocol Selection Decision Matrix"

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
        // When compute == IO, D_opt = ceil(1.0) + 1 = 2
        let d = calculate_d_opt(
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_compute_slower_than_io() {
        // T_compute=30ms, T_io=10ms → D_opt = ceil(3.0) + 1 = 4
        let d = calculate_d_opt(
            Duration::from_millis(30),
            Duration::from_millis(10),
        );
        assert_eq!(d, 4);
    }

    #[test]
    fn test_d_opt_io_slower_than_compute() {
        // T_compute=5ms, T_io=20ms → D_opt = ceil(0.25) + 1 = 2
        let d = calculate_d_opt(
            Duration::from_millis(5),
            Duration::from_millis(20),
        );
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_zero_io() {
        // Zero I/O time → minimal pipeline
        let d = calculate_d_opt(
            Duration::from_millis(10),
            Duration::ZERO,
        );
        assert_eq!(d, 2);
    }

    #[test]
    fn test_d_opt_clamped_to_bounds() {
        // Very slow IO, fast compute → would give D > 8, but clamped
        let d = calculate_d_opt(
            Duration::from_millis(100),
            Duration::from_millis(1),
        );
        assert_eq!(d, 8); // clamped to max
    }

    // ── B_opt Tests ──────────────────────────────────────────────

    #[test]
    fn test_b_opt_basic() {
        // T_io=10ms, T_kernel=2ms → B_opt = ceil(5.0) = 5
        let b = calculate_b_opt(
            Duration::from_millis(10),
            Duration::from_millis(2),
        );
        assert_eq!(b, 5);
    }

    #[test]
    fn test_b_opt_zero_kernel() {
        // Zero kernel time → B_opt = 1
        let b = calculate_b_opt(
            Duration::from_millis(10),
            Duration::ZERO,
        );
        assert_eq!(b, 1);
    }

    #[test]
    fn test_b_opt_clamped_to_64() {
        // Very slow IO → large B, but clamped to 64
        let b = calculate_b_opt(
            Duration::from_millis(1000),
            Duration::from_micros(100),
        );
        assert_eq!(b, 64);
    }

    // ── Rho Tests ────────────────────────────────────────────────

    #[test]
    fn test_rho_perfect_overlap() {
        // When compute == IO, ρ = 1.0
        let rho = calculate_rho(
            StreamingProtocol::SlipNvme,
            Duration::from_millis(10),
            Duration::from_millis(10),
            1,
        );
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rho_compute_bound() {
        // T_compute=20ms, T_io=5ms → ρ = 20/20 = 1.0 (compute dominates)
        let rho = calculate_rho(
            StreamingProtocol::SlipNvme,
            Duration::from_millis(20),
            Duration::from_millis(5),
            1,
        );
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rho_io_bound() {
        // T_compute=5ms, T_io=20ms → ρ = 5/20 = 0.25
        let rho = calculate_rho(
            StreamingProtocol::SlipNvme,
            Duration::from_millis(5),
            Duration::from_millis(20),
            1,
        );
        assert!((rho - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_rho_mist_batch() {
        // M.I.S.T.: T_compute=2ms, T_io=10ms, B=5 → ρ = (2*5)/10 = 1.0
        let rho = calculate_rho(
            StreamingProtocol::Mist,
            Duration::from_millis(2),
            Duration::from_millis(10),
            5,
        );
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
}

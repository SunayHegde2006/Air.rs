//! DriveInquisitor — Lightweight storage speed measurement & protocol helper.

use std::path::Path;
use std::time::{Duration, Instant};
use std::io::Read;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingProtocol {
    SlipNvme,
    SlipSata,
    Mist,
    MistDegraded,
}

pub fn select_protocol(speed_mbps: f64) -> StreamingProtocol {
    if speed_mbps >= 3000.0 { StreamingProtocol::SlipNvme }
    else if speed_mbps >= 400.0 { StreamingProtocol::SlipSata }
    else if speed_mbps >= 50.0 { StreamingProtocol::Mist }
    else { StreamingProtocol::MistDegraded }
}

pub fn calculate_d_opt(t_compute: Duration, t_io: Duration) -> usize {
    if t_io.is_zero() { return 2; }
    ((t_compute.as_secs_f64() / t_io.as_secs_f64()).ceil() as usize + 1).clamp(2, 8)
}

pub fn burst_read_50ms(path: &Path) -> anyhow::Result<f64> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total_bytes: u64 = 0;
    let start = Instant::now();
    
    while start.elapsed() < Duration::from_millis(50) {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => total_bytes += n as u64,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    Ok(if elapsed > 0.0 { (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed } else { 3000.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_protocol() {
        assert_eq!(select_protocol(3500.0), StreamingProtocol::SlipNvme);
        assert_eq!(select_protocol(500.0), StreamingProtocol::SlipSata);
        assert_eq!(select_protocol(100.0), StreamingProtocol::Mist);
    }
}

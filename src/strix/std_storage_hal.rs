//! Standard-library storage HAL implementation.
//!
//! `StdStorageHal` implements `StorageHal` using `std::fs` synchronous I/O.
//! All "async" operations complete immediately — this is a fallback for
//! platforms without io_uring / IOCP and for unit testing.
//!
//! From STRIX Protocol §12.3 — reference implementation.

use super::hal::{FileHandle, HalError, IoHandle, IoStatus, StorageHal, StorageType, ThroughputProfile};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

/// Per-IO-handle completion record.
struct IoCompletion {
    bytes_read: usize,
    failed: bool,
}

/// Synchronous `StorageHal` backed by `std::fs`.
///
/// All operations complete immediately (no actual async I/O).
/// Designed for testing and as a cross-platform baseline.
pub struct StdStorageHal {
    /// Open file handles, protected by a mutex for `Send + Sync`.
    files: Mutex<HashMap<u64, File>>,
    /// Completed I/O records.
    completions: Mutex<HashMap<u64, IoCompletion>>,
    /// Next file handle ID.
    next_file_id: Mutex<u64>,
    /// Next IO handle ID.
    next_io_id: Mutex<u64>,
}

impl StdStorageHal {
    /// Create a new `StdStorageHal`.
    pub fn new() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
            completions: Mutex::new(HashMap::new()),
            next_file_id: Mutex::new(1),
            next_io_id: Mutex::new(1),
        }
    }
}

impl Default for StdStorageHal {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageHal for StdStorageHal {
    fn open(&self, path: &Path, _direct_io: bool) -> Result<FileHandle, HalError> {
        let file = File::open(path)?;
        let mut id = self.next_file_id.lock().unwrap();
        let handle = FileHandle(*id);
        *id += 1;
        self.files.lock().unwrap().insert(handle.0, file);
        Ok(handle)
    }

    fn read_async(
        &self,
        handle: FileHandle,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<IoHandle, HalError> {
        // Synchronous read masquerading as async.
        let mut files = self.files.lock().unwrap();
        let file = files.get_mut(&handle.0).ok_or_else(|| HalError::DriverError {
            code: -1,
            message: format!("unknown FileHandle {:?}", handle),
        })?;

        file.seek(SeekFrom::Start(offset))?;
        let buf_len = buf.len();
        let result = file.read_exact(buf);

        let mut io_id = self.next_io_id.lock().unwrap();
        let io_handle = IoHandle(*io_id);
        *io_id += 1;

        let completion = match result {
            Ok(()) => IoCompletion {
                bytes_read: buf_len,
                failed: false,
            },
            Err(_) => IoCompletion {
                bytes_read: 0,
                failed: true,
            },
        };

        self.completions.lock().unwrap().insert(io_handle.0, completion);
        Ok(io_handle)
    }

    fn poll_io(&self, handle: IoHandle) -> IoStatus {
        let completions = self.completions.lock().unwrap();
        match completions.get(&handle.0) {
            Some(c) if c.failed => IoStatus::Failed,
            Some(c) => IoStatus::Complete(c.bytes_read),
            None => IoStatus::Failed,
        }
    }

    fn wait_io(&self, handle: IoHandle) -> Result<usize, HalError> {
        // Already complete (sync backend).
        let completions = self.completions.lock().unwrap();
        match completions.get(&handle.0) {
            Some(c) if !c.failed => Ok(c.bytes_read),
            Some(_) => Err(HalError::DriverError {
                code: -10,
                message: "I/O operation failed".into(),
            }),
            None => Err(HalError::DriverError {
                code: -11,
                message: format!("unknown IoHandle {:?}", handle),
            }),
        }
    }

    fn detect_storage_type(&self, _path: &Path) -> StorageType {
        // Heuristic: assume SSD on modern systems.
        StorageType::SataSsd
    }

    fn benchmark_throughput(&self, _path: &Path) -> ThroughputProfile {
        // Static defaults — no actual benchmarking.
        ThroughputProfile {
            seq_read_bps: 500_000_000,  // 500 MB/s (SATA SSD baseline)
            random_read_iops: 50_000,
            avg_latency_us: 100,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static STD_STORAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_file_with(content: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let id = STD_STORAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            "strix_std_test_{}_{}.bin",
            std::process::id(),
            id,
        ));
        let mut f = File::create(&path).unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        drop(f); // ensure file is fully closed before reads
        path
    }

    #[test]
    fn open_and_read() {
        let data = b"hello strix storage hal";
        let path = temp_file_with(data);
        let hal = StdStorageHal::new();

        let fh = hal.open(&path, false).unwrap();
        let mut buf = vec![0u8; data.len()];
        let io = hal.read_async(fh, 0, &mut buf).unwrap();

        let n = hal.wait_io(io).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(&buf[..n], data);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_at_offset() {
        let data = b"0123456789ABCDEF";
        let path = temp_file_with(data);
        let hal = StdStorageHal::new();

        let fh = hal.open(&path, false).unwrap();
        let mut buf = [0u8; 4];
        let io = hal.read_async(fh, 10, &mut buf).unwrap();

        let n = hal.wait_io(io).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf, b"ABCD");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn poll_completes_immediately() {
        let data = b"poll test";
        let path = temp_file_with(data);
        let hal = StdStorageHal::new();

        let fh = hal.open(&path, false).unwrap();
        let mut buf = [0u8; 9];
        let io = hal.read_async(fh, 0, &mut buf).unwrap();

        // Should complete immediately since it's synchronous.
        match hal.poll_io(io) {
            IoStatus::Complete(n) => assert_eq!(n, 9),
            other => panic!("expected Complete, got {:?}", other),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_not_found() {
        let hal = StdStorageHal::new();
        let result = hal.open(Path::new("__nonexistent_strix_test__.bin"), false);
        assert!(result.is_err());
    }

    #[test]
    fn benchmark_returns_defaults() {
        let hal = StdStorageHal::new();
        let tp = hal.benchmark_throughput(Path::new("."));
        assert!(tp.seq_read_bps > 0);
        assert!(tp.random_read_iops > 0);
        assert!(tp.avg_latency_us > 0);
        assert_eq!(hal.detect_storage_type(Path::new(".")), StorageType::SataSsd);
    }
}

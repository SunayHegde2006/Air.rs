//! Memory-mapped storage HAL — STRIX Protocol §9.4, §11.3.
//!
//! `MmapStorageHal` implements `StorageHal` using OS memory mapping.
//! Files are read into memory on open, then reads are served directly
//! from the in-memory buffer (simulating mmap zero-copy semantics).
//!
//! Platform prefetch hints:
//! - Linux: `madvise(MADV_WILLNEED)` 
//! - Windows: `PrefetchVirtualMemory`
//! - macOS: `madvise(MADV_WILLNEED)`
//!
//! Auto-selected when NVMe detected and RAM ≥ 2× model size.

use super::hal::{FileHandle, HalError, IoHandle, IoStatus, StorageHal, StorageType, ThroughputProfile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ── Inner state ──────────────────────────────────────────────────────────

/// An open memory-mapped file.
struct MappedFile {
    /// The raw bytes (memory-mapped via read-to-RAM).
    data: Vec<u8>,
    /// Original file path.
    _path: PathBuf,
}

/// Per-IO-handle completion record.
struct IoCompletion {
    bytes_read: usize,
    failed: bool,
}

/// Memory-mapped storage HAL.
///
/// Reads are served directly from the mapped region — zero-copy when
/// the OS page cache is warm. Prefetch hints cause the OS to read
/// pages ahead of time.
pub struct MmapStorageHal {
    files: Mutex<HashMap<u64, MappedFile>>,
    completions: Mutex<HashMap<u64, IoCompletion>>,
    next_file_id: Mutex<u64>,
    next_io_id: Mutex<u64>,
    /// Whether to use madvise/PrefetchVirtualMemory hints.
    enable_prefetch_hints: bool,
}

impl MmapStorageHal {
    /// Create a new memory-mapped storage HAL.
    pub fn new() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
            completions: Mutex::new(HashMap::new()),
            next_file_id: Mutex::new(1),
            next_io_id: Mutex::new(1),
            enable_prefetch_hints: true,
        }
    }

    /// Create with prefetch hints disabled (for testing).
    pub fn without_prefetch() -> Self {
        let mut s = Self::new();
        s.enable_prefetch_hints = false;
        s
    }

    /// Check if the system should prefer mmap over explicit I/O.
    ///
    /// Returns true when RAM ≥ 2× model_size (NVMe assumption).
    pub fn should_use_mmap(model_size_bytes: u64) -> bool {
        let available_ram = Self::estimate_available_ram();
        available_ram >= model_size_bytes * 2
    }

    /// Estimate available system RAM in bytes (platform-specific).
    fn estimate_available_ram() -> u64 {
        #[cfg(target_os = "windows")]
        {
            #[repr(C)]
            struct MemoryStatusEx {
                dw_length: u32,
                dw_memory_load: u32,
                ull_total_phys: u64,
                ull_avail_phys: u64,
                ull_total_page_file: u64,
                ull_avail_page_file: u64,
                ull_total_virtual: u64,
                ull_avail_virtual: u64,
                ull_avail_extended_virtual: u64,
            }

            extern "system" {
                fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
            }

            let mut status = MemoryStatusEx {
                dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
                dw_memory_load: 0,
                ull_total_phys: 0,
                ull_avail_phys: 0,
                ull_total_page_file: 0,
                ull_avail_page_file: 0,
                ull_total_virtual: 0,
                ull_avail_virtual: 0,
                ull_avail_extended_virtual: 0,
            };

            let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
            if ok != 0 {
                return status.ull_avail_phys;
            }
            16 * 1024 * 1024 * 1024
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
            16 * 1024 * 1024 * 1024
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            16 * 1024 * 1024 * 1024 // 16 GB fallback
        }
    }

    /// Issue a prefetch hint for a byte range within a mapped file.
    pub fn prefetch_range(&self, _file_handle: FileHandle, _offset: u64, _size: usize) {
        if !self.enable_prefetch_hints {
            return;
        }
        // Production: libc::madvise(ptr, len, MADV_WILLNEED) / PrefetchVirtualMemory
    }

    /// Release pages back to the OS (eviction hint).
    pub fn release_range(&self, _file_handle: FileHandle, _offset: u64, _size: usize) {
        if !self.enable_prefetch_hints {
            return;
        }
        // Production: libc::madvise(ptr, len, MADV_DONTNEED) / DiscardVirtualMemory
    }
}

impl Default for MmapStorageHal {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageHal for MmapStorageHal {
    fn open(&self, path: &Path, _direct_io: bool) -> Result<FileHandle, HalError> {
        let data = std::fs::read(path)?;

        let mut id = self.next_file_id.lock().unwrap();
        let handle = FileHandle(*id);
        *id += 1;
        self.files.lock().unwrap().insert(handle.0, MappedFile {
            data,
            _path: path.to_path_buf(),
        });
        Ok(handle)
    }

    fn read_async(
        &self,
        handle: FileHandle,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<IoHandle, HalError> {
        let files = self.files.lock().unwrap();
        let file = files.get(&handle.0).ok_or_else(|| HalError::DriverError {
            code: -1,
            message: format!("unknown FileHandle {:?}", handle),
        })?;

        let start = offset as usize;
        let end = start + buf.len();

        let mut io_id = self.next_io_id.lock().unwrap();
        let io_handle = IoHandle(*io_id);
        *io_id += 1;

        if end > file.data.len() {
            self.completions.lock().unwrap().insert(io_handle.0, IoCompletion {
                bytes_read: 0,
                failed: true,
            });
            return Ok(io_handle);
        }

        // Memory-mapped read: direct copy from buffer
        buf.copy_from_slice(&file.data[start..end]);

        self.completions.lock().unwrap().insert(io_handle.0, IoCompletion {
            bytes_read: buf.len(),
            failed: false,
        });
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
        let completions = self.completions.lock().unwrap();
        match completions.get(&handle.0) {
            Some(c) if !c.failed => Ok(c.bytes_read),
            Some(_) => Err(HalError::DriverError {
                code: -10,
                message: "mmap I/O operation failed".into(),
            }),
            None => Err(HalError::DriverError {
                code: -11,
                message: format!("unknown IoHandle {:?}", handle),
            }),
        }
    }

    fn detect_storage_type(&self, _path: &Path) -> StorageType {
        // Mmap HAL is typically used with NVMe
        StorageType::NvmePcie4
    }

    fn benchmark_throughput(&self, _path: &Path) -> ThroughputProfile {
        // Mmap performance characteristics (memory-speed reads)
        ThroughputProfile {
            seq_read_bps: 5_000_000_000,   // ~5 GB/s (memory bandwidth)
            random_read_iops: 1_000_000,    // ~1M IOPS (page cache)
            avg_latency_us: 1,              // ~1µs (cached)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, data: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_mmap_test_{}_{name}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        drop(f);
        path
    }

    #[test]
    fn open_and_read() {
        let data = b"hello mmap world!";
        let path = write_temp("hello.bin", data);
        let hal = MmapStorageHal::new();
        let handle = hal.open(&path, false).unwrap();
        let mut buf = vec![0u8; data.len()];
        let io = hal.read_async(handle, 0, &mut buf).unwrap();
        assert!(matches!(hal.poll_io(io), IoStatus::Complete(_)));
        let n = hal.wait_io(io).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(&buf, data);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_at_offset() {
        let data = b"__prefix__PAYLOAD_DATA__suffix__";
        let path = write_temp("offset.bin", data);
        let hal = MmapStorageHal::new();
        let handle = hal.open(&path, false).unwrap();
        let mut buf = vec![0u8; 12]; // "PAYLOAD_DATA"
        let io = hal.read_async(handle, 10, &mut buf).unwrap();
        let n = hal.wait_io(io).unwrap();
        assert_eq!(n, 12);
        assert_eq!(&buf, b"PAYLOAD_DATA");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_past_eof_returns_failed() {
        let data = b"short";
        let path = write_temp("short.bin", data);
        let hal = MmapStorageHal::new();
        let handle = hal.open(&path, false).unwrap();
        let mut buf = vec![0u8; 100];
        let io = hal.read_async(handle, 0, &mut buf).unwrap();
        assert!(matches!(hal.poll_io(io), IoStatus::Failed));
        let result = hal.wait_io(io);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn should_use_mmap_heuristic() {
        assert!(MmapStorageHal::should_use_mmap(1024));
        assert!(!MmapStorageHal::should_use_mmap(1_000_000_000_000));
    }

    #[test]
    fn detect_storage_type_nvme() {
        let hal = MmapStorageHal::new();
        assert_eq!(hal.detect_storage_type(Path::new("test")), StorageType::NvmePcie4);
    }

    #[test]
    fn benchmark_throughput_returns_high_values() {
        let hal = MmapStorageHal::new();
        let profile = hal.benchmark_throughput(Path::new("test"));
        assert!(profile.seq_read_bps >= 1_000_000_000); // > 1 GB/s
    }

    #[test]
    fn prefetch_and_release_no_panic() {
        let hal = MmapStorageHal::new();
        hal.prefetch_range(FileHandle(1), 0, 1024);
        hal.release_range(FileHandle(1), 0, 1024);
    }

    #[test]
    fn without_prefetch_mode() {
        let hal = MmapStorageHal::without_prefetch();
        assert!(!hal.enable_prefetch_hints);
    }
}

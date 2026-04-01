//! Hardware Abstraction Layer (HAL) trait definitions.
//!
//! These traits define the contract between STRIX and vendor-specific
//! GPU/storage backends. **No implementations here** — concrete backends
//! (CUDA, Vulkan, Metal, etc.) will implement these in future modules.
//!
//! From STRIX Protocol §12.

use super::types::GpuPtr;
use std::fmt;
use std::path::Path;

// ── Error Types ──────────────────────────────────────────────────────────

/// Errors returned by HAL operations.
#[derive(Debug)]
pub enum HalError {
    /// Out of VRAM.
    OutOfMemory { requested: usize, available: usize },
    /// GPU driver error with vendor error code.
    DriverError { code: i32, message: String },
    /// The requested operation is not supported on this hardware.
    Unsupported(String),
    /// I/O error during storage operations.
    IoError(std::io::Error),
    /// Timeout waiting for an async operation.
    Timeout,
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "OOM: requested {requested} bytes, only {available} available")
            }
            Self::DriverError { code, message } => {
                write!(f, "GPU driver error {code}: {message}")
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::Timeout => write!(f, "HAL operation timed out"),
        }
    }
}

impl std::error::Error for HalError {}

impl From<std::io::Error> for HalError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

// ── GPU HAL Trait ────────────────────────────────────────────────────────

/// GPU capabilities reported at startup.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable device name (e.g. "NVIDIA RTX 4090").
    pub name: String,
    /// Total VRAM in bytes.
    pub vram_total: usize,
    /// Currently free VRAM in bytes.
    pub vram_free: usize,
    /// Compute capability or equivalent (CUDA: 89, Vulkan: spirv version, etc.).
    pub compute_capability: u32,
    /// Maximum PCIe/bus bandwidth in bytes/sec (approximate).
    pub bus_bandwidth: u64,
}

/// GPU Hardware Abstraction Layer (STRIX Protocol §12.1–12.2).
///
/// Implementors: `CudaHal`, `VulkanHal`, `MetalHal`, `CpuHal`, etc.
pub trait GpuHal: Send + Sync {
    /// Query GPU capabilities.
    fn info(&self) -> Result<GpuInfo, HalError>;

    /// Allocate VRAM with the given alignment (must be power of 2).
    fn allocate_vram(&self, size: usize, alignment: usize) -> Result<GpuPtr, HalError>;

    /// Free a previously allocated VRAM region.
    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError>;

    /// Copy `size` bytes from host RAM (`src`) to VRAM (`dst`).
    ///
    /// `stream` selects the async copy engine (0 = default).
    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError>;

    /// Copy `size` bytes from VRAM (`src`) to host RAM (`dst`).
    fn copy_from_vram(
        &self,
        dst: *mut u8,
        src: GpuPtr,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError>;

    /// Synchronise the given stream (wait for all pending copies to finish).
    fn sync_stream(&self, stream: u32) -> Result<(), HalError>;

    /// Current VRAM usage snapshot.
    fn vram_used(&self) -> Result<usize, HalError>;
}

// ── Storage HAL Trait ────────────────────────────────────────────────────

/// Opaque file handle returned by the storage HAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileHandle(pub u64);

/// Opaque async I/O handle for polling/waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IoHandle(pub u64);

/// Status of an async I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoStatus {
    /// Operation is still in progress.
    Pending,
    /// Operation completed successfully.
    Complete(usize),
    /// Operation failed.
    Failed,
}

/// Detected storage device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    NvmePcie4,
    NvmePcie3,
    SataSsd,
    Hdd,
    RamDisk,
    NetworkFs,
}

/// Measured throughput characteristics.
#[derive(Debug, Clone)]
pub struct ThroughputProfile {
    /// Sequential read bandwidth in bytes/sec.
    pub seq_read_bps: u64,
    /// Random 4K read IOPS.
    pub random_read_iops: u64,
    /// Average latency for a 1MB read in microseconds.
    pub avg_latency_us: u64,
}

/// Storage Hardware Abstraction Layer (STRIX Protocol §12.3).
///
/// Wraps OS-specific async I/O (io_uring on Linux, IOCP on Windows, kqueue on macOS).
pub trait StorageHal: Send + Sync {
    /// Open a file for direct I/O.
    fn open(&self, path: &Path, direct_io: bool) -> Result<FileHandle, HalError>;

    /// Submit an async read from `handle` at `offset` into `buf`.
    fn read_async(&self, handle: FileHandle, offset: u64, buf: &mut [u8])
        -> Result<IoHandle, HalError>;

    /// Non-blocking poll on an async I/O handle.
    fn poll_io(&self, handle: IoHandle) -> IoStatus;

    /// Blocking wait until I/O completes. Returns bytes read.
    fn wait_io(&self, handle: IoHandle) -> Result<usize, HalError>;

    /// Detect the storage device type for the given path.
    fn detect_storage_type(&self, path: &Path) -> StorageType;

    /// Benchmark throughput characteristics.
    fn benchmark_throughput(&self, path: &Path) -> ThroughputProfile;
}

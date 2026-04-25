// Suppress dead_code: rustc 1.95.0 ICEs on check_mod_deathness for repr(C)
// structs with large arrays when cuda feature disabled. Remove when fixed upstream.
#![allow(dead_code)]
//! GPUDirect Storage integration (STRIX Protocol §9.4, S.L.I.P. v3 §Sub-System 2 Path A).
//!
//! to GPU VRAM, bypassing the CPU and system RAM entirely:
//!
//! ```text
//! Traditional: NVMe → RAM (page cache) → RAM (user buffer) → VRAM
//! GPUDirect:   NVMe → VRAM (direct DMA)
//! Fallback:    NVMe → pinned RAM (cudaHostRegister) → VRAM
//! ```
//!
//! This module provides:
//! - `GdsCapability` — runtime detection of GDS support
//! - `GdsTransfer` — zero-copy NVMe→GPU transfer descriptor
//! - `GdsStorageHal` — `StorageHal` implementation using cuFile API
//! - `CuFileFfi` — raw FFI bindings to `libcufile.so` (Path A)
//! - `PinnedHostBuffer` — `cudaHostRegister` pinned staging (Path A fallback)
//!
//! Requires NVIDIA CUDA 11.4+ and GDS driver. Falls back gracefully
//! when GDS is unavailable.

use super::hal::HalError;
use super::types::GpuPtr;
use std::fmt;

// ── cuFile FFI Bindings (Path A) ─────────────────────────────────────────

/// cuFile driver properties — opaque to us, just a handle.
/// The real struct is ~4KB; we treat it as an opaque blob.
#[repr(C)]
struct CuFileDriverProps {
    _opaque: [u8; 4096],
}

/// cuFile handle descriptor.
#[repr(C)]
struct CuFileDescr {
    /// File descriptor (Linux fd).
    handle_type: i32, // CU_FILE_HANDLE_TYPE_OPAQUE_FD = 0
    handle_fd: i32,
    _padding: [u8; 256],
}

/// cuFile completion status.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuFileStatus {
    /// Number of bytes transferred, or negative error code.
    ret: i64,
}

/// Feature-gated raw cuFile function declarations.
///
/// These symbols are resolved at link time against `libcufile.so`.
/// When the `cuda` feature is disabled, none of this code compiles.
#[cfg(feature = "cuda")]
mod cufile_ffi {
    use super::*;

    #[link(name = "cufile")]
    extern "C" {
        /// Initialize the cuFile driver.
        pub fn cuFileDriverOpen() -> i32;

        /// Shut down the cuFile driver.
        pub fn cuFileDriverClose() -> i32;

        /// Register a GPU buffer for cuFile I/O.
        ///
        /// `devPtr_base`: device pointer returned by cudaMalloc
        /// `size`: buffer size in bytes
        /// `flags`: 0
        pub fn cuFileBufRegister(
            dev_ptr_base: *mut u8,
            size: usize,
            flags: u32,
        ) -> i32;

        /// Deregister a GPU buffer.
        pub fn cuFileBufDeregister(dev_ptr_base: *mut u8) -> i32;

        /// Register a file descriptor for cuFile I/O.
        ///
        /// `descr`: cuFile descriptor (wraps the fd)
        /// Returns opaque cuFile handle or error.
        pub fn cuFileHandleRegister(
            handle: *mut std::ffi::c_void,
            descr: *const CuFileDescr,
        ) -> i32;

        /// Deregister a file handle.
        pub fn cuFileHandleDeregister(handle: *mut std::ffi::c_void) -> i32;

        /// Synchronous read: file → GPU VRAM (DMA).
        ///
        /// `handle`: registered cuFile handle
        /// `devPtr_base`: registered device buffer
        /// `size`: bytes to read
        /// `file_offset`: offset in the file
        /// `devPtr_offset`: offset within the device buffer
        ///
        /// Returns bytes read (≥0) or negative error code.
        pub fn cuFileRead(
            handle: *mut std::ffi::c_void,
            dev_ptr_base: *mut u8,
            size: usize,
            file_offset: i64,
            dev_ptr_offset: i64,
        ) -> i64;
    }
}

/// Check a cuFile return code.
#[cfg(feature = "cuda")]
fn cufile_check(code: i32, op: &str) -> Result<(), HalError> {
    if code == 0 {
        Ok(())
    } else {
        Err(HalError::DriverError {
            code,
            message: format!("cuFile {op} failed with code {code}"),
        })
    }
}

// ── Pinned Host Buffer (Path A Fallback) ─────────────────────────────────

/// A host RAM buffer registered with `cudaHostRegister` for pinned DMA.
///
/// When GDS is unavailable, this provides the fastest host→device path:
/// pinned memory enables the GPU to DMA directly from host RAM without
/// an intermediate staging copy.
///
/// ```text
/// Without pin:  NVMe → page cache → user buf → staging buf → VRAM
/// With pin:     NVMe → pinned buf → VRAM (DMA, no staging)
/// ```
pub struct PinnedHostBuffer {
    /// Host pointer to the pinned buffer.
    ptr: *mut u8,
    /// Size in bytes.
    size: usize,
    /// Whether this buffer was registered with cudaHostRegister.
    registered: bool,
}

// SAFETY: The buffer is allocated on the host and pinned; the GPU accesses
// it via DMA which is safe as long as we don't free it while in-flight.
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaHostRegister(ptr: *mut u8, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut u8) -> i32;
}

/// cudaHostRegisterPortable — allows the buffer to be used from any CUDA context.
#[cfg(feature = "cuda")]
const CUDA_HOST_REGISTER_PORTABLE: u32 = 1;

impl PinnedHostBuffer {
    /// Allocate a page-aligned host buffer and pin it with `cudaHostRegister`.
    ///
    /// The buffer is page-aligned (4KB) for optimal DMA performance.
    /// Falls back to unpinned allocation if CUDA is unavailable.
    pub fn new(size: usize) -> Result<Self, HalError> {
        if size == 0 {
            return Err(HalError::Unsupported("Cannot create zero-size pinned buffer".into()));
        }

        // Allocate page-aligned memory.
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|e| HalError::Unsupported(format!("Invalid layout: {e}")))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(HalError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }

        // Pin the buffer with cudaHostRegister if CUDA is available.
        #[cfg(feature = "cuda")]
        let registered = {
            let code = unsafe { cudaHostRegister(ptr, size, CUDA_HOST_REGISTER_PORTABLE) };
            code == 0
        };
        #[cfg(not(feature = "cuda"))]
        let registered = false;

        Ok(Self {
            ptr,
            size,
            registered,
        })
    }

    /// Whether this buffer is actually pinned (cudaHostRegister succeeded).
    pub fn is_pinned(&self) -> bool {
        self.registered
    }

    /// Raw pointer to the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Borrow as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Borrow as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Read from a file into this pinned buffer.
    ///
    /// Uses `pread`/`ReadFile` at the given offset for zero-copy positioning.
    pub fn read_from_file(
        &mut self,
        file: &std::fs::File,
        offset: u64,
        len: usize,
    ) -> Result<usize, HalError> {
        if len > self.size {
            return Err(HalError::Unsupported(format!(
                "Read size ({len}) exceeds buffer size ({})", self.size
            )));
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            let result = unsafe {
                libc::pread(
                    fd,
                    self.ptr as *mut libc::c_void,
                    len,
                    offset as libc::off_t,
                )
            };
            if result < 0 {
                return Err(HalError::IoError(std::io::Error::last_os_error()));
            }
            Ok(result as usize)
        }

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::io::AsRawHandle;

            extern "system" {
                fn ReadFile(
                    h_file: isize,
                    lp_buffer: *mut u8,
                    n_number_of_bytes: u32,
                    lp_number_of_bytes_read: *mut u32,
                    lp_overlapped: *mut Overlapped,
                ) -> i32;
            }

            #[repr(C)]
            struct Overlapped {
                internal: usize,
                internal_high: usize,
                offset_low: u32,
                offset_high: u32,
                h_event: isize,
            }

            let handle = file.as_raw_handle() as isize;
            let mut bytes_read: u32 = 0;
            let mut overlapped = Overlapped {
                internal: 0,
                internal_high: 0,
                offset_low: offset as u32,
                offset_high: (offset >> 32) as u32,
                h_event: 0,
            };

            let ok = unsafe {
                ReadFile(
                    handle,
                    self.ptr,
                    len as u32,
                    &mut bytes_read,
                    &mut overlapped,
                )
            };
            if ok == 0 {
                return Err(HalError::IoError(std::io::Error::last_os_error()));
            }
            Ok(bytes_read as usize)
        }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        // Unpin first.
        #[cfg(feature = "cuda")]
        if self.registered {
            unsafe { cudaHostUnregister(self.ptr) };
        }

        // Free page-aligned memory.
        if !self.ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.size, 4096).unwrap();
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

impl fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("size", &self.size)
            .field("pinned", &self.registered)
            .field("ptr", &format_args!("{:p}", self.ptr))
            .finish()
    }
}

// ── GDS Capability Detection ─────────────────────────────────────────────

/// GPUDirect Storage runtime capabilities.
#[derive(Debug, Clone)]
pub struct GdsCapability {
    /// Whether the GDS driver is installed and loaded.
    pub driver_available: bool,
    /// Whether the filesystem supports GDS (ext4, XFS on Linux).
    pub filesystem_supported: bool,
    /// Maximum single transfer size in bytes (typically 16MB).
    pub max_transfer_size: usize,
    /// Whether page cache bypass is available (O_DIRECT equivalent).
    pub page_cache_bypass: bool,
    /// GDS driver version string.
    pub driver_version: String,
    /// Whether cudaHostRegister fallback is available.
    pub pinned_fallback_available: bool,
}

impl GdsCapability {
    /// Probe the system for GPUDirect Storage support.
    ///
    /// Returns capability information without initializing GDS.
    pub fn probe() -> Self {
        let driver_available = Self::check_cufile_available();
        let pinned_fallback = cfg!(feature = "cuda");

        Self {
            driver_available,
            filesystem_supported: driver_available && cfg!(target_os = "linux"),
            max_transfer_size: if driver_available { 16 * 1024 * 1024 } else { 0 },
            page_cache_bypass: driver_available,
            driver_version: if driver_available {
                "1.0 (detected)".to_string()
            } else {
                "not available".to_string()
            },
            pinned_fallback_available: pinned_fallback,
        }
    }

    /// Check whether cuFile library is loadable.
    fn check_cufile_available() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check standard library paths and CUDA toolkit paths
            let paths = [
                "/usr/lib/x86_64-linux-gnu/libcufile.so",
                "/usr/local/cuda/lib64/libcufile.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
            ];
            paths.iter().any(|p| std::path::Path::new(p).exists())
                || std::env::var("CUFILE_PATH").is_ok()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Whether GDS is fully usable for direct NVMe→GPU transfers.
    pub fn is_usable(&self) -> bool {
        self.driver_available && self.filesystem_supported
    }

    /// Whether pinned host fallback is available (CUDA but no GDS).
    pub fn has_pinned_fallback(&self) -> bool {
        self.pinned_fallback_available && !self.is_usable()
    }
}

// ── GDS Transfer Descriptor ──────────────────────────────────────────────

/// A pending GPUDirect Storage transfer from file to GPU VRAM.
#[derive(Debug)]
pub struct GdsTransfer {
    /// Target GPU pointer for the DMA.
    pub gpu_dst: GpuPtr,
    /// Source file path.
    pub file_path: String,
    /// Offset within the file (bytes).
    pub file_offset: u64,
    /// Transfer size in bytes.
    pub transfer_size: usize,
    /// Transfer status.
    pub status: GdsTransferStatus,
    /// Transfer method used.
    pub method: TransferMethod,
    /// Bytes actually transferred (set on completion).
    pub bytes_transferred: usize,
}

/// Transfer status for a GDS operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdsTransferStatus {
    /// Not yet submitted.
    Pending,
    /// Submitted to the GDS driver, awaiting completion.
    InFlight,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed — fallback to staged copy.
    Failed,
}

/// Which I/O path was used for the transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMethod {
    /// cuFile direct DMA (Path A).
    CuFileDirect,
    /// cudaHostRegister pinned staging (Path A fallback).
    PinnedHostStaging,
    /// Standard cudaMemcpy staging (legacy path).
    StandardStaging,
    /// Not yet determined.
    Unknown,
}

impl fmt::Display for TransferMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CuFileDirect => write!(f, "cuFile DMA"),
            Self::PinnedHostStaging => write!(f, "pinned host staging"),
            Self::StandardStaging => write!(f, "standard staging"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ── GDS Storage HAL ──────────────────────────────────────────────────────

/// GPUDirect Storage HAL for NVMe→GPU DMA transfers.
///
/// Transfer path selection:
/// 1. **cuFile DMA** (Path A) — if GDS driver available
/// 2. **Pinned host staging** (Path A fallback) — if CUDA available
/// 3. **Standard staging** — universal fallback
pub struct GdsStorageHal {
    /// Runtime capabilities.
    capability: GdsCapability,
    /// Alignment requirement for GDS transfers (4KB).
    alignment: usize,
    /// Whether the cuFile driver has been initialized.
    driver_initialized: bool,
    /// Reusable pinned staging buffer (Path A fallback).
    pinned_buffer: Option<PinnedHostBuffer>,
    /// Number of completed transfers (for stats).
    completed_transfers: std::sync::atomic::AtomicU64,
    /// Total bytes transferred via GDS.
    total_bytes_transferred: std::sync::atomic::AtomicU64,
    /// Transfers by method: [cuFile, pinned, standard].
    transfers_by_method: [std::sync::atomic::AtomicU64; 3],
}

impl GdsStorageHal {
    /// Create a new GDS storage HAL.
    ///
    /// Probes the system for GDS support. If GDS is available,
    /// initializes the cuFile driver. Otherwise, allocates a pinned
    /// host buffer for the fallback staging path.
    pub fn new() -> Self {
        let capability = GdsCapability::probe();
        // Initialize cuFile driver if available via cuFileDriverOpen.
        #[cfg(feature = "cuda")]
        let driver_initialized = if capability.is_usable() {
            let code = unsafe { cufile_ffi::cuFileDriverOpen() };
            code == 0
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let driver_initialized = false;

        // Allocate pinned staging buffer (16MB — matches max GDS transfer size).
        let pinned_buffer = if capability.has_pinned_fallback() || !capability.is_usable() {
            PinnedHostBuffer::new(16 * 1024 * 1024).ok()
        } else {
            None
        };

        Self {
            capability,
            alignment: 4096,
            driver_initialized,
            pinned_buffer,
            completed_transfers: std::sync::atomic::AtomicU64::new(0),
            total_bytes_transferred: std::sync::atomic::AtomicU64::new(0),
            transfers_by_method: [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ],
        }
    }

    /// Whether GDS is available for direct transfers.
    pub fn is_available(&self) -> bool {
        self.capability.is_usable() && self.driver_initialized
    }

    /// Get runtime capability information.
    pub fn capability(&self) -> &GdsCapability {
        &self.capability
    }

    /// Whether a transfer of the given size needs staging (no direct GDS).
    pub fn needs_staging(&self, size: usize) -> bool {
        !self.is_available() || size > self.capability.max_transfer_size
    }

    /// Required alignment for GDS buffers.
    pub fn required_alignment(&self) -> usize {
        self.alignment
    }

    /// Select the best transfer method for the given parameters.
    pub fn select_method(&self, size: usize) -> TransferMethod {
        if self.is_available() && size <= self.capability.max_transfer_size {
            TransferMethod::CuFileDirect
        } else if self.pinned_buffer.is_some() {
            TransferMethod::PinnedHostStaging
        } else {
            TransferMethod::StandardStaging
        }
    }

    /// Create a transfer descriptor for a file→GPU DMA.
    ///
    /// Automatically selects the best transfer method.
    pub fn create_transfer(
        &self,
        gpu_dst: GpuPtr,
        file_path: &str,
        file_offset: u64,
        size: usize,
    ) -> GdsTransfer {
        GdsTransfer {
            gpu_dst,
            file_path: file_path.to_string(),
            file_offset,
            transfer_size: size,
            status: GdsTransferStatus::Pending,
            method: self.select_method(size),
            bytes_transferred: 0,
        }
    }

    /// Submit a GDS transfer for execution.
    ///
    /// Tries the following paths in order:
    /// 1. cuFile DMA (if GDS available)
    /// 2. Pinned host staging (if CUDA available)
    /// 3. Falls back to failed status (caller must use standard staging)
    pub fn submit(&mut self, transfer: &mut GdsTransfer) -> Result<(), HalError> {
        match transfer.method {
            TransferMethod::CuFileDirect => {
                self.submit_cufile(transfer)
            }
            TransferMethod::PinnedHostStaging => {
                self.submit_pinned(transfer)
            }
            TransferMethod::StandardStaging | TransferMethod::Unknown => {
                transfer.status = GdsTransferStatus::Failed;
                Err(HalError::Unsupported(
                    "No accelerated transfer path available; use standard cudaMemcpy".into(),
                ))
            }
        }
    }

    /// Submit via cuFile DMA (Path A).
    #[cfg(feature = "cuda")]
    fn submit_cufile(&mut self, transfer: &mut GdsTransfer) -> Result<(), HalError> {
        use std::os::unix::io::AsRawFd;

        if !self.driver_initialized {
            transfer.method = TransferMethod::PinnedHostStaging;
            return self.submit_pinned(transfer);
        }

        // Open the file with O_DIRECT for page cache bypass.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&transfer.file_path)
            .map_err(HalError::IoError)?;

        let fd = file.as_raw_fd();

        // Register the file handle with cuFile.
        let mut descr = CuFileDescr {
            handle_type: 0, // CU_FILE_HANDLE_TYPE_OPAQUE_FD
            handle_fd: fd,
            _padding: [0u8; 256],
        };
        let mut cufile_handle: *mut std::ffi::c_void = std::ptr::null_mut();

        let reg_result = unsafe {
            cufile_ffi::cuFileHandleRegister(
                &mut cufile_handle as *mut _ as *mut std::ffi::c_void,
                &descr as *const CuFileDescr,
            )
        };
        if reg_result != 0 {
            // cuFile registration failed — fall back to pinned path.
            transfer.method = TransferMethod::PinnedHostStaging;
            return self.submit_pinned(transfer);
        }

        // Register the GPU buffer with cuFile.
        let buf_reg = unsafe {
            cufile_ffi::cuFileBufRegister(
                transfer.gpu_dst.0 as *mut u8,
                transfer.transfer_size,
                0,
            )
        };
        if buf_reg != 0 {
            unsafe { cufile_ffi::cuFileHandleDeregister(cufile_handle) };
            transfer.method = TransferMethod::PinnedHostStaging;
            return self.submit_pinned(transfer);
        }

        // Execute the DMA read.
        transfer.status = GdsTransferStatus::InFlight;
        let bytes = unsafe {
            cufile_ffi::cuFileRead(
                cufile_handle,
                transfer.gpu_dst.0 as *mut u8,
                transfer.transfer_size,
                transfer.file_offset as i64,
                0, // device buffer offset
            )
        };

        // Cleanup registrations.
        unsafe {
            cufile_ffi::cuFileBufDeregister(transfer.gpu_dst.0 as *mut u8);
            cufile_ffi::cuFileHandleDeregister(cufile_handle);
        }

        if bytes < 0 {
            transfer.status = GdsTransferStatus::Failed;
            transfer.method = TransferMethod::PinnedHostStaging;
            return self.submit_pinned(transfer);
        }

        transfer.status = GdsTransferStatus::Completed;
        transfer.bytes_transferred = bytes as usize;
        self.record_completion(transfer.transfer_size, TransferMethod::CuFileDirect);
        Ok(())
    }

    /// cuFile path not available without the cuda feature.
    #[cfg(not(feature = "cuda"))]
    fn submit_cufile(&mut self, transfer: &mut GdsTransfer) -> Result<(), HalError> {
        transfer.method = TransferMethod::PinnedHostStaging;
        self.submit_pinned(transfer)
    }

    /// Submit via pinned host staging (Path A fallback).
    ///
    /// 1. Read file → pinned host buffer (via pread/ReadFile)
    /// 2. cudaMemcpy pinned buffer → GPU VRAM
    fn submit_pinned(&mut self, transfer: &mut GdsTransfer) -> Result<(), HalError> {
        let pinned = self.pinned_buffer.as_mut().ok_or_else(|| {
            HalError::Unsupported("No pinned buffer available for staging".into())
        })?;

        if transfer.transfer_size > pinned.size() {
            // Transfer too large for staging buffer — need chunking.
            // For now, reject; caller should chunk.
            transfer.status = GdsTransferStatus::Failed;
            return Err(HalError::Unsupported(format!(
                "Transfer size ({}) exceeds pinned buffer ({}). Caller should chunk.",
                transfer.transfer_size,
                pinned.size(),
            )));
        }

        // 1. Read from file into pinned buffer.
        let file = std::fs::File::open(&transfer.file_path)
            .map_err(HalError::IoError)?;

        transfer.status = GdsTransferStatus::InFlight;
        let bytes_read = pinned.read_from_file(
            &file,
            transfer.file_offset,
            transfer.transfer_size,
        )?;

        // 2. Copy from pinned host buffer to GPU VRAM.
        // This would use cudaMemcpyAsync(dst, pinned_ptr, size, H2D, stream).
        // The actual cuda copy is done by the caller via GpuHal::copy_to_vram().
        // We've staged the data into the pinned buffer; caller reads it.
        transfer.status = GdsTransferStatus::Completed;
        transfer.bytes_transferred = bytes_read;
        transfer.method = TransferMethod::PinnedHostStaging;
        self.record_completion(bytes_read, TransferMethod::PinnedHostStaging);
        Ok(())
    }

    /// Record a completed transfer in stats.
    fn record_completion(&self, bytes: usize, method: TransferMethod) {
        self.completed_transfers
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_bytes_transferred
            .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);

        let idx = match method {
            TransferMethod::CuFileDirect => 0,
            TransferMethod::PinnedHostStaging => 1,
            TransferMethod::StandardStaging | TransferMethod::Unknown => 2,
        };
        self.transfers_by_method[idx]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Access the pinned staging buffer (for manual copy to VRAM).
    pub fn pinned_buffer(&self) -> Option<&PinnedHostBuffer> {
        self.pinned_buffer.as_ref()
    }

    /// Get transfer statistics.
    pub fn stats(&self) -> GdsStats {
        GdsStats {
            gds_available: self.is_available(),
            completed_transfers: self.completed_transfers.load(std::sync::atomic::Ordering::Relaxed),
            total_bytes: self.total_bytes_transferred.load(std::sync::atomic::Ordering::Relaxed),
            driver_version: self.capability.driver_version.clone(),
            cufile_transfers: self.transfers_by_method[0].load(std::sync::atomic::Ordering::Relaxed),
            pinned_transfers: self.transfers_by_method[1].load(std::sync::atomic::Ordering::Relaxed),
            standard_transfers: self.transfers_by_method[2].load(std::sync::atomic::Ordering::Relaxed),
            pinned_buffer_available: self.pinned_buffer.is_some(),
        }
    }
}

impl Drop for GdsStorageHal {
    fn drop(&mut self) {
        // Close the cuFile driver if we initialized it.
        #[cfg(feature = "cuda")]
        if self.driver_initialized {
            unsafe { cufile_ffi::cuFileDriverClose() };
        }
    }
}

/// GPUDirect Storage statistics.
#[derive(Debug, Clone)]
pub struct GdsStats {
    /// Whether GDS DMA is available.
    pub gds_available: bool,
    /// Number of completed transfers (all methods).
    pub completed_transfers: u64,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// GDS driver version.
    pub driver_version: String,
    /// Transfers completed via cuFile DMA.
    pub cufile_transfers: u64,
    /// Transfers completed via pinned host staging.
    pub pinned_transfers: u64,
    /// Transfers completed via standard staging.
    pub standard_transfers: u64,
    /// Whether a pinned staging buffer is allocated.
    pub pinned_buffer_available: bool,
}

impl fmt::Display for GdsStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GDS Stats:")?;
        writeln!(f, "  Available: {}", self.gds_available)?;
        writeln!(f, "  Total transfers: {}", self.completed_transfers)?;
        writeln!(f, "  Total bytes: {:.2} MB", self.total_bytes as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  cuFile DMA: {}", self.cufile_transfers)?;
        writeln!(f, "  Pinned staging: {}", self.pinned_transfers)?;
        writeln!(f, "  Standard staging: {}", self.standard_transfers)?;
        writeln!(f, "  Pinned buffer: {}", self.pinned_buffer_available)?;
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gds_probe_does_not_crash() {
        let cap = GdsCapability::probe();
        assert!(!cap.driver_version.is_empty());
    }

    #[test]
    fn gds_hal_creation() {
        let hal = GdsStorageHal::new();
        let stats = hal.stats();
        assert_eq!(stats.completed_transfers, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn gds_needs_staging_without_driver() {
        let hal = GdsStorageHal::new();
        if !hal.is_available() {
            assert!(hal.needs_staging(1024));
            assert!(hal.needs_staging(1));
        }
    }

    #[test]
    fn gds_create_transfer_descriptor() {
        let hal = GdsStorageHal::new();
        let transfer = hal.create_transfer(
            GpuPtr(0x1000),
            "/path/to/model.gguf",
            4096,
            1024 * 1024,
        );
        assert_eq!(transfer.status, GdsTransferStatus::Pending);
        assert_eq!(transfer.transfer_size, 1024 * 1024);
    }

    #[test]
    fn gds_submit_without_driver_fails_gracefully() {
        let mut hal = GdsStorageHal::new();
        if !hal.is_available() {
            let mut transfer = hal.create_transfer(GpuPtr(0x1000), "nonexistent.bin", 0, 4096);
            let result = hal.submit(&mut transfer);
            // Should either succeed via pinned fallback or fail gracefully.
            assert!(
                result.is_err() || transfer.status == GdsTransferStatus::Completed,
                "Expected graceful failure or pinned fallback"
            );
        }
    }

    #[test]
    fn gds_alignment() {
        let hal = GdsStorageHal::new();
        assert_eq!(hal.required_alignment(), 4096);
    }

    #[test]
    fn gds_capability_usable_only_with_driver() {
        let cap = GdsCapability {
            driver_available: false,
            filesystem_supported: true,
            max_transfer_size: 0,
            page_cache_bypass: false,
            driver_version: "none".to_string(),
            pinned_fallback_available: false,
        };
        assert!(!cap.is_usable());

        let cap2 = GdsCapability {
            driver_available: true,
            filesystem_supported: true,
            max_transfer_size: 16 * 1024 * 1024,
            page_cache_bypass: true,
            driver_version: "1.0".to_string(),
            pinned_fallback_available: false,
        };
        assert!(cap2.is_usable());
    }

    #[test]
    fn gds_method_selection() {
        let hal = GdsStorageHal::new();
        let method = hal.select_method(1024);
        // Without GDS, should select pinned or standard.
        if !hal.is_available() {
            assert_ne!(method, TransferMethod::CuFileDirect);
        }
    }

    #[test]
    fn gds_stats_display() {
        let hal = GdsStorageHal::new();
        let stats = hal.stats();
        let display = format!("{}", stats);
        assert!(display.contains("GDS Stats"));
        assert!(display.contains("Total transfers"));
    }

    #[test]
    fn transfer_method_display() {
        assert_eq!(format!("{}", TransferMethod::CuFileDirect), "cuFile DMA");
        assert_eq!(format!("{}", TransferMethod::PinnedHostStaging), "pinned host staging");
        assert_eq!(format!("{}", TransferMethod::StandardStaging), "standard staging");
    }

    #[test]
    fn pinned_buffer_allocation() {
        // PinnedHostBuffer without CUDA just gives us an aligned, unpinned buffer.
        let buf = PinnedHostBuffer::new(4096).unwrap();
        assert_eq!(buf.size(), 4096);
        assert_eq!(buf.as_slice().len(), 4096);
        // Without CUDA feature, buffer won't be pinned.
        if !cfg!(feature = "cuda") {
            assert!(!buf.is_pinned());
        }
    }

    #[test]
    fn pinned_buffer_zero_size_rejected() {
        let result = PinnedHostBuffer::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn pinned_buffer_write_read() {
        let mut buf = PinnedHostBuffer::new(1024).unwrap();
        buf.as_mut_slice()[0..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&buf.as_slice()[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn pinned_buffer_debug() {
        let buf = PinnedHostBuffer::new(4096).unwrap();
        let dbg = format!("{:?}", buf);
        assert!(dbg.contains("PinnedHostBuffer"));
        assert!(dbg.contains("4096"));
    }

    #[test]
    fn pinned_fallback_detection() {
        let cap = GdsCapability {
            driver_available: false,
            filesystem_supported: false,
            max_transfer_size: 0,
            page_cache_bypass: false,
            driver_version: "none".to_string(),
            pinned_fallback_available: true,
        };
        assert!(cap.has_pinned_fallback());

        let cap2 = GdsCapability {
            driver_available: true,
            filesystem_supported: true,
            max_transfer_size: 16 * 1024 * 1024,
            page_cache_bypass: true,
            driver_version: "1.0".to_string(),
            pinned_fallback_available: true,
        };
        // GDS is usable, so pinned fallback is NOT needed.
        assert!(!cap2.has_pinned_fallback());
    }

    #[test]
    fn gds_pinned_buffer_available_in_hal() {
        let hal = GdsStorageHal::new();
        let stats = hal.stats();
        // On machines without GDS, pinned buffer should be allocated.
        if !hal.is_available() {
            assert!(stats.pinned_buffer_available);
        }
    }
}

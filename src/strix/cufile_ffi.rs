//! cuFile API FFI bindings for GPUDirect Storage.
//!
//! Raw C bindings to `libcufile.so` — the safe wrapper is in `gpu_direct.rs`.
//! Feature-gated behind `#[cfg(feature = "cuda")]` and Linux-only
//! (`#[cfg(target_os = "linux")]`).
//!
//! From STRIX Protocol §9.4, hardware_implementation_guide.md Task 1.
//!
//! ## Link requirements
//!
//! ```text
//! rustc-link-lib=cufile
//! rustc-link-search=/usr/local/cuda/lib64
//! ```

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::fmt;

// ── Handle Types ─────────────────────────────────────────────────────────

/// Opaque cuFile handle (returned by `cuFileHandleRegister`).
pub type CUfileHandle_t = *mut c_void;

/// cuFile file descriptor union.
#[repr(C)]
pub union CUfileDescrUnion {
    /// Linux file descriptor (O_DIRECT recommended).
    pub fd: i32,
    /// Windows HANDLE (not supported by GDS — reserved).
    pub handle: *mut c_void,
}

/// cuFile handle type discriminant.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileHandleType {
    /// Linux file descriptor.
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
    /// Windows handle (unsupported by GDS).
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
}

/// cuFile file descriptor — wraps a raw fd/HANDLE.
#[repr(C)]
pub struct CUfileDescr {
    /// Handle type discriminant.
    pub handle_type: CUfileHandleType,
    /// The underlying OS handle.
    pub handle: CUfileDescrUnion,
}

// ── Error Codes ──────────────────────────────────────────────────────────

/// cuFile status / error codes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileStatus {
    CUFILE_SUCCESS = 0,
    CUFILE_DRIVER_NOT_INITIALIZED = 1,
    CUFILE_INVALID_VALUE = 2,
    CUFILE_INTERNAL_ERROR = 3,
    CUFILE_DRIVER_VERSION_MISMATCH = 4,
    CUFILE_DRIVER_UNSUPPORTED_LIMIT = 5,
    CUFILE_DRIVER_VERSION_READ_ERROR = 6,
    CUFILE_IO_NOT_SUPPORTED = 7,
    CUFILE_BATCH_IO_NO_MEM = 8,
    CUFILE_BATCH_IO_TIMEOUT = 9,
    CUFILE_BATCH_SUBMIT_FAILED = 10,
    CUFILE_CUDA_DRIVER_ERROR = 11,
    CUFILE_CUDA_POINTER_RANGE_ERROR = 12,
    CUFILE_CUDA_MEMORY_TYPE_ERROR = 13,
    CUFILE_CUDA_POINTER_NOT_REGISTERED = 14,
    CUFILE_GCOMPAT_NOTALLOWED = 15,
    CUFILE_IO_DISABLED = 16,
    CUFILE_PLATFORM_NOT_SUPPORTED = 17,
}

impl CUfileStatus {
    /// Whether this is a success status.
    pub fn is_ok(self) -> bool {
        self == Self::CUFILE_SUCCESS
    }

    /// Human readable description.
    pub fn message(self) -> &'static str {
        match self {
            Self::CUFILE_SUCCESS => "success",
            Self::CUFILE_DRIVER_NOT_INITIALIZED => "driver not initialized",
            Self::CUFILE_INVALID_VALUE => "invalid value",
            Self::CUFILE_INTERNAL_ERROR => "internal error",
            Self::CUFILE_DRIVER_VERSION_MISMATCH => "driver version mismatch",
            Self::CUFILE_DRIVER_UNSUPPORTED_LIMIT => "unsupported limit",
            Self::CUFILE_DRIVER_VERSION_READ_ERROR => "version read error",
            Self::CUFILE_IO_NOT_SUPPORTED => "I/O not supported",
            Self::CUFILE_BATCH_IO_NO_MEM => "batch I/O: no memory",
            Self::CUFILE_BATCH_IO_TIMEOUT => "batch I/O: timeout",
            Self::CUFILE_BATCH_SUBMIT_FAILED => "batch submit failed",
            Self::CUFILE_CUDA_DRIVER_ERROR => "CUDA driver error",
            Self::CUFILE_CUDA_POINTER_RANGE_ERROR => "CUDA pointer range error",
            Self::CUFILE_CUDA_MEMORY_TYPE_ERROR => "CUDA memory type error",
            Self::CUFILE_CUDA_POINTER_NOT_REGISTERED => "CUDA pointer not registered",
            Self::CUFILE_GCOMPAT_NOTALLOWED => "GCompat not allowed",
            Self::CUFILE_IO_DISABLED => "I/O disabled",
            Self::CUFILE_PLATFORM_NOT_SUPPORTED => "platform not supported",
        }
    }
}

impl fmt::Display for CUfileStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuFile error {}: {}", *self as i32, self.message())
    }
}

impl std::error::Error for CUfileStatus {}

// ── Driver Properties (opaque) ───────────────────────────────────────────

/// cuFile driver properties. Opaque — only used by `cuFileDriverGetProperties`.
#[repr(C)]
pub struct CUfileDriverProps {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
    /// Rest is opaque.
    _opaque: [u8; 4084],
}

impl CUfileDriverProps {
    /// Create a zeroed props struct for `cuFileDriverGetProperties`.
    pub fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }

    /// Format version string.
    pub fn version_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ── FFI Declarations ─────────────────────────────────────────────────────
//
// These symbols are resolved at link time against `libcufile.so`.
// The `cuda` feature flag gates compilation; this file does NOT compile
// without `--features cuda` on Linux.

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[link(name = "cufile")]
extern "C" {
    /// Initialize the cuFile library. Must be called once before any I/O.
    pub fn cuFileDriverOpen() -> CUfileStatus;

    /// Shut down the cuFile library. Call during cleanup.
    pub fn cuFileDriverClose() -> CUfileStatus;

    /// Query driver properties (version, limits).
    pub fn cuFileDriverGetProperties(props: *mut CUfileDriverProps) -> CUfileStatus;

    /// Register a file descriptor for cuFile I/O.
    ///
    /// `fh`: output — receives the opaque cuFile handle.
    /// `descr`: input — wraps the OS file descriptor.
    pub fn cuFileHandleRegister(
        fh: *mut CUfileHandle_t,
        descr: *const CUfileDescr,
    ) -> CUfileStatus;

    /// Deregister a previously registered file handle.
    pub fn cuFileHandleDeregister(fh: CUfileHandle_t);

    /// Register a GPU buffer for cuFile DMA.
    ///
    /// `devPtr` must be a CUDA device pointer returned by `cudaMalloc`.
    /// `size` is the buffer size in bytes.
    /// `flags` must be 0.
    pub fn cuFileBufRegister(
        devPtr: *mut c_void,
        size: usize,
        flags: u32,
    ) -> CUfileStatus;

    /// Deregister a previously registered GPU buffer.
    pub fn cuFileBufDeregister(devPtr: *mut c_void) -> CUfileStatus;

    /// Synchronous read: file → GPU VRAM (DMA).
    ///
    /// Returns bytes read (≥ 0) on success, or a negative error code.
    pub fn cuFileRead(
        fh: CUfileHandle_t,
        devPtr: *mut c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64,
    ) -> isize;

    /// Synchronous write: GPU VRAM → file (DMA).
    ///
    /// Returns bytes written (≥ 0) on success, or a negative error code.
    pub fn cuFileWrite(
        fh: CUfileHandle_t,
        devPtr: *const c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64,
    ) -> isize;
}

// ── Safe Wrapper Helpers ─────────────────────────────────────────────────

/// Check a cuFile status code, converting to `Result`.
pub fn check(status: CUfileStatus) -> Result<(), CUfileStatus> {
    if status.is_ok() { Ok(()) } else { Err(status) }
}

/// Check a cuFile read/write return value.
///
/// Positive = bytes transferred, negative = error code.
pub fn check_rw(ret: isize) -> Result<usize, CUfileStatus> {
    if ret >= 0 {
        Ok(ret as usize)
    } else {
        // cuFile negative returns map to status codes.
        Err(CUfileStatus::CUFILE_INTERNAL_ERROR)
    }
}

/// Create a cuFile descriptor for a Linux file descriptor.
#[cfg(target_os = "linux")]
pub fn make_descr(fd: i32) -> CUfileDescr {
    CUfileDescr {
        handle_type: CUfileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
        handle: CUfileDescrUnion { fd },
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_display() {
        let s = CUfileStatus::CUFILE_SUCCESS;
        assert!(s.is_ok());
        let display = format!("{}", s);
        assert!(display.contains("success"), "{}", display);
    }

    #[test]
    fn status_error_codes() {
        assert!(!CUfileStatus::CUFILE_DRIVER_NOT_INITIALIZED.is_ok());
        assert!(!CUfileStatus::CUFILE_INTERNAL_ERROR.is_ok());
        assert!(!CUfileStatus::CUFILE_PLATFORM_NOT_SUPPORTED.is_ok());
    }

    #[test]
    fn check_success() {
        assert!(check(CUfileStatus::CUFILE_SUCCESS).is_ok());
    }

    #[test]
    fn check_error() {
        let r = check(CUfileStatus::CUFILE_DRIVER_NOT_INITIALIZED);
        assert!(r.is_err());
    }

    #[test]
    fn check_rw_success() {
        assert_eq!(check_rw(4096).unwrap(), 4096);
        assert_eq!(check_rw(0).unwrap(), 0);
    }

    #[test]
    fn check_rw_error() {
        assert!(check_rw(-1).is_err());
    }

    #[test]
    fn driver_props_zeroed() {
        let props = CUfileDriverProps::zeroed();
        assert_eq!(props.major, 0);
        assert_eq!(props.version_string(), "0.0.0");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn make_descr_linux() {
        let descr = make_descr(42);
        assert_eq!(descr.handle_type, CUfileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD);
        assert_eq!(unsafe { descr.handle.fd }, 42);
    }

    #[test]
    fn all_error_messages_non_empty() {
        let codes = [
            CUfileStatus::CUFILE_SUCCESS,
            CUfileStatus::CUFILE_DRIVER_NOT_INITIALIZED,
            CUfileStatus::CUFILE_INTERNAL_ERROR,
            CUfileStatus::CUFILE_PLATFORM_NOT_SUPPORTED,
        ];
        for code in codes {
            assert!(!code.message().is_empty());
        }
    }
}

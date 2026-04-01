//! CUDA GPU HAL backend (STRIX Protocol §12.1).
//!
//! `CudaHal` implements `GpuHal` via CUDA Runtime API FFI bindings.
//! All CUDA functions are declared as `extern "C"` — no external crate
//! dependency required. The linker resolves symbols against `cudart` at
//! link time when the `cuda` feature is enabled.
//!
//! Gated behind `#[cfg(feature = "cuda")]`.

#![cfg(feature = "cuda")]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::ffi::CStr;
use std::ptr;

// ── CUDA Runtime API FFI ─────────────────────────────────────────────────

/// CUDA error codes (subset we care about).
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;

/// CUDA memory copy direction.
#[repr(i32)]
#[allow(dead_code)]
enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// CUDA device properties (subset).
#[repr(C)]
struct CudaDeviceProp {
    name: [u8; 256],
    total_global_mem: usize,
    _padding: [u8; 512], // remaining fields we don't use
}

#[link(name = "cudart")]
extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: i32) -> i32;
    fn cudaMalloc(dev_ptr: *mut *mut u8, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut u8) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut u8,
        src: *const u8,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert a CUDA error code into a `HalError`.
fn cuda_check(code: i32) -> Result<(), HalError> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    if code == CUDA_ERROR_OUT_OF_MEMORY {
        return Err(HalError::OutOfMemory {
            requested: 0,
            available: 0,
        });
    }
    let msg = unsafe {
        let ptr = cudaGetErrorString(code);
        if ptr.is_null() {
            "unknown CUDA error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(HalError::DriverError {
        code,
        message: msg,
    })
}

// ── CudaHal ──────────────────────────────────────────────────────────────

/// CUDA GPU backend implementing `GpuHal`.
///
/// Manages a single CUDA device with a configurable number of async
/// copy streams.
pub struct CudaHal {
    /// CUDA device ordinal (0-based).
    device_id: i32,
    /// Pre-created CUDA streams for async copies.
    /// Index 0 = default (null) stream; 1..N = dedicated streams.
    streams: Vec<*mut std::ffi::c_void>,
}

// SAFETY: CUDA streams are thread-safe when used with proper synchronisation.
unsafe impl Send for CudaHal {}
unsafe impl Sync for CudaHal {}

impl CudaHal {
    /// Create a new CUDA HAL on the given device.
    ///
    /// `num_streams` additional copy streams will be created (beyond the default).
    pub fn new(device_id: i32, num_streams: u32) -> Result<Self, HalError> {
        unsafe { cuda_check(cudaSetDevice(device_id))? };

        // Stream 0 = null (default CUDA stream).
        let mut streams: Vec<*mut std::ffi::c_void> = vec![ptr::null_mut()];

        // Create additional streams.
        for _ in 0..num_streams {
            let mut stream = ptr::null_mut();
            unsafe { cuda_check(cudaStreamCreate(&mut stream))? };
            streams.push(stream);
        }

        Ok(Self {
            device_id,
            streams,
        })
    }

    /// Get the CUDA device count.
    pub fn device_count() -> Result<i32, HalError> {
        let mut count = 0;
        unsafe { cuda_check(cudaGetDeviceCount(&mut count))? };
        Ok(count)
    }

    /// Resolve a stream index to the corresponding CUDA stream handle.
    fn get_stream(&self, stream: u32) -> *mut std::ffi::c_void {
        let idx = stream as usize;
        if idx < self.streams.len() {
            self.streams[idx]
        } else {
            // Fall back to default stream.
            ptr::null_mut()
        }
    }
}

impl Drop for CudaHal {
    fn drop(&mut self) {
        // Destroy non-default streams.
        for &stream in &self.streams[1..] {
            if !stream.is_null() {
                unsafe { cudaStreamDestroy(stream) };
            }
        }
    }
}

impl GpuHal for CudaHal {
    fn info(&self) -> Result<GpuInfo, HalError> {
        let mut prop = CudaDeviceProp {
            name: [0u8; 256],
            total_global_mem: 0,
            _padding: [0u8; 512],
        };
        unsafe { cuda_check(cudaGetDeviceProperties(&mut prop, self.device_id))? };

        let name_len = prop.name.iter().position(|&b| b == 0).unwrap_or(256);
        let name = String::from_utf8_lossy(&prop.name[..name_len]).into_owned();

        let mut free_mem = 0usize;
        let mut total_mem = 0usize;
        unsafe { cuda_check(cudaMemGetInfo(&mut free_mem, &mut total_mem))? };

        Ok(GpuInfo {
            name,
            vram_total: total_mem,
            vram_free: free_mem,
            compute_capability: 0, // retrieved from prop in production
            bus_bandwidth: 32_000_000_000, // PCIe 4.0 x16 ~32 GB/s
        })
    }

    fn allocate_vram(&self, size: usize, _alignment: usize) -> Result<GpuPtr, HalError> {
        let mut dev_ptr: *mut u8 = ptr::null_mut();
        let code = unsafe { cudaMalloc(&mut dev_ptr, size) };
        if code == CUDA_ERROR_OUT_OF_MEMORY {
            let mut free = 0usize;
            let mut _total = 0usize;
            unsafe { cudaMemGetInfo(&mut free, &mut _total) };
            return Err(HalError::OutOfMemory {
                requested: size,
                available: free,
            });
        }
        cuda_check(code)?;
        Ok(GpuPtr(dev_ptr as u64))
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        unsafe { cuda_check(cudaFree(ptr.0 as *mut u8)) }
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError> {
        let cuda_stream = self.get_stream(stream);
        unsafe {
            cuda_check(cudaMemcpyAsync(
                dst.0 as *mut u8,
                src,
                size,
                CudaMemcpyKind::HostToDevice as i32,
                cuda_stream,
            ))
        }
    }

    fn copy_from_vram(
        &self,
        dst: *mut u8,
        src: GpuPtr,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError> {
        let cuda_stream = self.get_stream(stream);
        unsafe {
            cuda_check(cudaMemcpyAsync(
                dst,
                src.0 as *const u8,
                size,
                CudaMemcpyKind::DeviceToHost as i32,
                cuda_stream,
            ))
        }
    }

    fn sync_stream(&self, stream: u32) -> Result<(), HalError> {
        let cuda_stream = self.get_stream(stream);
        unsafe { cuda_check(cudaStreamSynchronize(cuda_stream)) }
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { cuda_check(cudaMemGetInfo(&mut free, &mut total))? };
        Ok(total - free)
    }
}

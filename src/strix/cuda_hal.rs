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

/// CUDA device properties — struct layout matches `cudaDeviceProp`.
///
/// Only the fields we access are named; the rest is covered by `_rest`
/// padding. The full struct is ~720+ bytes depending on CUDA version,
/// but we only need the first handful of fields.
///
/// Layout (CUDA 12.x, 64-bit):
///   offset  0: name            [u8; 256]
///   offset 256: totalGlobalMem  usize (8 bytes on 64-bit)
///   offset 264: sharedMemPerBlock  usize
///   offset 272: regsPerBlock    i32
///   offset 276: warpSize        i32
///   offset 280: memPitch        usize
///   offset 288: maxThreadsPerBlock  i32
///   offset 292: maxThreadsDim   [i32; 3]
///   offset 304: maxGridSize     [i32; 3]
///   offset 316: clockRate       i32
///   offset 320: totalConstMem   usize
///   offset 328: major           i32   ← compute capability major
///   offset 332: minor           i32   ← compute capability minor
#[repr(C)]
struct CudaDeviceProp {
    name: [u8; 256],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: i32,
    warp_size: i32,
    mem_pitch: usize,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    clock_rate: i32,
    total_const_mem: usize,
    /// Compute capability major version (e.g. 8 for Ampere, 9 for Hopper).
    major: i32,
    /// Compute capability minor version (e.g. 9 for SM_89 Ada Lovelace).
    minor: i32,
    // Remaining fields we don't use — pad to a safe over-size.
    _rest: [u8; 4096],
}

/// CUDA device attribute IDs for `cudaDeviceGetAttribute`.
#[allow(dead_code)]
const CUDA_DEV_ATTR_PCIE_BUS_ID: i32 = 33;
#[allow(dead_code)]
const CUDA_DEV_ATTR_MEMORY_BUS_WIDTH: i32 = 37;
#[allow(dead_code)]
const CUDA_DEV_ATTR_MEMORY_CLOCK_RATE: i32 = 36;

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
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaMemsetAsync(
        dev_ptr: *mut u8,
        value: i32,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
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

/// Estimate PCIe bandwidth from memory bus width and clock rate.
///
/// Formula: `(bus_width_bits / 8) * 2 * mem_clock_khz * 1000`.
/// Falls back to 16 GB/s (PCIe 3.0 x16) if attributes are unavailable.
fn estimate_bus_bandwidth(device_id: i32) -> u64 {
    let mut bus_width = 0i32;
    let mut mem_clock = 0i32;
    unsafe {
        let r1 = cudaDeviceGetAttribute(&mut bus_width, CUDA_DEV_ATTR_MEMORY_BUS_WIDTH, device_id);
        let r2 = cudaDeviceGetAttribute(&mut mem_clock, CUDA_DEV_ATTR_MEMORY_CLOCK_RATE, device_id);
        if r1 != CUDA_SUCCESS || r2 != CUDA_SUCCESS || bus_width == 0 || mem_clock == 0 {
            return 16_000_000_000; // conservative PCIe 3.0 x16 fallback
        }
    }
    // memory bandwidth = bus_width_bytes * 2 (DDR) * clock_hz
    let bytes_per_cycle = (bus_width as u64) / 8 * 2;
    bytes_per_cycle * (mem_clock as u64) * 1000
}

// ── CudaHal ──────────────────────────────────────────────────────────────

/// CUDA GPU backend implementing `GpuHal`.
///
/// Manages a single CUDA device with a configurable number of async
/// copy streams. Reads real compute capability and estimates bus
/// bandwidth from device attributes.
pub struct CudaHal {
    /// CUDA device ordinal (0-based).
    device_id: i32,
    /// Pre-created CUDA streams for async copies.
    /// Index 0 = default (null) stream; 1..N = dedicated streams.
    streams: Vec<*mut std::ffi::c_void>,
    /// Cached compute capability (e.g. 89 = SM_89 Ada Lovelace).
    compute_cap: u32,
    /// Cached bus bandwidth estimate in bytes/sec.
    bus_bw: u64,
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

        // Query compute capability.
        let mut prop: CudaDeviceProp = unsafe { std::mem::zeroed() };
        unsafe { cuda_check(cudaGetDeviceProperties(&mut prop, device_id))? };
        let compute_cap = (prop.major as u32) * 10 + (prop.minor as u32);
        let bus_bw = estimate_bus_bandwidth(device_id);

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
            compute_cap,
            bus_bw,
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

    /// CUDA always has discrete device-local VRAM on NVIDIA GPUs.
    pub fn has_discrete_vram(&self) -> bool {
        true
    }

    /// Perform a staged copy to device-local VRAM.
    ///
    /// Matches the Vulkan `staged_copy_to_device_local()` API contract.
    /// CUDA is simpler — `cudaMalloc` always allocates device-local,
    /// and `cudaMemcpyAsync(HostToDevice)` handles the staging internally.
    ///
    /// Steps:
    /// 1. Allocate device VRAM via `cudaMalloc`
    /// 2. Copy host data → device via `cudaMemcpyAsync(HostToDevice)`
    /// 3. Synchronize the stream
    ///
    /// Returns the device VRAM pointer as `GpuPtr`.
    pub fn staged_copy_to_device(&self, data: &[u8], stream: u32) -> Result<GpuPtr, HalError> {
        let size = data.len();

        // 1. Allocate device memory
        let ptr = self.allocate_vram(size, 256)?;

        // 2. Async copy host → device
        let cuda_stream = self.get_stream(stream);
        let result = unsafe {
            cudaMemcpyAsync(
                ptr.0 as *mut u8,
                data.as_ptr(),
                size,
                CudaMemcpyKind::HostToDevice as i32,
                cuda_stream,
            )
        };
        if result != CUDA_SUCCESS {
            // Clean up allocation on failure
            let _ = self.free_vram(ptr);
            cuda_check(result)?;
        }

        // 3. Synchronize to ensure transfer is complete
        unsafe { cuda_check(cudaStreamSynchronize(cuda_stream))? };

        Ok(ptr)
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
        let mut prop: CudaDeviceProp = unsafe { std::mem::zeroed() };
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
            compute_capability: self.compute_cap,
            bus_bandwidth: self.bus_bw,
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

    /// Hardware-optimized VRAM zeroing using `cudaMemsetAsync`.
    ///
    /// Runs entirely on the GPU — no host→device copy needed.
    fn secure_zero_vram(&self, ptr: GpuPtr, size: usize) -> Result<(), HalError> {
        let cuda_stream = self.get_stream(0);
        unsafe {
            cuda_check(cudaMemsetAsync(
                ptr.0 as *mut u8,
                0, // zero fill
                size,
                cuda_stream,
            ))?;
            cuda_check(cudaStreamSynchronize(cuda_stream))?;
        }
        Ok(())
    }
}

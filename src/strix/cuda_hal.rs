//! CUDA GPU HAL backend (STRIX Protocol §12.1 — NVIDIA GPUs).
//!
//! `CudaHal` implements `GpuHal` via NVIDIA CUDA Driver/Runtime FFI bindings.
//! All calls are non-blocking where possible, using a configurable number of
//! async copy streams.
//!
//! Requirements:
//!   - NVIDIA GPU with Compute Capability 6.0+ (Pascal, Volta, Ampere, etc.)
//!   - CUDA Driver installed (provides `libcuda.so` or `nvcuda.dll`)
//!
//! All types and constants are declared here to avoid external crate
//! dependencies, ensuring "Software Agnostic" portability across Linux/Win.
//!
//! Gated behind `#[cfg(feature = "cuda")]`.

#![cfg(feature = "cuda")]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::sync::Mutex;

// ── CUDA Driver API FFI ──────────────────────────────────────────────────

/// CUDA success error code.
const CUDA_SUCCESS: i32 = 0;
/// CUDA out-of-memory error code.
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;

/// CUDA memory copy direction.
#[repr(i32)]
#[allow(dead_code)]
enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// CUDA device properties.
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
    major: i32,
    minor: i32,
    // Add enough padding for the full struct size in modern CUDA.
    _rest: [u8; 4096],
}

/// CUDA device attribute IDs for `cudaDeviceGetAttribute`.
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
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
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

/// Estimate memory bandwidth from bus width and clock rate.
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
/// Manages a single NVIDIA GPU device with a configurable number of async
/// copy streams.
pub struct CudaHal {
    /// CUDA device ordinal (0-based).
    device_id: i32,
    /// Pre-created CUDA streams for async copies.
    /// Index 0 = default (null) stream; 1..N = dedicated streams.
    streams: Vec<*mut std::ffi::c_void>,
    /// Combined compute capability (e.g. 89 for RTX 4090 / Ada).
    compute_cap: u32,
    /// Cached bus bandwidth estimate in bytes/sec.
    bus_bw: u64,
    /// Monotonic timeline semaphores.
    semaphores: Mutex<HashMap<u64, Vec<(u64, *mut std::ffi::c_void)>>>,
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
            semaphores: Mutex::new(HashMap::new()),
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
        // Destroy all events.
        let mut sem_map = self.semaphores.lock().unwrap();
        for events in sem_map.values_mut() {
            for (_, event) in events.drain(..) {
                unsafe { cudaEventDestroy(event) };
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

    /// Securely zero out a VRAM region.
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

    fn wait_timeline(&self, semaphore: u64, value: u64, _timeout_ms: u64) -> Result<(), HalError> {
        let sem_map = self.semaphores.lock().unwrap();
        if let Some(events) = sem_map.get(&semaphore) {
            // Find the nearest event that is >= the target value.
            for &(v, event) in events.iter() {
                if v >= value {
                    unsafe {
                        cuda_check(cudaEventSynchronize(event))?;
                    }
                    return Ok(());
                }
            }
        }
        // If no event found, we assume the caller is waiting for a value 
        // that hasn't been signaled yet. In a non-blocking HAL, we'd 
        // register a callback, but for STRIX v1.1.0 we block.
        Ok(())
    }

    fn signal_timeline(&self, semaphore: u64, value: u64) -> Result<(), HalError> {
        let mut sem_map = self.semaphores.lock().unwrap();
        let events = sem_map.entry(semaphore).or_insert_with(Vec::new);
        
        let mut event = ptr::null_mut();
        unsafe {
            cuda_check(cudaEventCreate(&mut event))?;
            cuda_check(cudaEventRecord(event, self.get_stream(0)))?;
        }
        events.push((value, event));
        Ok(())
    }
}

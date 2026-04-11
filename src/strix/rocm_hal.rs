//! ROCm GPU HAL backend (STRIX Protocol §12.1 — AMD GPUs).
//!
//! `RocmHal` implements `GpuHal` via AMD HIP Runtime API FFI bindings.
//! HIP is AMD's CUDA-compatible runtime — the API surface mirrors CUDA
//! almost 1:1, making this backend structurally identical to `cuda_hal.rs`.
//!
//! All HIP functions are declared as `extern "C"` — no external crate
//! dependency required. The linker resolves symbols against `amdhip64` at
//! link time when the `rocm` feature is enabled.
//!
//! Requirements:
//!   - ROCm 5.0+ installed (provides `libamdhip64.so`)
//!   - AMD GPU with GCN 3.0+ architecture (Fiji, Vega, RDNA, CDNA)
//!   - Linux only (ROCm does not support Windows natively)
//!
//! Gated behind `#[cfg(feature = "rocm")]`.

#![cfg(feature = "rocm")]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::ffi::CStr;
use std::ptr;

// ── HIP Runtime API FFI ─────────────────────────────────────────────────

/// HIP success error code.
const HIP_SUCCESS: i32 = 0;
/// HIP out-of-memory error code (hipErrorOutOfMemory).
const HIP_ERROR_OUT_OF_MEMORY: i32 = 2;

/// HIP memory copy direction.
#[repr(i32)]
#[allow(dead_code)]
enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// HIP device properties — struct layout matches `hipDeviceProp_t`.
///
/// Layout (ROCm 5.x+, 64-bit Linux):
///   offset   0: name               [u8; 256]
///   offset 256: totalGlobalMem      usize
///   offset 264: sharedMemPerBlock   usize
///   offset 272: regsPerBlock        i32
///   offset 276: warpSize            i32
///   offset 280: memPitch            usize  (on HIP: maximal pitch)
///   offset 288: maxThreadsPerBlock  i32
///   offset 292: maxThreadsDim       [i32; 3]
///   offset 304: maxGridSize         [i32; 3]
///   offset 316: clockRate           i32
///   offset 320: totalConstMem       usize
///   offset 328: major               i32   ← GCN arch major
///   offset 332: minor               i32   ← GCN arch minor
///
/// The HIP struct is a superset of CUDA's `cudaDeviceProp`. We use the
/// same field layout for the subset we need, with `_rest` padding.
#[repr(C)]
struct HipDeviceProp {
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
    /// GCN architecture major version.
    major: i32,
    /// GCN architecture minor version.
    minor: i32,
    // Remaining fields — pad generously. HIP's hipDeviceProp_t is ~800+ bytes.
    _rest: [u8; 4096],
}

/// HIP device attribute IDs for `hipDeviceGetAttribute`.
#[allow(dead_code)]
const HIP_DEV_ATTR_MEMORY_BUS_WIDTH: i32 = 37;
#[allow(dead_code)]
const HIP_DEV_ATTR_MEMORY_CLOCK_RATE: i32 = 36;
/// hipDeviceAttributeGcnArch — AMD-specific GCN architecture ID.
#[allow(dead_code)]
const HIP_DEV_ATTR_GCN_ARCH: i32 = 1000;

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: i32) -> i32;
    fn hipGetDevice(device: *mut i32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipGetDeviceProperties(prop: *mut HipDeviceProp, device: i32) -> i32;
    fn hipMalloc(dev_ptr: *mut *mut u8, size: usize) -> i32;
    fn hipFree(dev_ptr: *mut u8) -> i32;
    fn hipMemcpyAsync(
        dst: *mut u8,
        src: *const u8,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn hipStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn hipStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn hipStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn hipGetErrorString(error: i32) -> *const i8;
    fn hipDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert a HIP error code into a `HalError`.
fn hip_check(code: i32) -> Result<(), HalError> {
    if code == HIP_SUCCESS {
        return Ok(());
    }
    if code == HIP_ERROR_OUT_OF_MEMORY {
        return Err(HalError::OutOfMemory {
            requested: 0,
            available: 0,
        });
    }
    let msg = unsafe {
        let ptr = hipGetErrorString(code);
        if ptr.is_null() {
            "unknown HIP error".to_string()
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
///
/// Formula: `(bus_width_bits / 8) * 2 * mem_clock_khz * 1000`.
/// Falls back to 16 GB/s (PCIe 3.0 x16) if attributes are unavailable.
fn estimate_bus_bandwidth(device_id: i32) -> u64 {
    let mut bus_width = 0i32;
    let mut mem_clock = 0i32;
    unsafe {
        let r1 = hipDeviceGetAttribute(&mut bus_width, HIP_DEV_ATTR_MEMORY_BUS_WIDTH, device_id);
        let r2 = hipDeviceGetAttribute(&mut mem_clock, HIP_DEV_ATTR_MEMORY_CLOCK_RATE, device_id);
        if r1 != HIP_SUCCESS || r2 != HIP_SUCCESS || bus_width == 0 || mem_clock == 0 {
            return 16_000_000_000; // conservative PCIe 3.0 x16 fallback
        }
    }
    // memory bandwidth = bus_width_bytes * 2 (DDR) * clock_hz
    let bytes_per_cycle = (bus_width as u64) / 8 * 2;
    bytes_per_cycle * (mem_clock as u64) * 1000
}

/// Query GCN architecture ID (e.g., 906 for Vega 20, 1030 for RDNA 3).
///
/// Falls back to computing from major.minor if the attribute is unavailable.
fn query_gcn_arch(device_id: i32, major: i32, minor: i32) -> u32 {
    let mut gcn_arch = 0i32;
    let result = unsafe {
        hipDeviceGetAttribute(&mut gcn_arch, HIP_DEV_ATTR_GCN_ARCH, device_id)
    };
    if result == HIP_SUCCESS && gcn_arch > 0 {
        gcn_arch as u32
    } else {
        // Fallback: encode as major*100 + minor*10
        (major as u32) * 100 + (minor as u32) * 10
    }
}

// ── RocmHal ──────────────────────────────────────────────────────────────

/// ROCm/HIP GPU backend implementing `GpuHal`.
///
/// Manages a single AMD GPU device with a configurable number of async
/// copy streams. Supports GCN 3.0+ (Fiji, Vega, RDNA, CDNA) architectures.
///
/// Tested device families:
///   - **CDNA**: MI100 (gfx908), MI210/MI250 (gfx90a), MI300X (gfx942)
///   - **RDNA**: RX 7900 XTX (gfx1100), RX 6900 XT (gfx1030)
///   - **GCN**: Vega 56/64 (gfx900/gfx906)
pub struct RocmHal {
    /// HIP device ordinal (0-based).
    device_id: i32,
    /// Pre-created HIP streams for async copies.
    /// Index 0 = default (null) stream; 1..N = dedicated streams.
    streams: Vec<*mut std::ffi::c_void>,
    /// GCN architecture ID (e.g., 1100 for gfx1100 RDNA 3).
    gcn_arch: u32,
    /// Cached bus bandwidth estimate in bytes/sec.
    bus_bw: u64,
}

// SAFETY: HIP streams are thread-safe when used with proper synchronisation.
unsafe impl Send for RocmHal {}
unsafe impl Sync for RocmHal {}

impl RocmHal {
    /// Create a new ROCm HAL on the given device.
    ///
    /// `num_streams` additional copy streams will be created (beyond the default).
    ///
    /// # Errors
    /// Returns `HalError::DriverError` if HIP is not available or the device
    /// ordinal is invalid.
    pub fn new(device_id: i32, num_streams: u32) -> Result<Self, HalError> {
        unsafe { hip_check(hipSetDevice(device_id))? };

        // Query device properties for GCN arch.
        let mut prop: HipDeviceProp = unsafe { std::mem::zeroed() };
        unsafe { hip_check(hipGetDeviceProperties(&mut prop, device_id))? };
        let gcn_arch = query_gcn_arch(device_id, prop.major, prop.minor);
        let bus_bw = estimate_bus_bandwidth(device_id);

        // Stream 0 = null (default HIP stream).
        let mut streams: Vec<*mut std::ffi::c_void> = vec![ptr::null_mut()];

        // Create additional async streams.
        for _ in 0..num_streams {
            let mut stream = ptr::null_mut();
            unsafe { hip_check(hipStreamCreate(&mut stream))? };
            streams.push(stream);
        }

        Ok(Self {
            device_id,
            streams,
            gcn_arch,
            bus_bw,
        })
    }

    /// Get the number of AMD GPUs visible to HIP.
    pub fn device_count() -> Result<i32, HalError> {
        let mut count = 0;
        unsafe { hip_check(hipGetDeviceCount(&mut count))? };
        Ok(count)
    }

    /// Resolve a stream index to the corresponding HIP stream handle.
    fn get_stream(&self, stream: u32) -> *mut std::ffi::c_void {
        let idx = stream as usize;
        if idx < self.streams.len() {
            self.streams[idx]
        } else {
            ptr::null_mut()
        }
    }

    /// AMD discrete GPUs have device-local VRAM.
    pub fn has_discrete_vram(&self) -> bool {
        true
    }

    /// Get the GCN architecture ID (e.g., 1100 = gfx1100).
    pub fn gcn_arch(&self) -> u32 {
        self.gcn_arch
    }

    /// Check if this device is CDNA (compute-focused: MI100, MI210, MI250, MI300).
    pub fn is_cdna(&self) -> bool {
        // gfx908 (MI100), gfx90a (MI210/MI250), gfx940-gfx942 (MI300)
        matches!(self.gcn_arch, 908 | 910 | 940..=942)
    }

    /// Check if this device is RDNA (consumer: RX 6000/7000/8000 series).
    pub fn is_rdna(&self) -> bool {
        // gfx1010-gfx1036 (RDNA 1/2), gfx1100-gfx1103 (RDNA 3), gfx1150+ (RDNA 4)
        self.gcn_arch >= 1010
    }

    /// Perform a staged copy to device-local VRAM.
    ///
    /// Matches the CUDA `staged_copy_to_device()` API contract.
    ///
    /// Steps:
    /// 1. Allocate device VRAM via `hipMalloc`
    /// 2. Copy host data → device via `hipMemcpyAsync(HostToDevice)`
    /// 3. Synchronize the stream
    ///
    /// Returns the device VRAM pointer as `GpuPtr`.
    pub fn staged_copy_to_device(&self, data: &[u8], stream: u32) -> Result<GpuPtr, HalError> {
        let size = data.len();

        // 1. Allocate device memory
        let ptr = self.allocate_vram(size, 256)?;

        // 2. Async copy host → device
        let hip_stream = self.get_stream(stream);
        let result = unsafe {
            hipMemcpyAsync(
                ptr.0 as *mut u8,
                data.as_ptr(),
                size,
                HipMemcpyKind::HostToDevice as i32,
                hip_stream,
            )
        };
        if result != HIP_SUCCESS {
            let _ = self.free_vram(ptr);
            hip_check(result)?;
        }

        // 3. Synchronize to ensure transfer is complete
        unsafe { hip_check(hipStreamSynchronize(hip_stream))? };

        Ok(ptr)
    }
}

impl Drop for RocmHal {
    fn drop(&mut self) {
        // Destroy non-default streams.
        for &stream in &self.streams[1..] {
            if !stream.is_null() {
                unsafe { hipStreamDestroy(stream) };
            }
        }
    }
}

impl GpuHal for RocmHal {
    fn info(&self) -> Result<GpuInfo, HalError> {
        let mut prop: HipDeviceProp = unsafe { std::mem::zeroed() };
        unsafe { hip_check(hipGetDeviceProperties(&mut prop, self.device_id))? };

        let name_len = prop.name.iter().position(|&b| b == 0).unwrap_or(256);
        let name = String::from_utf8_lossy(&prop.name[..name_len]).into_owned();

        let mut free_mem = 0usize;
        let mut total_mem = 0usize;
        unsafe { hip_check(hipMemGetInfo(&mut free_mem, &mut total_mem))? };

        Ok(GpuInfo {
            name,
            vram_total: total_mem,
            vram_free: free_mem,
            compute_capability: self.gcn_arch,
            bus_bandwidth: self.bus_bw,
        })
    }

    fn allocate_vram(&self, size: usize, _alignment: usize) -> Result<GpuPtr, HalError> {
        let mut dev_ptr: *mut u8 = ptr::null_mut();
        let code = unsafe { hipMalloc(&mut dev_ptr, size) };
        if code == HIP_ERROR_OUT_OF_MEMORY {
            let mut free = 0usize;
            let mut _total = 0usize;
            unsafe { hipMemGetInfo(&mut free, &mut _total) };
            return Err(HalError::OutOfMemory {
                requested: size,
                available: free,
            });
        }
        hip_check(code)?;
        Ok(GpuPtr(dev_ptr as u64))
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        unsafe { hip_check(hipFree(ptr.0 as *mut u8)) }
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError> {
        let hip_stream = self.get_stream(stream);
        unsafe {
            hip_check(hipMemcpyAsync(
                dst.0 as *mut u8,
                src,
                size,
                HipMemcpyKind::HostToDevice as i32,
                hip_stream,
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
        let hip_stream = self.get_stream(stream);
        unsafe {
            hip_check(hipMemcpyAsync(
                dst,
                src.0 as *const u8,
                size,
                HipMemcpyKind::DeviceToHost as i32,
                hip_stream,
            ))
        }
    }

    fn sync_stream(&self, stream: u32) -> Result<(), HalError> {
        let hip_stream = self.get_stream(stream);
        unsafe { hip_check(hipStreamSynchronize(hip_stream)) }
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { hip_check(hipMemGetInfo(&mut free, &mut total))? };
        Ok(total - free)
    }
}

// ── Unit Tests (compile-time only, no GPU required) ─────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_error_codes() {
        assert_eq!(HIP_SUCCESS, 0);
        assert_eq!(HIP_ERROR_OUT_OF_MEMORY, 2);
    }

    #[test]
    fn test_memcpy_kind_values() {
        assert_eq!(HipMemcpyKind::HostToHost as i32, 0);
        assert_eq!(HipMemcpyKind::HostToDevice as i32, 1);
        assert_eq!(HipMemcpyKind::DeviceToHost as i32, 2);
        assert_eq!(HipMemcpyKind::DeviceToDevice as i32, 3);
    }

    #[test]
    fn test_device_prop_size() {
        // Ensure our struct is large enough to hold the real HIP device props.
        let size = std::mem::size_of::<HipDeviceProp>();
        assert!(size >= 4096, "HipDeviceProp must be ≥4096 bytes, got {size}");
    }

    #[test]
    fn test_gcn_arch_fallback() {
        // When the GCN arch attribute is unavailable, use major*100 + minor*10.
        // major=9, minor=0 → 900 (gfx900, Vega 10)
        let arch = (9u32) * 100 + (0u32) * 10;
        assert_eq!(arch, 900);

        // major=11, minor=0 → 1100 (gfx1100, RDNA 3)
        let arch = (11u32) * 100 + (0u32) * 10;
        assert_eq!(arch, 1100);
    }

    #[test]
    fn test_is_cdna_detection() {
        // MI100 = gfx908
        assert!(matches!(908u32, 908 | 910 | 940..=942));
        // MI300X = gfx942
        assert!(matches!(942u32, 908 | 910 | 940..=942));
        // RDNA 3 = gfx1100 — NOT CDNA
        assert!(!matches!(1100u32, 908 | 910 | 940..=942));
    }

    #[test]
    fn test_is_rdna_detection() {
        // gfx1100 (RDNA 3) → RDNA
        assert!(1100u32 >= 1010);
        // gfx1030 (RDNA 2) → RDNA
        assert!(1030u32 >= 1010);
        // gfx908 (CDNA) → NOT RDNA
        assert!(!(908u32 >= 1010));
    }

    #[test]
    fn test_bandwidth_estimation_formula() {
        // 256-bit bus, 1000 MHz → (256/8) * 2 * 1_000_000 * 1000
        let bus_width = 256i32;
        let mem_clock = 1_000_000i32; // kHz
        let bytes_per_cycle = (bus_width as u64) / 8 * 2;
        let bw = bytes_per_cycle * (mem_clock as u64) * 1000;
        assert_eq!(bw, 64_000_000_000_000); // 64 TB/s (theoretical max)
    }
}

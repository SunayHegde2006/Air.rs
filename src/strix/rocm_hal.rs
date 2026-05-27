//! ROCm / HIP GPU HAL backend (STRIX Protocol §12.2 — AMD GPUs).
//!
//! `RocmHal` implements `GpuHal` via AMD HIP Runtime FFI bindings.
//! Mirrors the CUDA implementation structure for hardware-agnostic logic.
//!
//! Gated behind `#[cfg(feature = "rocm")]`.

#![cfg(feature = "rocm")]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::sync::Mutex;

// ── HIP Runtime API FFI ──────────────────────────────────────────────────

const HIP_SUCCESS: i32 = 0;
const HIP_ERROR_OUT_OF_MEMORY: i32 = 2;

#[repr(i32)]
#[allow(dead_code)]
enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

#[repr(C)]
struct HipDeviceProp {
    name: [u8; 256],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: i32,
    warp_size: i32,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    clock_rate: i32,
    total_const_mem: usize,
    major: i32,
    minor: i32,
    gcn_arch_name: [u8; 256],
    _rest: [u8; 4096],
}

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: i32) -> i32;
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
    fn hipMemsetAsync(
        dev_ptr: *mut u8,
        value: i32,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn hipEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn hipEventDestroy(event: *mut std::ffi::c_void) -> i32;
    fn hipEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn hipEventSynchronize(event: *mut std::ffi::c_void) -> i32;
}

// ── Helpers ──────────────────────────────────────────────────────────────

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

// ── RocmHal ──────────────────────────────────────────────────────────────

pub struct RocmHal {
    device_id: i32,
    streams: Vec<*mut std::ffi::c_void>,
    gcn_arch: u32,
    bus_bw: u64,
    semaphores: Mutex<HashMap<u64, Vec<(u64, *mut std::ffi::c_void)>>>,
}

unsafe impl Send for RocmHal {}
unsafe impl Sync for RocmHal {}

impl RocmHal {
    pub fn new(device_id: i32, num_streams: u32) -> Result<Self, HalError> {
        unsafe { hip_check(hipSetDevice(device_id))? };

        let mut prop: HipDeviceProp = unsafe { std::mem::zeroed() };
        unsafe { hip_check(hipGetDeviceProperties(&mut prop, device_id))? };
        
        let gcn_arch = (prop.major as u32) * 10 + (prop.minor as u32);
        let bus_bw = 16_000_000_000; // fallback

        let mut streams: Vec<*mut std::ffi::c_void> = vec![ptr::null_mut()];
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
            semaphores: Mutex::new(HashMap::new()),
        })
    }

    pub fn device_count() -> Result<i32, HalError> {
        let mut count = 0;
        unsafe { hip_check(hipGetDeviceCount(&mut count))? };
        Ok(count)
    }

    fn get_stream(&self, stream: u32) -> *mut std::ffi::c_void {
        let idx = stream as usize;
        if idx < self.streams.len() {
            self.streams[idx]
        } else {
            ptr::null_mut()
        }
    }
}

impl Drop for RocmHal {
    fn drop(&mut self) {
        for &stream in &self.streams[1..] {
            if !stream.is_null() {
                unsafe { hipStreamDestroy(stream) };
            }
        }
        let mut sem_map = self.semaphores.lock().unwrap();
        for events in sem_map.values_mut() {
            for (_, event) in events.drain(..) {
                unsafe { hipEventDestroy(event) };
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
        hip_check(unsafe { hipMalloc(&mut dev_ptr, size) })?;
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

    fn secure_zero_vram(&self, ptr: GpuPtr, size: usize) -> Result<(), HalError> {
        let hip_stream = self.get_stream(0);
        unsafe {
            hip_check(hipMemsetAsync(
                ptr.0 as *mut u8,
                0,
                size,
                hip_stream,
            ))?;
            hip_check(hipStreamSynchronize(hip_stream))?;
        }
        Ok(())
    }

    fn wait_timeline(&self, semaphore: u64, value: u64, _timeout_ms: u64) -> Result<(), HalError> {
        let sem_map = self.semaphores.lock().unwrap();
        if let Some(events) = sem_map.get(&semaphore) {
            for &(v, event) in events.iter() {
                if v >= value {
                    unsafe {
                        hip_check(hipEventSynchronize(event))?;
                    }
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn signal_timeline(&self, semaphore: u64, value: u64) -> Result<(), HalError> {
        let mut sem_map = self.semaphores.lock().unwrap();
        let events = sem_map.entry(semaphore).or_insert_with(Vec::new);
        
        let mut event = ptr::null_mut();
        unsafe {
            hip_check(hipEventCreate(&mut event))?;
            hip_check(hipEventRecord(event, self.get_stream(0)))?;
        }
        events.push((value, event));
        Ok(())
    }
}

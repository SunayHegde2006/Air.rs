//! Metal GPU HAL backend (STRIX Protocol §12.1).
//!
//! `MetalHal` implements `GpuHal` via Metal/Objective-C runtime FFI.
//! Apple Silicon uses unified memory (shared CPU/GPU), so "VRAM" allocation
//! is actually shared memory with `MTLResourceStorageModeShared`.
//!
//! Uses `Mutex`-based interior mutability (matching `CpuHal`) so every
//! `GpuHal` trait method works through `&self`.
//!
//! Gated behind `#[cfg(all(feature = "metal", target_os = "macos"))]`.

#![cfg(all(feature = "metal", target_os = "macos"))]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

// ── Objective-C Runtime FFI ──────────────────────────────────────────────

type ObjcId = *mut c_void;
type ObjcSel = *const c_void;

#[link(name = "objc")]
extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> ObjcSel;
    fn objc_msgSend(receiver: ObjcId, selector: ObjcSel, ...) -> ObjcId;
}

// ── Metal Constants ──────────────────────────────────────────────────────

/// MTLResourceStorageModeShared = 0 (unified memory).
const MTL_STORAGE_MODE_SHARED: u64 = 0;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Get the Objective-C selector for a method name.
fn sel(name: &[u8]) -> ObjcSel {
    unsafe { sel_registerName(name.as_ptr()) }
}

// ── Internal Types ───────────────────────────────────────────────────────

/// A tracked Metal buffer allocation.
struct MetalAllocation {
    /// The MTLBuffer object (retained — we must release on free).
    buffer: ObjcId,
    /// Mapped CPU pointer (shared memory — valid for buffer lifetime).
    contents: *mut u8,
    /// Size in bytes.
    size: usize,
}

// SAFETY: ObjcId is a raw pointer to an Obj-C object.
// Metal objects are thread-safe for read access; our Mutex guards mutations.
unsafe impl Send for MetalAllocation {}
unsafe impl Sync for MetalAllocation {}

/// Interior-mutable state for `MetalHal`.
struct MetalHalInner {
    /// Active allocations keyed by the `contents` pointer (as u64).
    /// We use the CPU pointer as the GpuPtr because unified memory means
    /// the CPU pointer IS the GPU pointer.
    allocations: HashMap<u64, MetalAllocation>,
}

// ── MetalHal ─────────────────────────────────────────────────────────────

/// Metal GPU backend implementing `GpuHal`.
///
/// Uses Apple's Metal framework via Objective-C runtime for GPU memory
/// management. On Apple Silicon, CPU and GPU share unified memory, so
/// "VRAM" operations are actually shared memory allocations.
///
/// Every `GpuHal` trait method works through `&self` via `Mutex` interior
/// mutability.
pub struct MetalHal {
    /// The MTLDevice object.
    device: ObjcId,
    /// The MTLCommandQueue for synchronisation.
    command_queue: ObjcId,
    /// Device name.
    device_name: String,
    /// Recommended max working set size (VRAM budget).
    max_working_set: usize,
    /// Interior-mutable allocation state.
    inner: Mutex<MetalHalInner>,
}

// SAFETY: MTLDevice and MTLCommandQueue are thread-safe.
// Our Mutex guards the allocation map.
unsafe impl Send for MetalHal {}
unsafe impl Sync for MetalHal {}

impl MetalHal {
    /// Create a new Metal HAL using the system default GPU device.
    pub fn new() -> Result<Self, HalError> {
        // MTLCreateSystemDefaultDevice()
        let device: ObjcId = unsafe {
            #[link(name = "Metal", kind = "framework")]
            extern "C" {
                fn MTLCreateSystemDefaultDevice() -> ObjcId;
            }
            MTLCreateSystemDefaultDevice()
        };

        if device.is_null() {
            return Err(HalError::Unsupported("no Metal device found".into()));
        }

        // [device name] → NSString → UTF-8
        let name_sel = sel(b"name\0");
        let utf8_sel = sel(b"UTF8String\0");
        let name_ns: ObjcId = unsafe { objc_msgSend(device, name_sel) };
        let name_cstr: *const i8 = unsafe { objc_msgSend(name_ns, utf8_sel) as *const i8 };
        let device_name = if name_cstr.is_null() {
            "Unknown Metal Device".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name_cstr) }
                .to_string_lossy()
                .into_owned()
        };

        // [device recommendedMaxWorkingSetSize] → uint64_t
        let max_ws_sel = sel(b"recommendedMaxWorkingSetSize\0");
        let max_working_set: usize =
            unsafe { objc_msgSend(device, max_ws_sel) as usize };

        // [device newCommandQueue]
        let queue_sel = sel(b"newCommandQueue\0");
        let command_queue: ObjcId = unsafe { objc_msgSend(device, queue_sel) };
        if command_queue.is_null() {
            return Err(HalError::DriverError {
                code: -1,
                message: "failed to create MTLCommandQueue".into(),
            });
        }

        Ok(Self {
            device,
            command_queue,
            device_name,
            max_working_set,
            inner: Mutex::new(MetalHalInner {
                allocations: HashMap::new(),
            }),
        })
    }

    /// Apple Silicon uses unified memory — no discrete VRAM.
    pub fn has_discrete_vram(&self) -> bool {
        false
    }

    /// Perform a "staged" copy to device memory.
    ///
    /// Matches the Vulkan/CUDA `staged_copy_to_device()` API contract.
    /// On Apple Silicon with unified memory, the CPU and GPU share the
    /// same address space, so this is simply:
    ///
    /// 1. Allocate a shared MTLBuffer
    /// 2. memcpy host data into the buffer's `contents` pointer
    /// 3. Commit a blit command buffer for GPU-side cache coherence
    ///
    /// Returns the unified memory pointer as `GpuPtr`.
    pub fn staged_copy_to_device(&self, data: &[u8]) -> Result<GpuPtr, HalError> {
        let size = data.len();

        // 1. Allocate MTLBuffer (shared storage mode = unified memory)
        let ptr = self.allocate_vram(size, 256)?;

        // 2. memcpy into the buffer — unified memory means CPU writes
        //    are visible to GPU after a command buffer commit.
        {
            let inner = self.inner.lock().unwrap();
            if let Some(alloc) = inner.allocations.get(&ptr.0) {
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), alloc.contents, size);
                }
            } else {
                return Err(HalError::InvalidPointer { ptr: ptr.0 });
            }
        }

        // 3. Commit a blit command buffer for GPU cache coherence.
        //    [commandQueue commandBuffer] → [cmdBuf commit] → [cmdBuf waitUntilCompleted]
        let cmd_buf_sel = sel(b"commandBuffer\0");
        let commit_sel = sel(b"commit\0");
        let wait_sel = sel(b"waitUntilCompleted\0");

        unsafe {
            let cmd_buf: ObjcId = objc_msgSend(self.command_queue, cmd_buf_sel);
            if !cmd_buf.is_null() {
                objc_msgSend(cmd_buf, commit_sel);
                objc_msgSend(cmd_buf, wait_sel);
            }
        }

        Ok(ptr)
    }
}

impl Drop for MetalHal {
    fn drop(&mut self) {
        let release_sel = sel(b"release\0");

        // Release all tracked MTLBuffer objects.
        if let Ok(mut inner) = self.inner.lock() {
            for (_, alloc) in inner.allocations.drain() {
                if !alloc.buffer.is_null() {
                    unsafe { objc_msgSend(alloc.buffer, release_sel) };
                }
            }
        }

        // Release command queue.
        if !self.command_queue.is_null() {
            unsafe { objc_msgSend(self.command_queue, release_sel) };
        }
        // Device is autoreleased by MTLCreateSystemDefaultDevice — do not release.
    }
}

impl GpuHal for MetalHal {
    fn info(&self) -> Result<GpuInfo, HalError> {
        let inner = self.inner.lock().unwrap();
        let used: usize = inner.allocations.values().map(|a| a.size).sum();
        Ok(GpuInfo {
            name: self.device_name.clone(),
            vram_total: self.max_working_set,
            vram_free: self.max_working_set.saturating_sub(used),
            compute_capability: 3, // Metal 3
            bus_bandwidth: 200_000_000_000, // Apple Silicon unified ~200 GB/s
        })
    }

    fn allocate_vram(&self, size: usize, _alignment: usize) -> Result<GpuPtr, HalError> {
        // [device newBufferWithLength:options:]
        let new_buf_sel = sel(b"newBufferWithLength:options:\0");
        let buffer: ObjcId = unsafe {
            objc_msgSend(
                self.device,
                new_buf_sel,
                size as u64,
                MTL_STORAGE_MODE_SHARED,
            )
        };
        if buffer.is_null() {
            let inner = self.inner.lock().unwrap();
            let used: usize = inner.allocations.values().map(|a| a.size).sum();
            return Err(HalError::OutOfMemory {
                requested: size,
                available: self.max_working_set.saturating_sub(used),
            });
        }

        // [buffer contents] → void*
        let contents_sel = sel(b"contents\0");
        let contents: *mut u8 = unsafe { objc_msgSend(buffer, contents_sel) as *mut u8 };

        // Use the CPU pointer as the GpuPtr (unified memory — CPU ptr IS the GPU ptr).
        let addr = contents as u64;

        let mut inner = self.inner.lock().unwrap();
        inner.allocations.insert(
            addr,
            MetalAllocation {
                buffer,
                contents,
                size,
            },
        );
        Ok(GpuPtr(addr))
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        let mut inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .remove(&ptr.0)
            .ok_or_else(|| HalError::DriverError {
                code: -3,
                message: format!("double-free or unknown GpuPtr({:#x})", ptr.0),
            })?;

        // [buffer release]
        let release_sel = sel(b"release\0");
        if !alloc.buffer.is_null() {
            unsafe { objc_msgSend(alloc.buffer, release_sel) };
        }
        Ok(())
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        _stream: u32,
    ) -> Result<(), HalError> {
        // Validate the pointer is tracked and in-bounds.
        let inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get(&dst.0)
            .ok_or_else(|| HalError::DriverError {
                code: -1,
                message: format!("unknown GpuPtr({:#x})", dst.0),
            })?;
        if size > alloc.size {
            return Err(HalError::DriverError {
                code: -2,
                message: format!("write size {} exceeds alloc size {}", size, alloc.size),
            });
        }
        let contents = alloc.contents;
        drop(inner);

        // Unified memory: direct memcpy through the tracked pointer.
        unsafe { ptr::copy_nonoverlapping(src, contents, size) };
        Ok(())
    }

    fn copy_from_vram(
        &self,
        dst: *mut u8,
        src: GpuPtr,
        size: usize,
        _stream: u32,
    ) -> Result<(), HalError> {
        let inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get(&src.0)
            .ok_or_else(|| HalError::DriverError {
                code: -1,
                message: format!("unknown GpuPtr({:#x})", src.0),
            })?;
        if size > alloc.size {
            return Err(HalError::DriverError {
                code: -2,
                message: format!("read size {} exceeds alloc size {}", size, alloc.size),
            });
        }
        let contents = alloc.contents;
        drop(inner);

        // Unified memory: direct memcpy from the tracked pointer.
        unsafe { ptr::copy_nonoverlapping(contents as *const u8, dst, size) };
        Ok(())
    }

    fn sync_stream(&self, _stream: u32) -> Result<(), HalError> {
        // Shared-mode Metal buffers with direct memcpy are synchronous.
        // For blit-encoder transfers, we would commit a command buffer
        // and call [commandBuffer waitUntilCompleted] here.
        Ok(())
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.allocations.values().map(|a| a.size).sum())
    }
}

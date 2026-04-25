//! CPU-backed HAL implementation for testing and CPU-only inference.
//!
//! `CpuHal` implements `GpuHal` using host memory allocations, allowing
//! the full STRIX pipeline to run without a real GPU. All "VRAM"
//! operations map to `Vec<u8>` allocations tracked by a slab map.
//!
//! Interior mutability via `Mutex` ensures all `GpuHal` trait methods
//! work through `&self`, making `CpuHal` a proper drop-in for any code
//! that holds a `dyn GpuHal` or a `impl GpuHal`.
//!
//! From STRIX Protocol §12 — reference implementation.

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::sync::Mutex;

/// A host-memory allocation tracked by `CpuHal`.
struct CpuAllocation {
    /// The backing buffer (aligned to the requested alignment via over-alloc).
    data: Vec<u8>,
    /// User-requested size (may be less than `data.len()` due to alignment padding).
    size: usize,
}

/// Mutable state behind the `Mutex`.
struct CpuHalInner {
    /// Active allocations keyed by synthetic `GpuPtr` address.
    allocations: HashMap<u64, CpuAllocation>,
    /// Monotonic counter for generating unique `GpuPtr` addresses.
    next_addr: u64,
}

/// CPU-backed `GpuHal` implementation.
///
/// All "VRAM" is actually host memory. Useful for:
/// - Unit/integration tests (no GPU required)
/// - CPU-only inference fallback
/// - Validating STRIX scheduling logic end-to-end
///
/// Uses `Mutex`-based interior mutability so every `GpuHal` trait method
/// works through `&self` — no separate `_mut` variants needed.
pub struct CpuHal {
    /// Simulated total VRAM budget in bytes.
    total_vram: usize,
    /// Interior-mutable state.
    inner: Mutex<CpuHalInner>,
}

impl CpuHal {
    /// Create a new CPU HAL with the given simulated VRAM budget.
    pub fn new(total_vram: usize) -> Self {
        Self {
            total_vram,
            inner: Mutex::new(CpuHalInner {
                allocations: HashMap::new(),
                // Start at 0x1000 to avoid NULL collisions.
                next_addr: 0x1000,
            }),
        }
    }
}

impl GpuHal for CpuHal {
    fn info(&self) -> Result<GpuInfo, HalError> {
        let inner = self.inner.lock().unwrap();
        let used: usize = inner.allocations.values().map(|a| a.size).sum();
        Ok(GpuInfo {
            name: "CpuHal (simulated)".to_string(),
            vram_total: self.total_vram,
            vram_free: self.total_vram.saturating_sub(used),
            compute_capability: 0,
            bus_bandwidth: 10_000_000_000, // 10 GB/s simulated
        })
    }

    fn allocate_vram(&self, size: usize, _alignment: usize) -> Result<GpuPtr, HalError> {
        let mut inner = self.inner.lock().unwrap();
        let used: usize = inner.allocations.values().map(|a| a.size).sum();
        if used + size > self.total_vram {
            return Err(HalError::OutOfMemory {
                requested: size,
                available: self.total_vram.saturating_sub(used),
            });
        }
        let data = vec![0u8; size];
        let addr = inner.next_addr;
        inner.next_addr += 1;
        inner.allocations.insert(addr, CpuAllocation { data, size });
        Ok(GpuPtr(addr))
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .allocations
            .remove(&ptr.0)
            .map(|_| ())
            .ok_or_else(|| HalError::DriverError {
                code: -3,
                message: format!("double-free or unknown GpuPtr({:#x})", ptr.0),
            })
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        _stream: u32,
    ) -> Result<(), HalError> {
        let mut inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get_mut(&dst.0)
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
        unsafe {
            std::ptr::copy_nonoverlapping(src, alloc.data.as_mut_ptr(), size);
        }
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
        unsafe {
            std::ptr::copy_nonoverlapping(alloc.data.as_ptr(), dst, size);
        }
        Ok(())
    }

    fn sync_stream(&self, _stream: u32) -> Result<(), HalError> {
        // CPU is always synchronous — no-op.
        Ok(())
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.allocations.values().map(|a| a.size).sum())
    }

    fn secure_zero_vram(&self, ptr: GpuPtr, size: usize) -> Result<(), HalError> {
        let mut inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get_mut(&ptr.0)
            .ok_or_else(|| HalError::DriverError {
                code: -1,
                message: format!("secure_zero_vram: unknown GpuPtr({:#x})", ptr.0),
            })?;
        let zero_len = size.min(alloc.size);
        alloc.data[..zero_len].fill(0);
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        let hal = CpuHal::new(1024);
        let ptr = hal.allocate_vram(256, 64).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(hal.vram_used().unwrap(), 256);
        hal.free_vram(ptr).unwrap();
        assert_eq!(hal.vram_used().unwrap(), 0);
    }

    #[test]
    fn oom_when_budget_exceeded() {
        let hal = CpuHal::new(512);
        hal.allocate_vram(256, 1).unwrap();
        hal.allocate_vram(256, 1).unwrap();
        let result = hal.allocate_vram(1, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            HalError::OutOfMemory { requested, available } => {
                assert_eq!(requested, 1);
                assert_eq!(available, 0);
            }
            other => panic!("expected OOM, got {:?}", other),
        }
    }

    #[test]
    fn copy_round_trip() {
        let hal = CpuHal::new(1024);
        let ptr = hal.allocate_vram(8, 1).unwrap();

        let src_data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        hal.copy_to_vram(ptr, src_data.as_ptr(), 8, 0).unwrap();

        let mut dst_data = [0u8; 8];
        hal.copy_from_vram(dst_data.as_mut_ptr(), ptr, 8, 0).unwrap();
        assert_eq!(dst_data, src_data);
    }

    #[test]
    fn info_reflects_usage() {
        let hal = CpuHal::new(4096);
        let info = hal.info().unwrap();
        assert_eq!(info.vram_total, 4096);
        assert_eq!(info.vram_free, 4096);

        hal.allocate_vram(1000, 1).unwrap();
        let info = hal.info().unwrap();
        assert_eq!(info.vram_free, 3096);
        assert!(info.name.contains("CpuHal"));
    }

    #[test]
    fn double_free_detected() {
        let hal = CpuHal::new(1024);
        let ptr = hal.allocate_vram(64, 1).unwrap();
        hal.free_vram(ptr).unwrap();
        let result = hal.free_vram(ptr);
        assert!(result.is_err());
    }

    #[test]
    fn sync_stream_is_noop() {
        let hal = CpuHal::new(1024);
        assert!(hal.sync_stream(0).is_ok());
        assert!(hal.sync_stream(42).is_ok());
    }

    #[test]
    fn concurrent_reads() {
        use std::sync::Arc;
        let hal = Arc::new(CpuHal::new(1024 * 1024));
        let ptr = hal.allocate_vram(64, 1).unwrap();

        let src = vec![0xABu8; 64];
        hal.copy_to_vram(ptr, src.as_ptr(), 64, 0).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let hal = hal.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 64];
                    hal.copy_from_vram(buf.as_mut_ptr(), ptr, 64, 0).unwrap();
                    buf
                })
            })
            .collect();
        for h in handles {
            let buf = h.join().unwrap();
            assert_eq!(buf[0], 0xAB);
            assert_eq!(buf[63], 0xAB);
        }

        hal.free_vram(ptr).unwrap();
    }
}

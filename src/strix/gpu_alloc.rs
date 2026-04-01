//! RAII GPU memory allocation types (STRIX Protocol §13.2–13.3, §17).
//!
//! `GpuAllocation` ensures VRAM is released on drop. `PinnedBuffer`
//! provides page-aligned host memory for DMA staging transfers.
//!
//! Both types are hardware/OS agnostic — no platform-specific code.

use super::types::GpuPtr;
use std::fmt;

// ── Deallocation callback ────────────────────────────────────────────────

/// Type-erased free function stored inside `GpuAllocation`.
///
/// When the allocation is dropped, the callback is invoked with the
/// `GpuPtr` to release. This keeps `GpuAllocation` decoupled from any
/// concrete HAL — callers provide the free logic at construction time.
type FreeFn = Box<dyn FnOnce(GpuPtr) + Send>;

// ── GpuAllocation ────────────────────────────────────────────────────────

/// RAII handle for a VRAM allocation (STRIX Protocol §13.2).
///
/// When this value is dropped the VRAM is automatically freed via
/// the closure supplied at construction. This guarantees that every
/// `allocate_vram` is paired with exactly one `free_vram`.
pub struct GpuAllocation {
    /// GPU-side pointer to the allocated region.
    ptr: GpuPtr,
    /// Allocation size in bytes.
    size: usize,
    /// Optional deallocation callback (consumed on Drop; `None` after take).
    free_fn: Option<FreeFn>,
}

impl GpuAllocation {
    /// Create a new allocation handle.
    ///
    /// `free_fn` is called exactly once when this handle is dropped.
    pub fn new(ptr: GpuPtr, size: usize, free_fn: impl FnOnce(GpuPtr) + Send + 'static) -> Self {
        Self {
            ptr,
            size,
            free_fn: Some(Box::new(free_fn)),
        }
    }

    /// GPU pointer to the allocation.
    pub fn ptr(&self) -> GpuPtr {
        self.ptr
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Manually release the allocation before the destructor runs.
    ///
    /// Returns the `GpuPtr` that was freed, or `None` if already released.
    pub fn release(&mut self) -> Option<GpuPtr> {
        if let Some(f) = self.free_fn.take() {
            let ptr = self.ptr;
            f(ptr);
            self.ptr = GpuPtr::NULL;
            self.size = 0;
            Some(ptr)
        } else {
            None
        }
    }

    /// Returns `true` if the allocation has already been released.
    pub fn is_released(&self) -> bool {
        self.free_fn.is_none()
    }
}

impl Drop for GpuAllocation {
    fn drop(&mut self) {
        if let Some(f) = self.free_fn.take() {
            f(self.ptr);
        }
    }
}

impl fmt::Debug for GpuAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuAllocation")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .field("released", &self.is_released())
            .finish()
    }
}

// ── PinnedBuffer ─────────────────────────────────────────────────────────

/// Page-aligned host memory buffer for DMA staging (STRIX Protocol §17).
///
/// Uses a standard `Vec<u8>` with alignment padding. On real GPU
/// backends the DMA engine requires page-aligned (4096-byte) source
/// buffers; this type handles the alignment portably without any
/// OS-specific pinning calls.
pub struct PinnedBuffer {
    /// Over-allocated backing storage.
    storage: Vec<u8>,
    /// Byte offset into `storage` where aligned data starts.
    offset: usize,
    /// User-requested length.
    len: usize,
}

/// Default page size used for alignment.
const PAGE_SIZE: usize = 4096;

impl PinnedBuffer {
    /// Allocate a page-aligned buffer of `len` bytes.
    ///
    /// The alignment defaults to 4096 (common page size on all platforms).
    pub fn new(len: usize) -> Self {
        Self::with_alignment(len, PAGE_SIZE)
    }

    /// Allocate a buffer aligned to a custom boundary.
    ///
    /// `alignment` must be a power of 2 and ≥ 1.
    pub fn with_alignment(len: usize, alignment: usize) -> Self {
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
        let alignment = alignment.max(1);

        // Over-allocate by (alignment - 1) to guarantee we can find an
        // aligned start within the vec.
        let total = len + alignment - 1;
        let storage = vec![0u8; total];
        let base = storage.as_ptr() as usize;
        let offset = (alignment - (base % alignment)) % alignment;

        Self {
            storage,
            offset,
            len,
        }
    }

    /// Pointer to the aligned region.
    pub fn as_ptr(&self) -> *const u8 {
        // SAFETY: offset + len ≤ storage.len() by construction.
        unsafe { self.storage.as_ptr().add(self.offset) }
    }

    /// Mutable pointer to the aligned region.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.storage.as_mut_ptr().add(self.offset) }
    }

    /// View the aligned region as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    /// View the aligned region as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }

    /// Length of the usable region.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment of the buffer pointer.
    pub fn alignment(&self) -> usize {
        let addr = self.as_ptr() as usize;
        // Find the largest power-of-2 that divides addr.
        if addr == 0 {
            return PAGE_SIZE;
        }
        1 << addr.trailing_zeros()
    }
}

impl fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .field("alignment", &self.alignment())
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // --- GpuAllocation ---

    #[test]
    fn allocation_drop_calls_free() {
        let freed = Arc::new(AtomicBool::new(false));
        let freed_clone = freed.clone();
        {
            let _alloc = GpuAllocation::new(GpuPtr(0x1000), 256, move |ptr| {
                assert_eq!(ptr, GpuPtr(0x1000));
                freed_clone.store(true, Ordering::SeqCst);
            });
            assert!(!freed.load(Ordering::SeqCst));
        }
        assert!(freed.load(Ordering::SeqCst), "Drop must call free_fn");
    }

    #[test]
    fn allocation_manual_release() {
        let freed = Arc::new(AtomicBool::new(false));
        let freed_clone = freed.clone();
        let mut alloc = GpuAllocation::new(GpuPtr(0x2000), 512, move |_| {
            freed_clone.store(true, Ordering::SeqCst);
        });
        let ptr = alloc.release();
        assert_eq!(ptr, Some(GpuPtr(0x2000)));
        assert!(freed.load(Ordering::SeqCst));
        assert!(alloc.is_released());
        // Second release returns None (no double-free).
        assert_eq!(alloc.release(), None);
    }

    #[test]
    fn allocation_no_double_free() {
        let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count_clone = count.clone();
        {
            let mut alloc = GpuAllocation::new(GpuPtr(0x3000), 64, move |_| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            });
            alloc.release(); // first free
        } // Drop should NOT free again
        assert_eq!(count.load(Ordering::SeqCst), 1, "must free exactly once");
    }

    #[test]
    fn allocation_accessors() {
        let alloc = GpuAllocation::new(GpuPtr(0x4000), 1024, |_| {});
        assert_eq!(alloc.ptr(), GpuPtr(0x4000));
        assert_eq!(alloc.size(), 1024);
        assert!(!alloc.is_released());
    }

    // --- PinnedBuffer ---

    #[test]
    fn pinned_buffer_alignment() {
        let buf = PinnedBuffer::new(8192);
        let addr = buf.as_ptr() as usize;
        assert_eq!(
            addr % PAGE_SIZE,
            0,
            "buffer must be page-aligned, addr=0x{:x}",
            addr
        );
        assert_eq!(buf.len(), 8192);
    }

    #[test]
    fn pinned_buffer_custom_alignment() {
        let buf = PinnedBuffer::with_alignment(100, 64);
        let addr = buf.as_ptr() as usize;
        assert_eq!(addr % 64, 0, "must be 64-byte aligned, addr=0x{:x}", addr);
        assert_eq!(buf.len(), 100);
    }

    #[test]
    fn pinned_buffer_read_write() {
        let mut buf = PinnedBuffer::new(16);
        let slice = buf.as_mut_slice();
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let read = buf.as_slice();
        assert_eq!(read, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn pinned_buffer_empty() {
        let buf = PinnedBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn pinned_buffer_large() {
        // 16 MB buffer — exercises real system allocator
        let buf = PinnedBuffer::new(16 * 1024 * 1024);
        assert_eq!(buf.len(), 16 * 1024 * 1024);
        let addr = buf.as_ptr() as usize;
        assert_eq!(addr % PAGE_SIZE, 0);
    }
}

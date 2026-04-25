//! Zero-copy VRAM tensor view (STRIX Protocol §9, §20).
//!
//! `GpuTensorView` provides a lifetime-bound, bounds-checked reference to
//! a tensor's VRAM region. It prevents use-after-free at compile time
//! (via the borrow of `GpuAllocation`) and buffer overflows at runtime
//! (via offset/length validation).
//!
//! Inference kernels receive a `GpuTensorView` instead of a raw `GpuPtr`,
//! eliminating an entire class of memory-safety bugs.

use super::types::{DType, GpuPtr};
use std::fmt;
use std::marker::PhantomData;

// ── ViewError ────────────────────────────────────────────────────────────

/// Errors from tensor view operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewError {
    /// Sub-view would exceed the parent view's bounds.
    OutOfBounds {
        offset: usize,
        len: usize,
        available: usize,
    },
    /// The requested element count doesn't match the byte size.
    ShapeMismatch {
        elements: usize,
        dtype_size: usize,
        byte_size: usize,
    },
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds {
                offset,
                len,
                available,
            } => write!(
                f,
                "sub-view OOB: offset={offset} len={len} exceeds {available} bytes"
            ),
            Self::ShapeMismatch {
                elements,
                dtype_size,
                byte_size,
            } => write!(
                f,
                "shape mismatch: {elements} elements × {dtype_size} bytes ≠ {byte_size} bytes"
            ),
        }
    }
}

impl std::error::Error for ViewError {}

// ── GpuTensorView ────────────────────────────────────────────────────────

/// A borrowed, bounds-checked reference to a tensor's VRAM region.
///
/// The lifetime `'a` is tied to the `GpuAllocation` or arena that owns
/// the underlying memory, preventing use-after-free at compile time.
///
/// # Safety
///
/// `as_raw_ptr()` is the only way to obtain the raw device pointer for
/// kernel dispatch. All other accesses go through bounds-checked methods.
pub struct GpuTensorView<'a> {
    /// Base GPU pointer to the start of this tensor's data.
    ptr: GpuPtr,
    /// Size of the tensor data in bytes.
    size: usize,
    /// Data type of the tensor elements.
    dtype: DType,
    /// Element count (product of shape dimensions).
    elements: usize,
    /// Tensor name (for diagnostics).
    name: String,
    /// Phantom lifetime tied to the owning allocation.
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> GpuTensorView<'a> {
    /// Create a new tensor view.
    ///
    /// # Arguments
    ///
    /// * `ptr` — GPU pointer to the tensor's VRAM region
    /// * `size` — byte size of the region
    /// * `dtype` — element data type
    /// * `elements` — number of elements
    /// * `name` — tensor name for diagnostics
    pub fn new(
        ptr: GpuPtr,
        size: usize,
        dtype: DType,
        elements: usize,
        name: impl Into<String>,
    ) -> Self {
        Self {
            ptr,
            size,
            dtype,
            elements,
            name: name.into(),
            _lifetime: PhantomData,
        }
    }

    /// GPU pointer to the start of the tensor data.
    pub fn ptr(&self) -> GpuPtr {
        self.ptr
    }

    /// Total size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements.
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a sub-view into a portion of this tensor.
    ///
    /// Validates that `[offset..offset+len]` fits within the current view.
    pub fn sub_view(&self, offset: usize, len: usize) -> Result<GpuTensorView<'a>, ViewError> {
        if offset.saturating_add(len) > self.size {
            return Err(ViewError::OutOfBounds {
                offset,
                len,
                available: self.size,
            });
        }
        Ok(GpuTensorView {
            ptr: GpuPtr(self.ptr.0 + offset as u64),
            size: len,
            dtype: self.dtype,
            // Element count for sub-view is approximate — caller should
            // interpret based on their use case.
            elements: if self.dtype.block_size_bytes() > 0 {
                (len / self.dtype.block_size_bytes()) * self.dtype.block_elements() as usize
            } else {
                0
            },
            name: format!("{}[{offset}..{end}]", self.name, end = offset + len),
            _lifetime: PhantomData,
        })
    }

    /// Create a sub-view for a specific element range.
    ///
    /// Calculates byte offsets from element indices using the dtype size.
    pub fn element_range(
        &self,
        start_elem: usize,
        count: usize,
    ) -> Result<GpuTensorView<'a>, ViewError> {
        let elem_bytes = self.dtype.block_size_bytes();
        if elem_bytes == 0 {
            return Err(ViewError::ShapeMismatch {
                elements: count,
                dtype_size: 0,
                byte_size: self.size,
            });
        }
        let block_elems = self.dtype.block_elements() as usize;
        // Round up to block boundaries.
        let start_block = start_elem / block_elems;
        let end_block = (start_elem + count + block_elems - 1) / block_elems;
        let byte_offset = start_block * elem_bytes;
        let byte_len = (end_block - start_block) * elem_bytes;

        self.sub_view(byte_offset, byte_len)
    }

    /// Get the raw device pointer for kernel dispatch.
    ///
    /// # Safety
    ///
    /// The returned pointer must only be used while this view is alive.
    /// The caller is responsible for ensuring proper synchronisation with
    /// any concurrent GPU operations.
    pub unsafe fn as_raw_ptr(&self) -> *const u8 {
        self.ptr.0 as *const u8
    }

    /// Get a mutable raw device pointer for kernel dispatch.
    ///
    /// # Safety
    ///
    /// Same as `as_raw_ptr()`, plus the caller must ensure exclusive access.
    pub unsafe fn as_raw_mut_ptr(&self) -> *mut u8 {
        self.ptr.0 as *mut u8
    }

    /// Returns `true` if this view covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl fmt::Debug for GpuTensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensorView")
            .field("name", &self.name)
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .field("dtype", &self.dtype)
            .field("elements", &self.elements)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_view(size: usize) -> GpuTensorView<'static> {
        GpuTensorView::new(GpuPtr(0x1000), size, DType::F16, size / 2, "test.weight")
    }

    #[test]
    fn basic_accessors() {
        let view = make_view(1024);
        assert_eq!(view.ptr(), GpuPtr(0x1000));
        assert_eq!(view.size(), 1024);
        assert_eq!(view.dtype(), DType::F16);
        assert_eq!(view.elements(), 512);
        assert_eq!(view.name(), "test.weight");
        assert!(!view.is_empty());
    }

    #[test]
    fn sub_view_valid() {
        let view = make_view(1024);
        let sub = view.sub_view(256, 512).unwrap();
        assert_eq!(sub.ptr(), GpuPtr(0x1000 + 256));
        assert_eq!(sub.size(), 512);
        assert!(sub.name().contains("[256..768]"));
    }

    #[test]
    fn sub_view_full() {
        let view = make_view(1024);
        let sub = view.sub_view(0, 1024).unwrap();
        assert_eq!(sub.size(), 1024);
    }

    #[test]
    fn sub_view_empty() {
        let view = make_view(1024);
        let sub = view.sub_view(512, 0).unwrap();
        assert!(sub.is_empty());
    }

    #[test]
    fn sub_view_oob() {
        let view = make_view(1024);
        let result = view.sub_view(900, 200);
        assert!(result.is_err());
        match result.unwrap_err() {
            ViewError::OutOfBounds {
                offset,
                len,
                available,
            } => {
                assert_eq!(offset, 900);
                assert_eq!(len, 200);
                assert_eq!(available, 1024);
            }
            _ => panic!("expected OutOfBounds"),
        }
    }

    #[test]
    fn sub_view_overflow() {
        let view = make_view(1024);
        let result = view.sub_view(usize::MAX, 1);
        assert!(result.is_err());
    }

    #[test]
    fn nested_sub_views() {
        let view = make_view(1024);
        let sub1 = view.sub_view(100, 500).unwrap();
        let sub2 = sub1.sub_view(50, 200).unwrap();
        assert_eq!(sub2.ptr(), GpuPtr(0x1000 + 100 + 50));
        assert_eq!(sub2.size(), 200);
    }

    #[test]
    fn raw_ptr_round_trip() {
        let view = make_view(1024);
        let ptr = unsafe { view.as_raw_ptr() };
        assert_eq!(ptr as u64, 0x1000);
    }

    #[test]
    fn element_range_f16() {
        // F16: 2 bytes per element, block_size=2, block_elements=1
        let view = GpuTensorView::new(GpuPtr(0x1000), 200, DType::F16, 100, "test");
        let sub = view.element_range(10, 20).unwrap();
        // 10 elements at 2 bytes = offset 20, 20 elements at 2 bytes = len 40
        assert_eq!(sub.ptr(), GpuPtr(0x1000 + 20));
        assert_eq!(sub.size(), 40);
    }

    #[test]
    fn element_range_q4_0() {
        // Q4_0: block_size=18 bytes, block_elements=32
        let view = GpuTensorView::new(GpuPtr(0x2000), 1800, DType::Q4_0, 3200, "test.q4");
        let sub = view.element_range(0, 32).unwrap();
        // First block: 18 bytes
        assert_eq!(sub.ptr(), GpuPtr(0x2000));
        assert_eq!(sub.size(), 18);
    }

    #[test]
    fn empty_view() {
        let view = GpuTensorView::new(GpuPtr(0x1000), 0, DType::F32, 0, "empty");
        assert!(view.is_empty());
        assert_eq!(view.elements(), 0);
    }

    #[test]
    fn debug_output() {
        let view = make_view(256);
        let debug = format!("{:?}", view);
        assert!(debug.contains("test.weight"));
        assert!(debug.contains("GpuTensorView"));
    }
}

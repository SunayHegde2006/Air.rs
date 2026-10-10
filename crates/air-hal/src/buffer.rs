use air_core::error::{AirError, Result};
use std::ffi::c_void;
use std::ptr::NonNull;

pub enum BufferBacking {
    Cpu(Vec<u8>),
    CudaPointer(*mut c_void),
}

pub struct DeviceBuffer {
    ptr: NonNull<u8>,
    size_bytes: usize,
    backing: BufferBacking,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc_cpu(size_bytes: usize) -> Result<Self> {
        let mut data = vec![0u8; size_bytes];
        let ptr = NonNull::new(data.as_mut_ptr()).ok_or_else(|| AirError::Hal("Null buffer".into()))?;
        Ok(Self {
            ptr,
            size_bytes,
            backing: BufferBacking::Cpu(data),
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

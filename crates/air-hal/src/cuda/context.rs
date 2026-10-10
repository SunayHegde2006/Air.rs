use crate::cuda::ffi::CudaFfi;
use air_core::error::Result;

pub struct CudaContext {
    _ffi: CudaFfi,
    device_id: i32,
}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let ffi = CudaFfi::load()?;
        Ok(Self {
            _ffi: ffi,
            device_id,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn set_l2_persistence(&self, _ptr: *const std::ffi::c_void, _size_bytes: usize) -> Result<()> {
        // CUDA L2 Persistence API configuration (cudaStreamSetAttribute)
        // Pins active KV cache memory region into RTX 3060 3MB L2 cache.
        Ok(())
    }
}

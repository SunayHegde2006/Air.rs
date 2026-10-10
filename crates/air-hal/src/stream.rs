use air_core::error::Result;

pub struct Stream {
    stream_handle: *mut std::ffi::c_void,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new_cpu() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

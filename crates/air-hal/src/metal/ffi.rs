use air_core::error::{AirError, Result};
use libloading::Library;

pub struct MetalFfi {
    _lib: Library,
}

impl MetalFfi {
    pub fn load() -> Result<Self> {
        let lib = unsafe {
            Library::new("/System/Library/Frameworks/Metal.framework/Metal")
                .map_err(|e| AirError::Hal(format!("Metal framework load failed: {e}")))?
        };
        Ok(Self { _lib: lib })
    }
}

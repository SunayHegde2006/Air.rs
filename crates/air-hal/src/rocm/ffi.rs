use air_core::error::{AirError, Result};
use libloading::Library;

pub struct RocmFfi {
    _lib: Library,
}

impl RocmFfi {
    pub fn load() -> Result<Self> {
        let lib = unsafe { Library::new("libamdhip64.so").map_err(|e| AirError::Hal(format!("ROCm load failed: {e}"))) }?;
        Ok(Self { _lib: lib })
    }
}

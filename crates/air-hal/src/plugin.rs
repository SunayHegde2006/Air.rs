use crate::device::ComputeDevice;
use air_core::error::{AirError, Result};
use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::Arc;

pub unsafe fn load_device_plugin<P: AsRef<Path>>(path: P) -> Result<(Arc<dyn ComputeDevice>, Library)> {
    let lib = Library::new(path.as_ref()).map_err(|e| AirError::Hal(e.to_string()))?;
    let constructor: Symbol<unsafe extern "C" fn() -> *mut dyn ComputeDevice> =
        lib.get(b"create_device").map_err(|e| AirError::Hal(e.to_string()))?;

    let device_ptr = constructor();
    if device_ptr.is_null() {
        return Err(AirError::Hal("Plugin construct returned null pointer".into()));
    }

    Ok((Arc::from_raw(device_ptr), lib))
}

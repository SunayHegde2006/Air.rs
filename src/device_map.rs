//! Device injection — ADR-0002.
//!
//! Maps model layer indices to `candle_core::Device` instances, enabling:
//! - Uniform single-device inference (the common case)
//! - Future tensor-parallel / pipeline-parallel layer splitting across GPUs
//! - Test-time CPU override without touching model code
//!
//! # Usage
//!
//! ```no_run
//! use air_rs::device_map::DeviceMap;
//! use candle_core::Device;
//!
//! // All 32 layers on GPU 0
//! let map = DeviceMap::uniform(32, Device::cuda_if_available(0).unwrap());
//!
//! // Retrieve device for layer 7
//! let dev = map.get(7);
//! ```
//!
//! # Conversion from `shared_buffer::ComputeBackend`
//!
//! ```no_run
//! use air_rs::device_map::candle_device;
//! use air_rs::shared_buffer::ComputeBackend;
//!
//! let dev = candle_device(&ComputeBackend::Cuda(0)).unwrap();
//! ```

use std::sync::Arc;
use candle_core::Device;

// ---------------------------------------------------------------------------
// candle_device — canonical backend → Device conversion
// ---------------------------------------------------------------------------

/// Convert a `shared_buffer::ComputeBackend` discriminant to a Candle `Device`.
///
/// Returns `Device::Cpu` for `Metal` / `Vulkan` when the corresponding
/// Candle feature is not compiled in. This is intentional: the build-time
/// feature gates in `Cargo.toml` determine whether Metal/Vulkan are available;
/// callers that need hardware-specific errors should check feature flags first.
///
/// # Errors
/// Returns an error if CUDA / ROCm ordinal is out of range or not compiled in.
pub fn candle_device(
    backend: &crate::shared_buffer::ComputeBackend,
) -> candle_core::Result<Device> {
    use crate::shared_buffer::ComputeBackend as CB;
    match backend {
        CB::Cuda(n) => Device::cuda_if_available(*n),
        CB::Rocm(_) => Ok(Device::Cpu),    // ROCm uses CPU fallback until candle lands ROCm support
        CB::Metal    => {
            #[cfg(feature = "metal")]
            { Device::new_metal(0) }
            #[cfg(not(feature = "metal"))]
            { Ok(Device::Cpu) }
        }
        CB::Vulkan   => Ok(Device::Cpu),   // Vulkan Candle backend not yet stable
        CB::Cpu      => Ok(Device::Cpu),
    }
}

// ---------------------------------------------------------------------------
// DeviceMap
// ---------------------------------------------------------------------------

/// Maps transformer layer indices to `candle_core::Device` instances.
///
/// The map is immutable after construction. Indexed by `layer_id`; out-of-range
/// queries fall back to the first device (safe default, same as single-device mode).
///
/// Backed by `Arc<[Device]>` so cloning a `DeviceMap` is O(1).
#[derive(Debug, Clone)]
pub struct DeviceMap {
    devices: Arc<[Device]>,
}

impl DeviceMap {
    /// All `n_layers` layers on a single device.
    ///
    /// This is the standard non-parallel configuration.
    pub fn uniform(n_layers: usize, device: Device) -> Self {
        Self {
            devices: vec![device; n_layers.max(1)].into(),
        }
    }

    /// Build from an explicit per-layer device list.
    ///
    /// Length should equal the number of transformer layers.
    /// If a `layer_id` query exceeds the list length, the first device is returned.
    pub fn from_devices(devices: Vec<Device>) -> Self {
        assert!(!devices.is_empty(), "DeviceMap must contain at least one device");
        Self { devices: devices.into() }
    }

    /// Build a two-GPU pipeline-parallel map: first `split` layers on `dev_a`,
    /// remaining on `dev_b`.
    ///
    /// Typical use: half the layers on GPU 0, other half on GPU 1.
    pub fn pipeline_parallel(n_layers: usize, split: usize, dev_a: Device, dev_b: Device) -> Self {
        let devices: Vec<Device> = (0..n_layers)
            .map(|i| if i < split { dev_a.clone() } else { dev_b.clone() })
            .collect();
        Self { devices: devices.into() }
    }

    /// Get the device for `layer_id`.
    ///
    /// Falls back to device 0 if the index is out of range.
    pub fn get(&self, layer_id: usize) -> &Device {
        let idx = layer_id.min(self.devices.len() - 1);
        &self.devices[idx]
    }

    /// Number of layers this map covers explicitly.
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// True only if the map was built with zero devices (impossible via public API).
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// True if every layer is mapped to the same device.
    pub fn is_uniform(&self) -> bool {
        let first = &self.devices[0];
        self.devices.iter().all(|d| d.same_device(first))
    }

    /// Convenience: the single device when `is_uniform()` is true.
    /// Returns device 0 regardless.
    pub fn primary(&self) -> &Device {
        &self.devices[0]
    }
}

// ---------------------------------------------------------------------------
// DeviceMapBuilder — fluent constructor for tensor-parallel layouts
// ---------------------------------------------------------------------------

/// Fluent builder for constructing heterogeneous `DeviceMap` layouts.
///
/// ```no_run
/// use air_rs::device_map::DeviceMapBuilder;
/// use candle_core::Device;
///
/// let map = DeviceMapBuilder::new(32)
///     .layers(0..16, Device::cuda_if_available(0).unwrap())
///     .layers(16..32, Device::cuda_if_available(1).unwrap())
///     .build();
/// ```
pub struct DeviceMapBuilder {
    devices: Vec<Option<Device>>,
}

impl DeviceMapBuilder {
    /// Create a builder for a model with `n_layers` layers.
    ///
    /// All layers default to `None` (unassigned). Call `layers()` to assign.
    pub fn new(n_layers: usize) -> Self {
        Self { devices: vec![None; n_layers] }
    }

    /// Assign `device` to all layers in `range`.
    pub fn layers(mut self, range: std::ops::Range<usize>, device: Device) -> Self {
        for i in range {
            if i < self.devices.len() {
                self.devices[i] = Some(device.clone());
            }
        }
        self
    }

    /// Finalise. Unassigned layers fall back to `Device::Cpu`.
    pub fn build(self) -> DeviceMap {
        let resolved: Vec<Device> = self.devices
            .into_iter()
            .map(|d| d.unwrap_or(Device::Cpu))
            .collect();
        DeviceMap::from_devices(resolved)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn uniform_map_all_cpu() {
        let map = DeviceMap::uniform(32, Device::Cpu);
        assert_eq!(map.len(), 32);
        assert!(map.is_uniform());
        assert!(matches!(map.get(0), Device::Cpu));
        assert!(matches!(map.get(31), Device::Cpu));
    }

    #[test]
    fn uniform_oob_clamps_to_last() {
        let map = DeviceMap::uniform(4, Device::Cpu);
        // Query beyond range returns last device (index 3)
        assert!(matches!(map.get(100), Device::Cpu));
    }

    #[test]
    fn pipeline_parallel_layout() {
        let map = DeviceMap::pipeline_parallel(8, 4, Device::Cpu, Device::Cpu);
        assert_eq!(map.len(), 8);
        // Both halves are Cpu in test, so uniform
        assert!(map.is_uniform());
    }

    #[test]
    fn builder_assigns_ranges() {
        let map = DeviceMapBuilder::new(4)
            .layers(0..2, Device::Cpu)
            .layers(2..4, Device::Cpu)
            .build();
        assert_eq!(map.len(), 4);
        assert!(matches!(map.get(0), Device::Cpu));
        assert!(matches!(map.get(3), Device::Cpu));
    }

    #[test]
    fn builder_unassigned_falls_back_to_cpu() {
        let map = DeviceMapBuilder::new(4)
            .layers(0..2, Device::Cpu)
            // layers 2 & 3 are unassigned
            .build();
        assert!(matches!(map.get(2), Device::Cpu));
    }

    #[test]
    fn candle_device_cpu_variant() {
        use crate::shared_buffer::ComputeBackend;
        let dev = candle_device(&ComputeBackend::Cpu).unwrap();
        assert!(matches!(dev, Device::Cpu));
    }

    #[test]
    fn candle_device_cuda_fallback() {
        use crate::shared_buffer::ComputeBackend;
        // Ordinal 999 — no such GPU, should return Cpu
        let dev = candle_device(&ComputeBackend::Cuda(999)).unwrap_or(Device::Cpu);
        assert!(matches!(dev, Device::Cpu));
    }

    #[test]
    fn device_map_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DeviceMap>();
    }
}

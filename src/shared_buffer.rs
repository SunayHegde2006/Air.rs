//! Platform-agnostic buffer and compute backend types.
//!
//! Always compiled — no feature gate. Used by `pipeline.rs` and the VRAM
//! arena on all supported platforms (Linux/CUDA, macOS/Metal, Windows/Vulkan).
//!
//! `metal_compute.rs` depends on nothing in this module; this module depends
//! on nothing in `metal_compute.rs`. See ADR-0005.

use std::sync::Arc;

// ---------------------------------------------------------------------------
// SharedBuffer
// ---------------------------------------------------------------------------

/// Platform-agnostic CPU↔GPU shared memory buffer.
///
/// Wraps a reference-counted byte slice so pipeline slots can hold
/// weight data without copying. Used by `pipeline.rs` on all platforms.
#[derive(Debug, Clone)]
pub struct SharedBuffer {
    /// Raw byte data (layer weight storage, activation buffers, etc).
    pub data: Arc<Vec<u8>>,
    /// Size in bytes.
    pub size: usize,
}

impl SharedBuffer {
    /// Create from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let size = data.len();
        Self { data: Arc::new(data), size }
    }

    /// Create a zero-initialised buffer of `size` bytes.
    pub fn zeroed(size: usize) -> Self {
        Self { data: Arc::new(vec![0u8; size]), size }
    }

    /// Borrow the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// ComputeBackend — canonical discriminant (ADR-0005)
// ---------------------------------------------------------------------------

/// Canonical compute backend discriminant.
///
/// Single source of truth across `ModelMux`, `drive_inquisitor`, and
/// `metal_compute`. Replaces two inconsistent enums previously in
/// `ucal.rs` and `drive_inquisitor.rs`. See ADR-0005.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    /// NVIDIA CUDA GPU at the given device ordinal.
    Cuda(usize),
    /// AMD ROCm / HIP GPU at the given device ordinal.
    Rocm(usize),
    /// Apple Metal (macOS / iOS, unified memory).
    Metal,
    /// Vulkan cross-platform GPU (Intel Arc, AMD, etc).
    Vulkan,
    /// CPU fallback — no discrete GPU.
    Cpu,
}

impl ComputeBackend {
    /// True if any GPU backend.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Device ordinal for multi-GPU selection (`None` for Metal/Vulkan/Cpu).
    pub fn device_ordinal(&self) -> Option<usize> {
        match self {
            Self::Cuda(n) | Self::Rocm(n) => Some(*n),
            _ => None,
        }
    }

    /// Short backend identifier string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "cuda",
            Self::Rocm(_) => "rocm",
            Self::Metal   => "metal",
            Self::Vulkan  => "vulkan",
            Self::Cpu     => "cpu",
        }
    }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(n) => write!(f, "cuda:{}", n),
            Self::Rocm(n) => write!(f, "rocm:{}", n),
            Self::Metal   => write!(f, "metal"),
            Self::Vulkan  => write!(f, "vulkan"),
            Self::Cpu     => write!(f, "cpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_buffer_roundtrip() {
        let buf = SharedBuffer::from_bytes(vec![1u8, 2, 3]);
        assert_eq!(buf.size, 3);
        assert_eq!(buf.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn shared_buffer_zeroed() {
        let buf = SharedBuffer::zeroed(16);
        assert_eq!(buf.size, 16);
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn compute_backend_display() {
        assert_eq!(ComputeBackend::Cuda(0).to_string(), "cuda:0");
        assert_eq!(ComputeBackend::Metal.to_string(), "metal");
        assert_eq!(ComputeBackend::Cpu.to_string(), "cpu");
    }

    #[test]
    fn compute_backend_ordinal() {
        assert_eq!(ComputeBackend::Cuda(2).device_ordinal(), Some(2));
        assert_eq!(ComputeBackend::Metal.device_ordinal(), None);
        assert_eq!(ComputeBackend::Cpu.device_ordinal(), None);
    }

    #[test]
    fn compute_backend_is_gpu() {
        assert!(ComputeBackend::Cuda(0).is_gpu());
        assert!(ComputeBackend::Metal.is_gpu());
        assert!(!ComputeBackend::Cpu.is_gpu());
    }
}

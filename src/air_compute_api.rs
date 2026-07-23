//! Uniform FFI C-API definitions for external runtimes (Vulkan/SYCL/Mojo).
//!
//! Provides platform-agnostic bindings to perform heavy math computations
//! on host-visible or device-local mapped unified memory buffers.

extern "C" {
    /// Compute Root Mean Square Normalization.
    ///
    /// # Safety
    /// Pointer inputs must point to valid arrays of size `size`.
    pub fn air_compute_rmsnorm(
        x: *mut f32,
        weight: *const f32,
        size: usize,
        eps: f32,
    );

    /// Compute Rotary Position Embedding (RoPE).
    ///
    /// # Safety
    /// Pointer input must point to valid array of size `dim * seq_len`.
    pub fn air_compute_rope(
        x: *mut f32,
        pos: usize,
        theta: f32,
        dim: usize,
        seq_len: usize,
    );

    /// Compute General Matrix Multiplication: `out = lhs @ rhs`.
    ///
    /// out: [m, n] (f32)
    /// lhs: [m, k] (f32)
    /// rhs: [k, n] (f32)
    ///
    /// # Safety
    /// All pointer inputs must point to allocated buffers layout-aligned with dimensions.
    pub fn air_compute_matmul(
        out: *mut f32,
        lhs: *const f32,
        rhs: *const f32,
        m: usize,
        n: usize,
        k: usize,
    );
}

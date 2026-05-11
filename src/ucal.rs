//! Compatibility shim — `ucal` re-exports from the split modules.
//!
//! `ucal.rs` has been split into:
//! - `shared_buffer.rs`  — platform-agnostic `SharedBuffer` + `ComputeBackend`
//! - `metal_compute.rs`  — Metal kernels, context, command encoding
//!
//! This shim keeps existing `use crate::ucal::*` imports compiling.
//! Migrate to the direct module paths when convenient. See ADR-0005.

pub use crate::shared_buffer::{ComputeBackend, SharedBuffer};
pub use crate::metal_compute::{
    CommandBuffer, CommandBufferEncoder, CommandOp, ExecutionHandle, ExecutionStatus,
    GridSize, MetalBuffer, MetalContext, MetalDtype, MetalError, MetalKernel,
    MetalKernelLibrary, MetalResult, ThreadgroupSize, TieredExecutor,
};

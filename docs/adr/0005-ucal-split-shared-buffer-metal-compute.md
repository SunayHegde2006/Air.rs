# ADR-0005: Split `ucal.rs` into `shared_buffer.rs` + `metal_compute.rs`, consolidate `ComputeBackend`

**Date:** 2026-05-03
**Status:** Accepted

## Context

`src/ucal.rs` (1,629 lines) has two structural problems:

1. **Platform-agnostic types trapped in a Metal-specific file.** `SharedBuffer` is imported by `pipeline.rs` on all platforms without a feature gate. Yet `ucal.rs` is documented as "Metal-only" and contains Metal FFI types (`MTLDevice`, MSL shader sources). Non-Mac builds compile 1,600 lines of Metal infrastructure just to access `SharedBuffer`.

2. **Duplicate `ComputeBackend` enum.** `ucal::ComputeBackend { Metal, Remote, Cpu }` and `drive_inquisitor::ComputeBackend { Cuda, Metal, Cpu, Vulkan, Rocm, Npu }` are separate enums with the same name and overlapping but inconsistent variant sets. Code that needs to select between CUDA and Metal backends cannot import a single canonical type.

## Decision

### Delete `src/ucal.rs`

Split into two focused modules.

### New: `src/shared_buffer.rs` (always compiled — no feature gate)

Contents:
- `MetalDtype` — dtype enum for Metal buffer element types
- `MetalBuffer` — GPU buffer wrapper with typed element accessors
- `SharedBuffer` — CPU/GPU shared memory with reference counting (`Arc<MetalBuffer>`)
- `ThreadgroupSize`, `GridSize` — compute dispatch geometry types
- **Canonical `ComputeBackend` enum** (replaces both existing enums):

```rust
pub enum ComputeBackend {
    Cuda(usize),    // CUDA, device ordinal
    Rocm(usize),    // ROCm / HIP, device ordinal
    Metal,          // Apple Metal (macOS / iOS)
    Vulkan,         // Vulkan (cross-platform)
    Cpu,            // CPU fallback
}

impl ComputeBackend {
    pub fn is_gpu(&self) -> bool;
    pub fn device_ordinal(&self) -> Option<usize>;
    pub fn name(&self) -> &'static str;
}
```

### New: `src/metal_compute.rs` — `#[cfg(feature = "metal")]`

Contents:
- `MetalError`, `MetalResult<T>` — Metal-specific error handling
- `MetalKernel` — compiled MSL compute pipeline handle
- `MetalKernelLibrary` — static MSL shader source strings (softmax, rms_norm, attention, etc.)
- `MetalContext` — device + command queue + pipeline cache
- `CommandOp`, `CommandBufferEncoder`, `CommandBuffer` — command recording and submission
- `ExecutionStatus`, `ExecutionHandle`, `TiledExecutor` — execution lifecycle

### Migration: `drive_inquisitor.rs`

```rust
// Before:
pub enum ComputeBackend { Cuda, Metal, Cpu, Vulkan, Rocm, Npu }

// After:
use crate::shared_buffer::ComputeBackend;
// all existing match arms updated to new variant names
```

### `lib.rs`

```rust
// Before:
pub mod ucal;

// After:
pub mod shared_buffer;
#[cfg(feature = "metal")]
pub mod metal_compute;
```

## Consequences

**Positive:**
- `SharedBuffer` is accessible on all platforms without compiling Metal infrastructure — correct for Windows, Linux, non-Mac builds
- Single canonical `ComputeBackend` — `ModelMux` (issue #4), `drive_inquisitor`, `metal_compute`, and `strix/backend_detect.rs` all import one type
- Metal kernel infrastructure is properly feature-gated — `cargo check` on non-Mac without `--features metal` no longer compiles MSL shader strings
- `shared_buffer.rs` is testable on any platform without Metal hardware

**Negative:**
- All imports of `ucal::SharedBuffer` → `shared_buffer::SharedBuffer` — mechanical find-replace. One caller today (`pipeline.rs`).
- `ucal::ComputeBackend` and `drive_inquisitor::ComputeBackend` match arms in all call sites must be updated to new variant names. Grep-driven, no logic change.

## Alternatives rejected

**Keep `ucal.rs` as-is, add re-exports:** A `pub use ucal::SharedBuffer` in `shared_buffer.rs` avoids the rename but keeps the Metal infrastructure compiling on non-Mac. Rejected — the feature-gating problem remains.

**Consolidate in a later refactor:** Future contributors adding `ModelMux` backend selection would hit the duplicate `ComputeBackend` immediately. Consolidating now costs one extra pass; leaving it costs N future readers re-discovering the collision. Rejected.

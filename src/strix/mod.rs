//! # STRIX — Streamed Tensor Residence & Intelligent eXchange
//!
//! GPU offloading protocol for Air.rs, enabling 70B+ parameter models
//! on consumer hardware by intelligently managing a 3-tier memory
//! hierarchy (VRAM → RAM → Storage).
//!
//! ## Module layout
//!
//! | Module              | Phase | Purpose |
//! |---------------------|-------|---------|
//! | `types`             | 1     | Core type definitions (GpuPtr, DType, ResidencyState, TensorClass) |
//! | `meta`              | 1     | Per-tensor metadata (`TensorMeta`) |
//! | `score`             | 1     | Residency scoring function R(t,τ) |
//! | `config`            | 1     | Runtime configuration (`StrixConfig`) |
//! | `hal`               | 1     | Hardware Abstraction Layer trait contracts |
//! | `registry`          | 2     | Central tensor tracking (`TensorRegistry`) |
//! | `arena`             | 2     | VRAM budget allocation (`VramArena`) |
//! | `scheduler`         | 2     | Residency tick loop (`ResidencyScheduler`) |
//! | `io_engine`         | 2     | Priority async I/O queue (`IoEngine`) |
//! | `cold_boot`         | 2     | Staged cold-start loading (`ColdBootSequence`) |
//! | `cpu_hal`           | 3     | CpuHal — GpuHal backend using host memory |
//! | `std_storage_hal`   | 3     | StdStorageHal — StorageHal using std::fs |
//! | `bridge`            | 3     | StrixBridge — high-level orchestrator |
//! | `streamer_adapter`  | 3     | WeightStreamer → STRIX adapter |
//! | `compat`            | 4     | GGUF parser, tensor name normalizer, arch detection |
//! | `session`           | 4     | StrixSession — Air.rs integration API |
//! | `gpu_alloc`         | 5     | RAII VRAM allocation + DMA staging buffers |
//! | `vram_pressure`     | 5     | 5-level VRAM pressure manager + KV cache budget |
//! | `ram_pool`          | 5     | Recycling RAM buffer pool for staging transfers |
//! | `scheduler_thread`  | 5     | Dedicated 2ms scheduler tick thread |
//! | `security`          | 6     | SecureAllocator, ShardedRwLock, BoundsCheckedPtr, OwnerToken, audit log |
//! | `gpu_tensor_view`   | 6     | Lifetime-bound zero-copy VRAM tensor view |
//! | `cuda_hal`          | 6     | CUDA Runtime API HAL backend (feature-gated) |
//! | `vulkan_hal`        | 6     | Vulkan 1.2 HAL backend (feature-gated) |
//! | `metal_hal`         | 6     | Metal framework HAL backend (feature-gated, macOS only) |
//! | `rocm_hal`          | 6     | ROCm/HIP HAL backend (feature-gated, AMD GPUs) |
//! | `async_io`          | 6     | Platform async I/O: io_uring (Linux) / IOCP (Windows) |
//! | `backend_detect`    | 6     | Sub-100ms backend detection + ranking |
//! | `execution_cursor`  | 8     | ExecutionCursor + MoE expert activation hook |
//! | `gpu_direct`        | 8     | GPUDirect Storage NVMe→GPU DMA integration |
//! | `cufile_ffi`        | 9     | cuFile API FFI bindings for GDS (cuda+linux) |
//! | `multi_gpu`         | 9     | Multi-GPU topology, NVLink, shard strategies |
//!
//! ## Status
//!
//! Phases 1–8 complete. All modules are testable with in-memory mocks
//! and CPU-only HAL backends — no real GPU or storage driver needed.

// Phase 1: Foundation scaffold
pub mod types;
pub mod meta;
pub mod score;
pub mod config;
pub mod hal;

// Phase 2: Runtime modules
pub mod registry;
pub mod arena;
pub mod scheduler;
pub mod io_engine;
pub mod cold_boot;

// Phase 3: HAL backends + integration
pub mod cpu_hal;
pub mod std_storage_hal;
pub mod bridge;
pub mod streamer_adapter;

// Phase 4: Model compatibility + session API
pub mod compat;
pub mod session;

// Phase 5: Memory management + threading
pub mod gpu_alloc;
pub mod vram_pressure;
pub mod ram_pool;
pub mod scheduler_thread;

// Phase 6: GPU backends, async I/O, tensor views, security
pub mod security;
pub mod gpu_tensor_view;
#[cfg(feature = "cuda")]
pub mod cuda_hal;
#[cfg(feature = "vulkan")]
pub mod vulkan_hal;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_hal;
#[cfg(feature = "rocm")]
pub mod rocm_hal;
pub mod async_io;
pub mod backend_detect;

// Phase 7: Multi-format readers, mmap, tests, validation
pub mod safetensors;
pub mod pytorch;
pub mod onnx;
pub mod mmap_storage;
#[cfg(test)]
pub mod integration_tests;
#[cfg(test)]
pub mod chaos_tests;
#[cfg(test)]
pub mod benchmarks;
#[cfg(test)]
pub mod e2e_validation;

// Phase 8: ExecutionCursor (MoE), GPUDirect Storage, serde config
pub mod execution_cursor;
pub mod gpu_direct;

// Phase 9: Hardware polish — cuFile FFI, Multi-GPU, VRAM zeroing verification
pub mod cufile_ffi;
pub mod multi_gpu;

// Re-export the most commonly used types at crate level.
pub use types::{DType, GpuPtr, ResidencyState, TensorClass, TensorId, TierId};
pub use meta::TensorMeta;
pub use score::ScoreWeights;
pub use config::StrixConfig;
pub use registry::{TensorRegistry, RegistryStats, ResidencyGuard};
pub use arena::{VramArena, Allocation};
pub use scheduler::{ResidencyScheduler, SchedulerAction};
pub use io_engine::{IoEngine, IoRequest, IoTicket, IoCompletion, IoPriority};
pub use cold_boot::{ColdBootSequence, ColdBootPlan, ColdBootStep};
pub use cpu_hal::CpuHal;
pub use std_storage_hal::StdStorageHal;
pub use bridge::{StrixBridge, BridgeError, BridgeStats};
pub use streamer_adapter::{StrixStreamerAdapter, LayerMapping};
pub use compat::{
    GgufHeader, GgufMetadata, GgufTensorInfo, GgufModel, ModelArchitecture,
    CompatError, normalize_tensor_name, classify_tensor, parse_gguf_header,
    parse_metadata_kv, parse_tensor_index, parse_gguf_model,
    ModelFormat, UnifiedTensorInfo, UnifiedModel, detect_format, parse_model_file,
};
pub use session::{StrixSession, SessionGuard, SessionState, SessionError};
pub use gpu_alloc::{GpuAllocation, PinnedBuffer};
pub use vram_pressure::{VramPressureManager, PressureLevel, PressureAction, kv_cache_budget};
pub use ram_pool::{RamPool, RamBuffer};
pub use scheduler_thread::{SchedulerThread, SchedulerWork, SchedulerStats};
pub use security::{SecureAllocator, BoundsCheckedPtr, BoundsError, OwnerToken, ShardedRwLock, SecurityAuditLog, SecurityEvent};
pub use gpu_tensor_view::{GpuTensorView, ViewError};
pub use async_io::PlatformStorageHal;
pub use mmap_storage::MmapStorageHal;
pub use config::ConfigError;
pub use execution_cursor::{
    ExecutionCursor, LayerPhase, ExpertActivation,
    ExpertActivationHook, default_expert_hook,
};
pub use gpu_direct::{GdsCapability, GdsStorageHal, GdsTransfer, GdsTransferStatus, GdsStats, TransferMethod, PinnedHostBuffer};
pub use backend_detect::{
    BackendDetector, DetectionResult, GpuBackendKind, StorageBackendKind,
    GpuProbeResult, StorageProbeResult,
};
pub use cufile_ffi::CUfileStatus;
pub use multi_gpu::{GpuTopology, ShardStrategy, PeerTransfer, PeerTransferStatus, Interconnect};

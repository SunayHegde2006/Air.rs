# ADR-0002: Inject `candle_core::Device` into `InferenceGenerator` — extract device selection to `GpuTopology`

**Date:** 2026-05-03
**Status:** Accepted

## Context

`InferenceGenerator::new()` calls `Device::new_cuda(0)` directly inside its constructor, with a silent fallback to `Device::Cpu`. This means:

1. **No test seam** — unit tests of `generate_step` require real CUDA hardware; `Device::Cpu` cannot be injected.
2. **Ignores `GpuTopology`** — `gpu_pipeline.rs` already has `GpuTopology::discover()` which probes all available GPUs, but `InferenceGenerator` never consults it. On a multi-GPU machine the engine always uses GPU:0 regardless of load.
3. **Blocks `ModelMux`** — the upcoming Model Multiplexer (issue #4) needs to pin each Model Slot to a specific GPU ordinal. With device selection hardwired inside the constructor, there is no interface to set the ordinal per slot.
4. **Duplicated selection logic** — `generator.rs:80`, `gpu_pipeline.rs:48`, and `drive_inquisitor.rs` each have independent device probing code.

## Decision

### 1. `InferenceGenerator::new` accepts `device: candle_core::Device`

```rust
pub fn new(
    config: ModelConfig,
    sampler_config: SamplerConfig,
    device: Device,           // ← injected, not constructed here
) -> Result<Self>
```

The constructor no longer calls `Device::new_cuda`. Callers are responsible for device selection.

### 2. `GpuTopology` becomes the single device-selection point

Two new methods added to `GpuTopology`:

```rust
impl GpuTopology {
    /// Return the Device at the given ordinal (0-indexed across discovered GPUs).
    /// Returns None if ordinal is out of range.
    pub fn device_at(&self, ordinal: usize) -> Option<Device>;

    /// Convenience alias: device_at(0), or Device::Cpu if no GPU found.
    pub fn best_device(&self) -> Device;
}
```

Usage pattern at all callsites:
```rust
let topo = GpuTopology::discover();
let device = topo.device_at(gpu_id).unwrap_or(Device::Cpu);
let generator = InferenceGenerator::new(config, sampler_cfg, device)?;
```

### 3. Python API gains `gpu_id` parameter

```python
engine = Engine.from_gguf("model.gguf", gpu_id=0)   # default: 0
engine = Engine.from_gguf("model.gguf", gpu_id=1)   # route to second GPU
```

`PyEngine::from_gguf` calls `GpuTopology::discover().device_at(gpu_id)`.

### 4. `ModelMux::load` passes ordinal per slot

```rust
mux.load("model-a.gguf", gpu_id: 0)?;
mux.load("model-b.gguf", gpu_id: 0)?;  // same GPU, VRAM-partitioned
```

Ordinal is stored on the `EngineSlot`; used when constructing the slot's `InferenceGenerator`.

## Consequences

**Positive:**
- `generate_step` is now testable on CPU without any GPU hardware — inject `Device::Cpu` in tests
- Multi-GPU routing is possible: `device_at(1)` pins a Model Slot to GPU:1
- `ModelMux` per-slot GPU pinning works via the ordinal parameter
- Single device-selection path (`GpuTopology`) — `generator.rs` no longer has its own probing logic

**Negative:**
- `InferenceGenerator::new` signature changes — all three callsites (`python.rs`, `api.rs`, integration tests) must be updated. One-time migration.

## Deferred

`dyn GpuHal` as the injection type (instead of `candle_core::Device`) is deferred to v0.4.0. It would enable a fully uniform HAL dispatch across CUDA/ROCm/Metal for tensor allocation, but requires threading the HAL through all internal tensor ops where `Device` is currently used — a larger, separate refactor.

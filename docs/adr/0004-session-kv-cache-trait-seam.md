# ADR-0004: `SessionKvCache` trait seam — decouple generator from `KvCacheManager`

**Date:** 2026-05-03
**Status:** Accepted

## Context

`InferenceGenerator` holds a concrete `KvCacheManager` field and calls `load_to_device(layer_id)` / `save_from_device(layer_id, k, v)` / `clear()` directly. There is no trait seam.

Two upcoming features require intercepting these calls:
1. **Prefix KV Cache (issue #3)** — `PrefixAwareSessionCache` must intercept `load_layer` to check the prefix hash before falling through to the real KV storage.
2. **M.I.S.T. v4 (v0.4.0)** — `IsoQuantSessionCache` must intercept `save_layer` to apply TriAttention + IsoQuant-Fast + TurboQuant quantization, replacing the current Q8 path.

Without a trait seam, both features require patching inside `KvCacheManager` itself — concentrated change that breaks locality.

A `SlotKvCache` trait already exists in `batching/kernel.rs` with a `(slot, layer_id)` interface designed for the ARB Scheduler's multi-slot, multi-sequence view. Forcing this onto the generator (which never has a non-zero slot) would create a false invariant.

## Decision

### Define `SessionKvCache` in `src/kv_cache.rs`

```rust
pub trait SessionKvCache: Send {
    /// Load K/V tensors for `layer_id` to the compute device.
    fn load_layer(
        &self,
        layer_id: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)>;

    /// Save updated K/V tensors for `layer_id` back to storage.
    fn save_layer(
        &mut self,
        layer_id: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()>;

    /// Clear all cached state (called between conversations / on reset).
    fn reset(&mut self);

    /// Current cached sequence length at `layer_id`.
    /// Used by PrefixAwareSessionCache to determine cache hit depth.
    fn seq_len(&self, layer_id: usize) -> usize;
}
```

### `KvCacheManager` implements `SessionKvCache`

Method mapping:
- `load_layer(l)` → `self.load_to_device(l)` (rename at trait boundary)
- `save_layer(l, k, v)` → `self.save_from_device(l, k, v)`
- `reset()` → `self.clear()`
- `seq_len(l)` → `self.seq_len(l)` (already exists)

### `InferenceGenerator` holds `Box<dyn SessionKvCache>`

```rust
pub struct InferenceGenerator {
    kv_cache: Box<dyn SessionKvCache>,   // was: KvCacheManager
    blocks: Vec<Box<dyn TransformerBlock>>,
    ...
}
```

Constructor: `InferenceGenerator::new(config, sampler_cfg, device, kv_cache: Box<dyn SessionKvCache>)`. Default callsite passes `Box::new(KvCacheManager::new_for_device(device.clone(), config.n_layers))`.

### `PrefixAwareSessionCache` — composable wrapper

```rust
pub struct PrefixAwareSessionCache {
    inner: Box<dyn SessionKvCache>,
    cache: PrefixCache,   // content-addressed block pool (issue #3)
}

impl SessionKvCache for PrefixAwareSessionCache {
    fn load_layer(&self, layer_id: usize) -> Result<...> {
        // 1. check prefix cache → on hit, return cached blocks
        // 2. on miss, delegate to self.inner.load_layer(layer_id)
    }
    fn save_layer(&mut self, layer_id: usize, k, v) -> Result<()> {
        // 1. store in prefix cache if prefix segment
        // 2. delegate to self.inner.save_layer(layer_id, k, v)
    }
    fn reset(&mut self) { self.inner.reset(); }
    fn seq_len(&self, layer_id: usize) -> usize { self.inner.seq_len(layer_id) }
}
```

### `#[cfg(test)] MockSessionKvCache`

Returns empty `(None, None)` for `load_layer`, no-ops for `save_layer`/`reset`, `0` for `seq_len`. Used in `generate_step` unit tests alongside `MockTransformerBlock` (ADR-0001) — full layer loop testable without GPU.

### `SlotKvCache` unchanged

`SlotKvCache` in `batching/kernel.rs` retains its multi-slot, multi-sequence interface. It serves the ARB Scheduler's batching path. The two traits are intentionally different contracts for different module scopes.

## Consequences

**Positive:**
- `PrefixAwareSessionCache` is a composable wrapper — one adapter, no `KvCacheManager` surgery required
- M.I.S.T. v4 quantization path (`IsoQuantSessionCache`) is a second adapter — swap at construction time
- `generate_step` fully testable without GPU: inject `MockTransformerBlock` + `MockSessionKvCache`
- `SlotKvCache` and `SessionKvCache` are honest about their scope — no forced slot dimension on single-session code

**Negative:**
- `InferenceGenerator::new` signature gains a fourth parameter (`kv_cache: Box<dyn SessionKvCache>`) — all callsites updated. One-time migration.
- Dynamic dispatch adds one vtable call per `load_layer`/`save_layer` per layer per step — negligible vs tensor compute time.

## Alternatives rejected

**`KvCacheManager` implements `SlotKvCache` with `slot=0`:** Adds a meaningless slot parameter to every generator call. Creates a false invariant (slot is always 0 in the single-session path). The seam interface complexity matches the caller complexity — shallow, not deep. Rejected.

**Patch `PrefixCache` inside `KvCacheManager`:** No seam. Adding M.I.S.T. v4 later requires patching again. Rejected — locality is broken.

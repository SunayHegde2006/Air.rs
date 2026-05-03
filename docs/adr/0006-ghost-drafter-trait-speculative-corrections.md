# ADR-0006: `GhostDrafter` trait seam + speculative decoding protocol corrections

**Date:** 2026-05-03
**Status:** Accepted

## Context

`air_rs_speculative_decoding_protocol.md` defines a complete self-speculative layer-skipping implementation. The design is sound. Five gaps were identified during architectural review:

1. **EOS mid-draft** — algorithm terminates on `output.last() == EOS` after extending, which emits tokens past EOS if EOS lands inside accepted draft tokens.
2. **`ThreadRng` is `!Send`** — `Speculative` struct cannot cross `tokio` await points.
3. **Draft sampling temperature not specified** — losslessness proof requires draft uses same `SamplerConfig` as target.
4. **`truncate_to` missing from `SessionKvCache`** — rollback requires downcasting; kills the ADR-0004 seam.
5. **No trait seam for draft strategy** — `VramResidentDrafter` (small models, section 9) and `MockDrafter` (tests) have no interface to implement.

## Decision

### Gap 1 — EOS termination

In `rejection_sample`, break the acceptance loop the moment an accepted token is EOS:

```rust
for i in 0..k {
    // ... accept/reject logic ...
    accepted.push(token);
    if token == eos_token { break; }   // ← inner break, not outer
}
// bonus token emitted only if accepted.last() != EOS
if accepted.last() != Some(&eos_token) && accepted.len() == k {
    accepted.push(bonus_from(verify.logits[k]));
}
```

### Gap 2 — RNG type

```rust
// Before (in protocol doc):
rng: rand::rngs::ThreadRng,   // !Send

// After:
rng: rand::rngs::SmallRng,    // Send + seeded at construction
// construction: SmallRng::from_rng(rand::thread_rng()).unwrap()
```

### Gap 3 — Draft sampling config

`SpeculativeConfig` is amended:

```rust
pub struct SpeculativeConfig {
    pub sampler: SamplerConfig,   // ← injected from request; used by BOTH draft and target
    pub draft_layer_ratio: f32,
    pub lookahead_k_init: usize,
    // ... rest unchanged
}
```

The draft pass calls `sample(&logits, &self.config.sampler)` — same temperature/top_p as target. This is required by Leviathan et al. (2023) Theorem 1. Greedy draft (argmax) is explicitly **deferred** to a config flag in a future version.

### Gap 4 — `SessionKvCache::truncate_to` (amends ADR-0004)

`truncate_to` is added to the `SessionKvCache` trait:

```rust
pub trait SessionKvCache: Send {
    fn load_layer(&self, layer_id: usize) -> Result<(Option<Tensor>, Option<Tensor>)>;
    fn save_layer(&mut self, layer_id: usize, k: &Tensor, v: &Tensor) -> Result<()>;
    fn reset(&mut self);
    fn seq_len(&self, layer_id: usize) -> usize;
    fn truncate_to(&mut self, seq_len: usize);   // ← O(1) pointer rollback for speculative rejection
}
```

`KvCacheManager::truncate_to(n)` sets `seq_len_used[layer] = n` for all layers. `MockSessionKvCache::truncate_to` is a no-op. No heap allocation, no memcopy per Pointer-truncation rollback (protocol §4.5).

### Gap 5 / Candidate #6 — `GhostDrafter` trait

```rust
pub trait GhostDrafter: Send + Sync {
    /// Propose k draft tokens for the given context.
    /// Returns token IDs + per-token probability distributions [k, vocab_size].
    fn draft_pass(
        &mut self,
        context: &[u32],
        k: usize,
        sampler: &SamplerConfig,
    ) -> Result<DraftResult>;

    /// Called after rejection sampling — advance draft KV to the accepted prefix.
    fn on_accept(&mut self, n_accept: usize, context_len: usize);

    /// Reset draft KV cache (called at start of each round before draft_pass).
    fn reset(&mut self);
}
```

`Speculative` replaces its own `draft_pass` method with `Box<dyn GhostDrafter>`:

```rust
pub struct Speculative {
    config: SpeculativeConfig,
    drafter: Box<dyn GhostDrafter>,     // ← injected, not constructed here
    target_kv: Box<dyn SessionKvCache>,
    manifest: DraftManifest,
    adaptive_k: AdaptiveK,
    rng: SmallRng,
}
```

### Three `GhostDrafter` adapters

| Adapter | When | Wraps |
|---|---|---|
| `StreamingLayerSkipDrafter` | Always (v0.3.0) | Streamer + draft `SessionKvCache`; implements protocol §8 exactly |
| `VramResidentDrafter` | Models fitting in VRAM (v0.4.0+) | Full in-VRAM weight cache; no NVMe streaming in draft pass |
| `MockDrafter` (`#[cfg(test)]`) | Unit tests | Returns canned `DraftResult` — no GPU, no GGUF, no streaming |

### `stream_layers` primitive exposed in `generator.rs`

`StreamingLayerSkipDrafter::draft_pass` needs to stream **only the draft layer offsets** through the existing triple-buffer pipeline. `generator.rs` exposes:

```rust
pub fn stream_layers(
    &mut self,
    layer_offsets: &[u64],          // only draft layer offsets from DraftManifest
    input: &[u32],
    kv: &mut dyn SessionKvCache,
    device: &Device,
) -> Result<Tensor>                 // final hidden state
```

`generate_step` is refactored to call `stream_layers` internally — no behaviour change for the target path.

### `DraftManifest` location

- **`DraftManifest` struct + `load_or_derive_manifest`** → `src/speculative.rs`
- **`DraftManifestBuilder`** → `src/manifest.rs` (reads GGUF byte offsets, selects every Nth layer)
- **Sidecar path**: `{gguf_path}.draft.json` — never tracked in git (`.gitignore` excludes `*.draft.json`)

## Consequences

**Positive:**
- `rejection_sample` is testable via `MockDrafter` + `MockSessionKvCache` + `Device::Cpu` — no GPU
- `VramResidentDrafter` slots in as a drop-in adapter when 7B models fit in VRAM
- EOS correctness: tokens past EOS never emitted
- Losslessness guarantee preserved: same sampler for draft and target
- `truncate_to` on trait means rollback works through the seam without downcasting

**Negative:**
- `SessionKvCache` trait gains a 5th method (`truncate_to`) — all impls must add it. Mechanical change.
- `Speculative::new` takes `Box<dyn GhostDrafter>` — construction site must build the correct drafter. More verbose callsite.

## Protocol document status

`air_rs_speculative_decoding_protocol.md` remains the authoritative algorithm specification. This ADR documents only the **implementation interface decisions** (trait seams, type corrections) that layer on top of the protocol. The protocol's algorithm (§8) is unchanged except Gap 1 (EOS inner break).

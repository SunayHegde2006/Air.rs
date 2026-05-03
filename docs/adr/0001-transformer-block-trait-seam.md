# ADR-0001: Extract `TransformerBlock` trait seam from `ops.rs` / `model.rs`

**Date:** 2026-05-03
**Status:** Accepted

## Context

`src/ops.rs` exposes 44 `pub fn` in a flat namespace. `model::transformer_block` receives a `ModelConfig` and performs three runtime `match` dispatches on `norm_type`, `ffn_type`, and `sliding_window` to select the correct operation variant per call. Every new architecture requires adding arms to all three matches.

This is a **shallow module**: the complexity of "which attention pattern does this architecture use?" leaks through the interface into every caller. Adding new architectures (MLA, Mamba, RWKV) and the v0.3.0 `ModelMux` tick loop both require editing the same dispatch matches. `generate_step` is untestable without a real GGUF + GPU.

## Decision

Introduce a `TransformerBlock` trait in a new `src/blocks/` module:

```rust
pub trait TransformerBlock: Send + Sync {
    fn forward(
        &self,
        hidden: &Tensor,
        weights: &LayerWeights,   // streamed per-step by S.L.I.P., not owned
        kv: (Option<&Tensor>, Option<&Tensor>),
        pos: usize,
        rope: Option<&RopeCache>,
    ) -> Result<(Tensor, Tensor, Tensor)>;
}
```

**Module layout:**
```
src/blocks/
  mod.rs       ← pub trait TransformerBlock; pub fn build_blocks(config) -> Vec<Box<dyn TransformerBlock>>; #[cfg(test)] MockTransformerBlock
  llama.rs     ← LlamaBlock       (RMSNorm + SwiGLU + full GQA)
  mistral.rs   ← MistralSwaBlock  (RMSNorm + SwiGLU + sliding-window GQA)
  gemma.rs     ← GemmaBlock       (GemmaRMSNorm + GeGLU + full GQA)
  falcon.rs    ← FalconBlock      (LayerNorm + DenseMLP + parallel attn/FFN)
  phi3.rs      ← Phi3FullBlock, Phi3SwaBlock (partial RoPE, alternating per layer)
```

`InferenceGenerator` replaces the per-step `model::transformer_block(config)` call with:
```rust
struct InferenceGenerator {
    blocks: Vec<Box<dyn TransformerBlock>>,   // length = n_layers
    ...
}
// constructed at: InferenceGenerator::new() → blocks::build_blocks(&config)
// used at:        generate_step → block.forward(...)
```

**`ops.rs` visibility changes:**
- `pub fn softmax`, `pub fn zeroed` — remain `pub` (called from `speculative.rs`, external callers)
- All architecture-specific fns (`silu_ffn`, `geglu_ffn`, `grouped_query_attention`, `sliding_window_gqa`, `flash_grouped_query_attention`, `rms_norm_gemma`, `apply_alibi`, `parallel_attn_ffn`, `fp4_attention`, etc.) — become `pub(crate)`, used only by block impls
- `model::transformer_block` free fn — removed

**Test infrastructure:**
`#[cfg(test)] pub struct MockTransformerBlock` in `blocks/mod.rs` returns identity tensors. Used by `generate_step` unit tests and future `ModelMux` tick-loop tests (issue #4).

## Consequences

**Positive:**
- Adding a new architecture (MLA, Mamba, Kimi) requires one new file in `blocks/`, zero changes to `generate_step` or `model.rs`
- `generate_step` is testable without GPU hardware via `MockTransformerBlock`
- `ModelMux` tick loop (issue #4) calls `block.forward()` uniformly without knowing architecture
- `ops.rs` public surface shrinks from 44 to 2 exported names; callers have a much smaller interface to understand

**Negative:**
- Dynamic dispatch (`dyn TransformerBlock`) adds one vtable indirection per layer per step. Negligible vs layer compute time (~microseconds per call vs ~milliseconds per layer)
- Initial migration touches `generator.rs`, `model.rs`, `model_variant.rs` — one-time cost

## Alternatives rejected

**Enum dispatch** (`match block_type { Llama(b) => b.forward(...), ... }`): requires touching the enum for every new architecture. Rejected — same coupling as the match arms in `transformer_block`, just moved.

**Keep flat in `model.rs`**: no locality gain. Every architecture variation still lives in one file. Rejected.

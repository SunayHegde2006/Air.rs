# Air.rs Domain Glossary

## Core Concepts

**Inference Engine** — The top-level runtime that loads a model and generates tokens. The public face of the system from a caller's perspective. Currently realised by `InferenceGenerator` (Rust) and `PyEngine` (Python).

**Model Slot** — A single loaded model with its own weight budget, KV cache manager, prefix cache, and CUDA stream pair. The unit of concurrent serving in v0.3.0+.

**Tick Loop** — The decode step that advances every active Model Slot by one token per cycle. True interleaving: all slots emit a token before any slot advances to the next step.

**S.L.I.P. (Strix Layer-Interleaved Pipeline)** — The weight-streaming pipeline that overlaps layer prefetch (stream 1) with layer compute (stream 0), enabling models larger than residual VRAM to run without stalling.

**VRAM Arena** — The VRAM allocator (`VramArena`) that tracks byte allocation across all Model Slots and enforces the 80% safety cap.

**KV Cache** — Per-sequence key/value tensor storage managed by `KvCacheManager`. Hot tier in VRAM, warm tier in RAM (compressed).

**Prefix Cache** — A content-addressed pool of KV blocks indexed by token-hash chunks. Allows sessions with identical system prompts to skip prefill. Per-Model Slot.

**M.I.S.T. (Memory-Interleaved Streaming Tiers)** — The tiered KV compression protocol. v3: QJL 1-bit keys + Q8 values. v4 (planned): TriAttention + IsoQuant-Fast + TurboQuant TQ4_0.

**CompressionScheme** — A tag on each KV block indicating which M.I.S.T. version compressed it. Prevents stale blocks from being reused after a pipeline upgrade.

**HERMES Eviction** — The KV cache eviction policy: hierarchical importance scoring combining recency, attention density, and position. Lives in `kv_tier.rs`.

**Ghost Drafting** — Speculative decoding using a small "ghost" model (1B/3B) to draft tokens that the main model verifies. Reduces decode latency by ~2–3×.

**HAL (Hardware Abstraction Layer)** — The `GpuHal` / `StorageHal` traits in `strix/`. Adapters: `CudaHal`, `MetalHal`, `RocmHal`, `VulkanHal`, `CpuHal`, `StdStorageHal`.

**TransformerBlock** — A trait (`src/blocks/`) with one method: `forward(hidden, weights, kv, pos, rope) → (hidden, new_k, new_v)`. One concrete impl per architecture-variant per layer. `InferenceGenerator` holds `Vec<Box<dyn TransformerBlock>>` of length `n_layers`; `blocks::build_blocks(config)` constructs the vec at model-load time. Weights are NOT owned by the block — they are streamed per-step by S.L.I.P. and passed as a parameter. Concrete impls: `LlamaBlock`, `MistralSwaBlock`, `GemmaBlock`, `FalconBlock`, `Phi3FullBlock`, `Phi3SwaBlock`. See ADR-0001.

**Block Factory** — `blocks::build_blocks(config: &ModelConfig) -> Vec<Box<dyn TransformerBlock>>`. The single construction point for the per-layer block vec. Called inside `InferenceGenerator::new()`. Encapsulates all per-layer architecture variation (e.g. Phi-3 even/odd alternation) so `generate_step` sees a uniform interface.

**Strix** — The hardware subsystem (`src/strix/`) containing all HAL adapters, the VRAM arena, GPU tensor views, and the IO engine.

**ARB Scheduler** — The continuous-batching scheduler (`batching/arb.rs`) that groups requests into decode batches, manages sequence lifetimes, and interacts with the Tick Loop.

**GBNF Constraint** — A grammar object (`GbnfConstraint`) that restricts the token logit distribution to a context-free language at each sampling step. Used for structured output.

**Model Hub** — The model download and registry subsystem (`model_hub.rs`): download, SHA-256 verify, alias lookup, local cache at `~/.cache/air-rs/models/`.

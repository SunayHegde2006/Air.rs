# Air.rs Domain Glossary

## Core Concepts

**Inference Engine** — The top-level runtime that loads a model and generates tokens. The public face of the system from a caller's perspective. Currently realised by `InferenceGenerator` (Rust) and `PyEngine` (Python).

**Model Slot** — A single loaded model with its own weight budget, KV cache manager, prefix cache, and CUDA stream pair. The unit of concurrent serving in v0.3.0+.

**Tick Loop** — The decode step that advances every active Model Slot by one token per cycle. True interleaving: all slots emit a token before any slot advances to the next step.

**S.L.I.P. (Strix Layer-Interleaved Pipeline)** — The weight-streaming pipeline that overlaps layer prefetch (stream 1) with layer compute (stream 0), enabling models larger than residual VRAM to run without stalling.

**VRAM Arena** — The VRAM allocator (`VramArena`) that tracks byte allocation across all Model Slots and enforces the 80% safety cap.

**KV Cache** — Per-sequence key/value tensor storage managed by `KvCacheManager`. Hot tier in VRAM, warm tier in RAM (compressed).

**SessionKvCache** — A trait in `src/kv_cache.rs` with 4 methods: `load_layer`, `save_layer`, `reset`, `seq_len`. The seam between `InferenceGenerator` and KV storage. `KvCacheManager` is the primary implementation. `PrefixAwareSessionCache` wraps `Box<dyn SessionKvCache>` and intercepts `load_layer` to check the prefix hash before delegating. Future `IsoQuantSessionCache` intercepts `save_layer` for M.I.S.T. v4 quantization. Kept separate from `SlotKvCache` (the ARB Scheduler's multi-slot interface in `batching/`) which has a different contract. See ADR-0004.

**Prefix Cache** — A content-addressed pool of KV blocks indexed by token-hash chunks. Allows sessions with identical system prompts to skip prefill. Per-Model Slot.

**M.I.S.T. (Memory-Interleaved Streaming Tiers)** — The tiered KV compression protocol. v3: QJL 1-bit keys + Q8 values. v4 (planned): TriAttention + IsoQuant-Fast + TurboQuant TQ4_0.

**CompressionScheme** — A tag on each KV block indicating which M.I.S.T. version compressed it. Prevents stale blocks from being reused after a pipeline upgrade.

**HERMES Eviction** — The KV cache eviction policy: hierarchical importance scoring combining recency, attention density, and position. Lives in `kv_tier.rs`.

**Ghost Drafting** — Speculative decoding using a small "ghost" model (1B/3B) to draft tokens that the main model verifies. Reduces decode latency by ~2–3×.

**HAL (Hardware Abstraction Layer)** — The `GpuHal` / `StorageHal` traits in `strix/`. Adapters: `CudaHal`, `MetalHal`, `RocmHal`, `VulkanHal`, `CpuHal`, `StdStorageHal`.

**TransformerBlock** — A trait (`src/blocks/`) with one method: `forward(hidden, weights, kv, pos, rope) → (hidden, new_k, new_v)`. One concrete impl per architecture-variant per layer. `InferenceGenerator` holds `Vec<Box<dyn TransformerBlock>>` of length `n_layers`; `blocks::build_blocks(config)` constructs the vec at model-load time. Weights are NOT owned by the block — they are streamed per-step by S.L.I.P. and passed as a parameter. Concrete impls: `LlamaBlock`, `MistralSwaBlock`, `GemmaBlock`, `FalconBlock`, `Phi3FullBlock`, `Phi3SwaBlock`. See ADR-0001.

**Block Factory** — `blocks::build_blocks(config: &ModelConfig) -> Vec<Box<dyn TransformerBlock>>`. The single construction point for the per-layer block vec. Called inside `InferenceGenerator::new()`. Encapsulates all per-layer architecture variation (e.g. Phi-3 even/odd alternation) so `generate_step` sees a uniform interface.

**Device Selector** — `GpuTopology::discover()` followed by `GpuTopology::device_at(ordinal) -> Option<Device>` or `GpuTopology::best_device() -> Device`. The single point of device selection in the system. `InferenceGenerator::new()` accepts a `candle_core::Device` as a constructor parameter and never constructs one itself. Python exposes this as `Engine.from_gguf(path, gpu_id=0)`. See ADR-0002.

**Dispatcher** — A trait in `src/dispatcher.rs` with one method: `dispatch(model_id, prompt, config) -> BoxStream<TokenChunk>`. The seam between HTTP handlers and the inference runtime. `ApiState` holds `Arc<dyn Dispatcher>`. Concrete adapters: `SingleModelDispatcher` (wraps one `InferenceGenerator`), `ModelMuxDispatcher` (wraps `ModelMux`, v0.3.0), `MockDispatcher` (test double). Non-streaming responses collect the stream. See ADR-0003.

**TokenChunk** — A tagged enum emitted by `Dispatcher::dispatch()`: `Token(String)` for mid-stream tokens, `Done(FinishReason)` for the final signal. `FinishReason` variants: `Stop`, `Length`, `Error(String)`. The enum makes it a compile error to emit token text on the final chunk or a finish reason on a mid-stream chunk.

**Strix** — The hardware subsystem (`src/strix/`) containing all HAL adapters, the VRAM arena, GPU tensor views, and the IO engine.

**ARB Scheduler** — The continuous-batching scheduler (`batching/arb.rs`) that groups requests into decode batches, manages sequence lifetimes, and interacts with the Tick Loop.

**GBNF Constraint** — A grammar object (`GbnfConstraint`) that restricts the token logit distribution to a context-free language at each sampling step. Used for structured output.

**Model Hub** — The model download and registry subsystem (`model_hub.rs`): download, SHA-256 verify, alias lookup, local cache at `~/.cache/air-rs/models/`.

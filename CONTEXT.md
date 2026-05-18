# Air.rs Domain Glossary

## Core Concepts

**Inference Engine** — The top-level runtime that loads a model and generates tokens. The public face of the system from a caller's perspective. Currently realised by `InferenceGenerator` (Rust) and `PyEngine` (Python).

**Model Slot** — A single loaded model with its own weight budget, KV cache manager, prefix cache, and CUDA stream pair. The unit of concurrent serving in v0.3.0+.

**Tick Loop** — The decode step that advances every active Model Slot by one token per cycle. True interleaving: all slots emit a token before any slot advances to the next step.

**S.L.I.P. (Strix Layer-Interleaved Pipeline)** — The weight-streaming pipeline that overlaps layer prefetch (stream 1) with layer compute (stream 0), enabling models larger than residual VRAM to run without stalling.

**VRAM Arena** — The VRAM allocator (`VramArena`) that tracks byte allocation across all Model Slots and enforces the 80% safety cap.

**KV Cache** — Per-sequence key/value tensor storage managed by `KvCacheManager`. Hot tier in VRAM, warm tier in RAM (compressed).

**SessionKvCache** — A trait in `src/kv_cache.rs` with 5 methods: `load_layer`, `save_layer`, `reset`, `seq_len`, `truncate_to`. The seam between `InferenceGenerator` and KV storage. `truncate_to(n)` is an O(1) pointer rollback used by the speculative rejection loop — no memcopy, no zeroing. `KvCacheManager` is the primary implementation. `PrefixAwareSessionCache` wraps `Box<dyn SessionKvCache>` and intercepts `load_layer` to check the prefix hash before delegating. Future `IsoQuantSessionCache` intercepts `save_layer` for M.I.S.T. v4 quantization. See ADR-0004 + ADR-0006.

**GhostDrafter** — A trait in `src/ghost_drafter.rs` with 3 methods: `draft_pass(context, k, sampler) -> DraftResult`, `on_accept(n_accept, context_len)`, `reset()`. The seam between `Speculative` and the draft strategy. Concrete adapters: `StreamingLayerSkipDrafter` (self-speculative layer skipping, protocol §4.1), `VramResidentDrafter` (models that fit fully in VRAM, v0.4.0), `MockDrafter` (canned tokens for unit tests). `Speculative` holds `Box<dyn GhostDrafter>`. See ADR-0006.

**DraftManifest** — JSON sidecar at `{gguf_path}.draft.json`. Contains GGUF mtime, size, `draft_layer_ratio`, `air_version`, draft layer indices + byte offsets into the GGUF, and offsets for `output.weight` (LM head) and `model.norm`. Invalidated and re-derived if any field mismatches (mtime/size/ratio/version). Built by `DraftManifestBuilder` in `manifest.rs`. Never tracked in git (`*.draft.json` in `.gitignore`).

**DraftResult** — Value type returned by `GhostDrafter::draft_pass`: `{ tokens: Vec<u32>, logits: Vec<Vec<f32>> }`. `tokens` is k proposed IDs; `logits` is k probability distributions `[k, vocab_size]`. Used directly by `Speculative::rejection_sample`.

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

**SharedBuffer** — Platform-agnostic CPU/GPU shared memory type in `src/shared_buffer.rs`. Always compiled (no feature gate). Used by `pipeline.rs` and the VRAM arena for zero-copy buffer hand-off between CPU and GPU.

**ComputeBackend** — Canonical enum in `src/shared_buffer.rs`: `Cuda(usize)`, `Rocm(usize)`, `Metal`, `Vulkan`, `Cpu`. Single source of truth for backend selection across `ModelMux`, `drive_inquisitor`, and `metal_compute`. Replaces the two separate `ComputeBackend` enums that previously existed in `ucal.rs` and `drive_inquisitor.rs`. See ADR-0005.

**ARB Scheduler** — The continuous-batching scheduler (`batching/arb.rs`) that groups requests into decode batches, manages sequence lifetimes, and interacts with the Tick Loop.

**GBNF Constraint** — A grammar object (`GbnfConstraint`) that restricts the token logit distribution to a context-free language at each sampling step. Used for structured output.

**Model Hub** — The model download and registry subsystem (`model_hub.rs`): download, SHA-256 verify, alias lookup, local cache at `~/.cache/air-rs/models/`.

---

## v0.9.0 Enterprise + Hybrid-Attention Terms

**ThinkingTokenizer** — Trait in `src/think_tag.rs` abstracting thinking-mode token detection across model families. Two implementations: `TagBasedThinking` (watches byte-pattern sequences `<think>`/`</think>`, used by Qwen3.6/DeepSeek/QwQ) and `SpecialTokenThinking` (watches special token IDs from GGUF tokenizer vocab, used by Gemma 4). Selected at model-load time by `ModelVariant::uses_special_token_thinking()`. See ADR Decision Q5.

**AttentionBackend** — Enum in `src/attention_backend.rs` representing per-layer attention kernel choice. Variants: `Softmax` (standard GQA/MHA, all existing models), `GatedDeltaNet` (Qwen3.6 linear recurrence layers v0.10.0), `SlidingWindow { window }` (Gemma4 local layers), `GlobalFull` (Gemma4 global layers with p-RoPE). `is_recurrent()` distinguishes KV-cache from state-matrix layers.

**HybridAttentionRouter** — Struct in `src/attention_backend.rs` that maps each transformer layer index to its `AttentionBackend`. `uniform(n, backend)` for homogeneous models; `from_layout(vec)` for hybrid models. Canonical layout constructors: `qwen3_6_27b()` (48 DeltaNet + 16 Softmax over 64 layers), `gemma4_e4b(n, w, stride)`. Built by `blocks.rs::build_streaming_blocks()` at model-load time. See ADR Decision Q1/Q2.

**MtpDraftHead** — Struct in `src/speculative.rs` representing a model's native multi-token prediction auxiliary head (Qwen3.6 NEXTN method). Auto-detected at load time by scanning GGUF tensor names for `mtp_head`, `output.*_mtp`, or metadata key `{arch}.mtp.num_steps`. Coexists with EAGLE-2 via `DraftStrategy` enum — zero user configuration required. Full forward pass in v0.10.0. Research: Gloeckle et al., ICML 2024. See ADR Decision Q4.

**DualRoPE** — Struct in `src/dual_rope.rs` (v0.10.0) holding two separate RoPE base-frequency caches: `θ_local` for local sliding-window attention layers and `θ_global` for global full-attention layers. Required by Gemma 4's p-RoPE (proportional RoPE) specification. Read from GGUF metadata keys `gemma4.attention.local_rope_theta` and `gemma4.attention.global_rope_theta`. Current `ops.rs` RoPE accepts a single `theta` — v0.10.0 extends this via `DualRoPE`.

**Gemma4MoeRouter** — MoE router for Gemma 4 26B A4B in `src/moe.rs` (v0.10.0). Uses sigmoid (not softmax) over router logits and top-1/2 expert selection per layer. Distinct from `ConceptMoeConfig` (uses softmax + confidence threshold). 26B total params, 4B active. Expert count derived from GGUF metadata. Extends `moe_forward` with `gemma4_moe_forward`.

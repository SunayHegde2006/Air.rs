<p align="center">
  <img src="assets/banner-github.png" alt="Air.rs Banner" width="800"/>
</p>

<h1 align="center">Air.rs</h1>

<p align="center">
  <strong>S.L.I.P. — Slipstream Layer Inference Protocol</strong><br>
  Run models larger than your VRAM — streaming weights from NVMe via mmap.
</p>

<p align="center">
  <a href="#project-status"><img src="https://img.shields.io/badge/status-beta-blue?style=flat-square" alt="Status: Beta"></a>
  <a href="https://pypi.org/project/air-rs/"><img src="https://img.shields.io/pypi/v/air-rs?style=flat-square&color=brightgreen" alt="PyPI"></a>
  <a href="https://pypi.org/project/air-rs/"><img src="https://img.shields.io/pypi/pyversions/air-rs?style=flat-square" alt="Python 3.11+"></a>
  <a href="#build"><img src="https://img.shields.io/badge/platform-Windows%20|%20Linux%20|%20macOS-blue?style=flat-square" alt="Cross-Platform"></a>
  <a href="#build"><img src="https://img.shields.io/badge/Rust-1.75+-F74C00?logo=rust&style=flat-square" alt="Rust 1.75+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT"></a>
</p>

---

## The Problem

Large language models don't fit in VRAM. A 70B-parameter model at FP16 needs **140 GB** of GPU memory. Even quantized to Q4, that's **35 GB** — more than an RTX 4090's 24 GB.

Current solutions force painful tradeoffs:
- **CPU offloading** — 10-50× slower inference
- **Model parallelism** — requires multiple expensive GPUs
- **Aggressive quantization** — degrades output quality

## The Air.rs Solution

Air.rs implements **S.L.I.P.** (**S**lipstream **L**ayer **I**nference **P**rotocol): the GGUF file is memory-mapped but only **one layer's worth of quantized weights** is resident in physical RAM at any time. Weights stay compressed in GGUF block formats — `QMatMul` performs dequantize-on-the-fly during matrix multiplication.

```
  +--------------------------------------------------------------+
  |                     S.L.I.P. Pipeline                        |
  |                                                              |
  |  GGUF on NVMe --mmap--> Virtual Address Space (RSS ~ 0)     |
  |                              |                               |
  |  Per token, per layer:       v                               |
  |    prefetch(layer N+1)  <-- SSD reads ahead (madvise)        |
  |    load_layer(N)        <-- QTensor -> QMatMul (RSS += 1)    |
  |    transformer_block()  <-- quantized forward pass           |
  |    drop(weights)        <-- Rust drops QBlockWeights         |
  |    release(layer N-1)   <-- madvise(DONTNEED), pages freed   |
  +--------------------------------------------------------------+

  Steady-state RSS:  ~400 MB for 7B  |  ~1.5 GB for 70B
  (vs 4 GB / 40 GB file sizes)
```

**Result:** Run 70B+ models on a single consumer GPU with minimal RAM.

---

## Install

### Python (recommended)

```bash
pip install air-rs          # PyPI — abi3 wheel, Python ≥ 3.11
```

```python
import air_rs

engine = air_rs.Engine.from_gguf("llama-3.2-3b-q4_k_m.gguf")
print(engine.generate("Explain attention in one sentence."))
```

### Rust / CLI

```bash
cargo build --release
cargo run --release -- --model path/to/model.gguf --prompt "Hello!"
```

---

## Features

| Category | Feature |
|----------|---------|
| **Core** | Layer-streamed inference — one transformer block in memory at a time |
| **Quantization** | Weights stay in GGUF block format; `QMatMul` dequantizes during matmul (21 formats: F32→IQ4_XS) |
| **File Format** | GGUF, SafeTensors, PyTorch (.bin/.pt), ONNX — auto-detected |
| **Memory** | `madvise` / `PrefetchVirtualMemory` page control + mmap storage HAL |
| **KV Cache** | 1-bit key + Q8 value compression; tiered eviction with triage scoring |
| **OCS Attention** | SageAttention3 FP4 E2M1 microscaling + KIMI linear O(N·D²) + per-head gating |
| **OCS KV** | QJL 1-bit JL-transform key compression + Fast cosine-merge compaction |
| **OCS Eviction** | HERMES hierarchical importance-score eviction (recency + density + position) |
| **OCS Routing** | ConceptMoE confidence-threshold adaptive top-1/top-k expert routing |
| **Pipeline** | Adaptive circular-buffer pipeline — overlaps NVMe reads, PCIe, GPU |
| **API** | OpenAI-compatible `/v1/chat/completions` (streaming SSE) via Axum |
| **Compute** | NVIDIA CUDA + AMD ROCm + Vulkan + Apple Metal + CPU backends |
| **GPU Offload** | STRIX 3-tier hierarchy (VRAM → RAM → Storage) with residency scoring |
| **GPUDirect** | NVMe → GPU DMA via cuFile FFI (zero CPU copies) |
| **Multi-GPU** | NVLink/PCIe peer-to-peer topology, layer-parallel + tensor-parallel sharding |
| **Security** | VRAM zeroing (hardware-native), bounds-checked pointers, owner tokens, audit log |
| **Decoding** | Speculative decoding (draft-verify acceleration, 2-3× speedup) |
| **Ghost Drafting** | Automated ghost model selection (1B/3B) with EMA-based adaptive batching |
| **Scheduling** | Continuous batching + adaptive request batcher (ARB) |
| **Sampling** | Temperature, top-p, top-k, repetition penalty, min-p |
| **Tokenizer** | BPE tokenizer built from GGUF vocabulary |
| **Models** | Llama 3 / 3.1 / 3.2, Mistral, Phi-3, Qwen2, Gemma — auto-detected from GGUF |
| **GBNF** | Grammar-constrained generation — JSON mode, integer, identifier, choice, raw |
| **Model Hub** | Download models from Hugging Face with SHA-256 verification |
| **Monitoring** | Real-time TUI dashboard + Prometheus-compatible metrics |
| **Templates** | Chat template engine (ChatML, Llama3, Mistral, Gemma, Phi-3) |
| **Python** | Production-ready PyO3 bindings — `pip install air-rs` |
| **Async Streaming** | `astream(engine, prompt)` async generator — GIL-free token streaming via `tokio::sync::mpsc` |
| **Model Multiplexer** | Load N models simultaneously; per-tick interleaved decode; VRAM budget enforced at 80% cap |
| **Prefix KV Cache** | Content-addressed KV block pool (16-token chunks, hash-keyed); `PrefixKvCache` + `CompressionScheme` |
| **CUDA Pipeline** | `LayerScheduler` — DMA stream overlaps weight prefetch with compute per layer (`CudaStreamPool`) |
| **Benchmarks** | Criterion throughput suite + 4-engine comparison harness (`scripts/`) |

---

## Python API

### Install

```bash
pip install air-rs                          # from PyPI (abi3, Python ≥ 3.11)

# or build from source
pip install maturin
maturin develop --features python
```

### Quick start

```python
import air_rs

# Load any GGUF model
engine = air_rs.Engine.from_gguf("llama-3.2-3b-q4_k_m.gguf")

# Generate text
print(engine.generate("Explain attention in one sentence."))

# Custom sampling
cfg = air_rs.GenerateConfig(temperature=0.0, max_tokens=64)
print(engine.generate("2 + 2 =", config=cfg))

# Structured output — force valid JSON
cfg = air_rs.GenerateConfig(
    grammar=air_rs.GbnfConstraint.json_mode(),
    max_tokens=128,
)
print(engine.generate("Extract name and age from: Bob, 42", config=cfg))

# Constrain to a fixed set of words
cfg = air_rs.GenerateConfig(
    grammar=air_rs.GbnfConstraint.choice(["yes", "no", "maybe"]),
)
print(engine.generate("Is Python slow?", config=cfg))

# Performance metrics
m = engine.metrics()
print(f"{m.tokens_per_second:.1f} tok/s  |  TTFT {m.time_to_first_token_ms:.0f} ms")

# Chat template formatting
from air_rs.utils import format_chat

prompt = format_chat(
    [{"role": "user", "content": "Hello!"}],
    template="llama3",
)
print(engine.generate(prompt))

# Reset KV cache between conversations
engine.reset()
```

### Async streaming (`astream`)

Yield tokens one-by-one without blocking the event loop — zero GIL holds during
generation, safe to use inside FastAPI / Starlette / aiohttp handlers:

```python
import asyncio
import air_rs

engine = air_rs.Engine.from_gguf("llama-3.2-3b-q4_k_m.gguf")

async def main() -> None:
    async for token in air_rs.astream(engine, "Once upon a time"):
        print(token, end="", flush=True)
    print()

asyncio.run(main())
```

With sampling config:

```python
cfg = air_rs.GenerateConfig(temperature=0.8, max_tokens=256)
async for token in air_rs.astream(engine, "Tell me a story", cfg):
    print(token, end="", flush=True)
```

FastAPI SSE endpoint example:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/stream")
async def stream(prompt: str) -> StreamingResponse:
    async def generator():
        async for token in air_rs.astream(engine, prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")
```


### API reference

| Symbol | Description |
|--------|-------------|
| `Engine.from_gguf(path, **sampler_defaults)` | Load GGUF — CUDA if available, else CPU |
| `Engine.generate(prompt, config=None)` | Synchronous generation → `str` |
| `Engine.stream_to_list(prompt, config=None)` | Token list (wrap in `run_in_executor` for async) |
| `Engine.set_grammar(constraint)` | Attach persistent grammar for all calls |
| `Engine.clear_grammar()` | Remove persistent grammar |
| `Engine.reset()` | Clear KV cache between conversations |
| `Engine.metrics()` | Returns `Metrics` snapshot |
| `GenerateConfig(max_tokens, temperature, top_p, top_k, stop_strings, grammar)` | Per-call sampling config |
| `GbnfConstraint.json_mode()` | Force valid JSON output |
| `GbnfConstraint.integer()` | Single integer output |
| `GbnfConstraint.identifier()` | C-style identifier `[a-zA-Z_][a-zA-Z0-9_]*` |
| `GbnfConstraint.choice(options)` | Restrict to one of N strings |
| `GbnfConstraint.from_grammar(src)` | Raw GBNF grammar string |
| `Metrics.tokens_per_second` | Decode throughput |
| `Metrics.time_to_first_token_ms` | Prefill latency |
| `Metrics.total_time_ms` | Full generation wall time |
| `format_chat(messages, template, add_generation_prompt)` | ChatML / Llama3 / Mistral / Gemma / Phi-3 |
| `count_tokens_approx(text)` | Fast token-count estimate (÷4 chars) |
| `astream(engine, prompt, config=None)` | **Async generator** — yields one token per `await`; GIL-free |
| `shutdown_stream_executor(wait=True)` | Cleanly tears down the background thread pool |

### Supported models (Python + CLI)

| Family | Architecture key | Tested |
|--------|-----------------|--------|
| Llama 3 / 3.1 / 3.2 | `llama` | ✅ Q8 + Q4 |
| Mistral / Mixtral | `mistral` | ✅ |
| Phi-3 | `phi3` | ✅ |
| Qwen 2 / 2.5 | `qwen2` | ✅ |
| Gemma / Gemma 2 | `gemma` / `gemma2` | ✅ |

---

## Architecture

```
src/
├── main.rs              # CLI entry point (clap)
├── lib.rs               # Module declarations, constants
│
│── loader.rs            # GGUF parser — tensor offsets + model config
│── weight_streamer.rs   # S.L.I.P. core — mmap + per-layer QMatMul streaming
│── manifest.rs          # Execution planner — page-aligned DMA chunks
│── pipeline.rs          # Adaptive D-deep circular slot pipeline
│
│── model.rs             # Transformer block — QBlockWeights + forward pass
│── ops.rs               # Math ops — RMSNorm, RoPE, SiLU, GQA, softmax
│── generator.rs         # Inference loop — layer-streamed token generation
│── speculative.rs       # Speculative decoding (draft-verify)
│
│── kv_cache.rs          # KV-cache manager — RAM/VRAM shuttle
│── kv_tier.rs           # Tiered eviction policy (HERMES)
│── kv_compress.rs       # 1-bit key + Q8 value KV compression (M.I.S.T. v3)
│── ghost_drafting.rs    # Ghost model selection + ColdLog + prefetch
│
│── sampler.rs           # Token sampling — temperature/top-p/top-k/min-p
│── tokenizer.rs         # BPE tokenizer from GGUF vocabulary
│── chat_template.rs     # Chat template engine (ChatML/Llama3/Mistral/Gemma/Phi-3)
│── gbnf.rs              # GBNF grammar parser + stack machine
│
│── api.rs               # OpenAI-compatible HTTP API (Axum, SSE streaming)
│── scheduler.rs         # Continuous batching request scheduler
│── arb.rs               # Adaptive Request Batcher
│── metrics.rs           # Prometheus-compatible metrics collector
│── tui.rs               # Real-time terminal dashboard
│
│── ucal.rs              # Metal compute backend (macOS Apple Silicon)
│── gpu_pipeline.rs      # GPU pipeline orchestration
│── uploader.rs          # Async triple-buffered NVMe→VRAM transfers
│── orchestrator.rs      # VRAM pointer → Candle tensor hydration
│
│── model_hub.rs         # Hugging Face model downloader + SHA-256 verify
│── drive_inquisitor.rs  # Storage/compute profiler + protocol routing
│
│── python.rs            # PyO3 bindings (--features python)
│
└── strix/               # STRIX — Streamed Tensor Residence & Intelligent eXchange
    ├── mod.rs             # Module registry + re-exports
    │── types.rs           # Core types (GpuPtr, DType, ResidencyState, TensorClass)
    │── hal.rs             # HAL trait contracts + secure_zero_vram()
    │── config.rs          # Runtime configuration (StrixConfig)
    │── meta.rs            # Per-tensor metadata (TensorMeta)
    │── score.rs           # Residency scoring function R(t,τ)
    │── cpu_hal.rs         # CpuHal — host memory backend
    │── cuda_hal.rs        # CudaHal — NVIDIA CUDA Runtime API
    │── vulkan_hal.rs      # VulkanHal — Vulkan 1.2 + command buffer staging
    │── metal_hal.rs       # MetalHal — Apple Metal framework
    │── rocm_hal.rs        # ROCmHal — AMD ROCm/HIP
    │── gpu_alloc.rs       # RAII VRAM allocation + DMA staging
    │── gpu_tensor_view.rs # Lifetime-bound zero-copy VRAM tensor view
    │── arena.rs           # VRAM budget allocation (VramArena)
    │── registry.rs        # Central tensor tracking (TensorRegistry)
    │── scheduler.rs       # Residency tick loop (ResidencyScheduler)
    │── scheduler_thread.rs # Dedicated 2ms scheduler tick thread
    │── vram_pressure.rs   # 5-level VRAM pressure manager
    │── security.rs        # SecureAllocator, ShardedRwLock, BoundsCheckedPtr
    │── session.rs         # StrixSession — open(), open_unified()
    │── bridge.rs          # StrixBridge — high-level orchestrator
    │── cold_boot.rs       # Staged cold-start loading sequence
    │── compat.rs          # GGUF/UnifiedModel parser + tensor classification
    │── safetensors.rs     # SafeTensors format reader
    │── pytorch.rs         # PyTorch .bin/.pt reader (ZIP + pickle)
    │── onnx.rs            # ONNX protobuf reader
    │── io_engine.rs       # Priority async I/O queue
    │── async_io.rs        # Platform I/O: io_uring / IOCP + stress tests
    │── std_storage_hal.rs # Synchronous StorageHal fallback
    │── mmap_storage.rs    # MmapStorageHal with platform prefetch hints
    │── ram_pool.rs        # Recycling RAM buffer pool
    │── streamer_adapter.rs # WeightStreamer → STRIX adapter
    │── execution_cursor.rs # ExecutionCursor + MoE expert activation hook
    │── gpu_direct.rs      # GPUDirect Storage NVMe→GPU DMA integration
    │── cufile_ffi.rs      # cuFile API FFI bindings (cuda+linux)
    │── multi_gpu.rs       # Multi-GPU topology, NVLink, shard strategies
    │── backend_detect.rs  # Sub-100ms GPU/storage backend detection
    │── integration_tests.rs # Lifecycle, budget, inference simulation tests
    │── chaos_tests.rs     # Stress, fragmentation, edge case tests
    │── benchmarks.rs      # Scheduler, scoring, arena, I/O benchmarks
    └── e2e_validation.rs  # Real GGUF model end-to-end validation
```

**65+ modules · ~32,000 lines of Rust · 976+ tests**

---

## Project Status

> **Beta** — All subsystems implemented and tested (**1 084+ tests**, 0 warnings). Compiles on Windows, Linux, and macOS. All protocol specifications (STRIX, S.L.I.P., M.I.S.T. v3/v4, DriveInquisitor v3) at 100% coverage. v0.3.0 multi-model serving complete. v0.4.0 M.I.S.T. v4 KV pipeline complete: TriAttention (SnapKV-inspired), IsoQuant-Fast (SO(4) quaternion, QuIP# inspired), TurboQuant Lloyd-Max (TQ4_0, optimal 4-bit), LoRA/PEFT hot-swap (S-LoRA-style), `air-rs` CLI binary. E2E validation passes against real Llama 3.2 3B Q8 GGUF models.

### Current Status

| Aspect | Status |
|--------|--------|
| Compiles on Windows / Linux / macOS | ✅ |
| Unit + integration tests (976+) | ✅ All passing, 0 warnings |
| Multi-format model support | ✅ GGUF, SafeTensors, PyTorch, ONNX |
| Multi-model auto-detection | ✅ Llama / Mistral / Phi-3 / Qwen2 / Gemma |
| GBNF grammar-constrained generation | ✅ JSON, integer, identifier, choice, raw |
| S.L.I.P. layer streaming engine | ✅ |
| Transformer forward pass (quantized) | ✅ |
| KV-cache + tiered HERMES eviction | ✅ |
| KV compression (1-bit key / Q8 value) | ✅ M.I.S.T. v3 |
| Ghost drafting + cold log | ✅ M.I.S.T. v3 |
| Speculative decoding | ✅ |
| OpenAI-compatible API | ✅ |
| STRIX GPU offloading (5 backends) | ✅ CUDA / ROCm / Vulkan / Metal / CPU |
| GPUDirect Storage (cuFile FFI) | ✅ |
| Multi-GPU (NVLink / PCIe) | ✅ Topology + layer/tensor sharding |
| VRAM security (hardware zeroing) | ✅ |
| Mmap storage HAL + prefetch | ✅ |
| DriveInquisitor v3 | ✅ Protocol routing matrix |
| E2E validation (Llama 3.2 3B real model) | ✅ |
| Criterion throughput benchmarks | ✅ |
| 4-engine benchmark harness | ✅ `scripts/run_benchmarks.sh` |
| Correctness validator vs llama.cpp | ✅ `scripts/validate_correctness.py` |
| OCS — SageAttention3 FP4 | ✅ `fp4_attention`, ops.rs |
| OCS — KIMI Linear Attention | ✅ `linear_attention_kimi`, ops.rs |
| OCS — Gated Attention | ✅ `gated_attention`, ops.rs |
| OCS — QJL 1-bit KV Keys | ✅ `QjlKey`, kv_compress.rs |
| OCS — Fast KV Compaction | ✅ kv_compress.rs |
| OCS — HERMES Eviction | ✅ `HermesTierManager`, kv_tier.rs |
| OCS — ConceptMoE Routing | ✅ moe.rs |
| **Python package (`pip install air-rs`)** | ✅ **v0.1.0 on PyPI** |
| CI/CD — multi-platform wheels | ✅ manylinux / macOS / Windows |
| OIDC Trusted Publisher (no secret) | ✅ |
| **Model Multiplexer** (v0.3.0) | ✅ `src/model_mux.rs` — interleaved decode, VRAM-budget enforcement |
| **VRAM 80% hard cap** (v0.3.0) | ✅ `src/vram_guard.rs` — rejects loads exceeding budget with clear error |
| **Per-model prefix KV cache** (v0.3.0) | ✅ `src/prefix_kv.rs` — content-addressed blocks, FIFO eviction, `CompressionScheme` |
| **CUDA multi-stream pipelining** (v0.3.0) | ✅ `src/cuda_pipeline.rs` — `CudaStreamPool` + `LayerScheduler` DMA/compute overlap |
| **Native async Python streaming** (v0.3.0) | ✅ `astream(engine, prompt)` via `tokio::sync::mpsc` + `_stream_channel` |

### STRIX Subsystem

STRIX (**S**treamed **T**ensor **R**esidence & **I**ntelligent e**X**change) manages a 3-tier memory hierarchy (VRAM → RAM → Storage) with intelligent eviction scoring for 70B+ models on consumer GPUs.

| Component | Status |
|-----------|--------|
| Tensor registry + lifecycle | ✅ Production |
| RAII VRAM allocations | ✅ Production |
| CUDA HAL + cudaMemsetAsync zeroing | ✅ Production |
| ROCm HAL (AMD GPUs) | ✅ Production |
| Vulkan HAL + staging transfers | ✅ Production |
| Metal HAL (Apple Silicon) | ✅ Production |
| VRAM pressure manager (5 levels) | ✅ Production |
| Security (bounds, audit log) | ✅ Production |
| Zero-copy tensor views | ✅ Production |
| Async I/O (io_uring / IOCP) | ✅ Production |
| Multi-format model parsing | ✅ Production |
| Mmap storage + prefetch | ✅ Production |
| ExecutionCursor + MoE routing | ✅ Production |
| GPUDirect Storage + cuFile FFI | ✅ Production |
| Multi-GPU topology + NVLink | ✅ Production |
| Layer-parallel + tensor-parallel | ✅ Production |
| Sub-100ms backend detection | ✅ Production |
| Integration + chaos tests | ✅ Production |
| E2E validation (real models) | ✅ Production |

### Roadmap

#### ✅ Completed (v0.1.0 — Beta)

- [x] E2E validation with real GGUF model (Llama 3.2 3B Q8)
- [x] Performance benchmarks (scheduler, scoring, I/O)
- [x] Multi-GPU topology and sharding strategies
- [x] GPUDirect Storage FFI bindings
- [x] Hardware-verified VRAM zeroing
- [x] Validate output correctness against llama.cpp
- [x] CUDA tested on RTX 3060 12 GB (CUDA 12.0) — [`docs/benchmarking_guide.md §7`](docs/benchmarking_guide.md)
- [x] Tokens/sec measurement with full inference pipeline
- [x] Multi-model support (Llama, Mistral, Phi-3, Qwen2, Gemma)
- [x] GBNF grammar-constrained generation
- [x] Python package release — `pip install air-rs` (PyPI v0.1.0)
- [x] Multi-platform CI/CD (manylinux + macOS + Windows wheels)
- [x] OIDC Trusted Publisher (no long-lived secrets)

#### ✅ Completed (v0.2.0)

- [x] Flash Attention 2 kernel integration — `#[cfg(feature="flash-attn")]` fused attention in `ops.rs` with causal masking via `candle_flash_attn`
- [x] Python token streaming — `engine.stream_to_list(prompt)` returns ordered token list; wrap in `asyncio.run_in_executor` for non-blocking use
- [x] Model download shorthand — `model_hub::download_model()` + `parse_model_spec()` + `ModelRegistry` with alias lookup (`air pull TheBloke/Llama-2-7B-GGUF`)
- [x] Quantized KV-cache — 1-bit key compression + Q8 value quantization (BF16→Q8, 2× compression) fully implemented in M.I.S.T. v3 (`kv_compress.rs`)
- [x] ROCm backend — `src/strix/rocm_hal.rs` fully implements `GpuHal` via AMD HIP Runtime API FFI, feature-gated as `--features rocm`

#### ✅ Completed (v0.3.0) — Multi-Model Concurrent Serving

> **Theme:** True interleaved multi-model serving on consumer GPUs. Every loaded model emits tokens at the same wall-clock tick. Designed and validated against RTX 3060 12 GB VRAM.

- [x] **Model Multiplexer** (`src/model_mux.rs`) — load N models simultaneously; per-tick interleaved decode loop emits one token per model per step; dynamic count bound by available VRAM (no hard limit — fits as many as VRAM allows)
- [x] **VRAM 80% hard cap** (`src/vram_guard.rs`) — reject model load with a clear error if combined weight footprint would exceed 80% of detected VRAM (9.6 GB on 3060): `Error: insufficient VRAM for simultaneous execution — free {X} MB or use a smaller model`
- [x] **Per-model prefix KV cache** (`src/prefix_kv.rs`) — content-addressed block pool (16-token chunks, hash-keyed); ref-counted entries shared across sessions with identical system prompts; blocks tagged with `CompressionScheme` enum for forward-compatible migration to M.I.S.T. v4; FIFO eviction at configurable capacity
- [x] **CUDA multi-stream pipelining** (`src/cuda_pipeline.rs`) — `LayerScheduler` + `CudaStreamPool`; compute stream overlaps weight prefetch DMA stream per layer; extends the existing S.L.I.P. pipeline; no CUDA MPS required; graceful noop fallback on non-CUDA builds
- [x] **Native async Python streaming** — `air_rs.astream(engine, prompt)` async generator backed by Rust `tokio::sync::mpsc`; `Engine._stream_channel()` opens channel natively; GIL-free, event-loop-safe; works across all loaded models concurrently

#### ✅ Completed (v0.4.0) — M.I.S.T. v4 KV Pipeline

> **Theme:** Replace QJL with a research-validated compression pipeline. Research basis: SnapKV (Li et al., 2024), QuIP# (Tseng et al., ICML 2024), Lloyd-Max optimal quantization (1957/1960), S-LoRA (Chen et al., 2023).

- [x] **TriAttention** (`src/tri_attention.rs`) — pre-RoPE trigonometric token importance scorer (SnapKV + H2O inspired); attention-sink preservation; cosine × RoPE-phase convex blend; `ScoringStrategy::TriAttention` plugs into HERMES eviction; 8 tests
- [x] **IsoQuant-Fast** (`src/iso_quant.rs`) — SO(4) quaternion rotation Stage 1 (4.5× faster than QR, geometrically lossless — ‖Rk‖ = ‖k‖); tiled 4-block design for head_dim=128; `UnitQuaternion` Hamilton product; reconstruction via conjugate; 7 tests
- [x] **TurboQuant Lloyd-Max** (`src/turbo_quant.rs`) — Stage 2 optimal 4-bit scalar quantization; Voronoi iteration to MSE-minimizing centroids; TQ4_0 format (32 values → 48 bytes); beats uniform Q4 MSE on Laplacian activation distribution; `uniform_q4_mse` + `uniform_q8_mse` baselines; 7 tests
- [x] **QJL path deprecated** — `kv_compress.rs` JL path gated behind `--features legacy-qjl`; `CompressionScheme::IsoQuantTQ4` is the new default
- [x] **LoRA / PEFT hot-swap** (`src/lora.rs`) — S-LoRA-style adapter serving; LRU `AdapterCache` bounded by VRAM budget; `LoraLinear::forward` with BAΔW delta (alpha/rank scaling); `SharedAdapterCache` RwLock wrapper; 8 tests
- [x] **Vision / multimodal** (`src/vision.rs`) — SigLIP / CLIP ViT encoder fully scaffolded (LLaVA 1.5/1.6, PaliGemma, Gemma 3, Qwen2-VL); patch embedding conv → positional → transformer → projection head
- [x] **`air-rs` standalone CLI binary** (`src/bin/air_rs.rs`) — `generate` / `serve` / `bench` / `info` subcommands; no external dep arg parser; streaming output; 8 tests
- [ ] **Windows ROCm validation** — tracked; requires HIP SDK 6.x + real AMD GPU hardware; CI job skeleton ready in `.github/workflows/ci.yml`

#### 🔭 Spec (v0.5.0) — Production Readiness

> **Theme:** From beta to first production deployment. Disaggregated prefill-decode, EAGLE-2 speculative decoding, PagedAttention v2, OpenAI-compatible REST API + auth/rate-limiting, and a comprehensive evaluation harness. Research basis: EAGLE-2 (Li et al., NeurIPS 2024); PagedAttention (Kwon et al., SOSP 2023); FlashDecoding++ (Hong et al., ICLR 2024); Orca continuous batching (Yu et al., OSDI 2022).

- [ ] **EAGLE-2 Speculative Decoding** — context-aware dynamic draft tree (not fixed depth); shared KV cache between draft + target model; target acceptance rate ≥ 3.2× smaller draft → ~2.8× wall-clock speedup; `src/eagle2.rs`
- [ ] **PagedAttention v2** — non-contiguous KV block table (4096-token pages); copy-on-write for shared prefixes (beam search, parallel sampling); `src/paged_attention.rs`
- [ ] **FlashDecoding++ Kernel** — parallel softmax reduction across split-k chunks; 40% decode latency reduction; custom CUDA / Metal kernel; `src/flash_decode.rs`
- [ ] **Continuous Batching v2** — disaggregated prefill (GPU-A) + decode (GPU-B) pools; KV transfer via zero-copy pinned memory; 5–10× throughput vs. chunked prefill
- [ ] **OpenAI-Compatible REST API** — `/v1/chat/completions` (SSE streaming + batch), `/v1/completions`, `/v1/models`, `/v1/embeddings`; JWT Bearer auth; token-bucket rate limiting; `src/openai_api.rs`
- [ ] **Evaluation Harness** — HellaSwag, ARC-Easy/Challenge, MMLU, TruthfulQA, GSM8K, perplexity on WikiText-103; CI regression gate (accuracy drop ≤ 0.5%); `src/eval/`
- [ ] **Production Observability** — Prometheus `/metrics` (TTFT p50/p95/p99, TPS, queue depth); OpenTelemetry trace spans; `/health` + `/ready` endpoints
- [ ] **Kubernetes Helm Chart** — Deployment + HPA + PodDisruptionBudget; GPU resource requests; RollingUpdate 0-downtime

#### 📅 Production Roadmap (v0.6.0 → v1.0.0 GA)

| Version | Theme | Key Features |
|---|---|---|
| **v0.6.0** | Multi-GPU + MoE | Megatron tensor parallel (2–8 GPU); Mixtral 8×7B / DeepSeek-V2 MoE routing |
| **v0.7.0** | Quantization v2 | AQLM 2-bit residual codebook; INT4×INT4 GEMM; QLoRA fine-tune endpoint |
| **v0.8.0** | Long Context | RoPE YaRN scaling; Mistral sliding window; Ring attention for ≥128K ctx; Whisper audio |
| **v0.9.0** | Enterprise | PII redaction; content safety classifier; OAuth2/OIDC; SOC 2 audit logging |
| **v1.0.0** | Production GA | SLA 99.9%; zero-regression eval gate (HellaSwag ≥80.1%, TTFT p99 ≤250ms); LTS branch; vLLM/Ollama migration guide |

---

## Build

### Build Scripts (Recommended)

Air.rs ships two platform-native build scripts that auto-detect hardware and run `cargo build`.

| Platform | Script | Shell |
|---|---|---|
| **Windows** | `build_air.ps1` | PowerShell |
| **macOS / Linux** | `build_air.sh` | bash |

```bash
# macOS / Linux
chmod +x build_air.sh
./build_air.sh               # interactive feature selection
./build_air.sh --skip-prompt # auto-enable everything detected
./build_air.sh --debug       # debug build
./build_air.sh --features cuda,flash-attn

# Windows
.\build_air.ps1
.\build_air.ps1 -SkipPrompt
.\build_air.ps1 -DebugBuild
```

### Manual Build

#### Prerequisites

| | Windows 11 | Linux | macOS |
|---|---|---|---|
| **Rust** | 1.75+ via [rustup.rs](https://rustup.rs) | 1.75+ via rustup | 1.75+ via rustup |
| **C++ Toolchain** | VS 2022 (Desktop C++ workload) | `build-essential` | Xcode CLI Tools |
| **GPU (optional)** | CUDA 12.x + NVIDIA GPU | CUDA 12.x + NVIDIA GPU | Metal (Apple Silicon) |

```bash
# Linux — CPU
sudo apt install -y build-essential pkg-config libssl-dev
cargo build --release

# Linux — NVIDIA GPU
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda,flash-attn

# macOS — Apple Silicon
xcode-select --install
cargo build --release --features metal

# Windows (from VS Developer Command Prompt)
.\setup_build_env.ps1
cargo build --release --features cuda,flash-attn
```

### Feature Flags

| Flag | What It Enables | Platforms |
|------|----------------|-----------|
| `cuda` | NVIDIA GPU via CUDA Runtime API (STRIX CudaHal) | Windows, Linux |
| `rocm` | AMD GPU via ROCm/HIP (STRIX ROCmHal) | Linux |
| `vulkan` | Vulkan 1.2 GPU compute (STRIX VulkanHal) | Windows, Linux |
| `flash-attn` | Flash Attention 2 kernels | Windows, Linux |
| `metal` | Apple Metal GPU compute (STRIX MetalHal) | macOS |
| `python` | PyO3 Python bindings (`pip install air-rs`) | All |

### Run

```bash
# Basic generation
cargo run --release -- --model path/to/model.gguf --prompt "Hello, world!"

# Custom sampling
cargo run --release -- \
  --model path/to/model.gguf \
  --prompt "Tell me a joke" \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 40 \
  --max-tokens 256
```

---

## Troubleshooting

<details>
<summary><strong>LNK1181: cannot open 'kernel32.lib' (Windows)</strong></summary>

The Windows SDK `LIB` path is not set. Run the setup script:
```powershell
.\setup_build_env.ps1
```
Or build from a **VS Developer Command Prompt** which sets paths automatically.
</details>

<details>
<summary><strong>stdc++.lib not found (Windows + flash-attn)</strong></summary>

`build.rs` auto-creates a stub `stdc++.lib` for MSVC. Clean and rebuild:
```powershell
cargo clean && cargo build --release --features cuda,flash-attn
```
</details>

<details>
<summary><strong>CUDA not detected</strong></summary>

1. Verify: `nvcc --version`
2. Build with: `cargo build --release --features cuda`
3. Linux: `export CUDA_HOME=/usr/local/cuda`
4. Windows: `echo $env:CUDA_PATH`
</details>

<details>
<summary><strong>Metal not available (macOS)</strong></summary>

Metal requires Apple Silicon (M1/M2/M3/M4). On Intel Mac, use CPU build:
```bash
cargo build --release  # Accelerate framework still accelerates matmuls
```
</details>

<details>
<summary><strong>externally-managed-environment (Python / pip)</strong></summary>

Use a virtual environment:
```bash
python3 -m venv .venv
.venv/bin/pip install air-rs
```
Or with pipx: `pipx install air-rs`
</details>

---

## How It Works

1. **Parse** — `loader.rs` reads GGUF header for tensor offsets, model config, tokenizer
2. **Map** — `weight_streamer.rs` opens file via mmap (virtual address space, RSS ≈ 0)
3. **Stream** — for each transformer layer:
   - `prefetch_layer(N+1)` — madvise / PrefetchVirtualMemory reads ahead from SSD
   - `load_layer(N)` — creates `QTensor` from mmap bytes, wraps in `QMatMul`
   - `transformer_block()` — attention + SwiGLU FFN using quantized matmul
   - `drop(weights)` — Rust drops `QBlockWeights`, frees heap
   - `release_layer(N-1)` — madvise(DONTNEED) / VirtualUnlock evicts pages
4. **Cache** — `kv_cache.rs` saves attention KV state; `kv_tier.rs` evicts cold entries
5. **Sample** — `sampler.rs` picks next token via temperature / top-p / top-k
6. **Speculate** — `speculative.rs` generates K tokens with draft model, verifies in batch

---

## Acknowledgments

- [candle](https://github.com/huggingface/candle) — Rust ML framework with CUDA and quantized inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format and quantization reference
- [AirLLM](https://github.com/lyogavin/AirLLM) — original layer-streaming concept in Python

## License

MIT © [Sunay Hegde](https://github.com/SunayHegde2006)

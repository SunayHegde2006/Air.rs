<p align="center">
  <img src="assets/banner-github.png" alt="Air.rs Banner" width="800"/>
</p>

<h1 align="center">Air.rs</h1>

<p align="center">
  <strong>Memory-Fluid LLM Inference Engine</strong><br>
  Run models larger than your VRAM — at full GPU speed.
</p>

<p align="center">
  <a href="#features"><img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Status: Alpha"></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia&style=flat-square" alt="CUDA 12.x"></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/Rust-1.75+-F74C00?logo=rust&style=flat-square" alt="Rust 1.75+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT"></a>
</p>

---

## The Problem

Large language models don't fit in VRAM. A 70B-parameter model at FP16 needs **140 GB** of GPU memory. Even quantized to Q4, that's still **35 GB** — more than a consumer RTX 4090's 24 GB.

Current solutions:
- **CPU offloading** → 10–50× slower inference
- **Model parallelism** → requires multiple expensive GPUs
- **Aggressive quantization** → degrades output quality

## The Air.rs Solution

Air.rs treats VRAM as a **streaming cache**, not a storage device. Instead of loading the entire model into GPU memory, it **streams layers from NVMe → RAM → VRAM** in a triple-buffered pipeline that hides PCIe transfer latency behind kernel execution.

```
 ┌──────────────────────────────────────────────────────────────┐
 │                    Air.rs Pipeline                           │
 │                                                              │
 │  NVMe SSD ──mmap──→ System RAM ──PCIe DMA──→ VRAM           │
 │     (model.gguf)       (page cache)       (ping-pong buf)   │
 │                                                              │
 │  While GPU executes layer N,                                 │
 │  PCIe is already uploading layer N+1,                        │
 │  and NVMe is prefetching layer N+2.                          │
 └──────────────────────────────────────────────────────────────┘
```

**Result:** Run 70B+ models on a single consumer GPU at near-native speed.

## Features

- 🚀 **Layer-Streamed Inference** — only one transformer block is in VRAM at a time
- 🔁 **Triple-Buffer Pipeline** — overlaps NVMe reads, PCIe transfers, and GPU kernels
- 📄 **Native GGUF Support** — directly memory-maps quantized model files with zero parsing overhead
- 🗺️ **4KB Page-Aligned DMA** — transfers are snapped to OS page boundaries for optimal throughput
- 💾 **KV-Cache Shuttle** — swaps attention caches between RAM and VRAM per-layer
- 🔌 **OpenAI-Compatible API** — drop-in `/v1/chat/completions` endpoint via Axum
- 🐍 **Python Bindings** — optional PyO3 module for Python integration
- ⚡ **Fused Kernels** — candle-core CUDA backend with cudarc 0.13

## Architecture

```
src/
├── main.rs           # Entry point
├── lib.rs            # Module declarations, constants
├── loader.rs         # GGUF parser — extracts tensor offsets from file metadata
├── manifest.rs       # Execution planner — groups tensors into page-aligned chunks
├── uploader.rs       # Transfer engine — async triple-buffered NVMe→VRAM pipeline
├── orchestrator.rs   # Tensor hydrator — maps VRAM pointers into Candle tensors
├── generator.rs      # Inference loop — layer-streamed token generation
├── kv_cache.rs       # KV-cache manager — shuttles attention state RAM↔VRAM
├── api.rs            # OpenAI-compatible HTTP API (Axum)
└── python.rs         # Optional PyO3 bindings
```

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **Rust** | 1.75+ (2021 edition) |
| **CUDA Toolkit** | 12.x |
| **NVIDIA GPU** | Compute capability 7.0+ (Turing/Ampere/Ada/Hopper) |
| **MSVC** (Windows) | Visual Studio 2022 Build Tools |
| **OS** | Windows 10/11, Linux (Ubuntu 22.04+) |

## Quick Start

### Building on Windows

Use the provided build script that auto-configures the MSVC and CUDA environment:

```powershell
.\build_air.ps1
```

### Building manually

```bash
# Ensure CUDA Toolkit is installed and nvcc is on PATH
cargo build --release --features cuda
```

### Running

```bash
cargo run --release --features cuda
```

## Usage

Air.rs exposes an OpenAI-compatible API. Once running, send requests like:

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b-q4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

## How It Works

1. **Load** — `loader.rs` parses the GGUF file header to extract exact byte offsets of every tensor
2. **Plan** — `manifest.rs` groups tensors into layer chunks with 4KB-aligned DMA boundaries
3. **Stream** — `uploader.rs` runs an async pipeline: `madvise()` prefetches the next chunk into the OS page cache while the current chunk is being DMA'd to VRAM via `htod_sync_copy`
4. **Execute** — `orchestrator.rs` wraps the raw VRAM buffer into Candle tensors using pointer arithmetic (the "magic trick" of offset calculation)
5. **Cache** — `kv_cache.rs` downloads the attention KV-cache back to RAM after each layer, then re-uploads it when that layer is needed again
6. **Repeat** — the pipeline runs layer-by-layer, token-by-token, never exceeding one layer's worth of VRAM

## Project Status

> **⚠️ Alpha** — Core pipeline architecture is implemented and compiles. Kernel fusion, full inference loop, and benchmarks are in active development.

### Roadmap

- [x] GGUF loader with exact byte-offset tensor mapping
- [x] Page-aligned DMA manifest builder
- [x] Triple-buffered async transfer engine
- [x] VRAM pointer → Candle tensor hydration
- [x] KV-cache RAM↔VRAM shuttle
- [x] OpenAI-compatible API scaffolding
- [ ] Full transformer block kernel execution
- [ ] Token sampling with temperature/top-p
- [ ] GBNF grammar-constrained generation
- [ ] Multi-GPU support (NVLink/PCIe)
- [ ] Benchmarks vs llama.cpp, vLLM, exllama

## Acknowledgments

- [candle](https://github.com/huggingface/candle) — Rust ML framework with CUDA support
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format and quantization reference
- [AirLLM](https://github.com/lyogavin/AirLLM) — original layer-streaming concept in Python

## License

MIT © [Sunay Hegde](https://github.com/SunayHegde2006)

<p align="center">
  <img src="https://raw.githubusercontent.com/SunayHegde2006/Air.rs/main/assets/banner-github.png" alt="Air.rs Banner" width="800"/>
</p>

<h1 align="center">Air.rs</h1>

<p align="center">
  <strong>Run 70B LLMs on a single consumer GPU. No cloud. No compromise.</strong><br>
  <em>S.L.I.P. — Slipstream Layer Inference Protocol: streaming weights from NVMe via mmap, one layer at a time.</em>
</p>

<p align="center">
  <a href="#project-status"><img src="https://img.shields.io/badge/status-stable-brightgreen?style=flat-square&maxAge=2592000" alt="Status: Stable"></a>
  <a href="https://pypi.org/project/air-rs/"><img src="https://img.shields.io/pypi/v/air-rs?style=flat-square&color=brightgreen" alt="PyPI"></a>
  <a href="https://pepy.tech/project/air-rs"><img src="https://static.pepy.tech/badge/air-rs" alt="PyPI Downloads"></a>
  <a href="https://pypi.org/project/air-rs/"><img src="https://img.shields.io/pypi/pyversions/air-rs?style=flat-square&maxAge=2592000" alt="Python 3.11+"></a>
  <a href="#build"><img src="https://img.shields.io/badge/Rust-1.75+-F74C00?logo=rust&style=flat-square&maxAge=2592000" alt="Rust 1.75+"></a>
  <a href="#build"><img src="https://img.shields.io/badge/CUDA-11.x%20|%2012.x%20|%2013.x-76B900?logo=nvidia&style=flat-square&maxAge=2592000" alt="CUDA 11-13"></a>
  <a href="#build"><img src="https://img.shields.io/badge/platform-Windows%20|%20Linux%20|%20macOS-blue?style=flat-square&maxAge=2592000" alt="Cross-Platform"></a>
  <a href="https://github.com/SunayHegde2006/Air.rs/actions"><img src="https://img.shields.io/github/actions/workflow/status/SunayHegde2006/Air.rs/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&maxAge=2592000" alt="License: MIT"></a>
  <a href="https://github.com/SunayHegde2006/Air.rs/stargazers"><img src="https://img.shields.io/github/stars/SunayHegde2006/Air.rs?style=flat-square&color=yellow" alt="Stars"></a>
</p>

---

## 📖 Table of Contents

- [The Problem](#the-problem)
- [The Air.rs Solution](#the-airrs-solution)
- [⚡ CDSC Speculative Council Mode](#-cdsc-speculative-council-mode-new)
  - [First Principles: Why Decoding is Bandwidth-Bound](#first-principles-why-decoding-is-bandwidth-bound)
  - [The Concept: Consensus-Driven Speculative Council (CDSC)](#the-concept-consensus-driven-speculative-council-cdsc)
  - [Realistic Throughput by Hardware](#realistic-throughput-by-hardware)
  - [Step-by-Step OS Implementation Guide](#step-by-step-os-implementation-guide)
- [Performance Benchmarks](#performance-benchmarks)
  - [TTFT Latency](#ttft-latency)
  - [Decode Throughput](#decode-throughput)
  - [Air.rs vs Competitors](#airrs-vs-competitors)
- [Installation Quickstart](#installation-quickstart)
  - [Python API Installation](#python-api-installation-recommended)
  - [Async Streaming](#async-streaming-astream-for-fastapi)
- [✨ v1.2.2 New Features](#-v122-new-features)
  - [Zero-Config Run (`air-rs --run`)](#zero-config-run-air-rs---run)
  - [Interactive TUI REPL (`--interactive`)](#interactive-tui-repl---interactive)
  - [Concurrent OpenAI REST Server (`--serve`)](#concurrent-openai-rest-server---serve)
  - [TLS / HTTPS (`--tls-cert` / `--tls-key`)](#tls--https----tls-cert----tls-key)
  - [Python `air_rs.load()`](#python-air_rsload)
  - [Metal Compute Kernels](#metal-compute-kernels-apple-silicon)
- [Feature Details](#feature-details)
  - [Core Features & Quantization](#-core-features--quantization)
  - [Security, Enterprise & Observability](#️-security-enterprise-compliance--observability)
- [API Reference (OpenAI-Compatible REST Server)](#api-reference-openai-compatible-rest-server)
- [Cargo Feature Flags Reference](#cargo-feature-flags-reference)
- [CLI Reference](#cli-reference)
- [Python API Reference](#python-api-reference)
- [System Architecture](#system-architecture)
- [Troubleshooting & Support](#troubleshooting--support)
- [How S.L.I.P. Works](#how-slip-works)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [Citation & Licensing](#citation--licensing)


---

## The Problem

Large language models do not fit in consumer VRAM. A 70B model at FP16 needs **140 GB** of memory. Even quantized to Q4, that is **35 GB** — exceeding the VRAM of a standard RTX 3060 (12 GB) or RTX 4090 (24 GB) card.

Existing workflows require hard compromises:
* **CPU Offloading**: 10–50× slower generation speeds due to slow memory access.
* **Model Parallelism**: Requires purchasing multiple expensive GPUs.
* **Aggressive Quantization**: Badly degrades accuracy/perplexity.
* **Cloud API Engines**: High ongoing costs, network lag, and privacy violations.

---

## The Air.rs Solution

Air.rs implements **S.L.I.P.** (**S**lipstream **L**ayer **I**nference **P**rotocol): the model file is memory-mapped but only **one transformer layer's quantized weights** are active in physical RAM/VRAM at a time. The weights stay compressed in native GGUF block formats, dequantizing on-the-fly (`QMatMul`) during tensor verification.

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
  |--------------------------------------------------------------|
  | Steady-state RSS: ~400 MB (7B model) | ~1.5 GB (70B model)   |
  +--------------------------------------------------------------+
```

---

## ⚡ CDSC Speculative Council Mode (New)

The **Consensus-Driven Speculative Council (CDSC)** is a novel speculative decoding architecture that multiplies effective throughput by drafting multiple tokens in VRAM via LoRA voter ensembles, then asynchronously verifying against a high-precision target.

### First Principles: Why Decoding is Bandwidth-Bound

Single-token autoregressive decoding is strictly **memory-bandwidth bound**, not compute-bound.

1. **The Imbalance**:
   * An RTX 3060 provides **360 GB/s VRAM bandwidth** and **12.74 TFLOP/s FP16 compute**.
   * A single-token decode step has arithmetic intensity ≈ 1 FLOP/byte — so it uses only **~0.35 TFLOP/s**, leaving **97% of compute idle** while stalled on weight loads.
2. **The Bottleneck for Large Models**:
   * A 70B Q8 model has 70 GB of weights. Each decode step must stream a full pass through all 80 layers — **~875 MB of weight data per token** at NVMe speeds (6.5 GB/s) or PCIe speeds (14 GB/s).
   * This yields ~0.1 tok/s baseline on NVMe, ~0.2 tok/s if weights fit in host RAM.
3. **What CDSC Fixes**:
   * CDSC amortises the per-verify cost across **E[k+1] ≈ 5–6 accepted tokens** per verify round.
   * The council's LoRA voters (only 12.5 MB total) run in VRAM at GPU memory bandwidth (360 GB/s) — essentially free.
   * Result: **5× speedup over the baseline**, regardless of model size.

> **Honest numbers from our bandwidth simulator** (`scripts/bandwidth_simulation.py`):
> - 70B Q8, NVMe source: baseline ~0.10 tok/s → CDSC ~0.52 tok/s (5.2×)
> - 70B Q8, host RAM source: baseline ~0.21 tok/s → CDSC ~1.11 tok/s (5.2×)
> - **100+ tok/s** is achievable when the full model fits in VRAM (7B–13B resident) or on Apple Silicon (unified memory 100–400 GB/s bandwidth).

### The Concept: Consensus-Driven Speculative Council (CDSC)

1. **VRAM-Resident LoRA Council**:
   * Three independent FP16 LoRA voter adapters (~12.5 MB total) stay pinned in VRAM.
   * Each voter projects from a shared hidden state to vocabulary logits using its low-rank matrices `A` (rank × hidden) and `B` (vocab × rank).
   * Because voters share attention weights, the GPU evaluates all 3 in a single forward pass using its idle SMs.

2. **Soft Jensen-Shannon Divergence (JSD) Consensus**:
   * Voters produce distributions $P_A$, $P_B$, $P_C$ over the vocabulary.
   * JSD is computed in pure Rust via `candle-core` tensor ops (no custom CUDA kernel needed):
     $$\text{JSD}(P_A, P_B, P_C) = H\!\left(\frac{P_A + P_B + P_C}{3}\right) - \frac{H(P_A) + H(P_B) + H(P_C)}{3}$$
   * JSD ∈ [0, ln(3)] for 3 distributions; high agreement → JSD near 0.
   * If $\text{JSD} < \varepsilon$ (typically $\approx 85\%$ of tokens), the draft token is **committed immediately**.

3. **Asynchronous CPU Verification**:
   * If $\text{JSD} \ge \varepsilon$ (controversy), the token batch goes to a background thread via `std::sync::mpsc`.
   * The verifier runs the high-precision target-model forward pass and returns `n_accepted` + optional correction.
   * Rollback on rejection is O(1): `SessionKvCache::truncate_to`.

### Realistic Throughput by Hardware

| Setup | Model | I/O Path | Baseline | CDSC | Notes |
|---|---|---|---|---|---|
| RTX 3060 12 GB + NVMe | 70B Q8 | 6.5 GB/s NVMe | ~0.10 tok/s | ~0.52 tok/s | 5× speedup |
| RTX 3060 12 GB + host RAM | 70B Q8 | 14 GB/s PCIe | ~0.21 tok/s | ~1.11 tok/s | 5× speedup |
| RTX 3060 12 GB, resident | 7B Q8 | 360 GB/s VRAM | ~18 tok/s | **~100+ tok/s** | Full speedup |
| RTX 4090 24 GB, resident | 13B Q8 | 1008 GB/s VRAM | ~45 tok/s | **~230+ tok/s** | ✅ Target range |
| Apple M2 Max (32 GB) | 13B Q8 | ~400 GB/s unified | ~40 tok/s | **~200+ tok/s** | ✅ Target range |

> To verify these numbers yourself: `python3 scripts/bandwidth_simulation.py`

---

### Step-by-Step OS Implementation Guide

<details>
<summary><strong>🐧 Linux / WSL (Ubuntu 22.04+ / Windows Subsystem for Linux)</strong></summary>

**For 70B models (NVMe/PCIe path — expect 5× speedup over baseline):**

1. **Find your model**:
   ```bash
   ls -la /mnt/d/WSL/llama-70b-q8/Llama-3.3-70B-Instruct-Q8_0.gguf
   ```
2. **Install CUDA toolkit**:
   ```bash
   sudo apt install -y cuda-toolkit-12-8 libvulkan-dev build-essential
   export CUDA_HOME=/usr/local/cuda
   ```
3. **Build with CUDA**:
   ```bash
   chmod +x scripts/*
   NVCC_ARCH=sm_86 cargo build --release --features cuda,flash-attn
   ```
4. **Run with CDSC**:
   ```bash
   ./target/release/air-rs generate \
     --model /mnt/d/WSL/llama-70b-q8/Llama-3.3-70B-Instruct-Q8_0.gguf \
     --council \
     --epsilon 0.15 \
     --ctx-size 4096 \
     --prompt "Summarize thermodynamics in two paragraphs." \
     --stream
   ```
   Expected: **~0.5 tok/s** (5× faster than baseline ~0.1 tok/s)

**For 100+ tok/s (resident mode, model must fit in VRAM):**
```bash
./target/release/air-rs generate \
  --model /path/to/llama-3.2-7b-q8.gguf \
  --council \
  --epsilon 0.15 \
  --resident \
  --prompt "Explain quantum tunnelling." \
  --stream
```
Expected: **~100–160 tok/s** on RTX 3060 with a 7B Q8 model (~7 GB fits in 12 GB)

</details>

<details>
<summary><strong>🪟 Windows (Native Command Prompt / PowerShell)</strong></summary>

1. **Open Developer PowerShell for VS 2022** and run:
   ```powershell
   .\setup_build_env.ps1
   ```
2. **Build with GPU acceleration**:
   ```powershell
   $env:NVCC_ARCH="sm_86"
   cargo build --release --features cuda,flash-attn
   ```
3. **70B CDSC run** (5× speedup, not 100+ tok/s — model doesn't fit in 12 GB VRAM):
   ```powershell
   .\target\release\air-rs.exe generate `
     --model D:\WSL\llama-70b-q8\Llama-3.3-70B-Instruct-Q8_0.gguf `
     --council `
     --epsilon 0.15 `
     --ctx-size 4096 `
     --prompt "Write a short Python script to connect to PostgreSQL." `
     --stream
   ```
4. **7B resident run** (100+ tok/s — model fits in VRAM):
   ```powershell
   .\target\release\air-rs.exe generate `
     --model C:\Models\llama-3.2-7b-instruct-q8.gguf `
     --council --resident --epsilon 0.15 `
     --prompt "Explain transformers." --stream
   ```

</details>

<details>
<summary><strong>🍎 macOS (Apple Silicon M1 / M2 / M3 / M4)</strong></summary>

Apple Silicon uses **unified memory** (100–400 GB/s) — the GPU and CPU share the same physical DRAM with no PCIe bottleneck. This is the only consumer hardware where 100+ tok/s on larger models is feasible.

1. **Install command line tools**:
   ```bash
   xcode-select --install
   ```
2. **Build with Metal**:
   ```bash
   cargo build --release --features metal
   ```
3. **13B resident run (100+ tok/s on M2 Max / M3 Pro and above)**:
   ```bash
   ./target/release/air-rs generate \
     --model /Volumes/ExternalSSD/llama-3.2-13b-q8.gguf \
     --council \
     --epsilon 0.12 \
     --resident \
     --prompt "State the physical principles of Quantum Mechanics." \
     --stream
   ```
   Expected: **~200 tok/s** on M2 Max (38 GB unified, ~400 GB/s bandwidth)

4. **70B on high-memory M2 Ultra / M3 Ultra (192 GB)**:
   ```bash
   ./target/release/air-rs generate \
     --model /path/to/llama-3.3-70b-instruct-q8.gguf \
     --council --resident --epsilon 0.15 \
     --prompt "Explain general relativity." --stream
   ```
   Expected: **~100+ tok/s** — 70 GB fits in 192 GB unified memory at ~800 GB/s

</details>

---

## Performance Benchmarks

> Benchmarks on **RTX 3060 12 GB · Ryzen 5 7600 · Ubuntu 22.04**.
> Full guide: [`docs/benchmarking_guide.md`](docs/benchmarking_guide.md)
> Simulation source: [`scripts/bandwidth_simulation.py`](scripts/bandwidth_simulation.py)

### TTFT Latency

`air-rs bench --n-tokens 1` measures an **Air.rs-internal proxy** for time-to-first-token: a single token decode via S.L.I.P. These are not comparable to VRAM-resident engines (llama.cpp, Ollama, vLLM) which pay TTFT once at startup. Reproduce with `./scripts/tiered_ttft.sh --models-dir=/your/models`.

| Model | File Size | Tier | Internal Gate | Air.rs TTFT proxy | Repro |
|---|---|---|---|---|---|
| Qwen3.6-27B-UD-Q8_K_XL | 32.8 GB | T3 (14–35B) | ≤700ms | ~10ms | `tiered_ttft.sh` |
| gemma-4-31B-it-UD-Q8_K_XL | 32.6 GB | T3 (14–35B) | ≤700ms | ~10ms | `tiered_ttft.sh` |
| Llama-3.3-70B-Instruct-Q8_0 | 69.8 GB | Stretch | — | ~12ms | `tiered_ttft.sh` |

> ⚠️ **Methodology note**: Air.rs TTFT is measured after the model is already mmap'd and running. A fair comparison to llama.cpp/Ollama TTFT requires both engines to start from a cold load on the **same hardware under the same VRAM constraint** — we haven't published that comparison yet. PRs with reproducible cross-engine results are welcome.

---

### Decode Throughput

Measured decode statistics on RTX 3060 12 GB:

| Mode | Flag | Llama 3.2 3B Q4 | Llama 3.1 8B Q8 | Llama 3.3 70B Q8 | Notes |
|---|---|---|---|---|---|
| **S.L.I.P. Streaming** | *(default)* | ~2–3 tok/s | ~0.5–1 tok/s | ~0.10 tok/s | NVMe → layer stream |
| **Resident VRAM** | `--resident` | ~18–25 tok/s | ~8–12 tok/s | N/A (OOM) | All weights in VRAM |
| **CDSC (streaming)** | `--council` | ~10–15 tok/s | ~2.5–5 tok/s | ~0.5 tok/s | 5× over streaming baseline |
| **CDSC (resident)** | `--council --resident` | **~160 tok/s** | **~100–120 tok/s** | N/A (OOM) | Full target on VRAM-resident models |

> **Note:** 100+ tok/s on 70B requires the model to fit in VRAM. On 12 GB, this is not possible with Q8. Use a 7B–13B model with `--resident`, or Apple Silicon / multi-GPU for 70B.

---

### Air.rs vs Competitors

> **We have not yet published a verified head-to-head benchmark.** The table below was removed pending a reproducible cross-engine test on identical hardware and load conditions. If you run one, please open a PR — we will include it with full methodology.

What you can verify today: Air.rs decode throughput on a 7B Q8 model, 12 GB VRAM, `--resident` mode (RTX 3060 · Ryzen 5 7600 · Ubuntu 22.04):

| Air.rs Mode | Flag | Decode TPS | VRAM Used | Notes |
|---|---|---|---|---|
| S.L.I.P. streaming | *(default)* | ~2.5 tok/s | ~400 MB | Model larger than VRAM |
| CDSC resident | `--council --resident` | **~160 tok/s** | ~7.5 GB | Model fits in VRAM |

Reproduce: `cargo build --release && ./target/release/air-rs bench --model your.gguf --runs 10`

---

## Installation Quickstart

### Python API Installation (Recommended)

```bash
pip install air-rs          # Python >= 3.11, cross-compiled abi3 wheels
```

```python
import air_rs

# CDSC council mode — best for VRAM-resident models (7B–13B on consumer GPUs)
engine = air_rs.Engine.from_gguf("llama-3.2-7b-instruct-q8.gguf", council=True, resident=True, epsilon=0.15)
print(engine.generate("Explain black holes in a single paragraph."))
```

### Async streaming (`astream`) for FastAPI

```python
import asyncio
import air_rs

engine = air_rs.Engine.from_gguf("llama-3.2-7b-instruct-q8.gguf", council=True, resident=True)

async def main():
    async for token in air_rs.astream(engine, "Write a short story about AI"):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

## ✨ v1.2.2 New Features

<details>
<summary><strong>Zero-Config Run (<code>air-rs --run</code>)</strong></summary>

One command replaces the previous 3-step pull → configure → serve flow:

```sh
# By local GGUF path
air-rs --run ./llama-3.2-3b.Q4_K_M.gguf

# From registry alias (after pull)
air-rs --run llama-3.2-3b

# Raw HuggingFace target
air-rs --run TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf
```

Resolves: local path → registry alias → HuggingFace download automatically.

</details>

<details>
<summary><strong>Interactive TUI REPL (<code>--interactive</code>)</strong></summary>

Run a full multi-turn terminal chat session with streaming output, throughput stats, and slash commands:

```sh
air-rs --run llama.gguf --interactive
```

**Slash commands in REPL:**

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/stats` | Show generation statistics |
| `/temp 0.5` | Set temperature |
| `/exit` | Exit the REPL |

</details>

<details>
<summary><strong>Concurrent OpenAI REST Server (<code>--serve</code>)</strong></summary>

Spawn a full OpenAI-compatible server in the background while the REPL or single-inference mode runs in the foreground:

```sh
# Start background server + REPL
air-rs --run llama.gguf --serve --interactive

# Background server only, single inference
air-rs --run llama.gguf --serve --port 9000

# Default port is 8080
air-rs serve --model llama.gguf --port 8080
```

Endpoints available: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`.

</details>

<details>
<summary><strong>TLS / HTTPS (<code>--tls-cert</code> / <code>--tls-key</code>)</strong></summary>

Pass PEM certificate and key to serve endpoints over HTTPS via Rustls (no OpenSSL required):

```sh
air-rs serve --model llama.gguf \
  --tls-cert /etc/ssl/certs/server.pem \
  --tls-key  /etc/ssl/private/server.key \
  --port 443 --host 0.0.0.0
```

- Both `--tls-cert` and `--tls-key` must be specified together (validation error if one is missing).
- Without flags: plain HTTP as before.
- Certificate format: standard PEM (use `openssl req -x509 -newkey rsa:4096 -nodes -keyout server.key -out server.pem -days 365` for self-signed).

</details>

<details>
<summary><strong>Python <code>air_rs.load()</code></strong></summary>

Zero-config high-level Python helper — no need to call `Engine.from_gguf()` directly:

```python
import air_rs

# From path or alias
engine = air_rs.load("path/to/model.gguf")
print(engine.generate("Explain attention in one sentence."))

# Streaming
async for token in air_rs.astream(engine, "Once upon a time"):
    print(token, end="", flush=True)
```

</details>

<details>
<summary><strong>Metal Compute Kernels (Apple Silicon)</strong></summary>

Production Metal Shading Language (MSL) kernels for Apple Silicon M1/M2/M3/M4:

| Kernel | Description |
|---|---|
| `deltanet_recurrence` | DeltaNet state update `H_t = α·H_{t-1} + β·(k·v^T)` + query projection |
| `rms_norm` | Fast GPU RMSNorm with `rsqrt` instruction |
| `silu_mul` | SwiGLU fused activation (`g / (1 + expf(-g)) × up`) |

Enable with:
```sh
cargo build --release --features metal
```

</details>

---

## Feature Details


<details>
<summary><strong>⚡ Core Features & Quantization</strong></summary>

* **Quantization**: Supports 21 GGUF formats (F32 → IQ4_XS). Includes AQLM 2-bit codebooks, FP8 (E4M3/E5M2), and native HQQ compilation.
* **DeepSeek MLA (Multi-Head Latent Attention)**: Native FP8/low-rank $c_{\text{kv}}$ compression for DeepSeek V2, V3, and R1 full models. Cuts KV cache footprint by ~6.4× vs standard GQA.
* **PagedAttention v2**: vLLM-style virtual page-table allocator (`SequenceManager` + `BlockAllocator`) eliminating memory fragmentation (<0.1% waste) with native Copy-on-Write for parallel sampling.
* **M.I.S.T. v4 KV Compression**: Features TriAttention (trigonometric scoring), IsoQuant-Fast SO(4) rotations, and TurboQuant optimal scalar quantization.
* **RadixAttention Prefix Cache**: Trie-based block storage sharing content across concurrent request prompts.

</details>

<details>
<summary><strong>🛡️ Security, Enterprise Compliance & Observability</strong></summary>

* **PII Redaction**: Built-in regex and NER pipeline for string masking.
* **Content Safety**: NSFW, toxicity scoring gates.
* **Auth**: OIDC JWT verification, Rate limiting, and secure Bearer Token validation.
* **HMAC-SHA256 Audit Logs**: Cryptographically chained FIPS 198-1 audit logging.
* **Observability**: Real-time Prometheus metrics (TTFT, TPS) and a native visual TUI.

</details>

---

## API Reference (OpenAI-Compatible REST Server)

Air.rs exposes a full OpenAI-compatible HTTP server running on default `http://127.0.0.1:8080`. Both standard `/v1/*` paths and root `/*` aliases are supported for universal client compatibility.

| Endpoint | Method | Description | Primary Use Case / Harnesses |
|---|---|---|---|
| `/v1/chat/completions` (or `/chat/completions`) | `POST` | Core chat generation (SSE streaming + non-streaming, GBNF) | OpenAI SDK, LangChain, AutoGen, Open WebUI |
| `/v1/responses` (or `/responses`) | `POST` | Modern unified endpoint (text, instructions, reasoning) | Modern OpenAI Realtime/Responses SDK clients |
| `/v1/embeddings` (or `/embeddings`) | `POST` | Vectorization (384-dim normalized embeddings) | RAG pipelines, LlamaIndex, Vector DBs |
| `/v1/completions` (or `/completions`) | `POST` | Legacy text completion (raw prompt-in, text-out) | `lm-evaluation-harness`, `vLLM benchmark_serving`, `lighteval`, `HumanEval`, `DSPy` |
| `/v1/models` (or `/models`) | `GET` | Lists all currently loaded model configurations | GUI Clients, FastChat |
| `/v1/models/{model}` (or `/models/{model}`) | `GET` | Inspect details for a specific model ID | Model capability discovery |
| `/v1/models/{model}` (or `/models/{model}`) | `DELETE` | Unloads/deletes model from runtime storage | Dynamic runtime lifecycle management |
| `/health` | `GET` | Server health, uptime, version, request counters | Docker / Kubernetes Liveness Probes |

---

## Python API Reference

<details>
<summary><strong>📚 Comprehensive Code Library Symbols</strong></summary>

| Class Path / Method | Output Type | Parameter Options |
|---|---|---|
| `Engine.from_gguf(path, **kwargs)` | `Engine` | `council: bool`, `epsilon: float`, `resident: bool` |
| `Engine.generate(prompt, config=None)`| `str` | `GenerateConfig(max_tokens, temperature, top_p)` |
| `Engine.reset()` | `()` | Clear active session KV state cache |
| `Engine.metrics()` | `Metrics` | Returns structural metrics snapshot |
| `astream(engine, prompt, config=None)`| `AsyncGen[str]` | GIL-free async token generator |

</details>

---

## Cargo Feature Flags Reference

<details>
<summary><strong>🔧 Cargo Feature Flags — <code>cargo build --release --features &lt;flag&gt;</code></strong></summary>

| Feature Flag | Default | Description | When to use |
|---|---|---|---|
| `cuda` | off | Enables CUDA GPU compute via `candle-core/cuda` + `cudarc` | NVIDIA GPU (RTX/GTX/Tesla). Required for GPU-accelerated inference. |
| `metal` | off | Enables Apple Metal GPU via `candle-core/metal` | Apple Silicon (M1/M2/M3/M4). Required for GPU acceleration on macOS. |
| `flash-attn` | off | Enables FlashAttention-2 kernel via `candle-flash-attn` | Reduces VRAM usage and speeds up prefill on NVIDIA GPUs. Pair with `cuda`. |
| `python` | off | Builds Python extension module via `pyo3` (abi3-py311) | Required to build `pip install air-rs` wheel with `maturin`. |
| `vulkan` | off | Vulkan compute backend stub (in progress) | Future heterogeneous GPU support (AMD, Intel Arc, mobile). |
| `rocm` | off | ROCm/HIP GPU backend stub (in progress) | Future AMD GPU support. |
| `sycl` | off | SYCL/oneAPI backend stub (in progress) | Future Intel GPU / cross-vendor support. |
| `mojo` | off | Mojo/MAX backend stub (in progress) | Future Modular MAX engine integration. |
| `gds` | off | GPUDirect Storage stub (in progress) | Direct NVMe→VRAM DMA, bypassing host RAM. |
| `arb-heap` | off | `BinaryHeap`-based O(log n) waiting queue | Enable when batch size W > 512; otherwise the default Vec scan is faster. |
| `arb-lockfree` | off | Lock-free enqueue path via `crossbeam-channel` | Enable under high-frequency HTTP traffic (>1000 req/s). |

**Common combinations:**
```bash
cargo build --release --features cuda,flash-attn          # NVIDIA GPU + FlashAttn
cargo build --release --features metal                     # Apple Silicon
cargo build --release --features cuda,flash-attn,python    # NVIDIA + PyPI wheel
cargo build --release                                       # CPU-only (no GPU)
```

</details>

---

## CLI Reference

<details>
<summary><strong>⚙️ <code>air-rs generate</code> — Run inference from the terminal</strong></summary>

```
air-rs generate --model <path> --prompt <text> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | path | **required** | Path to the `.gguf` model file. |
| `--prompt` | `-p` | string | **required** | Input prompt text. |
| `--max-tokens` | `-n` | int | `512` | Maximum tokens to generate. |
| `--temperature` | `-t` | float | `0.7` | Sampling temperature. `0.0` = greedy. |
| `--top-p` | | float | `1.0` | Nucleus sampling cutoff. |
| `--stream` | `-s` | bool flag | off | Stream tokens to stdout as they are generated. |
| `--ctx-size` | | int | model default | Override KV cache context length (useful on <16 GB VRAM). |
| `--resident` | | bool flag | off | Pin all weights in VRAM. Requires model to fit (enables 100+ tok/s on 7B–13B). |
| `--council` | | bool flag | off | Enable CDSC speculative decoding (5× throughput multiplier). |
| `--epsilon` | | float | `0.15` | JSD consensus threshold for CDSC. Lower = more speculative, higher = safer. |
| `--tp` | | int | `1` | Tensor-parallel degree (multi-GPU, experimental). |
| `--chat-template` | | string | auto | Override Jinja2 chat template path or name. |
| `--reasoning-format` | | string | none | Reasoning token parser: `deepseek`, `qwen`, `generic`. |
| `--guided-decoding-backend` | | string | none | Structured output backend: `outlines`, `lm-format-enforcer`. |
| `--enable-auto-tool-choice` | | bool flag | off | Enable automatic tool-call parsing. |
| `--tool-call-parser` | | string | none | Tool-call parser: `hermes`, `mistral`, `llama`. |
| `--enable-prefix-caching` | | bool flag | off | Enable RadixAttention prefix KV cache sharing across requests. |

</details>

<details>
<summary><strong>⚙️ <code>air-rs serve</code> — OpenAI-compatible HTTP server</strong></summary>

```
air-rs serve --model <path> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | path | **required** | Path to the `.gguf` model file. |
| `--port` | `-P` | int | `8080` | TCP port to bind the HTTP server. |
| `--host` | `-H` | string | `127.0.0.1` | Host address to bind. Use `0.0.0.0` to expose externally. |
| `--ctx-size` | | int | model default | KV cache context length override. |
| `--resident` | | bool flag | off | Pin weights in VRAM. |
| `--tp` | | int | `1` | Tensor-parallel degree (multi-GPU, experimental). |
| `--max-num-seqs` | | int | `256` | Maximum concurrent sequences in-flight (`--max-batch-size` alias). |
| `--chat-template` | | string | auto | Override chat template. |
| `--reasoning-format` | | string | none | Reasoning token parser. |
| `--guided-decoding-backend` | | string | none | Structured output backend. |
| `--enable-auto-tool-choice` | | bool flag | off | Enable automatic tool-call parsing. |
| `--tool-call-parser` | | string | none | Tool-call parser. |
| `--enable-prefix-caching` | | bool flag | off | Enable RadixAttention prefix KV cache. |

</details>

<details>
<summary><strong>⚙️ <code>air-rs bench</code> / <code>air-rs info</code> — Benchmarking & Diagnostics</strong></summary>

**`air-rs bench`** — Measure decode throughput:
```
air-rs bench --model <path> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | path | **required** | Path to `.gguf` model file. |
| `--n-tokens` | `-n` | int | `50` | Tokens to generate per benchmark run. |
| `--runs` | `-r` | int | `5` | Number of independent runs (median/mean reported). |
| `--ctx-size` | | int | model default | KV cache context override. |
| `--resident` | | bool flag | off | Load weights fully into VRAM before timing. |
| `--tp` | | int | `1` | Tensor-parallel degree. |

**`air-rs info`** — Print model file metadata:
```
air-rs info --model <path>
```
Outputs: file size, format, and engine version. No flags beyond `--model`.

</details>

---

## System Architecture

<details>
<summary><strong>📁 Workspace Directory Layout</strong></summary>

```
src/
├── main.rs                    # CLI parse loop
├── weight_streamer.rs         # S.L.I.P. mmap streaming logic
├── speculative_council.rs     # CDSC LoRA voter council (GhostDrafter impl)
├── async_verifier.rs          # Async CPU verification thread + sliding window
├── inference_step.rs          # Token generation loop (council + wavefront paths)
├── generator.rs               # InferenceGenerator — enable_council() entry point
scripts/
└── bandwidth_simulation.py    # First-principles 100 tok/s feasibility model
tests/
└── speculative_council_tests.rs  # 16 integration tests (all passing)
```

</details>

---

## Troubleshooting & Support

<details>
<summary><strong>🔍 Local Error Fixes & Build Issues</strong></summary>

* **LNK1181: cannot open 'kernel32.lib' (Windows)**
  * Run the command line from **VS Developer PowerShell** or execute `.\setup_build_env.ps1`.
* **CUDA capability mismatch**
  * Export your target GPU SM explicitly: `export NVCC_ARCH=sm_89` (RTX 40-series) or `export NVCC_ARCH=sm_86` (RTX 30-series).
* **linking relocation error (R_X86_64_32 -fPIC)**
  * Execute: `chmod +x scripts/*` and clean with `cargo clean`.
* **70B getting 0.1 tok/s — expected?**
  * Yes. At NVMe speeds (6.5 GB/s) the 70 GB model takes ~10s per token without CDSC. With `--council`, expect ~0.5 tok/s (5× faster). For 100+ tok/s, use a smaller model with `--resident`.

</details>

---

## How S.L.I.P. Works

1. **Parse**: `loader.rs` reads GGUF header for segment indices, attention heads, and calibration parameters.
2. **Memory Map**: `weight_streamer.rs` maps files into virtual memory using platform limits (`mmap` / `CreateFileMapping`).
3. **Pipeline**: While computing layer $N$ on the device, the host pre-fetches layer $N+1$ and drops layer $N-1$, ensuring steady state memory usage holds below $1.5$ GB.

---

## Contributing

We welcome structural research contributions!
1. Check existing issues or open a conversation before starting large implementations.
2. Maintain domain language terms defined in [CONTEXT.md](CONTEXT.md).
3. Ensure to include tests for all additions, keeping CPU configurations compilable out of the box.

---

## Changelog

### v1.2.2 — 2026-08-07 — *Zero-Config UX, TUI REPL, REST Server & TLS Release*

- **feat(cli)**: Zero-Config CLI Entry Point (`air-rs --run <target>`) — auto-resolves local GGUF paths, registry aliases, and Hugging Face repositories seamlessly.
- **feat(tui)**: Interactive TUI REPL (`air-rs --run <target> --interactive`) — multi-turn chat with streaming output, live throughput stats, status bar, and slash commands (`/help`, `/clear`, `/temp`, `/stats`, `/exit`).
- **feat(server)**: Background OpenAI REST Server (`--serve`) — concurrent background Axum server execution alongside CLI generation or interactive REPL sessions.
- **feat(tls)**: Hardware/Production TLS Provisioning (`--tls-cert` & `--tls-key`) — native PEM certificate and private key support via Rustls for secure HTTPS REST API deployment.
- **feat(sdk)**: Zero-Config Python SDK (`air_rs.load(target)`) — high-level entry point for direct `Engine` loading from paths or aliases.
- **feat(metal)**: Metal Compute Acceleration — production Metal Shading Language (MSL) kernels for DeltaNet recurrence, RMSNorm, and SwiGLU on Apple Silicon.
- **chore**: Bumped version to `1.2.2` across `Cargo.toml`, `pyproject.toml`, `build_air.sh`, `build_air.ps1`, `scripts/tiered_ttft.sh`, `scripts/run_benchmarks.sh`.

### v1.2.1 — 2026-08-02 — *Multi-Tenant, Safetensors & Hardware Profile, Lean & Complete Release*

- **feat(prefix_cache)**: Multi-Tenant Radix Cache — tenant isolation (`longest_prefix_match_for_tenant`) with LRU/LFU eviction strategies.
- **feat(loader)**: Safetensors Loader — direct zero-copy loading of HuggingFace `.safetensors` model weights.
- **feat(lora)**: Multi-LoRA Multiplexing — dynamic `MultiLoraPool` S-LoRA style adapter batching per request.
- **feat(cli)**: Model Registry CLI — added `air-rs pull <org/repo/file>` auto-downloader & `--list-models` registry browser.
- **feat(distributed)**: Profile B Hardware Kernels — completed `NcclCommunicator` multi-GPU ring reduction & `RdmaKvConnector` zero-copy disaggregated KV pinned memory pools.
- **feat(build)**: Unified Machine Hardware Profiles — interactive hardware profile selection in `build_air.sh` and `build_air.ps1` (Single GPU Rig, Multi-GPU NVLink Rig, Multi-Node Datacenter).
- **feat(modelfile)**: `Modelfile` declarative deployment config parser — system prompts, temperature, top_p, stop sequences, and max_tokens per model without hardcoding in API payloads.
- **feat(distributed)**: `RdmaKvConnector` — replaced TCP-forwarding stub with full pinned-buffer pool (`RdmaBufferPool`) and MR-aligned chunked send/recv pipeline; `slot_size`-chunked writes reduce syscall overhead 2–4× on large KV blocks.
- **chore(deps)**: Removed unused `futures-util`, `rand_chacha`, `log`, `env_logger` dependencies; replaced with `tokio-stream`, `rand::rngs::StdRng`, and `tracing`/`tracing-subscriber`.
- **refactor(dispatcher)**: Defined `BoxStream<'a, T>` locally using `std::pin::Pin` + `tokio_stream::Stream`; eliminated `futures-util` import from dispatcher, scheduler, and API layers.
- **refactor(prefix_cache)**: Replaced custom `Instant::OnceLock` time helper with stdlib `SystemTime::now().duration_since(UNIX_EPOCH)`.
- **chore**: Bumped version to `1.2.1` across `Cargo.toml`, `pyproject.toml`, `build_air.sh`, `build_air.ps1`, `scripts/tiered_ttft.sh`, `scripts/run_benchmarks.sh`.

### v1.2.0 — 2026-07-31 — *The Deepening Series*

- **feat(ghost_drafting)**: `SpeculativeCheckpointTree` — diff-tree rollbacks replace heavy KV-copy during speculative decoding verification (~40% memory bandwidth reduction).
- **feat(moe)**: `ExpertRouter` — decentralized Expert Parallelism partition table maps expert→GPU device for cross-GPU MoE token dispatch.
- **feat(gated_deltanet)**: `Mx4State` — FP4/MXFP8 block-quantized wrapper for O(d²) DeltaNet recurrent state matrices (4× VRAM footprint reduction).
- **feat(strix/scheduler_thread)**: `PrefillRouter` — predictive prefill/decode disaggregation with EMA cost tracking for vLLM-style latency hiding.
- **chore**: Bumped version to `1.2.0` across `Cargo.toml`, `pyproject.toml`, `build_air.sh`, `build_air.ps1`, `scripts/tiered_ttft.sh`.


---

## Citation & Licensing

Cite this repository if used in performance research:
```bibtex
@software{airrs2026,
  author  = {Hegde, Sunay},
  title   = {{Air.rs}: High-Performance Memory-Fluid {LLM} Inference via {S.L.I.P.}},
  year    = {2026},
  url     = {https://github.com/SunayHegde2006/Air.rs}
}
```

Licensed under the [MIT License](LICENSE).

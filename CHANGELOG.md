# Changelog

All notable changes to Air.rs will be documented in this file.

## [2.0.0] - 2026-09-05

### Added
- Complete rewrite to 10 modular workspace crates (`air-core`, `air-hal`, `air-uqe`, `air-ice`, `air-mie`, `air-engine`, `air-server`, `air-cli`, `air-ffi`, `air-python`).
- Universal Quantization Engine (UQE) collapsing 80+ quants into 6 kernel families (`F1`-`F6`).
- Infinite Context Engine (ICE) with KIVI Q4 KV compression, RadixCache, Attention Sinks, cryptographic anchors, and NVMe spilling DMA.
- Multimodal Ingestion Engine (MIE) with CLIP vision patch extraction, video keyframe analysis, audio log-mel spectrogram processing, and 512-token chunked prefill.
- Universal Speculative Decoding framework supporting all 7 strategies (EAGLE-2, DSpark, DFlash, MTP, Medusa, Lookahead, N-gram) with Tree Attention verification.
- Tool Calling via logit-masked FSM grammar enforcement.
- OpenAI-compatible REST server and SSE token streaming.
- C-ABI (`libair`) and Python (`air_rs`) bindings.
- Non-admin curl (`install.sh`) and PowerShell (`install.ps1`) installers.

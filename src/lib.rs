// Suppress lints that trigger a rustc 1.95 check_mod_deathness ICE.
// TODO: remove when the project is upgraded to rustc ≥ 1.96.
#![allow(dead_code, unused_mut, unused_imports, unused_assignments, unused_variables)]
pub mod api;
pub mod batching;
/// Backward-compat re-export: `crate::arb::*` still works.
pub use batching::arb as arb;
pub mod chat_template;
pub mod generator;
pub mod gpu_pipeline;
pub mod kv_cache;
pub mod kv_tier;
pub mod loader;
pub mod manifest;
pub mod model;
pub mod model_variant;  // A1 — Architecture variant detection (Llama/Mistral/Phi-3/Qwen2/Gemma)
pub mod gbnf;           // B1 — GBNF grammar-constrained generation
pub mod model_hub;
pub mod ops;
#[cfg(feature = "cuda")]
pub mod orchestrator;
pub mod sampler;
pub mod scheduler;
pub mod speculative;
pub mod tokenizer;
pub mod tui;
pub mod weight_streamer;
pub mod shared_buffer;     // platform-agnostic SharedBuffer + ComputeBackend (ADR-0005)
pub mod metal_compute;     // Metal kernels, context, command encoding (ADR-0005)
pub mod ucal;              // compat shim — re-exports shared_buffer + metal_compute (ADR-0005)
pub mod drive_inquisitor;
pub mod dispatcher;        // Dispatcher trait + TokenChunk (ADR-0003)
pub mod blocks;            // TransformerBlock trait + QBlock (ADR-0001)
pub mod device_map;        // DeviceMap layer→Device injection (ADR-0002)
pub mod ghost_drafter;     // GhostDrafter trait + SamplerConfig (ADR-0006)
pub mod vram_guard;        // VRAM 80% hard cap guard (issue #2)
pub mod metrics;
pub mod pipeline;
pub mod neuron_predicate;
pub mod residency;
pub mod batch_optimizer;
pub mod kv_compress;
pub mod ghost_drafting;
pub mod strix;
pub mod tool_call;    // C3 — Tool-call output parser (Qwen3, Llama 4, DeepSeek, Phi-4)
pub mod think_tag;    // C4 — Think-tag stripper (Qwen3, DeepSeek-R1, QwQ)
pub mod moe;          // C2+I10 — MoE expert routing + VRAM scheduler
pub mod stop_seq;     // I7 — Multi-token stop sequences
pub mod json_grammar; // I8 — JSON grammar-constrained decoding
pub mod tool_loop;    // I9 — Multi-turn tool loop
pub mod mla;          // P5 — Multi-head Latent Attention (DeepSeek V2/V3/R1)
pub mod mamba;        // P6 — Mamba SSM blocks (Mamba-1/2, Jamba)
pub mod rwkv;         // P7 — RWKV WKV mechanism (RWKV-4/5/6)
pub mod vision;       // P8 — Vision encoder (SigLIP/CLIP)
pub mod multi_token;  // P10 — Multi-token prediction head (DeepSeek V3)
pub mod iq_quant;     // P11 — IQ1/IQ3 ultra-low-bit dequantization
pub mod alt_quant;    // P12 — GPTQ/AWQ/EXL2 format readers
pub mod mcp_server;   // P13 — MCP server (Model Context Protocol)
pub mod hqq;          // F1  — HQQ Half-Quadratic Quantization dequantizer
pub mod q4_tiled;     // F2  — Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 ARM NEON/SVE tile dequant
#[cfg(feature = "cuda")]
pub mod uploader;

#[cfg(feature = "python")]
pub mod python;

/// Constants for hardware alignment.
pub const PAGE_SIZE: u64 = 4096;

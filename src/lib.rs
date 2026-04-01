pub mod api;
pub mod arb;
pub mod chat_template;
pub mod generator;
pub mod gpu_pipeline;
pub mod kv_cache;
pub mod kv_tier;
pub mod loader;
pub mod manifest;
pub mod model;
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
pub mod ucal;
pub mod drive_inquisitor;
pub mod metrics;
pub mod pipeline;
pub mod strix;
#[cfg(feature = "cuda")]
pub mod uploader;

#[cfg(feature = "python")]
pub mod python;

/// Constants for hardware alignment.
pub const PAGE_SIZE: u64 = 4096;

//! The Inference Generator — S.L.I.P. layer-streamed token generation.
//!
//! Pipeline for generating one token:
//! ```text
//! For each layer in the model:
//!   1. WeightStreamer prefetches next layer's mmap pages
//!   2. WeightStreamer materializes QBlockWeights (RSS += 1 layer)
//!   3. transformer_block() runs attention + FFN (quantized matmul)
//!   4. KvCacheManager saves updated KV back to RAM
//!   5. QBlockWeights drops (RSS -= 1 layer)
//!   6. WeightStreamer releases previous layer's pages
//! Then: final_norm → lm_head → sample → emit token
//!
//! Steady-state RSS ≈ embedding + 1–2 layers + KV cache
//! ```
//!
//! Token delivery modes:
//! - `generate()`: Synchronous, blocking. Prints tokens to stdout.
//! - `generate_stream()`: Async. Sends `GenerationEvent`s via mpsc channel.

use std::sync::Arc;
use crate::gbnf::GbnfConstraint;
use crate::blocks::{LayerUnit, build_streaming_blocks};
use crate::vram_guard::VramBudget;
use crate::kv_cache::{KvCacheManager, SessionKvCache};
use crate::metrics::{InferenceMetrics, LayerTiming};
use crate::model::{self, ModelConfig};
use crate::ops::{self, RopeCache};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor};
use std::time::Instant;
use tokio::sync::mpsc;

/// Maximum tokens processed in a single prefill chunk.
/// Keeps attention matrix memory bounded for long prompts.
const PREFILL_CHUNK_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Generation Events — async token delivery
// ---------------------------------------------------------------------------

/// Events emitted during async generation via `generate_stream()`.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A decoded token string (may be a partial UTF-8 character for BPE tokens).
    Token(String),
    /// Generation finished successfully. Contains final metrics.
    Done(GenerationMetricsSummary),
    /// Generation encountered an error.
    Error(String),
}

/// Lightweight metrics snapshot sent with `GenerationEvent::Done`.
#[derive(Debug, Clone)]
pub struct GenerationMetricsSummary {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_secs: f64,
}

use crate::ghost_drafter::GhostDrafter;

use crate::session_context::SessionContext;
use crate::execution_policy::ExecutionPolicy;

#[derive(Default)]
pub enum DrafterState {
    #[default]
    None,
    WarpingUp,
    Ready(Box<dyn GhostDrafter>),
}

/// The main inference orchestrator. 
/// Deeply refactored to delegate state management and execution policies to sub-modules.
pub struct InferenceGenerator {
    pub session: SessionContext,
    pub policy: ExecutionPolicy,
    pub config: ModelConfig,
    pub device: Device,
    pub blocks: Vec<Box<dyn crate::layer_pipeline::LayerUnit>>,
    pub dispatcher: Arc<crate::slip::SlipDispatcher>,
    pub rope_cache: RopeCache,
}

impl InferenceGenerator {
    /// Create with auto-detected device (CUDA → CPU fallback).
    pub fn new(
        config: ModelConfig,
        sampler_config: SamplerConfig,
    ) -> Result<Self> {
        let device = Device::new_cuda(0)
            .or_else(|e| {
                println!("CUDA not available ({e}), falling back to CPU");
                Ok::<Device, candle_core::Error>(Device::Cpu)
            })
            .map_err(|e| anyhow::anyhow!("Failed to create device: {e}"))?;
        Self::with_device(config, sampler_config, device)
    }

    /// Create with an explicitly injected device (ADR-0002).
    pub fn with_device(
        config: ModelConfig,
        sampler_config: SamplerConfig,
        device: Device,
    ) -> Result<Self> {
        let kv_cache: Box<dyn SessionKvCache> =
            Box::new(KvCacheManager::new_for_device(device.clone(), config.n_layers));
        
        let tp_config = crate::tensor_parallel::TensorParallelConfig::new(1, 0, None);
        let session = SessionContext::new(kv_cache, tp_config);
        
        let sampler = Sampler::new(sampler_config);
        let policy = ExecutionPolicy::new(sampler, 8); // Default draft size 8
        
        let dispatcher = Arc::new(crate::slip::SlipDispatcher::new(
            None,
            Arc::new(config.clone()),
            device.clone(),
        ));

        Ok(Self {
            session,
            policy,
            config,
            device,
            blocks: Vec::new(),
            dispatcher,
            rope_cache: RopeCache::new(),
        })
    }

    /// Create with device injection + pre-built block stack (ADR-0001 + ADR-0002).
    pub fn with_streamer(
        config: ModelConfig,
        sampler_config: SamplerConfig,
        device: Device,
        streamer: std::sync::Arc<WeightStreamer>,
        _rope: Option<std::sync::Arc<crate::ops::RopeCache>>,
        mut dual_rope: Option<crate::dual_rope::DualRopeCache>,
    ) -> Result<Self> {
        // Initialize DualRope tensors on the target device
        if let Some(ref mut dr) = dual_rope {
            dr.init_on_device(config.context_length, &device)?;
        }

        // ── VRAM 80% hard cap guard (issue #2) ───────────────────────────
        let budget = crate::vram_guard::VramBudget::check(
            &device,
            0,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            config.hidden_dim,
            config.context_length,
        )?;
        eprintln!("{}", budget.summary());

        let mut gen = Self::with_device(config.clone(), sampler_config, device.clone())?;
        gen.session.dual_rope = dual_rope;
        
        // Re-initialize dispatcher with the streamer
        gen.dispatcher = std::sync::Arc::new(crate::slip::SlipDispatcher::new(
            Some(streamer),
            std::sync::Arc::new(config),
            device,
        ));
        Ok(gen)
    }

    /// Create with device injection + pre-built block stack (ADR-0001 + ADR-0002).
    ///
    /// Builds a [`StreamingQBlock`] for every layer in the GGUF file and stores
    /// them in `self.blocks`. The `generate_step` layer loop will use
    /// `block.forward()` instead of direct `streamer.load_layer()` calls,
    /// enabling future heterogeneous stacks (MoE, device-split, etc.).
    ///
    pub fn set_grammar(&mut self, constraint: GbnfConstraint) {
        self.policy.set_grammar(constraint);
    }

    pub fn clear_grammar(&mut self) {
        self.policy.clear_grammar();
    }

    pub fn set_tp(&mut self, rank: usize, tp_size: usize) {
        self.session.tp_config = crate::tensor_parallel::TensorParallelConfig::new(tp_size, rank, None);
    }

    pub fn set_communicator(&mut self, comm: Arc<dyn crate::distributed::Communicator>) {
        self.session.tp_config.comm = Some(comm);
    }

    pub fn enable_wavefront(&mut self, draft_size: usize, dense_only: bool, streamer: &WeightStreamer) -> Result<()> {
        let medusa_cfg = crate::medusa_heads::MedusaConfig {
            n_heads: draft_size,
            hidden_dim: self.config.hidden_dim,
            vocab_size: self.config.vocab_size,
            ..Default::default()
        };
        
        // Try loading native MTP heads for Qwen 3.6
        self.policy.medusa_heads = Some(crate::medusa_heads::MedusaHeads::load_native(
            medusa_cfg, streamer, &self.device
        )?);

        if !dense_only {
            let sparsity_cfg = crate::sparsity_predictor::SparsityConfig {
                hidden_dim: self.config.hidden_dim,
                intermediate_dim: self.config.intermediate_dim,
                ..Default::default()
            };
            self.policy.sparsity_bank = Some(crate::sparsity_predictor::SparsityPredictorBank::new(
                self.config.n_layers, sparsity_cfg
            ));
        }

        self.policy.wavefront_health = crate::wavefront::WavefrontHealthMonitor::new(draft_size);
        
        // Load actual lm_head (output.weight) for draft projection
        let head_name = "output.weight";
        
        if let Ok(t) = streamer.load_tensor(head_name, &self.device) {
            self.session.lm_head_tensor = Some(t.to_dtype(candle_core::DType::F16)?);
        } else {
            eprintln!("⚠️ Warning: Failed to load '{}' for Medusa drafting. Using dummy zeros.", head_name);
        }

        Ok(())
    }

    pub fn warp_up_drafter(&mut self, streamer: &WeightStreamer) {
        if !matches!(self.policy.drafter, DrafterState::None) { return; }
        
        self.policy.drafter = DrafterState::WarpingUp;
        
        let config = self.config.clone();
        let device = self.device.clone();
        
        if let Ok(mut d) = crate::tq2_drafter::TQ2GhostDrafter::new(config, device) {
            let _ = d.warp_up(streamer);
            self.policy.drafter = DrafterState::Ready(Box::new(d));
            eprintln!("\n🚀 Gemma 4 TQ2 Ghost Drafter WARPED UP — Switching to Speculative Mode");
        }
    }

    pub fn has_grammar(&self) -> bool {
        self.policy.has_grammar()
    }

}

#[derive(Debug, Clone)]
pub struct WavefrontResult {
    pub accepted_tokens: Vec<u32>,
    pub eos_reached: bool,
}

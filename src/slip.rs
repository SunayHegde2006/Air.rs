//! S.L.I.P. Dispatcher — Streaming Layer-wise Inference Protocol (Candidate 3).
//!
//! Orchestrates the high-performance streaming pipeline by managing:
//!   - Asymmetric Residency: Keeping attention/router fixed while experts stream.
//!   - Prefetch Overlap: Ensuring Layer N+1 is loading while Layer N computes.
//!   - Demand-paging for MoE experts (v0.10.1).
//!
//! # Resident Mode (`--resident`)
//!
//! When a model fits in available VRAM, `new_resident()` pre-loads all layer
//! weights into VRAM at startup. This eliminates the per-layer NVMe I/O on
//! every decode step, unlocking ~5–25× higher decode throughput vs. streaming.
//!
//! Trade-off: the model must fit in VRAM. If it does not, fall back to `new()`
//! (S.L.I.P. streaming), which can run 32–70 GB models on a 12 GB card.

use crate::model::{ModelConfig, QBlockWeights, transformer_block};
use crate::kv_cache::LayerCache;
use crate::weight_streamer::WeightStreamer;
use crate::moe::{ExpertWeights, ExpertVramScheduler};
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// The central dispatcher for the Air.rs inference engine.
use crate::layer_pipeline::LayerPipeline;

/// The central dispatcher for the Air.rs inference engine.
/// Deeply refactored to delegate layer execution to a LayerPipeline.
pub struct SlipDispatcher {
    streamer: Option<Arc<WeightStreamer>>,
    config: Arc<ModelConfig>,
    device: Device,
    pipeline: LayerPipeline,
    /// VRAM scheduler for MoE experts across all layers.
    moe_scheduler: std::sync::Mutex<ExpertVramScheduler>,
    /// Resident cache: all layer weights pre-loaded into VRAM.
    ///
    /// `Some` when `--resident` is active; `None` in default S.L.I.P. streaming mode.
    /// When populated, `forward_layer` reads from this map and skips the NVMe I/O path.
    resident_cache: Option<HashMap<usize, QBlockWeights>>,
}

impl SlipDispatcher {
    pub fn new(streamer: Option<Arc<WeightStreamer>>, config: Arc<ModelConfig>, device: Device) -> Self {
        let capacity = if config.n_experts > 0 { 24 } else { 1 };
        let mut pipeline = LayerPipeline::new(config.clone(), device.clone());
        pipeline.add_unit(Box::new(crate::model::StandardLayerUnit));
        Self {
            streamer,
            config,
            device,
            pipeline,
            moe_scheduler: std::sync::Mutex::new(ExpertVramScheduler::new(capacity)),
            resident_cache: None,
        }
    }

    /// Construct a resident-VRAM dispatcher: pre-loads ALL layer weights into VRAM
    /// at startup.
    ///
    /// Use this when the model fits in available VRAM (check with `--ctx-size` +
    /// `VramBudget::check`). Decode throughput is ~5–25× higher than streaming mode
    /// because `forward_layer` reads from an in-VRAM HashMap instead of issuing
    /// NVMe I/O on every call.
    ///
    /// # Errors
    /// Returns an error if any layer fails to load (e.g. VRAM OOM).
    pub fn new_resident(
        streamer: Arc<WeightStreamer>,
        config: Arc<ModelConfig>,
        device: Device,
        tp: Option<&crate::tensor_parallel::TensorParallelConfig>,
    ) -> anyhow::Result<Self> {
        let n_layers = config.n_layers;
        eprintln!("  [resident] Pre-loading {} layers into VRAM…", n_layers);

        let mut resident_cache: HashMap<usize, QBlockWeights> = HashMap::with_capacity(n_layers);

        for layer_id in 0..n_layers {
            let weights = streamer.load_layer(layer_id, &device, tp)?;
            resident_cache.insert(layer_id, weights);
            if layer_id % 8 == 7 || layer_id == n_layers - 1 {
                eprintln!("  [resident] Loaded {}/{} layers", layer_id + 1, n_layers);
            }
        }

        eprintln!("  [resident] All {} layers resident in VRAM ✓", n_layers);

        let capacity = if config.n_experts > 0 { 24 } else { 1 };
        let mut pipeline = LayerPipeline::new(config.clone(), device.clone());
        pipeline.add_unit(Box::new(crate::model::StandardLayerUnit));

        Ok(Self {
            streamer: Some(streamer),
            config,
            device,
            pipeline,
            moe_scheduler: std::sync::Mutex::new(ExpertVramScheduler::new(capacity)),
            resident_cache: Some(resident_cache),
        })
    }

    /// Forward pass for a single layer.
    pub fn forward_layer(
        &self,
        layer_id: usize,
        x: &Tensor,
        cache: Option<&LayerCache>,
        pos: usize,
        rope_cache: Option<&crate::ops::RopeCache>,
        dual_cache: Option<&crate::dual_rope::DualRopeCache>,
        custom_mask: Option<&Tensor>,
        tp: Option<&crate::tensor_parallel::TensorParallelConfig>,
    ) -> Result<(Tensor, LayerCache)> {
        // ── Resident mode: read from VRAM cache, no I/O ──────────────────
        let mut weights = if let Some(ref cache) = self.resident_cache {
            cache.get(&layer_id)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(
                    format!("ResidentCache: layer {layer_id} not loaded")
                ))?
        } else {
            // ── Streaming mode: load from NVMe via WeightStreamer ─────────
            let streamer = self.streamer.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("SlipDispatcher: No WeightStreamer connected".into())
            })?;
            streamer.load_layer(layer_id, &self.device, tp)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        };

        // ── MoE Demand-Paging (v0.10.1) ────────────────────────────────
        if self.config.n_experts > 0 {
            if let Some(router) = &weights.ffn_router {
                let (indices, _) = crate::ops::gemma_moe_route(x, router, self.config.moe_top_k)?;
                let needed_expert_ids: Vec<usize> = indices.iter().flatten().cloned().collect();

                let streamer = self.streamer.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("SlipDispatcher: No WeightStreamer for MoE experts".into())
                })?;
                let mut scheduler = self.moe_scheduler.lock().unwrap();
                scheduler.ensure_resident(&needed_expert_ids, |id| {
                    streamer.load_expert(layer_id, id, &self.device)
                        .map_err(|e| e.to_string())
                }).map_err(candle_core::Error::Msg)?;
                
                let mut w_gate = Vec::with_capacity(self.config.moe_top_k);
                let mut w_up = Vec::with_capacity(self.config.moe_top_k);
                let mut w_down = Vec::with_capacity(self.config.moe_top_k);
                
                for &id in &needed_expert_ids {
                    let exp = scheduler.get(id);
                    w_gate.push(exp.w_gate.clone());
                    w_up.push(exp.w_up.clone());
                    w_down.push(exp.w_down.clone());
                }
                
                weights.ffn_exps_gate = Some(w_gate);
                weights.ffn_exps_up = Some(w_up);
                weights.ffn_exps_down = Some(w_down);
            }
        }

        // ── Pipeline Execution (ADR-0005) ─────────────────────────────
        self.pipeline.execute(
            layer_id,
            x,
            &weights,
            cache,
            pos,
            rope_cache,
            dual_cache,
            custom_mask,
            tp,
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}


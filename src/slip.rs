//! S.L.I.P. Dispatcher — Streaming Layer-wise Inference Protocol (Candidate 3).
//!
//! Orchestrates the high-performance streaming pipeline by managing:
//!   - Asymmetric Residency: Keeping attention/router fixed while experts stream.
//!   - Prefetch Overlap: Ensuring Layer N+1 is loading while Layer N computes.
//!   - Demand-paging for MoE experts (v0.10.1).

use crate::model::{ModelConfig, QBlockWeights, transformer_block};
use crate::kv_cache::LayerCache;
use crate::weight_streamer::WeightStreamer;
use crate::moe::{ExpertWeights, ExpertVramScheduler};
use candle_core::{Device, Result, Tensor};
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
        }
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
        let streamer = self.streamer.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("SlipDispatcher: No WeightStreamer connected".into())
        })?;
        
        let mut weights = streamer.load_layer(layer_id, &self.device, tp)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // ── MoE Demand-Paging (v0.10.1) ────────────────────────────────
        if self.config.n_experts > 0 {
            if let Some(router) = &weights.ffn_router {
                let (indices, _) = crate::ops::gemma_moe_route(x, router, self.config.moe_top_k)?;
                let needed_expert_ids: Vec<usize> = indices.iter().flatten().cloned().collect();
                
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

//! LayerPipeline — Functional Layer Orchestration (ADR-0005).
//!
//! Deeply refactored from the procedural SlipDispatcher.
//! Encapsulates the complete lifecycle of a single layer's execution,
//! hiding MoE, Dense, and Recurrent complexity from the main orchestrator.

use candle_core::{Tensor, Device, Result};
use std::sync::Arc;
use crate::model::{ModelConfig, QBlockWeights};
use crate::kv_cache::LayerCache;
use crate::weight_streamer::WeightStreamer;
use crate::tensor_parallel::TensorParallelConfig;
use crate::ops::RopeCache;
use crate::dual_rope::DualRopeCache;

/// A functional unit of computation within a layer (e.g. Attention, FFN, MoE).
///
/// Interface is purely functional: (input, stage) -> (output, updated_state)
pub trait LayerUnit: Send + Sync {
    fn execute(
        &self,
        x: &Tensor,
        weights: &QBlockWeights,
        state: Option<&LayerCache>,
        pos: usize,
        config: &ModelConfig,
        rope_cache: Option<&RopeCache>,
        dual_cache: Option<&DualRopeCache>,
        mask: Option<&Tensor>,
        tp: Option<&TensorParallelConfig>,
    ) -> Result<(Tensor, LayerCache)>;

    fn clone_box(&self) -> Box<dyn LayerUnit>;
}

impl Clone for Box<dyn LayerUnit> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// The main pipeline for executing a single transformer layer.
/// Hides MoE streaming, demand-paging, and hybrid architecture details.
pub struct LayerPipeline {
    pub units: Vec<Box<dyn LayerUnit>>,
    pub config: Arc<ModelConfig>,
    pub device: Device,
}

impl LayerPipeline {
    pub fn new(config: Arc<ModelConfig>, device: Device) -> Self {
        Self {
            units: Vec::new(),
            config,
            device,
        }
    }

    /// Add a computation unit to the pipeline stack.
    pub fn add_unit(&mut self, unit: Box<dyn LayerUnit>) {
        self.units.push(unit);
    }

    /// Run the entire pipeline for one layer.
    pub fn execute(
        &self,
        layer_id: usize,
        x: &Tensor,
        weights: &QBlockWeights,
        state: Option<&LayerCache>,
        pos: usize,
        rope_cache: Option<&RopeCache>,
        dual_cache: Option<&DualRopeCache>,
        mask: Option<&Tensor>,
        tp: Option<&TensorParallelConfig>,
    ) -> Result<(Tensor, LayerCache)> {
        // In a deep pipeline, we iterate through units in sequence.
        // For architectural indifference, the 'x' tensor is updated by each unit.
        let mut current_x = x.clone();
        let mut current_state = state.cloned().unwrap_or(LayerCache::Empty);

        for unit in &self.units {
            let (out_x, out_state) = unit.execute(
                &current_x,
                weights,
                Some(&current_state),
                pos,
                &self.config,
                rope_cache,
                dual_cache,
                mask,
                tp,
            )?;
            current_x = out_x;
            current_state = out_state;
        }

        // Fallback for empty pipeline (legacy transformer_block)
        if self.units.is_empty() {
             let mut delta_state = match state { Some(LayerCache::Recurrent(s)) => Some(s.clone()), _ => None };
             let (out_x, out_k, out_v) = crate::model::transformer_block(
                layer_id,
                x,
                weights,
                match state { Some(LayerCache::Attention { k, .. }) => Some(k), _ => None },
                match state { Some(LayerCache::Attention { v, .. }) => Some(v), _ => None },
                delta_state.as_mut(),
                &self.config,
                pos,
                rope_cache,
                dual_cache,
                mask,
                tp,
            ).map_err(|e| candle_core::Error::Msg(format!("Layer {} execution failed: {}", layer_id, e)))?;
            
            let next_cache = if let Some(s) = delta_state {
                LayerCache::Recurrent(s)
            } else {
                LayerCache::Attention { k: out_k, v: out_v }
            };
            
            return Ok((out_x, next_cache));
        }

        Ok((current_x, current_state))
    }
}

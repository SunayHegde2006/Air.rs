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

/// Execution context for a single layer unit.
/// Bundles environmental invariants and data to reduce interface noise.
pub struct LayerExecutionContext<'a> {
    pub x: &'a Tensor,
    pub weights: Option<&'a QBlockWeights>,
    pub state: Option<&'a LayerCache>,
    pub pos: usize,
    pub config: &'a ModelConfig,
    pub rope_cache: Option<&'a RopeCache>,
    pub dual_cache: Option<&'a DualRopeCache>,
    pub mask: Option<&'a Tensor>,
    pub tp: Option<&'a TensorParallelConfig>,
}

/// A functional unit of computation within a layer (e.g. Attention, FFN, MoE).
///
/// Interface is purely functional: context -> (output, updated_state)
pub trait LayerUnit: Send + Sync {
    fn execute(
        &self,
        ctx: &LayerExecutionContext,
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
            let ctx = LayerExecutionContext {
                x: &current_x,
                weights: Some(weights),
                state: Some(&current_state),
                pos,
                config: &self.config,
                rope_cache,
                dual_cache,
                mask,
                tp,
            };
            let (out_x, out_state) = unit.execute(&ctx)?;
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

/// A LayerUnit adapter that manages weight residency via a WeightStreamer.
///
/// Hides the loading/releasing lifecycle from the compute units.
pub struct ResidentLayer {
    pub inner: Box<dyn LayerUnit>,
    pub streamer: Arc<WeightStreamer>,
    pub layer_id: usize,
    pub device: Device,
}

impl LayerUnit for ResidentLayer {
    fn execute(&self, ctx: &LayerExecutionContext) -> Result<(Tensor, LayerCache)> {
        // Ensure weights are resident
        let weights = self.streamer.load_layer(self.layer_id, &self.device, ctx.tp)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        // Build a new context with the resident weights
        let local_ctx = LayerExecutionContext {
            weights: Some(&weights),
            ..*ctx
        };
        
        self.inner.execute(&local_ctx)
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(Self {
            inner: self.inner.clone(),
            streamer: Arc::clone(&self.streamer),
            layer_id: self.layer_id,
            device: self.device.clone(),
        })
    }
}

// Ensure LayerExecutionContext is Cloneable (shallow) for context patching.
impl<'a> Clone for LayerExecutionContext<'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a> Copy for LayerExecutionContext<'a> {}

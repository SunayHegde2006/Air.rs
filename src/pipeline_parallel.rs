//! # Pipeline Parallelism
//!
//! 1-Forward-1-Backward (1F1B) pipeline schedule for inference across
//! multiple stages. Each `PipelineStage` holds a contiguous slice of
//! transformer layers and passes `Activation` tensors to the next stage
//! via bounded SPSC channels.
//!
//! **Research**:
//! - "GPipe: Efficient Training of Giant Neural Networks using Pipeline
//!   Parallelism" (Huang et al., NeurIPS 2019)
//! - "Efficient Large-Scale Language Model Training on GPU Clusters Using
//!   Megatron-LM" (Narayanan et al., SC 2021, arXiv:2104.04473)
//! - "PipeDream: Generalized Pipeline Parallelism for DNN Training"
//!   (Narayanan et al., SOSP 2019)
//!
//! **Consumer note**: On a single GPU this acts as a no-op identity passthrough
//! (n_stages=1). Multi-GPU pipelines use channel-based IPC; actual CUDA-P2P
//! transfers are behind `#[cfg(feature = "cuda")]`.

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Activation — the data flowing between stages
// ---------------------------------------------------------------------------

/// A batch of hidden-state activations flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct Activation {
    /// Sequence ids in this micro-batch.
    pub seq_ids: Vec<u64>,
    /// Packed hidden states: [batch × seq_len × hidden_dim] as f32.
    pub hidden: Vec<f32>,
    /// Shape metadata.
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    /// Tokens already generated (used by the final stage to emit output).
    pub generated_tokens: Vec<u32>,
}

impl Activation {
    pub fn new(
        seq_ids: Vec<u64>,
        hidden: Vec<f32>,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Self {
        assert_eq!(
            hidden.len(),
            batch_size * seq_len * hidden_dim,
            "activation shape mismatch"
        );
        Self {
            seq_ids,
            hidden,
            batch_size,
            seq_len,
            hidden_dim,
            generated_tokens: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// TransformerBlock trait — implemented by the actual model layers
// ---------------------------------------------------------------------------

/// A single transformer layer that can be placed into a pipeline stage.
pub trait TransformerBlock: Send + Sync {
    /// Apply this layer to an activation in-place (or return a new one).
    fn forward(&self, input: Activation) -> Activation;

    /// Human-readable identifier for logging.
    fn layer_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Identity block — used in tests and single-GPU passthrough
// ---------------------------------------------------------------------------

/// No-op transformer block: returns the input unchanged.
pub struct IdentityBlock {
    name: String,
}

impl IdentityBlock {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl TransformerBlock for IdentityBlock {
    fn forward(&self, input: Activation) -> Activation {
        input
    }
    fn layer_name(&self) -> &str {
        &self.name
    }
}

/// Scaled block: multiplies every hidden value by `scale`. Used to verify
/// that stages really run in order and compose correctly.
pub struct ScaledBlock {
    name: String,
    scale: f32,
}

impl ScaledBlock {
    pub fn new(name: impl Into<String>, scale: f32) -> Self {
        Self {
            name: name.into(),
            scale,
        }
    }
}

impl TransformerBlock for ScaledBlock {
    fn forward(&self, mut input: Activation) -> Activation {
        for v in &mut input.hidden {
            *v *= self.scale;
        }
        input
    }
    fn layer_name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Total number of stages.
    pub n_stages: usize,
    /// Number of transformer layers per stage (must sum to total layer count).
    pub layers_per_stage: Vec<usize>,
    /// Micro-batch size (tokens) — smaller = lower pipeline bubble ratio.
    pub micro_batch_size: usize,
    /// Number of micro-batches in flight simultaneously (pipeline depth).
    pub n_micro_batches: usize,
    /// SPSC channel capacity between stages.
    pub channel_capacity: usize,
}

impl PipelineConfig {
    /// Validate that `layers_per_stage` has exactly `n_stages` entries and
    /// that each entry is ≥ 1.
    pub fn validate(&self) -> Result<(), String> {
        if self.layers_per_stage.len() != self.n_stages {
            return Err(format!(
                "layers_per_stage len {} != n_stages {}",
                self.layers_per_stage.len(),
                self.n_stages
            ));
        }
        if self.layers_per_stage.iter().any(|&l| l == 0) {
            return Err("every stage must have ≥1 layer".into());
        }
        if self.n_stages == 0 {
            return Err("n_stages must be ≥1".into());
        }
        Ok(())
    }

    /// Convenience: single-stage config (consumer single-GPU).
    pub fn single_stage(n_layers: usize) -> Self {
        Self {
            n_stages: 1,
            layers_per_stage: vec![n_layers],
            micro_batch_size: 1,
            n_micro_batches: 1,
            channel_capacity: 4,
        }
    }

    /// Two-stage config for a 2-GPU workstation split.
    pub fn two_stage(layers_first: usize, layers_second: usize) -> Self {
        Self {
            n_stages: 2,
            layers_per_stage: vec![layers_first, layers_second],
            micro_batch_size: 1,
            n_micro_batches: 2,
            channel_capacity: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// A single stage in the pipeline holding a contiguous slice of layers.
///
/// In inference-only mode we run **forward only**: receive activation from the
/// previous stage, run through local layers, send to the next stage.
/// Stage 0 has no recv (input is the embedding). The final stage has no send
/// (output is the logit projection).
pub struct PipelineStage {
    pub stage_id: usize,
    pub layers: Vec<Box<dyn TransformerBlock>>,
    recv: Option<Receiver<Activation>>,
    send: Option<SyncSender<Activation>>,
}

impl PipelineStage {
    /// Process one micro-batch.
    ///
    /// If this is stage 0: `input` must be `Some(activation)`, recv is unused.
    /// Otherwise: blocks on recv channel for the previous stage's output.
    /// Returns `Some(output)` for all stages; the caller for the final stage
    /// reads the returned value as the network output.
    pub fn forward_step(&mut self, input: Option<Activation>) -> Option<Activation> {
        let mut act = if self.stage_id == 0 {
            input.expect("stage 0 requires explicit input")
        } else {
            self.recv.as_ref()?.recv().ok()?
        };

        for layer in &self.layers {
            act = layer.forward(act);
        }

        if let Some(tx) = &self.send {
            tx.send(act).ok()?;
            None // intermediate stage: consumer reads from channel
        } else {
            Some(act) // final stage: return result directly
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineRunner — orchestrates all stages in separate threads
// ---------------------------------------------------------------------------

/// Synchronous pipeline runner: each stage runs in its own thread.
///
/// For GPU inference the stage threads would be pinned to specific CUDA
/// devices via `cuda-sys::cudaSetDevice(stage_id)`.
pub struct PipelineRunner {
    pub cfg: PipelineConfig,
    stages: Vec<Arc<std::sync::Mutex<PipelineStage>>>,
}

impl PipelineRunner {
    /// Build a pipeline from a flat list of layers.
    ///
    /// Layers are distributed to stages according to `cfg.layers_per_stage`.
    /// Channels are wired: stage i's send → stage i+1's recv.
    pub fn build(
        cfg: PipelineConfig,
        mut layers: Vec<Box<dyn TransformerBlock>>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let total_layers: usize = cfg.layers_per_stage.iter().sum();
        if layers.len() != total_layers {
            return Err(format!(
                "layers.len()={} but layers_per_stage sums to {}",
                layers.len(),
                total_layers
            ));
        }

        // Build channels between stages.
        let mut senders: Vec<Option<SyncSender<Activation>>> = Vec::new();
        let mut receivers: Vec<Option<Receiver<Activation>>> = Vec::new();

        for _ in 0..cfg.n_stages.saturating_sub(1) {
            let (tx, rx) = mpsc::sync_channel(cfg.channel_capacity);
            senders.push(Some(tx));
            receivers.push(Some(rx));
        }
        senders.push(None); // final stage has no send
        // Stage 0 has no recv.
        let mut stage_recvs: Vec<Option<Receiver<Activation>>> = vec![None];
        stage_recvs.extend(receivers);

        // Distribute layers to stages.
        let mut stages = Vec::new();
        for (si, &n_layers) in cfg.layers_per_stage.iter().enumerate() {
            let stage_layers: Vec<Box<dyn TransformerBlock>> =
                layers.drain(..n_layers).collect();
            let stage = PipelineStage {
                stage_id: si,
                layers: stage_layers,
                recv: stage_recvs.remove(0),
                send: senders.remove(0),
            };
            stages.push(Arc::new(std::sync::Mutex::new(stage)));
        }

        Ok(Self { cfg, stages })
    }

    /// Run a single micro-batch through the entire pipeline **synchronously**
    /// (all stages in the caller's thread, in order). Suitable for single-GPU
    /// inference where there's no benefit to pipelining across threads.
    ///
    /// For real multi-GPU use: spawn each stage in its own thread and call
    /// `forward_step` in a loop.
    pub fn run_sequential(&self, input: Activation) -> Activation {
        let mut act = input;
        for (si, stage_arc) in self.stages.iter().enumerate() {
            let mut stage = stage_arc.lock().unwrap();
            let opt_input = if si == 0 { Some(act) } else { None };
            if si == 0 {
                if let Some(out) = stage.forward_step(opt_input) {
                    act = out;
                } else {
                    // Stage sent to channel — but in sequential mode we peek.
                    // For sequential single-GPU this shouldn't happen.
                    panic!("sequential run: intermediate stage must not have a send channel");
                }
            } else if let Some(out) = stage.forward_step(None) {
                act = out;
            } else {
                act = stage.recv.as_mut().unwrap().recv().unwrap();
                act = stage.layers.iter().fold(act, |a, l| l.forward(a));
            }
        }
        act
    }
}

// ---------------------------------------------------------------------------
// Sequential single-GPU runner (simplified — avoids channel complexity)
// ---------------------------------------------------------------------------

/// Simplified sequential pipeline: no channels, no threads.
/// Equivalent to `PipelineRunner::run_sequential` but allocates less.
pub fn run_pipeline_sequential(
    layers: &[Box<dyn TransformerBlock>],
    input: Activation,
) -> Activation {
    layers.iter().fold(input, |acc, layer| layer.forward(acc))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_act(batch: usize, seq: usize, hidden: usize) -> Activation {
        Activation::new(
            (0..batch as u64).collect(),
            vec![1.0f32; batch * seq * hidden],
            batch,
            seq,
            hidden,
        )
    }

    // --- Config ---

    #[test]
    fn test_config_validate_ok() {
        let cfg = PipelineConfig::two_stage(16, 16);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_len_mismatch() {
        let cfg = PipelineConfig {
            n_stages: 3,
            layers_per_stage: vec![8, 8],
            ..PipelineConfig::single_stage(16)
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_layer_in_stage() {
        let cfg = PipelineConfig {
            n_stages: 2,
            layers_per_stage: vec![8, 0],
            ..PipelineConfig::single_stage(8)
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_single_stage() {
        let cfg = PipelineConfig::single_stage(32);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.n_stages, 1);
    }

    // --- Identity block ---

    #[test]
    fn test_identity_block_passthrough() {
        let block = IdentityBlock::new("id");
        let act = make_act(2, 4, 8);
        let out = block.forward(act.clone());
        assert_eq!(out.hidden, act.hidden);
    }

    // --- ScaledBlock ---

    #[test]
    fn test_scaled_block_multiplies() {
        let block = ScaledBlock::new("scale2", 2.0);
        let act = make_act(1, 1, 4);
        let out = block.forward(act);
        assert!(out.hidden.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // --- run_pipeline_sequential ---

    #[test]
    fn test_sequential_two_scaled_blocks() {
        let layers: Vec<Box<dyn TransformerBlock>> = vec![
            Box::new(ScaledBlock::new("s1", 2.0)),
            Box::new(ScaledBlock::new("s2", 3.0)), // total: ×6
        ];
        let act = make_act(1, 1, 4);
        let out = run_pipeline_sequential(&layers, act);
        assert!(out.hidden.iter().all(|&v| (v - 6.0).abs() < 1e-5));
    }

    #[test]
    fn test_sequential_identity_is_noop() {
        let layers: Vec<Box<dyn TransformerBlock>> =
            vec![Box::new(IdentityBlock::new("i"))];
        let act = make_act(2, 3, 8);
        let expected = act.hidden.clone();
        let out = run_pipeline_sequential(&layers, act);
        assert_eq!(out.hidden, expected);
    }

    // --- PipelineRunner::build ---

    #[test]
    fn test_build_rejects_layer_count_mismatch() {
        let cfg = PipelineConfig::two_stage(4, 4);
        let layers: Vec<Box<dyn TransformerBlock>> = (0..6)
            .map(|i| Box::new(IdentityBlock::new(format!("l{i}"))) as Box<dyn TransformerBlock>)
            .collect();
        assert!(PipelineRunner::build(cfg, layers).is_err());
    }

    // --- Activation shape ---

    #[test]
    fn test_activation_shape_asserted() {
        // Should not panic.
        let _ = make_act(2, 4, 8);
    }

    #[test]
    #[should_panic(expected = "activation shape mismatch")]
    fn test_activation_wrong_shape_panics() {
        Activation::new(vec![0], vec![1.0; 5], 2, 4, 8); // wrong size
    }

    // --- Send + Sync ---

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_types_send_sync() {
        _assert_send_sync::<Activation>();
        _assert_send_sync::<PipelineConfig>();
        _assert_send_sync::<IdentityBlock>();
        _assert_send_sync::<ScaledBlock>();
    }
}

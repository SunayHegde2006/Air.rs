//! Execution cursor and MoE expert activation hook (STRIX Protocol §9, §20).
//!
//! The `ExecutionCursor` tracks the exact position within an inference pass:
//! which layer, which sub-operation (attn/ffn), and which expert (for MoE).
//!
//! The `expert_activation_hook` callback allows the scheduler to prefetch
//! only the active experts' tensors for Mixture-of-Experts models, avoiding
//! wasted VRAM bandwidth on unused expert weights.

use super::types::TensorId;

// ── Execution Phase ──────────────────────────────────────────────────────

/// Sub-operation within a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerPhase {
    /// Attention query/key/value projection.
    AttnQKV,
    /// Attention output projection.
    AttnOutput,
    /// Attention normalization (pre or post).
    AttnNorm,
    /// Feed-forward network gate/up projection (or expert dispatch for MoE).
    FfnGateUp,
    /// Feed-forward network down projection.
    FfnDown,
    /// FFN normalization.
    FfnNorm,
    /// Complete — layer finished, cursor advancing.
    Done,
}

// ── Expert Activation ────────────────────────────────────────────────────

/// Which experts are active in the current MoE layer.
#[derive(Debug, Clone)]
pub struct ExpertActivation {
    /// Total number of experts in this MoE layer.
    pub num_experts: usize,
    /// Top-K experts selected by the router for this token.
    pub active_expert_ids: Vec<usize>,
    /// Router confidence scores for each active expert (same order).
    pub router_weights: Vec<f32>,
}

impl ExpertActivation {
    /// Create a dense (non-MoE) activation — all experts active.
    pub fn dense() -> Self {
        Self {
            num_experts: 1,
            active_expert_ids: vec![0],
            router_weights: vec![1.0],
        }
    }

    /// Create a sparse MoE activation from router output.
    pub fn sparse(num_experts: usize, active_ids: Vec<usize>, weights: Vec<f32>) -> Self {
        Self {
            num_experts,
            active_expert_ids: active_ids,
            router_weights: weights,
        }
    }

    /// Whether this is a sparse MoE layer (more than 1 expert total).
    pub fn is_moe(&self) -> bool {
        self.num_experts > 1
    }

    /// Fraction of experts that are active (for VRAM budget estimation).
    pub fn sparsity(&self) -> f32 {
        if self.num_experts == 0 {
            return 1.0;
        }
        self.active_expert_ids.len() as f32 / self.num_experts as f32
    }
}

// ── Execution Cursor ─────────────────────────────────────────────────────

/// Tracks the exact position within a transformer inference pass.
///
/// Used by the scheduler to determine which tensors to prefetch and
/// which can be evicted. For MoE models, the cursor includes expert
/// activation information so only needed expert weights are loaded.
#[derive(Debug, Clone)]
pub struct ExecutionCursor {
    /// Current layer index (0-based).
    pub layer: usize,
    /// Total layers in the model.
    pub total_layers: usize,
    /// Current sub-operation within the layer.
    pub phase: LayerPhase,
    /// Expert activation for the current layer (dense or MoE).
    pub expert_activation: ExpertActivation,
    /// Token step counter (monotonically increasing).
    pub step: u64,
    /// Whether the cursor has completed a full forward pass.
    pub completed: bool,
}

impl ExecutionCursor {
    /// Create a new cursor at the start of inference.
    pub fn new(total_layers: usize) -> Self {
        Self {
            layer: 0,
            total_layers,
            phase: LayerPhase::AttnNorm,
            expert_activation: ExpertActivation::dense(),
            step: 0,
            completed: false,
        }
    }

    /// Advance to the next phase within the current layer.
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            LayerPhase::AttnNorm => LayerPhase::AttnQKV,
            LayerPhase::AttnQKV => LayerPhase::AttnOutput,
            LayerPhase::AttnOutput => LayerPhase::FfnNorm,
            LayerPhase::FfnNorm => LayerPhase::FfnGateUp,
            LayerPhase::FfnGateUp => LayerPhase::FfnDown,
            LayerPhase::FfnDown => LayerPhase::Done,
            LayerPhase::Done => LayerPhase::Done,
        };
    }

    /// Advance to the next layer (resets phase to AttnNorm).
    pub fn advance_layer(&mut self) {
        if self.layer + 1 < self.total_layers {
            self.layer += 1;
            self.phase = LayerPhase::AttnNorm;
            self.expert_activation = ExpertActivation::dense();
            self.step += 1;
        } else {
            self.completed = true;
        }
    }

    /// Set MoE expert activation for the current layer.
    ///
    /// This should be called after the router network runs but before
    /// the expert FFN weights are needed. The scheduler uses this to
    /// prefetch only the active experts' tensors.
    pub fn set_expert_activation(&mut self, activation: ExpertActivation) {
        self.expert_activation = activation;
    }

    /// Distance (in layers) from the current position to a target layer.
    pub fn distance_to(&self, target_layer: usize) -> usize {
        if target_layer >= self.layer {
            target_layer - self.layer
        } else {
            // Already past this layer in the current pass
            self.total_layers - self.layer + target_layer
        }
    }

    /// Whether a given expert ID is active in the current cursor position.
    pub fn is_expert_active(&self, expert_id: usize) -> bool {
        self.expert_activation.active_expert_ids.contains(&expert_id)
    }
}

// ── Expert Activation Hook ───────────────────────────────────────────────

/// Callback type for the expert activation hook.
///
/// When a MoE router selects experts, this hook is called so the STRIX
/// scheduler can adjust prefetch priorities:
///
/// - **Active experts**: boost urgency score to ensure weights are in VRAM
/// - **Inactive experts**: lower urgency to allow eviction
///
/// The hook receives the layer index, the activation info, and returns
/// a list of tensor IDs that should be boosted for prefetch.
pub type ExpertActivationHook = Box<dyn Fn(usize, &ExpertActivation) -> Vec<TensorId> + Send + Sync>;

/// Creates the default expert activation hook for MoE models.
///
/// This hook generates tensor names for the active experts and returns
/// their IDs from the name lookup. It follows the standard MoE naming
/// convention: `blk.{layer}.ffn_{gate,up,down}.{expert_id}.weight`.
pub fn default_expert_hook(
    layer: usize,
    activation: &ExpertActivation,
) -> Vec<String> {
    let mut tensor_names = Vec::with_capacity(activation.active_expert_ids.len() * 3);
    for &expert_id in &activation.active_expert_ids {
        tensor_names.push(format!("blk.{layer}.ffn_gate.{expert_id}.weight"));
        tensor_names.push(format!("blk.{layer}.ffn_up.{expert_id}.weight"));
        tensor_names.push(format!("blk.{layer}.ffn_down.{expert_id}.weight"));
    }
    tensor_names
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_advance_through_phases() {
        let mut cursor = ExecutionCursor::new(32);
        assert_eq!(cursor.phase, LayerPhase::AttnNorm);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::AttnQKV);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::AttnOutput);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::FfnNorm);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::FfnGateUp);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::FfnDown);

        cursor.advance_phase();
        assert_eq!(cursor.phase, LayerPhase::Done);
    }

    #[test]
    fn cursor_advance_layer() {
        let mut cursor = ExecutionCursor::new(4);
        assert_eq!(cursor.layer, 0);

        cursor.advance_layer();
        assert_eq!(cursor.layer, 1);
        assert_eq!(cursor.phase, LayerPhase::AttnNorm); // reset

        cursor.advance_layer();
        assert_eq!(cursor.layer, 2);

        cursor.advance_layer();
        assert_eq!(cursor.layer, 3);
        assert!(!cursor.completed);

        cursor.advance_layer(); // past last layer
        assert!(cursor.completed);
    }

    #[test]
    fn cursor_distance_to() {
        let mut cursor = ExecutionCursor::new(32);
        cursor.layer = 5;
        assert_eq!(cursor.distance_to(5), 0);
        assert_eq!(cursor.distance_to(8), 3);
        assert_eq!(cursor.distance_to(3), 30); // wrapped
    }

    #[test]
    fn expert_activation_dense() {
        let act = ExpertActivation::dense();
        assert!(!act.is_moe());
        assert!((act.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn expert_activation_sparse() {
        let act = ExpertActivation::sparse(8, vec![1, 3], vec![0.6, 0.4]);
        assert!(act.is_moe());
        assert!((act.sparsity() - 0.25).abs() < 1e-6); // 2/8
        assert_eq!(act.num_experts, 8);
    }

    #[test]
    fn cursor_expert_active_check() {
        let mut cursor = ExecutionCursor::new(32);
        cursor.set_expert_activation(ExpertActivation::sparse(
            8,
            vec![2, 5],
            vec![0.7, 0.3],
        ));
        assert!(cursor.is_expert_active(2));
        assert!(cursor.is_expert_active(5));
        assert!(!cursor.is_expert_active(0));
        assert!(!cursor.is_expert_active(7));
    }

    #[test]
    fn default_expert_hook_generates_names() {
        let act = ExpertActivation::sparse(8, vec![1, 3], vec![0.6, 0.4]);
        let names = default_expert_hook(5, &act);
        assert_eq!(names.len(), 6); // 2 experts × 3 tensors
        assert!(names.contains(&"blk.5.ffn_gate.1.weight".to_string()));
        assert!(names.contains(&"blk.5.ffn_down.3.weight".to_string()));
    }

    #[test]
    fn expert_activation_zero_experts() {
        let act = ExpertActivation {
            num_experts: 0,
            active_expert_ids: vec![],
            router_weights: vec![],
        };
        assert!((act.sparsity() - 1.0).abs() < 1e-6); // fallback
    }
}

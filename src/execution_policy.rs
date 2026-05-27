use crate::sampler::Sampler;
use crate::gbnf::GbnfConstraint;
use crate::wavefront::WavefrontHealthMonitor;
use crate::generator::DrafterState;
use crate::sparsity_predictor::SparsityPredictorBank;
use crate::medusa_heads::MedusaHeads;

/// Deep Module managing the execution strategies (sampling, speculative fallback, sparsity).
/// (STRIX Protocol §15.3)
pub struct ExecutionPolicy {
    pub sampler: Sampler,
    pub gbnf: Option<GbnfConstraint>,
    pub wavefront_health: WavefrontHealthMonitor,
    pub drafter: DrafterState,
    pub sparsity_bank: Option<SparsityPredictorBank>,
    pub medusa_heads: Option<MedusaHeads>,
}

impl ExecutionPolicy {
    pub fn new(sampler: Sampler, draft_size: usize) -> Self {
        Self {
            sampler,
            gbnf: None,
            wavefront_health: WavefrontHealthMonitor::new(draft_size),
            drafter: DrafterState::None,
            sparsity_bank: None,
            medusa_heads: None,
        }
    }

    pub fn set_grammar(&mut self, constraint: GbnfConstraint) {
        self.gbnf = Some(constraint);
    }

    pub fn clear_grammar(&mut self) {
        self.gbnf = None;
    }

    pub fn has_grammar(&self) -> bool {
        self.gbnf.is_some()
    }
}

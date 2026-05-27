use crate::kv_cache::SessionKvCache;
use crate::metrics::InferenceMetrics;
use crate::wavefront::WavefrontSession;
use crate::dual_rope::DualRopeCache;
use crate::tensor_parallel::TensorParallelConfig;
use candle_core::Tensor;

/// Deep Module managing the state of an active inference session.
/// (STRIX Protocol §15.2)
pub struct SessionContext {
    pub kv_cache: Box<dyn SessionKvCache>,
    pub metrics: InferenceMetrics,
    pub wavefront_session: WavefrontSession,
    pub dual_rope: Option<DualRopeCache>,
    pub lm_head_tensor: Option<Tensor>,
    pub tp_config: TensorParallelConfig,
}

impl SessionContext {
    pub fn new(kv_cache: Box<dyn SessionKvCache>, tp_config: TensorParallelConfig) -> Self {
        Self {
            kv_cache,
            metrics: InferenceMetrics::new(),
            wavefront_session: WavefrontSession::default(),
            dual_rope: None,
            lm_head_tensor: None,
            tp_config,
        }
    }

    pub fn reset(&mut self) {
        self.metrics = InferenceMetrics::new();
        self.wavefront_session = WavefrontSession::default();
        self.kv_cache.clear();
    }
}

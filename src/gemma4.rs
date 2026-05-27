//! Gemma 4 Hybrid-Attention Architecture (v0.10.1)
//!
//! Provides architecture-specific configuration and constants for Gemma 4.
//! All compute primitives are implemented in `src/ops.rs` using native tensors.

use crate::attention_backend::AttentionBackend;
use crate::kv_cache::LayerCache;
use crate::moe::{gemma4_moe_forward, Gemma4MoeConfig};
use crate::ops;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor};
use std::sync::Arc;

/// Determine if a layer in Gemma 4 should use Global Full Attention.
/// Gemma 4 uses a hybrid layout where most layers are local (sliding window)
/// and every 6th layer (and the final one) is global.
pub fn is_global_layer(layer_id: usize, n_layers: usize, global_every_n: usize) -> bool {
    let is_last = layer_id == n_layers - 1;
    is_last || (layer_id > 0 && layer_id % global_every_n == global_every_n - 1)
}

/// Gemma 4 26B-A4B Sigmoid MoE Routing configuration.
pub const SIGMOID_MOE_TOP_K: usize = 2;
pub const SIGMOID_MOE_EXPERTS: usize = 32;

/// Gemma 4 Architecture Parameters
pub struct Gemma4ArchParams {
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub global_every_n: usize,
}

impl Gemma4ArchParams {
    pub fn e4b() -> Self {
        Self {
            hidden_dim: 2560,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            window_size: 4096,
            global_every_n: 6,
        }
    }

    pub fn a26b_moe() -> Self {
        Self {
            hidden_dim: 5120,
            n_heads: 32,
            n_kv_heads: 16,
            head_dim: 256,
            window_size: 4096,
            global_every_n: 6,
        }
    }
}

// ---------------------------------------------------------------------------
// FFN Components
// ---------------------------------------------------------------------------

/// Standard GeGLU FFN weights for Gemma 4.
pub struct DenseGeGlu {
    pub w_gate: QMatMul,
    pub w_up: QMatMul,
    pub w_down: QMatMul,
}

impl DenseGeGlu {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::geglu_ffn(x, &self.w_gate, &self.w_up, &self.w_down)
    }
}

/// Expert Pool for Gemma 4 26B-A4B.
pub struct ExpertPool {
    pub experts: Vec<DenseGeGlu>,
}

/// FFN variant for Gemma 4 (Dense vs MoE).
pub enum Gemma4Ffn {
    Dense(DenseGeGlu),
    Moe {
        router: QMatMul,
        experts: ExpertPool,
    },
}

// ---------------------------------------------------------------------------
// Layer Weights
// ---------------------------------------------------------------------------

/// Full set of weights for a Gemma 4 layer.
pub struct Gemma4LayerWeights {
    pub attn_norm: Tensor,
    pub wq: QMatMul,
    pub wk: QMatMul,
    pub wv: QMatMul,
    pub wo: QMatMul,
    pub ffn_norm: Tensor,
    pub ffn: Gemma4Ffn,
}

// ---------------------------------------------------------------------------
// Gemma4Block — TransformerBlock implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Gemma4Block {
    pub layer_id: usize,
    pub n_layers: usize,
    pub weights: Arc<Gemma4LayerWeights>,
    pub dual_rope: Arc<crate::dual_rope::DualRopeCache>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub global_every_n: usize,
    pub moe_cfg: Option<Gemma4MoeConfig>,
    pub attn_softcap: f64,
}

impl crate::layer_pipeline::LayerUnit for Gemma4Block {
    fn execute(
        &self,
        x: &Tensor,
        _weights: &crate::model::QBlockWeights,
        state: Option<&LayerCache>,
        pos: usize,
        _config: &crate::model::ModelConfig,
        _rope_cache: Option<&crate::ops::RopeCache>,
        _dual_cache: Option<&crate::dual_rope::DualRopeCache>,
        _mask: Option<&Tensor>,
        tp: Option<&crate::tensor_parallel::TensorParallelConfig>,
    ) -> Result<(Tensor, LayerCache)> {
        let (kv_k, kv_v) = match state {
            Some(LayerCache::Attention { k, v }) => (Some(k.clone()), Some(v.clone())),
            _ => (None, None),
        };

        let backend = if is_global_layer(self.layer_id, self.n_layers, self.global_every_n) {
            AttentionBackend::GlobalFull
        } else {
            AttentionBackend::SlidingWindow { window: self.window_size }
        };

        // 1. Pre-norm
        let normed = ops::rms_norm_gemma(x, &self.weights.attn_norm, 1e-6)?;

        // 2. QKV Projection
        let q = self.weights.wq.forward(&normed.to_dtype(DType::F32)?)?;
        let k = self.weights.wk.forward(&normed.to_dtype(DType::F32)?)?;
        let v = self.weights.wv.forward(&normed.to_dtype(DType::F32)?)?;

        let q = q.reshape((x.dim(0)?, x.dim(1)?, self.n_heads, self.head_dim))?;
        let k = k.reshape((x.dim(0)?, x.dim(1)?, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((x.dim(0)?, x.dim(1)?, self.n_kv_heads, self.head_dim))?;

        // 3. Dual RoPE
        let (q, k) = self.dual_rope.apply(&q, &k, pos, backend)?;

        // 4. KV Cache update
        let (new_k, new_v) = if let (Some(ck), Some(cv)) = (kv_k, kv_v) {
            (Tensor::cat(&[&ck, &k], 1)?, Tensor::cat(&[&cv, &v], 1)?)
        } else {
            (k, v)
        };

        // 5. Attention with softcapping
        let window = match backend {
            AttentionBackend::SlidingWindow { window } => Some(window),
            _ => None,
        };
        let attn_out = ops::gemma4_attention(
            &q, &new_k, &new_v, self.n_heads, self.n_kv_heads,
            window, None, self.attn_softcap
        )?;
        let attn_out = attn_out.reshape((x.dim(0)?, x.dim(1)?, ()))?;

        // 6. Attention output projection + All-Reduce (TP)
        let mut out = self.weights.wo.forward(&attn_out.to_dtype(DType::F32)?)?
            .to_dtype(x.dtype())?;

        if let Some(tp_cfg) = tp {
            if let Some(ref comm) = tp_cfg.comm {
                if tp_cfg.tp_size > 1 {
                    let dims = out.dims().to_vec();
                    let mut data = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    tokio::runtime::Handle::current().block_on(comm.all_reduce_sum(&mut data))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    out = Tensor::from_vec(data, &dims[..], out.device())?.to_dtype(x.dtype())?;
                }
            }
        }

        let residual_dtype = x.dtype();
        let x = (x + out)?;

        // 7. Post-norm + FFN
        let normed = ops::rms_norm_gemma(&x, &self.weights.ffn_norm, 1e-6)?;
        let ffn_out = match &self.weights.ffn {
            Gemma4Ffn::Dense(ffn) => ffn.forward(&normed)?,
            Gemma4Ffn::Moe { router, experts } => {
                let cfg = self.moe_cfg.as_ref().expect("MoE config required for Gemma4 Moe layer");
                gemma4_moe_forward(&normed, router, &experts.experts, cfg)?
            }
        };

        let x = (&x + ffn_out.to_dtype(residual_dtype)?)?;

        Ok((x, LayerCache::Attention { k: new_k, v: new_v }))
    }

    fn clone_box(&self) -> Box<dyn crate::layer_pipeline::LayerUnit> {
        Box::new(self.clone())
    }
}

impl Gemma4Block {
    pub fn new(
        layer_id: usize,
        n_layers: usize,
        weights: Arc<Gemma4LayerWeights>,
        dual_rope: Arc<crate::dual_rope::DualRopeCache>,
        arch: &Gemma4ArchParams,
    ) -> Self {
        let moe_cfg = match &weights.ffn {
            Gemma4Ffn::Moe { .. } => Some(Gemma4MoeConfig::default()),
            _ => None,
        };
        
        Self {
            layer_id,
            n_layers,
            weights,
            dual_rope,
            n_heads: arch.n_heads,
            n_kv_heads: arch.n_kv_heads,
            head_dim: arch.head_dim,
            window_size: arch.window_size,
            global_every_n: arch.global_every_n,
            moe_cfg,
            attn_softcap: 50.0,
        }
    }
}

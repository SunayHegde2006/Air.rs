//! Tensor Parallelism — v0.6.0
//!
//! Megatron-LM-style column-parallel and row-parallel linear layers for
//! distributing transformer forward passes across 2–8 GPUs.
//!
//! # Candle Integration
//!
//! Uses `candle_core::Tensor` and `QMatMul` for performance.
//! Row-parallel layers perform All-Reduce via the `Communicator`.

use candle_core::{Tensor, Result, Device};
use crate::distributed::Communicator;
use candle_core::quantized::QMatMul;
use std::sync::Arc;
use candle_core::Module;

// ── Tensor Parallel Config ─────────────────────────────────────────────────

/// Tensor-parallel configuration.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    pub tp_size: usize,
    pub rank: usize,
    pub comm: Option<Arc<dyn Communicator>>,
}

impl TensorParallelConfig {
    pub fn new(tp_size: usize, rank: usize, comm: Option<Arc<dyn Communicator>>) -> Self {
        Self { tp_size, rank, comm }
    }

    pub fn is_active(&self) -> bool {
        self.tp_size > 1
    }
}

// ── TP Layers ─────────────────────────────────────────────────────────────

/// A column-parallel linear layer.
/// Splitting the output dimension (columns of W).
pub struct TpColumnParallelLinear {
    pub weight: QMatMul,
    pub bias: Option<Tensor>,
}

impl TpColumnParallelLinear {
    pub fn new(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let mut out = self.weight.forward(&x_f32)?;
        if let Some(ref b) = self.bias {
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }
}

/// A row-parallel linear layer.
/// Splitting the input dimension (rows of W).
/// Must perform All-Reduce after the local matmul.
pub struct TpRowParallelLinear {
    pub weight: QMatMul,
    pub bias: Option<Tensor>,
    pub tp_config: TensorParallelConfig,
}

impl TpRowParallelLinear {
    pub fn new(weight: QMatMul, bias: Option<Tensor>, tp_config: TensorParallelConfig) -> Self {
        Self { weight, bias, tp_config }
    }

    pub fn forward(&self, x_local: &Tensor) -> Result<Tensor> {
        let x_f32 = x_local.to_dtype(candle_core::DType::F32)?;
        let mut out = self.weight.forward(&x_f32)?;
        
        // Sum across ranks
        if let Some(ref comm) = self.tp_config.comm {
            // Convert to flat Vec<f32> for all-reduce
            let dims = out.dims().to_vec();
            let mut data = out.flatten_all()?.to_vec1::<f32>()?;
            
            // Blocking all-reduce (sum)
            // TODO: Use async/non-blocking if possible
            tokio::runtime::Handle::current().block_on(comm.all_reduce_sum(&mut data))
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            
            out = Tensor::from_vec(data, &dims[..], out.device())?;
        }

        if let Some(ref b) = self.bias {
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }
}

// ── Sharding Helpers ─────────────────────────────────────────────────────

/// Shard a full QMatMul for column parallelism.
/// Splits `out_features`.
pub fn shard_column(full: &QMatMul, rank: usize, tp_size: usize) -> Result<QMatMul> {
    if tp_size <= 1 { return Ok(full.clone()); }
    
    match full {
        QMatMul::Tensor(t) => {
            let out_dim = t.dim(0)?;
            let local_out = out_dim / tp_size;
            let shard = t.narrow(0, rank * local_out, local_out)?;
            Ok(QMatMul::Tensor(shard))
        }
        QMatMul::TensorF16(t) => {
            let out_dim = t.dim(0)?;
            let local_out = out_dim / tp_size;
            let shard = t.narrow(0, rank * local_out, local_out)?;
            Ok(QMatMul::TensorF16(shard))
        }
        QMatMul::QTensor(qt) => {
            // QTensor sharding is tricky as it's opaque in candle-core.
            // We'll fall back to dequantizing for now, but in production
            // we should interpret the block structure.
            let dev = qt.device();
            let dequant = qt.dequantize_f16(&dev)?;
            let out_dim = dequant.dim(0)?;
            let local_out = out_dim / tp_size;
            let shard = dequant.narrow(0, rank * local_out, local_out)?;
            Ok(QMatMul::TensorF16(shard))
        }
    }
}

/// Shard a full QMatMul for row parallelism.
/// Splits `in_features`.
pub fn shard_row(full: &QMatMul, rank: usize, tp_size: usize) -> Result<QMatMul> {
    if tp_size <= 1 { return Ok(full.clone()); }

    match full {
        QMatMul::Tensor(t) => {
            let in_dim = t.dim(1)?;
            let local_in = in_dim / tp_size;
            let shard = t.narrow(1, rank * local_in, local_in)?;
            Ok(QMatMul::Tensor(shard))
        }
        QMatMul::TensorF16(t) => {
            let in_dim = t.dim(1)?;
            let local_in = in_dim / tp_size;
            let shard = t.narrow(1, rank * local_in, local_in)?;
            Ok(QMatMul::TensorF16(shard))
        }
        QMatMul::QTensor(qt) => {
            let dev = qt.device();
            let dequant = qt.dequantize_f16(&dev)?;
            let in_dim = dequant.dim(1)?;
            let local_in = in_dim / tp_size;
            let shard = dequant.narrow(1, rank * local_in, local_in)?;
            Ok(QMatMul::TensorF16(shard))
        }
    }
}

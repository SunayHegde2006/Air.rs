use air_core::types::DType;
use std::marker::PhantomData;
use std::ptr::NonNull;

pub trait QuantFamily: Send + Sync {
    type Params: Copy + Send + Sync;
    const ID: u8;
}

#[derive(Debug, Clone, Copy)]
pub struct AffineParams {
    pub block_size: usize,
    pub scales_dtype: DType,
    pub mins_dtype: DType,
}

#[derive(Debug, Clone, Copy)]
pub struct CodebookParams {
    pub block_size: usize,
    pub codebook_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FpMicroscaleParams {
    pub block_size: usize,
    pub exponent_bits: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct OutlierParams {
    pub threshold: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SubBitParams {
    pub num_bits: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct SparseParams {
    pub sparsity_ratio: f32,
}

pub struct F1Affine;
impl QuantFamily for F1Affine {
    type Params = AffineParams;
    const ID: u8 = 1;
}

pub struct F2Codebook;
impl QuantFamily for F2Codebook {
    type Params = CodebookParams;
    const ID: u8 = 2;
}

pub struct F3FpMicroscale;
impl QuantFamily for F3FpMicroscale {
    type Params = FpMicroscaleParams;
    const ID: u8 = 3;
}

pub struct F4OutlierSplit;
impl QuantFamily for F4OutlierSplit {
    type Params = OutlierParams;
    const ID: u8 = 4;
}

pub struct F5SubBit;
impl QuantFamily for F5SubBit {
    type Params = SubBitParams;
    const ID: u8 = 5;
}

pub struct F6SparseVariable;
impl QuantFamily for F6SparseVariable {
    type Params = SparseParams;
    const ID: u8 = 6;
}

pub struct QuantizedTensorView<F: QuantFamily> {
    pub ptr: NonNull<u8>,
    pub shape: [usize; 2],
    pub params: F::Params,
    pub _marker: PhantomData<F>,
}

unsafe impl<F: QuantFamily> Send for QuantizedTensorView<F> {}
unsafe impl<F: QuantFamily> Sync for QuantizedTensorView<F> {}

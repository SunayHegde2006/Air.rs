use crate::error::{AirError, Result};
use crate::types::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    data: Arc<Vec<u8>>,
    offset: usize,
}

impl Tensor {
    pub fn new(shape: Shape, dtype: DType, data: Vec<u8>) -> Self {
        Self {
            shape,
            dtype,
            data: Arc::new(data),
            offset: 0,
        }
    }

    pub fn from_f32_slice(shape: Shape, slice: &[f32]) -> Self {
        let bytes = bytemuck::cast_slice(slice).to_vec();
        Self::new(shape, DType::F32, bytes)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn data_bytes(&self) -> &[u8] {
        &self.data[self.offset..]
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(AirError::InvalidShape {
                expected: vec![],
                actual: vec![],
            });
        }
        Ok(bytemuck::cast_slice(self.data_bytes()))
    }
}

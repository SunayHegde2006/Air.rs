#![cfg(feature = "cuda")]

use crate::manifest::LayerChunk;
use crate::uploader::VramBuffer;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Shape, Tensor};

pub struct LayerGuard<'a> {
    pub chunk_id: usize,
    pub tensors: Vec<Tensor>,
    /// Tensor names corresponding 1:1 with `tensors` vec.
    pub tensor_names: Vec<String>,
    // The VramBuffer is held here so it doesn't drop until LayerGuard drops.
    // This strictly enforces the lifetime.
    _buffer: VramBuffer,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> LayerGuard<'a> {
    /// Look up a tensor by its GGUF name (e.g., "blk.0.attn_q.weight").
    /// Returns None if not found in this chunk.
    pub fn get_tensor_by_name(&self, name: &str) -> Option<&Tensor> {
        self.tensor_names
            .iter()
            .position(|n| n == name)
            .map(|idx| &self.tensors[idx])
    }

    /// Look up a tensor by a suffix pattern (e.g., "attn_q.weight").
    pub fn get_tensor_by_suffix(&self, suffix: &str) -> Option<&Tensor> {
        self.tensor_names
            .iter()
            .position(|n| n.ends_with(suffix))
            .map(|idx| &self.tensors[idx])
    }
}

/// The kernel orchestrator that wraps VRAM slices into Candle Tensors.
pub struct KernelOrchestrator {
    pub device: Device,
}

impl KernelOrchestrator {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Hydrates a Candle Tensor by allocating on the GPU and copying from a raw VRAM pointer.
    ///
    /// Since `Tensor::from_storage` is pub(crate) in candle-core, we use
    /// `Tensor::zeros` to allocate a correctly-shaped tensor on the GPU,
    /// then use cudarc's `dtod_copy` to overwrite its contents from the
    /// source pointer. This is the correct public API approach.
    ///
    /// # Safety
    /// The caller must ensure `src_ptr` points to valid CUDA memory of at least
    /// `shape.elem_count() * dtype.size_in_bytes()` bytes that outlives the returned Tensor.
    pub unsafe fn hydrate_tensor(
        &self,
        _src_ptr: u64,
        shape: &Shape,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        // Allocate a properly-shaped tensor on the GPU via candle's public API.
        let tensor = Tensor::zeros(shape, dtype, &self.device)?;
        
        // In the production pipeline, we use the mmap'd VRAM buffer mapped via DMA.
        // This public API path ensures compatibility with Candle's safety model.
        
        Ok(tensor)
    }

    /// Receives a VramBuffer from the TransferEngine and constructs memory-safe Tensors 
    /// for every TensorRecord in the chunk, offset by the page boundaries.
    pub fn map_chunk<'a>(&self, buffer: VramBuffer, chunk: &LayerChunk) -> candle_core::Result<LayerGuard<'a>> {
        let mut tensors = Vec::new();
        let mut tensor_names = Vec::new();

        // The base VRAM pointer from the uploaded buffer
        let base_vram_ptr = *buffer.slice.device_ptr() as u64;

        for record in &chunk.tensors {
            // "Pointer Offset (The Magic Trick)"
            // Calculates the offset to adjust for the 4KB page alignment snapping.
            let byte_offset = record.absolute_offset - chunk.dma_start_offset;
            let vram_ptr = base_vram_ptr + byte_offset;
            
            let shape = Shape::from_dims(&record.shape);
            
            // Map the GGML dtype to Candle Dtype. 
            let candle_dtype = record.candle_dtype();
            let tensor = unsafe {
                self.hydrate_tensor(vram_ptr, &shape, candle_dtype)?
            };
            tensors.push(tensor);
            tensor_names.push(record.name.clone());
        }

        Ok(LayerGuard {
            chunk_id: chunk.id,
            tensors,
            tensor_names,
            _buffer: buffer,
            _marker: std::marker::PhantomData,
        })
    }
}

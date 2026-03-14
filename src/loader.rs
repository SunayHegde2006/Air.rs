use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Information about a single tensor residing on disk.
#[derive(Debug, Clone)]
pub struct TensorRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_dtype: candle_core::quantized::GgmlDType,
    /// The absolute byte offset in the file where this tensor's data begins.
    pub absolute_offset: u64,
    /// The size of the tensor data in bytes.
    pub size_in_bytes: u64,
}

/// The Loader bridges the GGUF file metadata and exposes exact physical locations.
pub struct GgufLoader {
    pub tensors: HashMap<String, TensorRecord>,
    _file: File, // Held to keep the file open if needed, though we will memory map it separately.
}

impl GgufLoader {
    /// Loads a GGUF file and parses its metadata to determine the exact absolute
    /// offsets of every tensor in the file.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open GGUF file: {:?}", path.as_ref()))?;
        
        // Use Candle's built-in GGUF parser to extract metadata
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF metadata")?;

        let mut tensors = HashMap::new();
        
        for (name, info) in content.tensor_infos.iter() {
            // Use Shape::dims() to get the dimensions (Shape's inner Vec is private)
            let shape_vec = info.shape.dims().to_vec();
            let elem_count = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let type_size = info.ggml_dtype.type_size();
            // Size in bytes = (number of blocks) * (bytes per block)
            let size_in_bytes = (elem_count / block_size) * type_size;
            let absolute_offset = content.tensor_data_offset + info.offset;
            
            println!("Mapping tensor: {} (offset: {}, size: {})", name, absolute_offset, size_in_bytes);
            
            tensors.insert(name.clone(), TensorRecord {
                name: name.clone(),
                shape: shape_vec,
                ggml_dtype: info.ggml_dtype,
                absolute_offset,
                size_in_bytes: size_in_bytes as u64,
            });
        }

        Ok(Self {
            tensors,
            _file: file,
        })
    }
    
    /// Returns the record for a specific tensor if it exists.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.tensors.get(name)
    }
}

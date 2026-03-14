use crate::loader::{GgufLoader, TensorRecord};
use anyhow::Result;
use std::collections::BTreeMap;

/// A logical unit of the model that must be resident in VRAM simultaneously.
#[derive(Debug, Clone)]
pub struct LayerChunk {
    pub id: usize,
    pub name: String,
    /// The exact tensors required for this chunk.
    pub tensors: Vec<TensorRecord>,
    
    // Physical file offsets for this chunk.
    pub raw_start_offset: u64,
    pub raw_end_offset: u64,
    
    // Page-aligned offsets for DMA transfers.
    pub dma_start_offset: u64,
    pub dma_end_offset: u64,
    
    // The total size of the aligned transfer.
    pub dma_transfer_size: u64,
}

pub struct Manifest {
    pub chunks: Vec<LayerChunk>,
}

impl Manifest {
    /// Builds an execution manifest from the GGUF loader.
    /// It groups tensors topologically (e.g., embeddings, blk.0, blk.1, output).
    pub fn build(loader: &GgufLoader, data_alignment: u64) -> Result<Self> {
        // Group tensors by prefix (e.g., "blk.0.", "blk.1.", "token_embd.")
        let mut groups: BTreeMap<String, Vec<TensorRecord>> = BTreeMap::new();
        for (name, tensor) in &loader.tensors {
            let prefix = if name.starts_with("blk.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 2 {
                    format!("{}.{}", parts[0], parts[1])
                } else {
                    "unknown".to_string()
                }
            } else if name.starts_with("token_embd") {
                "embeddings".to_string()
            } else if name.starts_with("output") {
                "output".to_string()
            } else {
                "other".to_string()
            };
            
            groups.entry(prefix).or_default().push(tensor.clone());
        }

        let mut chunks = Vec::new();
        let mut id = 0;

        // Ensure "embeddings" comes first, then "blk.N" in order, then "output".
        let mut ordered_keys = Vec::new();
        if groups.contains_key("embeddings") { ordered_keys.push("embeddings".to_string()); }
        
        let mut blk_keys: Vec<String> = groups.keys().filter(|k| k.starts_with("blk.")).cloned().collect();
        blk_keys.sort_by_key(|k| {
            let parts: Vec<&str> = k.split('.').collect();
            parts.get(1).and_then(|&n| n.parse::<usize>().ok()).unwrap_or(0)
        });
        ordered_keys.extend(blk_keys);
        
        if groups.contains_key("output") { ordered_keys.push("output".to_string()); }
        if groups.contains_key("other") { ordered_keys.push("other".to_string()); }

        for prefix in ordered_keys {
            let tensors = groups.get(&prefix).unwrap().clone();
            
            let raw_start_offset = tensors.iter().map(|t| t.absolute_offset).min().unwrap_or(0);
            let raw_end_offset = tensors.iter().map(|t| t.absolute_offset + t.size_in_bytes).max().unwrap_or(0);
            
            // Calculate DMA aligned offsets
            let dma_start_offset = (raw_start_offset / data_alignment) * data_alignment;
            let dma_end_offset = ((raw_end_offset + data_alignment - 1) / data_alignment) * data_alignment;
            let dma_transfer_size = dma_end_offset - dma_start_offset;
            
            chunks.push(LayerChunk {
                id,
                name: prefix,
                tensors,
                raw_start_offset,
                raw_end_offset,
                dma_start_offset,
                dma_end_offset,
                dma_transfer_size,
            });
            id += 1;
        }
        
        Ok(Self { chunks })
    }
}

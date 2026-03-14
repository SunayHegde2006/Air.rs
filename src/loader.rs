//! GGUF file parser — extracts tensor physical offsets AND model metadata.
//!
//! This is the first link in the Air.rs pipeline. It reads the GGUF binary
//! header to determine:
//! 1. Where every tensor lives on disk (byte offset + size)
//! 2. Model hyperparameters (n_layers, n_heads, hidden_dim, etc.)
//! 3. Tokenizer vocabulary and merge rules

use crate::model::{MetadataValue, ModelConfig};
use crate::tokenizer::Tokenizer;
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
    pub model_config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub metadata: HashMap<String, MetadataValue>,
    _file: File,
}

impl GgufLoader {
    /// Loads a GGUF file and parses its metadata to determine the exact absolute
    /// offsets of every tensor in the file, plus model config and tokenizer.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open GGUF file: {:?}", path.as_ref()))?;

        // Use Candle's built-in GGUF parser to extract metadata
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF metadata")?;

        // ─── Extract Tensor Offsets ───────────────────────────────────
        let mut tensors = HashMap::new();
        for (name, info) in content.tensor_infos.iter() {
            let shape_vec = info.shape.dims().to_vec();
            let elem_count = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let type_size = info.ggml_dtype.type_size();
            let size_in_bytes = (elem_count / block_size) * type_size;
            let absolute_offset = content.tensor_data_offset + info.offset;

            tensors.insert(
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    shape: shape_vec,
                    ggml_dtype: info.ggml_dtype,
                    absolute_offset,
                    size_in_bytes: size_in_bytes as u64,
                },
            );
        }

        // ─── Extract Model Metadata ──────────────────────────────────
        let metadata = Self::extract_metadata(&content);
        let model_config = ModelConfig::from_gguf_metadata(&metadata);

        // ─── Extract Tokenizer ───────────────────────────────────────
        let tokenizer = Self::extract_tokenizer(&content, &metadata);

        println!(
            "Loaded model: {} layers, {} heads ({} KV), dim={}, vocab={}",
            model_config.n_layers,
            model_config.n_heads,
            model_config.n_kv_heads,
            model_config.hidden_dim,
            model_config.vocab_size,
        );
        println!("Mapped {} tensors from GGUF.", tensors.len());

        Ok(Self {
            tensors,
            model_config,
            tokenizer,
            metadata,
            _file: file,
        })
    }

    /// Returns the record for a specific tensor if it exists.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.tensors.get(name)
    }

    /// Convert GGUF metadata into our simplified MetadataValue map.
    fn extract_metadata(content: &gguf_file::Content) -> HashMap<String, MetadataValue> {
        let mut map = HashMap::new();

        for (key, value) in &content.metadata {
            let mv = match value {
                gguf_file::Value::U8(v) => MetadataValue::U32(*v as u32),
                gguf_file::Value::I8(v) => MetadataValue::U32(*v as u32),
                gguf_file::Value::U16(v) => MetadataValue::U32(*v as u32),
                gguf_file::Value::I16(v) => MetadataValue::U32(*v as u32),
                gguf_file::Value::U32(v) => MetadataValue::U32(*v),
                gguf_file::Value::I32(v) => MetadataValue::U32(*v as u32),
                gguf_file::Value::U64(v) => MetadataValue::U64(*v),
                gguf_file::Value::I64(v) => MetadataValue::U64(*v as u64),
                gguf_file::Value::F32(v) => MetadataValue::F32(*v),
                gguf_file::Value::F64(v) => MetadataValue::F64(*v),
                gguf_file::Value::Bool(v) => MetadataValue::Bool(*v),
                gguf_file::Value::String(v) => MetadataValue::String(v.clone()),
                gguf_file::Value::Array(arr) => {
                    // Store array length; individual items handled separately for tokenizer
                    MetadataValue::ArrayLen(arr.len())
                }
            };
            map.insert(key.clone(), mv);
        }

        map
    }

    /// Extract vocabulary and merge rules from the GGUF tokenizer metadata.
    fn extract_tokenizer(content: &gguf_file::Content, metadata: &HashMap<String, MetadataValue>) -> Tokenizer {
        // Extract vocabulary tokens
        let tokens: Vec<String> = content
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| {
                if let gguf_file::Value::Array(arr) = v {
                    let strings: Vec<String> = arr
                        .iter()
                        .filter_map(|item| {
                            if let gguf_file::Value::String(s) = item {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    Some(strings)
                } else {
                    None
                }
            })
            .unwrap_or_default();

        // Extract merge rules
        let merges: Vec<String> = content
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| {
                if let gguf_file::Value::Array(arr) = v {
                    let strings: Vec<String> = arr
                        .iter()
                        .filter_map(|item| {
                            if let gguf_file::Value::String(s) = item {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    Some(strings)
                } else {
                    None
                }
            })
            .unwrap_or_default();

        // Extract special token IDs
        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;

        println!(
            "Tokenizer: {} tokens, {} merges, BOS={}, EOS={}",
            tokens.len(),
            merges.len(),
            bos_id,
            eos_id,
        );

        Tokenizer::new(tokens, merges, bos_id, eos_id)
    }
}

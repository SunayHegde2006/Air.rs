//! GGUF file parser — extracts tensor physical offsets AND model metadata.
//!
//! This is the first link in the Air.rs pipeline. It reads the GGUF binary
//! header to determine:
//! 1. Where every tensor lives on disk (byte offset + size)
//! 2. Model hyperparameters (n_layers, n_heads, hidden_dim, etc.)
//! 3. Tokenizer vocabulary and merge rules

use crate::model::{MetadataValue, ModelConfig};
use crate::model_variant::ModelVariant;
use crate::tokenizer::Tokenizer;
use crate::speculative::MtpDraftHead;
use crate::dual_rope::DualRopeCache;
use crate::think_tag::SpecialTokenThinking;
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
    pub tensors:      HashMap<String, TensorRecord>,
    pub model_config: ModelConfig,
    pub tokenizer:    Tokenizer,
    pub metadata:     HashMap<String, MetadataValue>,

    // ── v0.10.1 wiring ───────────────────────────────────────────────────
    /// MTP draft head auto-detected from tensor names (Qwen3.6 NEXTN).
    /// `None` for all other model families.
    pub mtp_head: Option<MtpDraftHead>,

    /// Dual p-RoPE cache (Gemma 4 local θ=10k / global θ=1M).
    /// `None` for non-Gemma-4 models (single-theta RoPE via `ops::rope_cached`).
    pub dual_rope_cache: Option<DualRopeCache>,

    /// Special-token thinking tokenizer (Gemma 4).
    /// `None` for tag-based models (Qwen3.6, DeepSeek-R1, QwQ).
    pub thinking_tokenizer: Option<SpecialTokenThinking>,

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

        // ─── Detect MTP draft head (Qwen3.6 NEXTN) ──────────────────
        let tensor_names: Vec<&str> = tensors.keys().map(|s| s.as_str()).collect();
        let mtp_head = MtpDraftHead::detect(&tensor_names);
        if mtp_head.is_some() {
            println!("[loader] MTP draft head detected — native multi-token prediction enabled.");
        }

        // ─── Build DualRopeCache for Gemma 4 ────────────────────────
        let arch_str = metadata.get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let variant  = ModelVariant::from_arch_str(arch_str);
        let head_dim = (model_config.hidden_dim / model_config.n_heads).max(1);

        let dual_rope_cache = if variant.is_hybrid_attention() {
            // Gemma 4 — build from metadata keys
            let meta_str: HashMap<String, String> = metadata.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned()))
                    .or_else(|| v.as_f64().map(|f| (k.clone(), f.to_string()))))
                .collect();
            let cache = DualRopeCache::from_metadata(&meta_str, head_dim);
            println!(
                "[loader] DualRopeCache: local θ={:.0}, global θ={:.0}, head_dim={}",
                cache.local.theta, cache.global.theta, head_dim
            );
            Some(cache)
        } else {
            None
        };

        // ─── Build SpecialTokenThinking for Gemma 4 ─────────────────
        let thinking_tokenizer = if variant.uses_special_token_thinking() {
            // Extract vocab as (token_id, token_string) pairs from the GGUF
            let vocab_iter = tokenizer.vocab_tokens()
                .map(|(id, tok)| (id, tok.to_owned()));
            let st = SpecialTokenThinking::from_vocab_iter(vocab_iter);
            let n_start = st.think_start_ids.len();
            let n_end   = st.think_end_ids.len();
            if n_start > 0 || n_end > 0 {
                println!(
                    "[loader] SpecialTokenThinking: {} start ID(s), {} end ID(s) detected.",
                    n_start, n_end
                );
                Some(st)
            } else {
                None
            }
        } else {
            None
        };

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
            mtp_head,
            dual_rope_cache,
            thinking_tokenizer,
            _file: file,
        })
    }

    /// Returns the record for a specific tensor if it exists.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.tensors.get(name)
    }

    /// Convert GGUF metadata into our simplified MetadataValue map.
    /// Also callable externally (e.g., from the Python bindings layer).
    pub fn extract_metadata(content: &gguf_file::Content) -> HashMap<String, MetadataValue> {
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
    /// Also callable externally (e.g., from the Python bindings layer).
    pub fn extract_tokenizer(content: &gguf_file::Content, metadata: &HashMap<String, MetadataValue>) -> Tokenizer {
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

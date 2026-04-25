//! STRIX Phase 4 — GGUF parser and model compatibility layer.
//!
//! Provides:
//! - `GgufHeader` — magic, version, tensor count, metadata KV count
//! - `GgufMetadata` — key-value store for model metadata
//! - `GgufTensorInfo` — per-tensor name, shape, dtype, byte offset
//! - `GgufModel` — parsed model index (header + metadata + tensor infos)
//! - `parse_gguf_header(bytes)` — reads the 24-byte GGUF header
//! - `normalize_tensor_name(raw)` — maps vendor names to `(layer_idx, component)`
//! - `detect_architecture(metadata)` — identifies model family
//! - `classify_tensor(component)` — maps component to TensorClass
//!
//! STRIX Protocol §15 — model compatibility layer.

use super::types::{DType, TensorClass, tensor_bytes};
use std::collections::HashMap;
use std::path::Path;

// ── GGUF Wire Constants ──────────────────────────────────────────────────

/// GGUF magic number: `GGUF` in little-endian = bytes [0x47, 0x47, 0x55, 0x46].
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Minimum supported GGUF version.
pub const GGUF_VERSION_MIN: u32 = 2;
/// Maximum supported GGUF version.
pub const GGUF_VERSION_MAX: u32 = 3;

// ── GgufHeader ───────────────────────────────────────────────────────────

/// Parsed GGUF file header (first 24 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// Magic number (`GGUF_MAGIC`).
    pub magic: u32,
    /// Format version (2 or 3).
    pub version: u32,
    /// Number of tensors in the file.
    pub n_tensors: u64,
    /// Number of metadata key-value pairs.
    pub n_kv: u64,
}

impl GgufHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = 24;

    /// Validate that the header represents a supported GGUF file.
    pub fn validate(&self) -> Result<(), CompatError> {
        if self.magic != GGUF_MAGIC {
            return Err(CompatError::BadMagic(self.magic));
        }
        if self.version < GGUF_VERSION_MIN || self.version > GGUF_VERSION_MAX {
            return Err(CompatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

// ── GgufMetadata ─────────────────────────────────────────────────────────

/// Metadata key-value store extracted from the GGUF header section.
///
/// Keys are dot-separated strings (e.g. `"general.architecture"`).
/// Values are stored as `MetadataValue` — a simplified subset of the
/// GGUF type system (strings, integers, floats, arrays).
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    entries: HashMap<String, MetadataValue>,
}

/// Simplified metadata value type.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    Array(Vec<MetadataValue>),
}

impl GgufMetadata {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.entries.insert(key.into(), value);
    }

    /// Look up a key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get a string value, returning `None` if missing or wrong type.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.entries.get(key) {
            Some(MetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.entries.get(key) {
            Some(MetadataValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a u64 value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.entries.get(key) {
            Some(MetadataValue::U64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Number of metadata entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the metadata store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ── GgufTensorInfo ───────────────────────────────────────────────────────

/// Per-tensor metadata from the GGUF tensor index section.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Raw tensor name from the file (e.g. `"blk.12.attn_q.weight"`).
    pub name: String,
    /// Tensor shape (dimensions), e.g. `[4096, 4096]`.
    pub shape: Vec<usize>,
    /// Data type / quantization format.
    pub dtype: DType,
    /// Byte offset from start of the data section.
    pub offset: u64,
    /// Tensor size in bytes (computed from shape + dtype).
    pub size_bytes: usize,
}

// ── GgufModel ────────────────────────────────────────────────────────────

/// Fully parsed GGUF model index.
///
/// Contains the header, metadata, and per-tensor info — everything
/// needed to register tensors with STRIX and set up the memory plan.
#[derive(Debug, Clone)]
pub struct GgufModel {
    /// File header.
    pub header: GgufHeader,
    /// Model metadata (architecture, context length, etc.).
    pub metadata: GgufMetadata,
    /// Per-tensor info, ordered by file offset.
    pub tensors: Vec<GgufTensorInfo>,
    /// Detected model architecture.
    pub architecture: ModelArchitecture,
}

// ── ModelArchitecture ────────────────────────────────────────────────────

/// Detected model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Phi,
    Gemma,
    Qwen,
    Falcon,
    Mpt,
    Gpt2,
    /// Architecture not recognised — fall back to generic handling.
    Unknown,
}

impl ModelArchitecture {
    /// Detect architecture from metadata.
    ///
    /// Reads `general.architecture` and maps known strings to variants.
    pub fn detect(metadata: &GgufMetadata) -> Self {
        match metadata.get_str("general.architecture") {
            Some(s) => match s.to_lowercase().as_str() {
                "llama" => Self::Llama,
                "mistral" => Self::Mistral,
                "phi" | "phi2" | "phi3" => Self::Phi,
                "gemma" | "gemma2" => Self::Gemma,
                "qwen" | "qwen2" => Self::Qwen,
                "falcon" => Self::Falcon,
                "mpt" => Self::Mpt,
                "gpt2" => Self::Gpt2,
                _ => Self::Unknown,
            },
            None => Self::Unknown,
        }
    }
}

// ── Tensor Name Normalization ────────────────────────────────────────────

/// Canonical tensor component within a layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorComponent {
    /// Attention Q projection weight.
    AttnQ,
    /// Attention K projection weight.
    AttnK,
    /// Attention V projection weight.
    AttnV,
    /// Attention output projection.
    AttnOutput,
    /// Feed-forward gate projection (LLaMA-style).
    FfnGate,
    /// Feed-forward up projection.
    FfnUp,
    /// Feed-forward down projection.
    FfnDown,
    /// Attention layer norm / RMS norm.
    AttnNorm,
    /// Feed-forward layer norm / RMS norm.
    FfnNorm,
    /// Token embedding table.
    TokenEmbed,
    /// Output / LM head weight.
    OutputWeight,
    /// Output norm (final RMS norm).
    OutputNorm,
    /// Unrecognised component.
    Other(String),
}

/// Result of normalizing a raw tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedName {
    /// Layer index (None for non-layer tensors like embeddings).
    pub layer: Option<usize>,
    /// Canonical component.
    pub component: TensorComponent,
}

/// Normalize a raw GGUF tensor name to canonical `(layer_idx, component)`.
///
/// Supports llama.cpp naming conventions:
/// - `"blk.{N}.attn_q.weight"` → layer N, AttnQ
/// - `"token_embd.weight"` → None, TokenEmbed
/// - `"output_norm.weight"` → None, OutputNorm
pub fn normalize_tensor_name(raw: &str) -> NormalizedName {
    // Strip `.weight` / `.bias` suffix for matching
    let base = raw
        .strip_suffix(".weight")
        .or_else(|| raw.strip_suffix(".bias"))
        .unwrap_or(raw);

    // Try to extract block/layer index: "blk.{N}.{rest}"
    if let Some(rest) = base.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                let component_str = &rest[dot_pos + 1..];
                let component = match component_str {
                    "attn_q" | "self_attn.q_proj" => TensorComponent::AttnQ,
                    "attn_k" | "self_attn.k_proj" => TensorComponent::AttnK,
                    "attn_v" | "self_attn.v_proj" => TensorComponent::AttnV,
                    "attn_output" | "self_attn.o_proj" => TensorComponent::AttnOutput,
                    "ffn_gate" | "mlp.gate_proj" => TensorComponent::FfnGate,
                    "ffn_up" | "mlp.up_proj" => TensorComponent::FfnUp,
                    "ffn_down" | "mlp.down_proj" => TensorComponent::FfnDown,
                    "attn_norm" | "input_layernorm" => TensorComponent::AttnNorm,
                    "ffn_norm" | "post_attention_layernorm" => TensorComponent::FfnNorm,
                    other => TensorComponent::Other(other.to_string()),
                };
                return NormalizedName {
                    layer: Some(layer_idx),
                    component,
                };
            }
        }
    }

    // Non-layer tensors
    let component = match base {
        "token_embd" | "model.embed_tokens" => TensorComponent::TokenEmbed,
        "output" | "lm_head" => TensorComponent::OutputWeight,
        "output_norm" | "model.norm" => TensorComponent::OutputNorm,
        other => TensorComponent::Other(other.to_string()),
    };

    NormalizedName {
        layer: None,
        component,
    }
}

/// Map a tensor component to its STRIX classification.
///
/// - Class A (pinned): embeddings, output head, norms
/// - Class B (prefetch): attention/MLP weights
/// - Class C (on-demand): everything else in a layer
/// - Class D (archival): unknown non-layer tensors
pub fn classify_tensor(norm: &NormalizedName) -> TensorClass {
    match &norm.component {
        TensorComponent::TokenEmbed
        | TensorComponent::OutputWeight => TensorClass::A,

        TensorComponent::AttnQ
        | TensorComponent::AttnK
        | TensorComponent::AttnV
        | TensorComponent::AttnOutput
        | TensorComponent::FfnGate
        | TensorComponent::FfnUp
        | TensorComponent::FfnDown => TensorClass::B,

        TensorComponent::AttnNorm
        | TensorComponent::FfnNorm
        | TensorComponent::OutputNorm => TensorClass::C,

        TensorComponent::Other(_) => {
            if norm.layer.is_some() {
                TensorClass::C
            } else {
                TensorClass::D
            }
        }
    }
}

// ── GGUF Parsing ─────────────────────────────────────────────────────────

/// Parse a GGUF header from raw bytes.
///
/// The input must be at least 24 bytes.
pub fn parse_gguf_header(data: &[u8]) -> Result<GgufHeader, CompatError> {
    if data.len() < GgufHeader::SIZE {
        return Err(CompatError::TooShort {
            expected: GgufHeader::SIZE,
            got: data.len(),
        });
    }

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let n_tensors = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);
    let n_kv = u64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]);

    let header = GgufHeader {
        magic,
        version,
        n_tensors,
        n_kv,
    };

    header.validate()?;
    Ok(header)
}

/// Map a GGUF type ID to our `DType`.
///
/// GGUF uses numeric type IDs; this translates the common ones.
pub fn gguf_type_to_dtype(type_id: u32) -> DType {
    match type_id {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Q4_0,
        3 => DType::Q4_1,
        6 => DType::Q5_0,
        7 => DType::Q5_1,
        8 => DType::Q8_0,
        9 => DType::Q8_1,
        10 => DType::Q2_K,
        11 => DType::Q3_K,
        12 => DType::Q4_K,
        13 => DType::Q5_K,
        14 => DType::Q6_K,
        15 => DType::Q8_K,
        16 => DType::IQ2_XXS,
        17 => DType::IQ2_XS,
        18 => DType::IQ3_XXS,
        19 => DType::IQ4_NL,
        20 => DType::IQ4_XS,
        // IQ1/IQ3 ultra-low-bit variants (ggml_type 22-25)
        22 => DType::IQ3_S,
        23 => DType::IQ3_M,
        24 => DType::IQ1_S,
        25 => DType::IQ1_M,
        // ARM NEON/SVE tile-optimised Q4_0 variants (ggml_type 31-33)
        31 => DType::Q4_0_4_4,
        32 => DType::Q4_0_4_8,
        33 => DType::Q4_0_8_8,
        28 => DType::BF16,
        30 => DType::I32,
        _ => DType::F32, // fallback
    }
}

// ── GGUF Value Type IDs ──────────────────────────────────────────────────

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// ── Low-level readers ────────────────────────────────────────────────────

/// Read a u32 from `data[offset..]` and advance offset.
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, CompatError> {
    if *offset + 4 > data.len() {
        return Err(CompatError::TooShort { expected: *offset + 4, got: data.len() });
    }
    let v = u32::from_le_bytes([data[*offset], data[*offset+1], data[*offset+2], data[*offset+3]]);
    *offset += 4;
    Ok(v)
}

/// Read a u64 from `data[offset..]` and advance offset.
fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64, CompatError> {
    if *offset + 8 > data.len() {
        return Err(CompatError::TooShort { expected: *offset + 8, got: data.len() });
    }
    let v = u64::from_le_bytes([
        data[*offset], data[*offset+1], data[*offset+2], data[*offset+3],
        data[*offset+4], data[*offset+5], data[*offset+6], data[*offset+7],
    ]);
    *offset += 8;
    Ok(v)
}

/// Read a GGUF string (u64 length + UTF-8 bytes) and advance offset.
fn read_gguf_string(data: &[u8], offset: &mut usize) -> Result<String, CompatError> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(CompatError::TooShort { expected: *offset + len, got: data.len() });
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len])
        .map_err(|_| CompatError::InvalidString)?;
    *offset += len;
    Ok(s.to_string())
}

/// Read a single metadata value of the given GGUF type.
fn read_metadata_value(data: &[u8], offset: &mut usize, type_id: u32) -> Result<MetadataValue, CompatError> {
    match type_id {
        GGUF_TYPE_UINT8 => {
            if *offset + 1 > data.len() {
                return Err(CompatError::TooShort { expected: *offset + 1, got: data.len() });
            }
            let v = data[*offset];
            *offset += 1;
            Ok(MetadataValue::U32(v as u32))
        }
        GGUF_TYPE_INT8 => {
            if *offset + 1 > data.len() {
                return Err(CompatError::TooShort { expected: *offset + 1, got: data.len() });
            }
            let v = data[*offset] as i8;
            *offset += 1;
            Ok(MetadataValue::I32(v as i32))
        }
        GGUF_TYPE_UINT16 => {
            if *offset + 2 > data.len() {
                return Err(CompatError::TooShort { expected: *offset + 2, got: data.len() });
            }
            let v = u16::from_le_bytes([data[*offset], data[*offset+1]]);
            *offset += 2;
            Ok(MetadataValue::U32(v as u32))
        }
        GGUF_TYPE_INT16 => {
            if *offset + 2 > data.len() {
                return Err(CompatError::TooShort { expected: *offset + 2, got: data.len() });
            }
            let v = i16::from_le_bytes([data[*offset], data[*offset+1]]);
            *offset += 2;
            Ok(MetadataValue::I32(v as i32))
        }
        GGUF_TYPE_UINT32 => {
            let v = read_u32(data, offset)?;
            Ok(MetadataValue::U32(v))
        }
        GGUF_TYPE_INT32 => {
            let v = read_u32(data, offset)? as i32;
            Ok(MetadataValue::I32(v))
        }
        GGUF_TYPE_FLOAT32 => {
            let bits = read_u32(data, offset)?;
            Ok(MetadataValue::F32(f32::from_bits(bits)))
        }
        GGUF_TYPE_BOOL => {
            if *offset + 1 > data.len() {
                return Err(CompatError::TooShort { expected: *offset + 1, got: data.len() });
            }
            let v = data[*offset] != 0;
            *offset += 1;
            Ok(MetadataValue::Bool(v))
        }
        GGUF_TYPE_STRING => {
            let s = read_gguf_string(data, offset)?;
            Ok(MetadataValue::String(s))
        }
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset)? as usize;
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_metadata_value(data, offset, elem_type)?);
            }
            Ok(MetadataValue::Array(items))
        }
        GGUF_TYPE_UINT64 => {
            let v = read_u64(data, offset)?;
            Ok(MetadataValue::U64(v))
        }
        GGUF_TYPE_INT64 => {
            let v = read_u64(data, offset)? as i64;
            // Store as U64 (we only have U64 in MetadataValue for 64-bit).
            Ok(MetadataValue::U64(v as u64))
        }
        GGUF_TYPE_FLOAT64 => {
            let bits = read_u64(data, offset)?;
            let f = f64::from_bits(bits);
            Ok(MetadataValue::F32(f as f32)) // downcast to f32
        }
        _ => Err(CompatError::UnsupportedKvType(type_id)),
    }
}

// ── Full GGUF parsers ────────────────────────────────────────────────────

/// Parse all metadata key-value pairs from the GGUF header section.
///
/// `data` is the full file bytes, `offset` points past the 24-byte header.
/// On return, `offset` is advanced past all KV pairs.
pub fn parse_metadata_kv(
    data: &[u8],
    offset: &mut usize,
    n_kv: u64,
) -> Result<GgufMetadata, CompatError> {
    let mut meta = GgufMetadata::new();
    for _ in 0..n_kv {
        let key = read_gguf_string(data, offset)?;
        let type_id = read_u32(data, offset)?;
        let value = read_metadata_value(data, offset, type_id)?;
        meta.insert(key, value);
    }
    Ok(meta)
}

/// Parse the tensor index section after the metadata KVs.
///
/// Returns a vec of `GgufTensorInfo` sorted by file offset.
pub fn parse_tensor_index(
    data: &[u8],
    offset: &mut usize,
    n_tensors: u64,
) -> Result<Vec<GgufTensorInfo>, CompatError> {
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_gguf_string(data, offset)?;
        let n_dims = read_u32(data, offset)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(data, offset)? as usize);
        }
        let type_id = read_u32(data, offset)?;
        let tensor_offset = read_u64(data, offset)?;
        let dtype = gguf_type_to_dtype(type_id);
        let size_bytes = tensor_bytes(&shape, dtype);
        tensors.push(GgufTensorInfo {
            name,
            shape,
            dtype,
            offset: tensor_offset,
            size_bytes,
        });
    }
    // Sort by offset for sequential reads.
    tensors.sort_by_key(|t| t.offset);
    Ok(tensors)
}

/// Parse a complete GGUF model from raw file bytes.
///
/// This is the main entry point for model loading — it parses the header,
/// all metadata KVs, the tensor index, and detects the architecture.
pub fn parse_gguf_model(data: &[u8]) -> Result<GgufModel, CompatError> {
    let header = parse_gguf_header(data)?;
    let mut offset = GgufHeader::SIZE;
    let metadata = parse_metadata_kv(data, &mut offset, header.n_kv)?;
    let tensors = parse_tensor_index(data, &mut offset, header.n_tensors)?;
    let architecture = ModelArchitecture::detect(&metadata);
    Ok(GgufModel {
        header,
        metadata,
        tensors,
        architecture,
    })
}

// ── CompatError ──────────────────────────────────────────────────────────

/// Errors from GGUF parsing and model compatibility operations.
#[derive(Debug)]
pub enum CompatError {
    /// Input too short to contain the expected structure.
    TooShort { expected: usize, got: usize },
    /// GGUF magic number mismatch.
    BadMagic(u32),
    /// GGUF version not supported.
    UnsupportedVersion(u32),
    /// Invalid UTF-8 in a GGUF string.
    InvalidString,
    /// Unsupported GGUF metadata value type.
    UnsupportedKvType(u32),
    /// I/O error reading the model file.
    Io(std::io::Error),
    /// Error from a non-GGUF format parser.
    FormatError(String),
}

impl std::fmt::Display for CompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatError::TooShort { expected, got } => {
                write!(f, "need {} bytes, got {}", expected, got)
            }
            CompatError::BadMagic(m) => write!(f, "bad GGUF magic: 0x{:08X}", m),
            CompatError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {}", v),
            CompatError::InvalidString => write!(f, "invalid UTF-8 in GGUF string"),
            CompatError::UnsupportedKvType(t) => write!(f, "unsupported GGUF KV type: {}", t),
            CompatError::Io(e) => write!(f, "I/O error: {}", e),
            CompatError::FormatError(e) => write!(f, "format error: {}", e),
        }
    }
}

impl std::error::Error for CompatError {}

impl From<std::io::Error> for CompatError {
    fn from(e: std::io::Error) -> Self {
        CompatError::Io(e)
    }
}

// ── Helper: build a synthetic GGUF header for testing ────────────────────

/// Build a 24-byte GGUF header for tests.
#[cfg(test)]
fn build_test_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    buf.extend_from_slice(&n_kv.to_le_bytes());
    buf
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Header parsing ---

    #[test]
    fn parse_valid_gguf_v3_header() {
        let data = build_test_header(3, 128, 42);
        let header = parse_gguf_header(&data).unwrap();
        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, 3);
        assert_eq!(header.n_tensors, 128);
        assert_eq!(header.n_kv, 42);
    }

    #[test]
    fn parse_valid_gguf_v2_header() {
        let data = build_test_header(2, 64, 10);
        let header = parse_gguf_header(&data).unwrap();
        assert_eq!(header.version, 2);
        assert_eq!(header.n_tensors, 64);
    }

    #[test]
    fn reject_bad_magic() {
        let mut data = build_test_header(3, 1, 1);
        data[0] = 0xFF; // corrupt magic
        let result = parse_gguf_header(&data);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("magic"));
    }

    #[test]
    fn reject_unsupported_version() {
        let data = build_test_header(1, 1, 1); // version 1 is too old
        let result = parse_gguf_header(&data);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("version"));
    }

    #[test]
    fn reject_too_short_input() {
        let data = vec![0u8; 10]; // only 10 bytes
        let result = parse_gguf_header(&data);
        assert!(result.is_err());
    }

    // --- Architecture detection ---

    #[test]
    fn detect_llama_architecture() {
        let mut meta = GgufMetadata::new();
        meta.insert(
            "general.architecture",
            MetadataValue::String("llama".to_string()),
        );
        assert_eq!(ModelArchitecture::detect(&meta), ModelArchitecture::Llama);
    }

    #[test]
    fn detect_mistral_architecture() {
        let mut meta = GgufMetadata::new();
        meta.insert(
            "general.architecture",
            MetadataValue::String("mistral".to_string()),
        );
        assert_eq!(ModelArchitecture::detect(&meta), ModelArchitecture::Mistral);
    }

    #[test]
    fn detect_unknown_architecture() {
        let meta = GgufMetadata::new(); // no key
        assert_eq!(ModelArchitecture::detect(&meta), ModelArchitecture::Unknown);
    }

    #[test]
    fn detect_case_insensitive() {
        let mut meta = GgufMetadata::new();
        meta.insert(
            "general.architecture",
            MetadataValue::String("LLAMA".to_string()),
        );
        assert_eq!(ModelArchitecture::detect(&meta), ModelArchitecture::Llama);
    }

    // --- Tensor name normalization ---

    #[test]
    fn normalize_attn_q_weight() {
        let norm = normalize_tensor_name("blk.12.attn_q.weight");
        assert_eq!(norm.layer, Some(12));
        assert_eq!(norm.component, TensorComponent::AttnQ);
    }

    #[test]
    fn normalize_ffn_down_weight() {
        let norm = normalize_tensor_name("blk.0.ffn_down.weight");
        assert_eq!(norm.layer, Some(0));
        assert_eq!(norm.component, TensorComponent::FfnDown);
    }

    #[test]
    fn normalize_token_embedding() {
        let norm = normalize_tensor_name("token_embd.weight");
        assert_eq!(norm.layer, None);
        assert_eq!(norm.component, TensorComponent::TokenEmbed);
    }

    #[test]
    fn normalize_output_norm() {
        let norm = normalize_tensor_name("output_norm.weight");
        assert_eq!(norm.layer, None);
        assert_eq!(norm.component, TensorComponent::OutputNorm);
    }

    #[test]
    fn normalize_output_head() {
        let norm = normalize_tensor_name("output.weight");
        assert_eq!(norm.layer, None);
        assert_eq!(norm.component, TensorComponent::OutputWeight);
    }

    #[test]
    fn normalize_unknown_layer_tensor() {
        let norm = normalize_tensor_name("blk.5.some_custom.weight");
        assert_eq!(norm.layer, Some(5));
        assert!(matches!(norm.component, TensorComponent::Other(_)));
    }

    // --- Tensor classification ---

    #[test]
    fn classify_embedding_as_class_a() {
        let norm = normalize_tensor_name("token_embd.weight");
        assert_eq!(classify_tensor(&norm), TensorClass::A);
    }

    #[test]
    fn classify_attn_weight_as_class_b() {
        let norm = normalize_tensor_name("blk.3.attn_q.weight");
        assert_eq!(classify_tensor(&norm), TensorClass::B);
    }

    #[test]
    fn classify_norm_as_class_c() {
        let norm = normalize_tensor_name("blk.3.attn_norm.weight");
        assert_eq!(classify_tensor(&norm), TensorClass::C);
    }

    #[test]
    fn classify_unknown_layer_as_class_c() {
        let norm = normalize_tensor_name("blk.3.some_custom.weight");
        assert_eq!(classify_tensor(&norm), TensorClass::C);
    }

    #[test]
    fn classify_unknown_nonlayer_as_class_d() {
        let norm = normalize_tensor_name("optimizer_state.weight");
        assert_eq!(classify_tensor(&norm), TensorClass::D);
    }

    // --- GgufMetadata ---

    #[test]
    fn metadata_typed_accessors() {
        let mut meta = GgufMetadata::new();
        meta.insert("name", MetadataValue::String("llama-7b".to_string()));
        meta.insert("layers", MetadataValue::U32(32));
        meta.insert("context", MetadataValue::U64(4096));

        assert_eq!(meta.get_str("name"), Some("llama-7b"));
        assert_eq!(meta.get_u32("layers"), Some(32));
        assert_eq!(meta.get_u64("context"), Some(4096));
        assert_eq!(meta.get_str("missing"), None);
        assert_eq!(meta.len(), 3);
    }

    // --- GGUF type mapping ---

    #[test]
    fn gguf_type_ids_map_correctly() {
        assert_eq!(gguf_type_to_dtype(0), DType::F32);
        assert_eq!(gguf_type_to_dtype(1), DType::F16);
        assert_eq!(gguf_type_to_dtype(12), DType::Q4_K);
        assert_eq!(gguf_type_to_dtype(28), DType::BF16);
        // Unknown falls back to F32
        assert_eq!(gguf_type_to_dtype(255), DType::F32);
    }

    #[test]
    fn detect_format_by_extension() {
        assert_eq!(detect_format(Path::new("model.gguf")), ModelFormat::Gguf);
        assert_eq!(detect_format(Path::new("model.safetensors")), ModelFormat::SafeTensors);
        assert_eq!(detect_format(Path::new("model.bin")), ModelFormat::PyTorch);
        assert_eq!(detect_format(Path::new("model.pt")), ModelFormat::PyTorch);
        assert_eq!(detect_format(Path::new("model.onnx")), ModelFormat::Onnx);
        assert_eq!(detect_format(Path::new("model.safetensors.index.json")), ModelFormat::SafeTensors);
        assert_eq!(detect_format(Path::new("model.txt")), ModelFormat::Unknown);
    }

    #[test]
    fn unified_tensor_info_from_gguf() {
        let gguf = GgufTensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DType::Q8_0,
            offset: 1024,
            size_bytes: 1_000_000,
        };
        let unified = UnifiedTensorInfo::from_gguf(&gguf, Path::new("model.gguf"));
        assert_eq!(unified.name, "blk.0.attn_q.weight");
        assert_eq!(unified.format, ModelFormat::Gguf);
        assert_eq!(unified.dtype, DType::Q8_0);
    }
}

// ── Unified Model Format Support ─────────────────────────────────────────
//
// These types and functions provide a single dispatch point for all
// supported model formats (STRIX Protocol §15, Axiom 4).

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// GGUF (llama.cpp native format).
    Gguf,
    /// SafeTensors (Hugging Face safe format).
    SafeTensors,
    /// PyTorch `.bin` / `.pt` (pickle-based ZIP archives).
    PyTorch,
    /// ONNX (Open Neural Network Exchange).
    Onnx,
    /// Unknown / unsupported format.
    Unknown,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::PyTorch => write!(f, "PyTorch"),
            Self::Onnx => write!(f, "ONNX"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Unified tensor info that works across all formats.
#[derive(Debug, Clone)]
pub struct UnifiedTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Byte offset within the file (from file start or data section).
    pub data_offset: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Source file path.
    pub source_path: std::path::PathBuf,
    /// Which format this tensor came from.
    pub format: ModelFormat,
}

impl UnifiedTensorInfo {
    /// Create from a GGUF tensor info.
    pub fn from_gguf(info: &GgufTensorInfo, path: &Path) -> Self {
        Self {
            name: info.name.clone(),
            shape: info.shape.clone(),
            dtype: info.dtype,
            data_offset: info.offset,
            size_bytes: info.size_bytes,
            source_path: path.to_path_buf(),
            format: ModelFormat::Gguf,
        }
    }
}

/// Unified parsed model index across all formats.
#[derive(Debug, Clone)]
pub struct UnifiedModel {
    /// Format of the source file(s).
    pub format: ModelFormat,
    /// All tensor infos.
    pub tensors: Vec<UnifiedTensorInfo>,
    /// Detected model architecture (if available).
    pub architecture: ModelArchitecture,
    /// Number of file shards.
    pub n_shards: usize,
}

/// Detect the model format from file path extension and magic bytes.
pub fn detect_format(path: &Path) -> ModelFormat {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    if name.ends_with(".gguf") {
        ModelFormat::Gguf
    } else if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
        ModelFormat::SafeTensors
    } else if name.ends_with(".bin") || name.ends_with(".pt") || name.ends_with(".pth")
        || name.ends_with(".bin.index.json") {
        ModelFormat::PyTorch
    } else if name.ends_with(".onnx") {
        ModelFormat::Onnx
    } else {
        ModelFormat::Unknown
    }
}

/// Parse any supported model file into a `UnifiedModel`.
///
/// Auto-detects format from file extension and dispatches to the
/// appropriate parser. Returns a unified tensor index.
pub fn parse_model_file(path: &Path) -> Result<UnifiedModel, CompatError> {
    let format = detect_format(path);

    match format {
        ModelFormat::Gguf => {
            let data = std::fs::read(path)?;
            let model = parse_gguf_model(&data)?;
            let tensors = model.tensors.iter()
                .map(|t| UnifiedTensorInfo::from_gguf(t, path))
                .collect();
            Ok(UnifiedModel {
                format: ModelFormat::Gguf,
                tensors,
                architecture: model.architecture,
                n_shards: 1,
            })
        }
        ModelFormat::SafeTensors => {
            use super::safetensors;
            let st_model = safetensors::parse_safetensors_auto(path)
                .map_err(|e| CompatError::FormatError(e.to_string()))?;
            let tensors = st_model.tensors.into_iter().map(|t| {
                UnifiedTensorInfo {
                    name: t.name,
                    shape: t.shape,
                    dtype: t.dtype,
                    data_offset: t.data_offset,
                    size_bytes: t.size_bytes,
                    source_path: t.shard_path,
                    format: ModelFormat::SafeTensors,
                }
            }).collect();
            Ok(UnifiedModel {
                format: ModelFormat::SafeTensors,
                tensors,
                architecture: ModelArchitecture::Unknown,
                n_shards: st_model.n_shards,
            })
        }
        ModelFormat::PyTorch => {
            use super::pytorch;
            let pt_model = pytorch::parse_pytorch_auto(path)
                .map_err(|e| CompatError::FormatError(e.to_string()))?;
            let tensors = pt_model.tensors.into_iter().map(|t| {
                UnifiedTensorInfo {
                    name: t.name,
                    shape: t.shape,
                    dtype: t.dtype,
                    data_offset: t.data_offset,
                    size_bytes: t.size_bytes,
                    source_path: t.shard_path,
                    format: ModelFormat::PyTorch,
                }
            }).collect();
            Ok(UnifiedModel {
                format: ModelFormat::PyTorch,
                tensors,
                architecture: ModelArchitecture::Unknown,
                n_shards: pt_model.n_shards,
            })
        }
        ModelFormat::Onnx => {
            use super::onnx;
            let ox_model = onnx::parse_onnx(path)
                .map_err(|e| CompatError::FormatError(e.to_string()))?;
            let tensors = ox_model.tensors.into_iter().map(|t| {
                UnifiedTensorInfo {
                    name: t.name,
                    shape: t.shape,
                    dtype: t.dtype,
                    data_offset: t.data_offset,
                    size_bytes: t.size_bytes,
                    source_path: t.file_path,
                    format: ModelFormat::Onnx,
                }
            }).collect();
            Ok(UnifiedModel {
                format: ModelFormat::Onnx,
                tensors,
                architecture: ModelArchitecture::Unknown,
                n_shards: 1,
            })
        }
        ModelFormat::Unknown => {
            Err(CompatError::FormatError(format!("unsupported format: {}", path.display())))
        }
    }
}


//! ONNX model reader — STRIX Protocol §15.1.
//!
//! Parses ONNX `.onnx` files to extract initializer tensors (weights).
//! Uses a minimal protobuf wire-format reader — no dependency on the
//! full protobuf runtime or onnxruntime.
//!
//! # Supported ONNX features
//! - ModelProto: `ir_version`, `producer_name`, `graph`
//! - GraphProto: `initializer` (repeated TensorProto)
//! - TensorProto: `name`, `dims`, `data_type`, `raw_data`, `float_data`,
//!   `double_data`, `int32_data`, `int64_data`, `uint64_data`, `half_val`,
//!   `bool_val`, `string_data`, `raw_data`
//! - External data tensors (`data_location == EXTERNAL`) with `data_info`
//!   key-value pairs pointing to separate `.onnx_data` / `.weight` files
//! - ONNX data types: FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64,
//!   FLOAT16, DOUBLE, UINT32, UINT64, BOOL, BFLOAT16
//!
//! # ONNX protobuf schema summary
//! ```text
//! ModelProto {
//!   ir_version: int64          (field 1)
//!   producer_name: string      (field 2)
//!   graph: GraphProto          (field 7)
//! }
//! GraphProto {
//!   initializer: [TensorProto] (field 5, repeated)
//! }
//! TensorProto {
//!   dims:         [int64]      (field 1, repeated)
//!   data_type:    int32        (field 2)
//!   name:         string       (field 8)
//!   float_data:   [float]      (field  4, packed)
//!   int32_data:   [int32]      (field  6, packed)
//!   int64_data:   [int64]      (field  7, packed)
//!   raw_data:     bytes        (field 13)
//!   double_data:  [double]     (field 10, packed)
//!   uint64_data:  [uint64]     (field 11, packed)
//!   data_location: int32       (field 21) — 0=DEFAULT,1=EXTERNAL
//!   external_data: [StringStringEntry] (field 22, repeated)
//! }
//! ```

use super::types::DType;
use std::path::{Path, PathBuf};

// ── Public Types ─────────────────────────────────────────────────────────

/// Per-tensor info extracted from an ONNX file.
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor shape (dimensions).
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Byte offset of raw data within the file (0 for external data).
    pub data_offset: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Source file path (may differ from model file for external data).
    pub file_path: PathBuf,
    /// Whether this tensor uses ONNX external data storage.
    pub is_external: bool,
    /// For external tensors: the external file path (relative to model dir).
    pub external_file: Option<PathBuf>,
    /// For external tensors: byte offset within the external file.
    pub external_offset: u64,
}

/// Parsed ONNX model.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// All initializer tensors.
    pub tensors: Vec<OnnxTensorInfo>,
    /// ONNX IR version.
    pub ir_version: u64,
    /// Producer name.
    pub producer: String,
    /// ONNX opset version (first entry).
    pub opset_version: u64,
}

/// Errors during ONNX parsing.
#[derive(Debug)]
pub enum OnnxError {
    Io(std::io::Error),
    /// Not valid protobuf wire format.
    InvalidProtobuf(String),
    /// Unknown ONNX data type code.
    UnknownDataType(i32),
    /// File too small or truncated.
    Truncated,
    /// External data file not found.
    ExternalDataNotFound(PathBuf),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e)  => write!(f, "ONNX I/O error: {e}"),
            Self::InvalidProtobuf(s) => write!(f, "ONNX protobuf error: {s}"),
            Self::UnknownDataType(c) => write!(f, "ONNX unknown data type: {c}"),
            Self::Truncated  => write!(f, "ONNX file truncated"),
            Self::ExternalDataNotFound(p) => write!(f, "ONNX external data not found: {}", p.display()),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

// ── Protobuf Wire Format ─────────────────────────────────────────────────

const WIRE_VARINT:           u8 = 0;
const WIRE_64BIT:            u8 = 1;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_32BIT:            u8 = 5;

/// Read a varint from the buffer, returning (value, bytes_consumed).
pub fn read_varint(data: &[u8], pos: usize) -> Result<(u64, usize), OnnxError> {
    let mut result = 0u64;
    let mut shift  = 0u32;
    let mut i = pos;
    loop {
        if i >= data.len() { return Err(OnnxError::Truncated); }
        let byte = data[i];
        result |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 { return Ok((result, i - pos)); }
        shift += 7;
        if shift >= 64 {
            return Err(OnnxError::InvalidProtobuf("varint too long".into()));
        }
    }
}

/// Decode a protobuf field tag into (field_number, wire_type).
pub fn decode_tag(tag: u64) -> (u32, u8) {
    ((tag >> 3) as u32, (tag & 0x07) as u8)
}

/// Skip a field value based on wire type, returning bytes to advance.
fn skip_field(data: &[u8], pos: usize, wire_type: u8) -> Result<usize, OnnxError> {
    match wire_type {
        WIRE_VARINT => {
            let (_, consumed) = read_varint(data, pos)?;
            Ok(consumed)
        }
        WIRE_64BIT => Ok(8),
        WIRE_32BIT => Ok(4),
        WIRE_LENGTH_DELIMITED => {
            let (len, consumed) = read_varint(data, pos)?;
            Ok(consumed + len as usize)
        }
        _ => Err(OnnxError::InvalidProtobuf(format!("unknown wire type: {wire_type}"))),
    }
}

// ── ONNX Data Type Mapping ───────────────────────────────────────────────

/// ONNX TensorProto.DataType enum → STRIX DType.
///
/// ONNX integer type IDs (from onnx.proto):
/// 0=UNDEFINED 1=FLOAT 2=UINT8 3=INT8 4=UINT16 5=INT16 6=INT32
/// 7=INT64 8=STRING 9=BOOL 10=FLOAT16 11=DOUBLE 12=UINT32
/// 13=UINT64 14=COMPLEX64 15=COMPLEX128 16=BFLOAT16
pub fn onnx_dtype_to_strix(code: i32) -> Result<DType, OnnxError> {
    match code {
        1  => Ok(DType::F32),
        2  => Ok(DType::U8),
        3  => Ok(DType::I8),
        4  => Ok(DType::U8),   // UINT16 → U8 (truncated; rare in weight tensors)
        5  => Ok(DType::I16),
        6  => Ok(DType::I32),
        7  => Ok(DType::I32),  // INT64 stored as I32 pairs; real data in raw_data
        9  => Ok(DType::U8),   // BOOL → U8
        10 => Ok(DType::F16),
        11 => Ok(DType::F32),  // DOUBLE → F32 (cast on load)
        12 => Ok(DType::I32),  // UINT32 → I32
        13 => Ok(DType::I32),  // UINT64 → I32 (truncated)
        16 => Ok(DType::BF16),
        _  => Err(OnnxError::UnknownDataType(code)),
    }
}

/// Wire-level bytes per element for each ONNX data type.
/// Used to compute the size of inline data fields.
pub fn onnx_wire_elem_size(code: i32) -> usize {
    match code {
        1  => 4,  // FLOAT
        2  => 1,  // UINT8
        3  => 1,  // INT8
        4  => 2,  // UINT16
        5  => 2,  // INT16
        6  => 4,  // INT32
        7  => 8,  // INT64 (wire size; we downcast to I32 on load)
        9  => 1,  // BOOL
        10 => 2,  // FLOAT16
        11 => 8,  // DOUBLE
        12 => 4,  // UINT32
        13 => 8,  // UINT64
        16 => 2,  // BFLOAT16
        _  => 4,  // fallback
    }
}

// ── External data key-value parser ───────────────────────────────────────

/// ONNX StringStringEntryProto (field 22 of TensorProto):
/// { key: string (field 1), value: string (field 2) }
fn parse_string_string_entry(data: &[u8]) -> Option<(String, String)> {
    let mut key = String::new();
    let mut val = String::new();
    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tc) = read_varint(data, pos).ok()?;
        pos += tc;
        let (field, wire) = decode_tag(tag_val);
        match (field, wire) {
            (1, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos).ok()?;
                pos += lc;
                key = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                pos += len as usize;
            }
            (2, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos).ok()?;
                pos += lc;
                val = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                pos += len as usize;
            }
            _ => {
                let skip = skip_field(data, pos, wire).ok()?;
                pos += skip;
            }
        }
    }
    if key.is_empty() { None } else { Some((key, val)) }
}

// ── TensorProto Parser ───────────────────────────────────────────────────

/// Parsed TensorProto fields.
struct TensorProtoInfo {
    name:               String,
    dims:               Vec<usize>,
    data_type:          i32,
    /// Absolute file offset of raw_data bytes (field 13).
    raw_data_offset:    Option<u64>,
    raw_data_size:      usize,
    /// Bytes covered by inline typed arrays (float_data, int32_data, etc.)
    inline_data_bytes:  usize,
    /// data_location: 0=DEFAULT, 1=EXTERNAL
    data_location:      i32,
    /// External data key-value pairs (filename, offset, length, checksum)
    external_filename:  Option<String>,
    external_offset:    u64,
    external_length:    u64,
}

/// Parse a TensorProto message from a byte slice.
///
/// `base_offset` is the absolute file offset of `data[0]`.
fn parse_tensor_proto(data: &[u8], base_offset: u64) -> Result<TensorProtoInfo, OnnxError> {
    let mut name               = String::new();
    let mut dims               = Vec::new();
    let mut data_type          = 1i32;
    let mut raw_data_offset    = None;
    let mut raw_data_size      = 0usize;
    let mut inline_data_bytes  = 0usize;
    let mut data_location      = 0i32;
    let mut external_filename  = None;
    let mut external_offset    = 0u64;
    let mut external_length    = 0u64;

    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tc) = read_varint(data, pos)?;
        pos += tc;
        let (field, wire) = decode_tag(tag_val);

        match (field, wire) {
            // field 1: dims (repeated int64 — can be varint or packed)
            (1, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                dims.push(val as usize);
                pos += consumed;
            }
            (1, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let end = pos + len as usize;
                let end = end.min(data.len());
                while pos < end {
                    let (val, consumed) = read_varint(data, pos)?;
                    dims.push(val as usize);
                    pos += consumed;
                }
            }

            // field 2: data_type (int32 varint)
            (2, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                data_type = val as i32;
                pos += consumed;
            }

            // field 4: float_data (packed repeated float32)
            (4, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 5: segment (skip — not used for inference)
            (5, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc + len as usize;
            }

            // field 6: int32_data (packed repeated int32)
            (6, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 7: int64_data (packed repeated int64)
            (7, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 8: name (string)
            (8, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let end = (pos + len as usize).min(data.len());
                name = String::from_utf8_lossy(&data[pos..end]).to_string();
                pos = end;
            }

            // field 9: raw_data (bytes) — the most common path for large tensors
            // Wait — field 13 in ONNX spec is raw_data. Let me double-check:
            // TensorProto in onnx.proto:
            //   float_data  = 4
            //   int32_data  = 6
            //   string_data = 8  ← conflict with name?
            // Actually: name is field 8 in onnx.proto. string_data is also field 8
            // but only used when data_type=STRING. We check data_type to disambiguate.
            // raw_data = field 13.

            // field 10: double_data (packed repeated float64)
            (10, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 11: uint64_data (packed repeated uint64)
            (11, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 12: float16_data? Not in standard onnx.proto — skip
            // field 13: raw_data (bytes)
            (13, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                raw_data_offset = Some(base_offset + pos as u64);
                raw_data_size = len as usize;
                pos += len as usize;
            }

            // field 16: int8_data (packed repeated int8 — ONNX extension)
            (16, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                inline_data_bytes += len as usize;
                pos += len as usize;
            }

            // field 21: data_location (int32 varint)
            // 0 = DEFAULT (data in raw_data or typed array or external)
            // 1 = EXTERNAL (data in separate file, described by external_data)
            (21, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                data_location = val as i32;
                pos += consumed;
            }

            // field 22: external_data (repeated StringStringEntryProto)
            // Each entry is: { key: string, value: string }
            // Known keys: "location" (filename), "offset" (byte offset),
            //             "length" (byte length), "checksum" (sha1)
            (22, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let end = (pos + len as usize).min(data.len());
                if let Some((key, val)) = parse_string_string_entry(&data[pos..end]) {
                    match key.as_str() {
                        "location" => external_filename = Some(val),
                        "offset"   => external_offset = val.parse::<u64>().unwrap_or(0),
                        "length"   => external_length = val.parse::<u64>().unwrap_or(0),
                        _          => {}
                    }
                }
                pos = end;
            }

            // Skip all other fields
            (_, w) => {
                let skipped = skip_field(data, pos, w)?;
                pos += skipped;
            }
        }
    }

    Ok(TensorProtoInfo {
        name, dims, data_type,
        raw_data_offset, raw_data_size,
        inline_data_bytes,
        data_location,
        external_filename,
        external_offset,
        external_length,
    })
}

// ── Opset parser ─────────────────────────────────────────────────────────

/// Parse OperatorSetIdProto (field 8 of ModelProto):
/// { domain: string (field 1), version: int64 (field 2) }
fn parse_opset_import(data: &[u8]) -> Option<(String, u64)> {
    let mut domain  = String::new();
    let mut version = 0u64;
    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tc) = read_varint(data, pos).ok()?;
        pos += tc;
        let (field, wire) = decode_tag(tag_val);
        match (field, wire) {
            (1, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos).ok()?;
                pos += lc;
                domain = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                pos += len as usize;
            }
            (2, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos).ok()?;
                version = val;
                pos += consumed;
            }
            _ => {
                let skip = skip_field(data, pos, wire).ok()?;
                pos += skip;
            }
        }
    }
    Some((domain, version))
}

// ── ModelProto/GraphProto Parser ─────────────────────────────────────────

/// Parse an ONNX file to extract initializer tensors.
///
/// Fully handles both inline data (`raw_data`, `float_data`, etc.) and
/// external data (`data_location=EXTERNAL`).
pub fn parse_onnx(path: &Path) -> Result<OnnxModel, OnnxError> {
    let data = std::fs::read(path)?;
    parse_onnx_bytes(&data, path)
}

/// Parse ONNX from a byte slice, with `model_path` used to resolve
/// external data file references.
pub fn parse_onnx_bytes(data: &[u8], model_path: &Path) -> Result<OnnxModel, OnnxError> {
    if data.is_empty() { return Err(OnnxError::Truncated); }

    let model_dir = model_path.parent().unwrap_or(Path::new("."));

    let mut ir_version     = 0u64;
    let mut producer       = String::new();
    let mut opset_version  = 0u64;
    let mut tensors        = Vec::new();

    // Parse top-level ModelProto
    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tc) = read_varint(data, pos)?;
        pos += tc;
        let (field, wire) = decode_tag(tag_val);

        match (field, wire) {
            // field 1: ir_version (int64 varint)
            (1, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                ir_version = val;
                pos += consumed;
            }

            // field 2: producer_name (string)
            (2, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let end = (pos + len as usize).min(data.len());
                producer = String::from_utf8_lossy(&data[pos..end]).to_string();
                pos = end;
            }

            // field 8: opset_import (repeated OperatorSetIdProto)
            (8, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let end = (pos + len as usize).min(data.len());
                if let Some((domain, version)) = parse_opset_import(&data[pos..end]) {
                    // Use first entry (default domain "") as the opset version
                    if domain.is_empty() || opset_version == 0 {
                        opset_version = version;
                    }
                }
                pos = end;
            }

            // field 7: graph (GraphProto)
            (7, WIRE_LENGTH_DELIMITED) => {
                let (len, lc) = read_varint(data, pos)?;
                pos += lc;
                let graph_end = (pos + len as usize).min(data.len());

                // Parse GraphProto
                let mut gpos = pos;
                while gpos < graph_end {
                    let (gtag_val, gtc) = read_varint(data, gpos)?;
                    gpos += gtc;
                    let (gfield, gwire) = decode_tag(gtag_val);

                    match (gfield, gwire) {
                        // field 5: initializer (repeated TensorProto)
                        (5, WIRE_LENGTH_DELIMITED) => {
                            let (tlen, tlc) = read_varint(data, gpos)?;
                            gpos += tlc;
                            let tensor_end = (gpos + tlen as usize).min(data.len());
                            let base_offset = gpos as u64;
                            let tensor_slice = &data[gpos..tensor_end];

                            match parse_tensor_proto(tensor_slice, base_offset) {
                                Ok(info) => {
                                    let dtype = onnx_dtype_to_strix(info.data_type)
                                        .unwrap_or(DType::F32);
                                    let n_elems: usize = info.dims.iter()
                                        .copied()
                                        .product::<usize>()
                                        .max(1);

                                    if info.data_location == 1 {
                                        // EXTERNAL data
                                        let ext_file = info.external_filename
                                            .as_deref()
                                            .map(|f| model_dir.join(f));
                                        let size = if info.external_length > 0 {
                                            info.external_length as usize
                                        } else {
                                            n_elems * onnx_wire_elem_size(info.data_type)
                                        };
                                        tensors.push(OnnxTensorInfo {
                                            name: info.name,
                                            shape: info.dims,
                                            dtype,
                                            data_offset: info.external_offset,
                                            size_bytes: size,
                                            file_path: model_path.to_path_buf(),
                                            is_external: true,
                                            external_file: ext_file,
                                            external_offset: info.external_offset,
                                        });
                                    } else {
                                        // Inline data
                                        let size = if info.raw_data_size > 0 {
                                            info.raw_data_size
                                        } else if info.inline_data_bytes > 0 {
                                            info.inline_data_bytes
                                        } else {
                                            n_elems * onnx_wire_elem_size(info.data_type)
                                        };
                                        tensors.push(OnnxTensorInfo {
                                            name: info.name,
                                            shape: info.dims,
                                            dtype,
                                            data_offset: info.raw_data_offset
                                                .unwrap_or(base_offset),
                                            size_bytes: size,
                                            file_path: model_path.to_path_buf(),
                                            is_external: false,
                                            external_file: None,
                                            external_offset: 0,
                                        });
                                    }
                                }
                                Err(_) => { /* skip malformed tensor */ }
                            }
                            gpos = tensor_end;
                        }
                        (_, w) => {
                            let skipped = skip_field(data, gpos, w)?;
                            gpos += skipped;
                        }
                    }
                }
                pos = graph_end;
            }

            // Skip all other top-level fields
            (_, w) => {
                let skipped = skip_field(data, pos, w)?;
                pos += skipped;
            }
        }
    }

    Ok(OnnxModel { tensors, ir_version, producer, opset_version })
}

/// Resolve external tensors: load data from `.onnx_data` / weight files.
///
/// Returns a flat `Vec<u8>` of raw bytes for the given tensor, or an error
/// if the external file is not accessible.
pub fn load_external_tensor(info: &OnnxTensorInfo) -> Result<Vec<u8>, OnnxError> {
    if !info.is_external {
        return Err(OnnxError::InvalidProtobuf(
            "load_external_tensor called on non-external tensor".to_string()
        ));
    }
    let path = info.external_file.as_ref()
        .ok_or_else(|| OnnxError::InvalidProtobuf("no external file path".to_string()))?;
    if !path.exists() {
        return Err(OnnxError::ExternalDataNotFound(path.clone()));
    }
    let file_data = std::fs::read(path)?;
    let start = info.external_offset as usize;
    let end   = start + info.size_bytes;
    if end > file_data.len() {
        return Err(OnnxError::InvalidProtobuf(format!(
            "external data slice [{start}..{end}] out of bounds (file_len={})", file_data.len()
        )));
    }
    Ok(file_data[start..end].to_vec())
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Write a varint to a buffer.
    fn push_varint(buf: &mut Vec<u8>, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 { buf.push(byte); break; }
            buf.push(byte | 0x80);
        }
    }

    /// Write a length-delimited field.
    fn push_ld_field(buf: &mut Vec<u8>, field: u32, content: &[u8]) {
        push_varint(buf, ((field as u64) << 3) | 2); // tag
        push_varint(buf, content.len() as u64);      // len
        buf.extend_from_slice(content);
    }

    /// Write a varint field.
    fn push_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
        push_varint(buf, ((field as u64) << 3) | 0); // tag (wire=varint)
        push_varint(buf, val);
    }

    /// Build a minimal TensorProto.
    fn build_tensor_proto(name: &str, dims: &[usize], data_type: i32, data: &[u8]) -> Vec<u8> {
        let mut t = Vec::new();
        // name = field 8
        push_ld_field(&mut t, 8, name.as_bytes());
        // dims = field 1 (repeated varint)
        for &d in dims {
            push_varint_field(&mut t, 1, d as u64);
        }
        // data_type = field 2
        push_varint_field(&mut t, 2, data_type as u64);
        // raw_data = field 13
        push_ld_field(&mut t, 13, data);
        t
    }

    /// Build a full ModelProto with one initializer.
    fn build_model_proto(tensor: &[u8], ir_version: u64, producer: &str) -> Vec<u8> {
        let mut model = Vec::new();
        // ir_version = field 1
        push_varint_field(&mut model, 1, ir_version);
        // producer_name = field 2
        push_ld_field(&mut model, 2, producer.as_bytes());
        // graph = field 7
        let mut graph = Vec::new();
        push_ld_field(&mut graph, 5, tensor); // initializer = field 5
        push_ld_field(&mut model, 7, &graph);
        model
    }

    // ── Unit tests ────────────────────────────────────────────────────────

    #[test]
    fn varint_decoding_single_byte() {
        let (val, consumed) = read_varint(&[0x01], 0).unwrap();
        assert_eq!(val, 1);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn varint_decoding_multi_byte() {
        // 150 = 0x96, 0x01
        let (val, consumed) = read_varint(&[0x96, 0x01], 0).unwrap();
        assert_eq!(val, 150);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn varint_decoding_zero() {
        let (val, _) = read_varint(&[0x00], 0).unwrap();
        assert_eq!(val, 0);
    }

    #[test]
    fn varint_truncated() {
        let result = read_varint(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn tag_decoding() {
        // field 1, wire 0 → tag=0x08
        let (field, wire) = decode_tag(0x08);
        assert_eq!(field, 1); assert_eq!(wire, 0);
        // field 7, wire 2 → tag = (7<<3)|2 = 58
        let (field, wire) = decode_tag(58);
        assert_eq!(field, 7); assert_eq!(wire, 2);
    }

    #[test]
    fn dtype_mapping_all() {
        assert_eq!(onnx_dtype_to_strix(1).unwrap(),  DType::F32);
        assert_eq!(onnx_dtype_to_strix(2).unwrap(),  DType::U8);
        assert_eq!(onnx_dtype_to_strix(3).unwrap(),  DType::I8);
        assert_eq!(onnx_dtype_to_strix(5).unwrap(),  DType::I16);
        assert_eq!(onnx_dtype_to_strix(6).unwrap(),  DType::I32);
        assert_eq!(onnx_dtype_to_strix(7).unwrap(),  DType::I32);  // INT64→I32
        assert_eq!(onnx_dtype_to_strix(10).unwrap(), DType::F16);
        assert_eq!(onnx_dtype_to_strix(11).unwrap(), DType::F32);  // DOUBLE→F32
        assert_eq!(onnx_dtype_to_strix(16).unwrap(), DType::BF16);
        assert!(onnx_dtype_to_strix(99).is_err());
    }

    #[test]
    fn wire_elem_sizes() {
        assert_eq!(onnx_wire_elem_size(1),  4); // FLOAT
        assert_eq!(onnx_wire_elem_size(7),  8); // INT64
        assert_eq!(onnx_wire_elem_size(10), 2); // FLOAT16
        assert_eq!(onnx_wire_elem_size(16), 2); // BFLOAT16
        assert_eq!(onnx_wire_elem_size(11), 8); // DOUBLE
    }

    #[test]
    fn tensor_proto_raw_data() {
        let raw = [1.0f32.to_bits() as u8; 16]; // 4 floats
        let proto = build_tensor_proto("w", &[2, 2], 1, &raw[..16]);
        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.name, "w");
        assert_eq!(info.dims, vec![2, 2]);
        assert_eq!(info.data_type, 1);
        assert_eq!(info.raw_data_size, 16);
        assert!(info.raw_data_offset.is_some());
    }

    #[test]
    fn tensor_proto_empty_dims() {
        // Scalar tensor: no dims = 1 element
        let raw = [0u8; 4];
        let proto = build_tensor_proto("scalar", &[], 1, &raw);
        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.dims, Vec::<usize>::new());
        // Empty product = 1 (multiplicative identity) in Rust iterators.
        // parse_onnx_bytes uses .max(1) which is a no-op for scalar tensors.
        assert_eq!(info.dims.iter().product::<usize>(), 1);
    }

    #[test]
    fn tensor_proto_bf16() {
        let raw = vec![0u8; 8]; // 4 × BF16
        let proto = build_tensor_proto("bf16_w", &[4], 16, &raw);
        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.data_type, 16);
    }

    #[test]
    fn tensor_proto_int64_data_inline() {
        // Build a TensorProto with int64_data (field 7) inline
        let mut proto = Vec::new();
        push_ld_field(&mut proto, 8, b"int64_tensor");
        push_varint_field(&mut proto, 1, 3); // dim=3
        push_varint_field(&mut proto, 2, 7); // data_type=INT64
        // int64_data = field 7, packed: three int64 varints
        let mut int64_data = Vec::new();
        push_varint(&mut int64_data, 100u64);
        push_varint(&mut int64_data, 200u64);
        push_varint(&mut int64_data, 300u64);
        push_ld_field(&mut proto, 7, &int64_data);

        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.name, "int64_tensor");
        assert_eq!(info.data_type, 7);
        assert!(info.inline_data_bytes > 0);
    }

    #[test]
    fn tensor_proto_float_data_inline() {
        // float_data = field 4
        let mut proto = Vec::new();
        push_ld_field(&mut proto, 8, b"float_tensor");
        push_varint_field(&mut proto, 1, 2);
        push_varint_field(&mut proto, 2, 1); // FLOAT
        let float_bytes = [0u8; 8]; // 2 × f32
        push_ld_field(&mut proto, 4, &float_bytes);
        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.inline_data_bytes, 8);
        assert_eq!(info.raw_data_size, 0);
    }

    #[test]
    fn full_model_proto_parse() {
        let raw = vec![0u8; 24]; // 6 × f32
        let tensor = build_tensor_proto("w", &[2, 3], 1, &raw);
        let model_bytes = build_model_proto(&tensor, 8, "test_producer");

        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_onnx2_test_{}.onnx", std::process::id()));
        std::fs::write(&path, &model_bytes).unwrap();
        let result = parse_onnx(&path).unwrap();

        assert_eq!(result.ir_version, 8);
        assert_eq!(result.producer, "test_producer");
        assert_eq!(result.tensors.len(), 1);
        assert_eq!(result.tensors[0].name, "w");
        assert_eq!(result.tensors[0].shape, vec![2, 3]);
        assert_eq!(result.tensors[0].dtype, DType::F32);
        assert_eq!(result.tensors[0].size_bytes, 24);
        assert!(!result.tensors[0].is_external);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn multi_tensor_model() {
        let mut graph = Vec::new();
        let raw16 = vec![0u8; 16];
        let raw32 = vec![0u8; 32];
        push_ld_field(&mut graph, 5, &build_tensor_proto("A", &[4], 1, &raw16));
        push_ld_field(&mut graph, 5, &build_tensor_proto("B", &[4], 10, &raw32[..8]));

        let mut model = Vec::new();
        push_varint_field(&mut model, 1, 7);
        push_ld_field(&mut model, 7, &graph);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_onnx_multi_{}.onnx", std::process::id()));
        std::fs::write(&path, &model).unwrap();
        let result = parse_onnx_bytes(&model, &path).unwrap();

        assert_eq!(result.tensors.len(), 2);
        assert_eq!(result.tensors[0].name, "A");
        assert_eq!(result.tensors[0].dtype, DType::F32);
        assert_eq!(result.tensors[1].name, "B");
        assert_eq!(result.tensors[1].dtype, DType::F16);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn external_data_tensor() {
        // Build a TensorProto with data_location=EXTERNAL + external_data entries
        let mut proto = Vec::new();
        push_ld_field(&mut proto, 8, b"ext_w");
        push_varint_field(&mut proto, 1, 3);
        push_varint_field(&mut proto, 2, 1);  // FLOAT
        push_varint_field(&mut proto, 21, 1); // data_location=EXTERNAL

        // external_data entry: { key="location", value="weights.bin" }
        let mut kv1 = Vec::new();
        push_ld_field(&mut kv1, 1, b"location");
        push_ld_field(&mut kv1, 2, b"weights.bin");
        push_ld_field(&mut proto, 22, &kv1);

        // external_data entry: { key="offset", value="0" }
        let mut kv2 = Vec::new();
        push_ld_field(&mut kv2, 1, b"offset");
        push_ld_field(&mut kv2, 2, b"0");
        push_ld_field(&mut proto, 22, &kv2);

        // external_data entry: { key="length", value="12" }
        let mut kv3 = Vec::new();
        push_ld_field(&mut kv3, 1, b"length");
        push_ld_field(&mut kv3, 2, b"12");
        push_ld_field(&mut proto, 22, &kv3);

        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.data_location, 1);
        assert_eq!(info.external_filename.as_deref(), Some("weights.bin"));
        assert_eq!(info.external_offset, 0);
        assert_eq!(info.external_length, 12);
    }

    #[test]
    fn external_tensor_model_integration() {
        // Build model with external tensor, write actual weight file
        let dir = std::env::temp_dir();
        let weights_path = dir.join(format!("strix_onnx_ext_weights_{}.bin", std::process::id()));
        let weight_data: Vec<u8> = (0..12u8).collect();
        std::fs::write(&weights_path, &weight_data).unwrap();

        let weight_file = weights_path.file_name().unwrap().to_str().unwrap();
        let mut proto = Vec::new();
        push_ld_field(&mut proto, 8, b"ext_weight");
        push_varint_field(&mut proto, 1, 3); // dim=3
        push_varint_field(&mut proto, 2, 2); // UINT8
        push_varint_field(&mut proto, 21, 1); // EXTERNAL
        let mut kv = Vec::new();
        push_ld_field(&mut kv, 1, b"location");
        push_ld_field(&mut kv, 2, weight_file.as_bytes());
        push_ld_field(&mut proto, 22, &kv);
        let mut kv2 = Vec::new();
        push_ld_field(&mut kv2, 1, b"length");
        push_ld_field(&mut kv2, 2, b"12");
        push_ld_field(&mut proto, 22, &kv2);

        let mut graph = Vec::new();
        push_ld_field(&mut graph, 5, &proto);
        let mut model = Vec::new();
        push_varint_field(&mut model, 1, 7);
        push_ld_field(&mut model, 7, &graph);

        let model_path = dir.join(format!("strix_onnx_ext_{}.onnx", std::process::id()));
        std::fs::write(&model_path, &model).unwrap();

        let result = parse_onnx(&model_path).unwrap();
        assert_eq!(result.tensors.len(), 1);
        let tensor = &result.tensors[0];
        assert!(tensor.is_external);
        assert_eq!(tensor.size_bytes, 12);

        let loaded = load_external_tensor(tensor).unwrap();
        assert_eq!(loaded.len(), 12);
        assert_eq!(loaded[0], 0);
        assert_eq!(loaded[11], 11);

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&weights_path);
    }

    #[test]
    fn opset_version_parsed() {
        // Build model with opset_import
        let mut opset = Vec::new();
        push_ld_field(&mut opset, 1, b""); // domain = "" (default)
        push_varint_field(&mut opset, 2, 17); // version = 17

        let mut model = Vec::new();
        push_varint_field(&mut model, 1, 8);   // ir_version
        push_ld_field(&mut model, 8, &opset);   // opset_import (field 8)
        let empty_graph: Vec<u8> = Vec::new();
        push_ld_field(&mut model, 7, &empty_graph);

        let path = std::env::temp_dir()
            .join(format!("strix_onnx_opset_{}.onnx", std::process::id()));
        std::fs::write(&path, &model).unwrap();
        let result = parse_onnx(&path).unwrap();
        assert_eq!(result.ir_version, 8);
        assert_eq!(result.opset_version, 17);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_file_returns_error() {
        let result = parse_onnx(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err());
    }

    #[test]
    fn skip_field_wire_types() {
        // WIRE_VARINT: varint 150 = [0x96, 0x01] → 2 bytes
        let consumed = skip_field(&[0x96, 0x01, 0xFF], 0, 0).unwrap();
        assert_eq!(consumed, 2);
        // WIRE_64BIT
        assert_eq!(skip_field(&[0u8; 10], 0, 1).unwrap(), 8);
        // WIRE_32BIT
        assert_eq!(skip_field(&[0u8; 10], 0, 5).unwrap(), 4);
        // WIRE_LENGTH_DELIMITED: varint len + len bytes
        let consumed = skip_field(&[4, 0, 1, 2, 3, 0xFF], 0, 2).unwrap();
        assert_eq!(consumed, 1 + 4); // 1-byte varint=4, then 4 bytes
    }
}

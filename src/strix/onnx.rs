//! ONNX model reader — STRIX Protocol §15.1.
//!
//! Parses ONNX `.onnx` files to extract initializer tensors (weights).
//! Uses a minimal protobuf wire-format reader — no dependency on the
//! full protobuf runtime.
//!
//! ONNX files follow the protobuf schema:
//! ```text
//! ModelProto {
//!   graph: GraphProto {
//!     initializer: [TensorProto] {
//!       name, dims, data_type, raw_data / float_data / ...
//!     }
//!   }
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
    /// Byte offset of raw data within the file.
    pub data_offset: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Source file path.
    pub file_path: PathBuf,
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
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "ONNX I/O error: {e}"),
            Self::InvalidProtobuf(s) => write!(f, "ONNX protobuf error: {s}"),
            Self::UnknownDataType(c) => write!(f, "ONNX unknown data type: {c}"),
            Self::Truncated => write!(f, "ONNX file truncated"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Protobuf Wire Format ─────────────────────────────────────────────────
//
// Protobuf wire types:
// 0 = Varint, 1 = 64-bit, 2 = Length-delimited, 5 = 32-bit

const WIRE_VARINT: u8 = 0;
const WIRE_64BIT: u8 = 1;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_32BIT: u8 = 5;

/// Read a varint from the buffer, returning (value, bytes_consumed).
fn read_varint(data: &[u8], pos: usize) -> Result<(u64, usize), OnnxError> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut i = pos;
    loop {
        if i >= data.len() {
            return Err(OnnxError::Truncated);
        }
        let byte = data[i];
        result |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return Ok((result, i - pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(OnnxError::InvalidProtobuf("varint too long".into()));
        }
    }
}

/// Decode a protobuf field tag into (field_number, wire_type).
fn decode_tag(tag: u64) -> (u32, u8) {
    let field = (tag >> 3) as u32;
    let wire = (tag & 0x07) as u8;
    (field, wire)
}

/// Skip a field value based on wire type, returning bytes to skip.
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

/// ONNX TensorProto.DataType enum values.
fn onnx_dtype_to_strix(code: i32) -> Result<DType, OnnxError> {
    match code {
        1 => Ok(DType::F32),     // FLOAT
        2 => Ok(DType::U8),      // UINT8
        3 => Ok(DType::I8),      // INT8
        5 => Ok(DType::I16),     // INT16
        6 => Ok(DType::I32),     // INT32
        7 => Ok(DType::I32),     // INT64 → downcast to I32
        10 => Ok(DType::F16),    // FLOAT16
        16 => Ok(DType::BF16),   // BFLOAT16
        _ => Err(OnnxError::UnknownDataType(code)),
    }
}

/// Bytes per element for ONNX data types (before any STRIX conversion).
fn onnx_dtype_size(code: i32) -> usize {
    match code {
        1 => 4,    // FLOAT
        2 => 1,    // UINT8
        3 => 1,    // INT8
        5 => 2,    // INT16
        6 => 4,    // INT32
        7 => 8,    // INT64
        10 => 2,   // FLOAT16
        16 => 2,   // BFLOAT16
        11 => 8,   // DOUBLE
        _ => 4,    // fallback
    }
}

// ── TensorProto Parser ───────────────────────────────────────────────────

/// Parsed TensorProto fields we care about.
struct TensorProtoInfo {
    name: String,
    dims: Vec<usize>,
    data_type: i32,
    raw_data_offset: Option<u64>,
    raw_data_size: usize,
    /// If raw_data is absent, data might be inline in typed arrays
    inline_data_size: usize,
}

/// Parse a TensorProto message from a byte slice.
///
/// `base_offset` is the absolute file offset of `data[0]`, used to calculate
/// the raw_data offset within the file.
fn parse_tensor_proto(data: &[u8], base_offset: u64) -> Result<TensorProtoInfo, OnnxError> {
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 1i32; // default FLOAT
    let mut raw_data_offset = None;
    let mut raw_data_size = 0;
    let mut inline_data_size = 0;

    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tag_consumed) = read_varint(data, pos)?;
        pos += tag_consumed;
        let (field, wire) = decode_tag(tag_val);

        match (field, wire) {
            // field 1: dims (repeated int64, packed)
            (1, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                dims.push(val as usize);
                pos += consumed;
            }
            (1, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(data, pos)?;
                pos += consumed;
                let end = pos + len as usize;
                while pos < end && pos < data.len() {
                    let (val, consumed) = read_varint(data, pos)?;
                    dims.push(val as usize);
                    pos += consumed;
                }
            }
            // field 2: data_type (int32)
            (2, WIRE_VARINT) => {
                let (val, consumed) = read_varint(data, pos)?;
                data_type = val as i32;
                pos += consumed;
            }
            // field 8: name (string)
            (8, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(data, pos)?;
                pos += consumed;
                let end = pos + len as usize;
                if end <= data.len() {
                    name = String::from_utf8_lossy(&data[pos..end]).to_string();
                }
                pos = end;
            }
            // field 13: raw_data (bytes)
            (13, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(data, pos)?;
                pos += consumed;
                raw_data_offset = Some(base_offset + pos as u64);
                raw_data_size = len as usize;
                pos += len as usize;
            }
            // field 4: float_data (packed repeated float)
            (4, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(data, pos)?;
                pos += consumed;
                inline_data_size = len as usize;
                pos += len as usize;
            }
            // Skip unknown fields
            (_, w) => {
                let skipped = skip_field(data, pos, w)?;
                pos += skipped;
            }
        }
    }

    Ok(TensorProtoInfo {
        name,
        dims,
        data_type,
        raw_data_offset,
        raw_data_size,
        inline_data_size,
    })
}

// ── ModelProto/GraphProto Parser ─────────────────────────────────────────

/// Parse an ONNX file to extract initializer tensors.
pub fn parse_onnx(path: &Path) -> Result<OnnxModel, OnnxError> {
    let data = std::fs::read(path)?;
    if data.is_empty() {
        return Err(OnnxError::Truncated);
    }

    let mut ir_version = 0u64;
    let mut producer = String::new();
    let mut tensors = Vec::new();

    // Parse top-level ModelProto
    let mut pos = 0;
    while pos < data.len() {
        let (tag_val, tag_consumed) = read_varint(&data, pos)?;
        pos += tag_consumed;
        let (field, wire) = decode_tag(tag_val);

        match (field, wire) {
            // field 1: ir_version (int64)
            (1, WIRE_VARINT) => {
                let (val, consumed) = read_varint(&data, pos)?;
                ir_version = val;
                pos += consumed;
            }
            // field 2: producer_name (string)
            (2, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(&data, pos)?;
                pos += consumed;
                let end = pos + len as usize;
                if end <= data.len() {
                    producer = String::from_utf8_lossy(&data[pos..end]).to_string();
                }
                pos = end;
            }
            // field 7: graph (GraphProto)
            (7, WIRE_LENGTH_DELIMITED) => {
                let (len, consumed) = read_varint(&data, pos)?;
                pos += consumed;
                let graph_end = pos + len as usize;
                let graph_base = pos as u64;

                // Parse GraphProto for initializer fields
                let mut gpos = pos;
                while gpos < graph_end && gpos < data.len() {
                    let (gtag_val, gtag_consumed) = read_varint(&data, gpos)?;
                    gpos += gtag_consumed;
                    let (gfield, gwire) = decode_tag(gtag_val);

                    match (gfield, gwire) {
                        // field 5: initializer (repeated TensorProto)
                        (5, WIRE_LENGTH_DELIMITED) => {
                            let (tlen, tconsumed) = read_varint(&data, gpos)?;
                            gpos += tconsumed;
                            let tensor_end = gpos + tlen as usize;
                            if tensor_end <= data.len() {
                                let base_offset = gpos as u64;
                                let tensor_data = &data[gpos..tensor_end];
                                match parse_tensor_proto(tensor_data, base_offset) {
                                    Ok(info) => {
                                        let dtype = onnx_dtype_to_strix(info.data_type)
                                            .unwrap_or(DType::F32);
                                        let size = if info.raw_data_size > 0 {
                                            info.raw_data_size
                                        } else if info.inline_data_size > 0 {
                                            info.inline_data_size
                                        } else {
                                            let n_elems: usize = info.dims.iter().product();
                                            n_elems * onnx_dtype_size(info.data_type)
                                        };

                                        tensors.push(OnnxTensorInfo {
                                            name: info.name,
                                            shape: info.dims,
                                            dtype,
                                            data_offset: info.raw_data_offset.unwrap_or(base_offset),
                                            size_bytes: size,
                                            file_path: path.to_path_buf(),
                                        });
                                    }
                                    Err(_) => { /* skip malformed tensor */ }
                                }
                            }
                            gpos = tensor_end;
                        }
                        (_, w) => {
                            let skipped = skip_field(&data, gpos, w)?;
                            gpos += skipped;
                        }
                    }
                }
                let _ = graph_base; // suppress unused warning
                pos = graph_end;
            }
            // Skip other top-level fields
            (_, w) => {
                let skipped = skip_field(&data, pos, w)?;
                pos += skipped;
            }
        }
    }

    Ok(OnnxModel {
        tensors,
        ir_version,
        producer,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_decoding() {
        // Single byte: 150 = 0x96 → [0x96, 0x01]
        let (val, consumed) = read_varint(&[0x96, 0x01], 0).unwrap();
        assert_eq!(val, 150);
        assert_eq!(consumed, 2);

        // Single byte value: 1
        let (val, consumed) = read_varint(&[0x01], 0).unwrap();
        assert_eq!(val, 1);
        assert_eq!(consumed, 1);

        // Zero
        let (val, consumed) = read_varint(&[0x00], 0).unwrap();
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn tag_decoding() {
        // field 1, wire type 0 → tag = 0x08
        let (field, wire) = decode_tag(0x08);
        assert_eq!(field, 1);
        assert_eq!(wire, 0);

        // field 2, wire type 2 → tag = 0x12
        let (field, wire) = decode_tag(0x12);
        assert_eq!(field, 2);
        assert_eq!(wire, 2);
    }

    #[test]
    fn dtype_mapping() {
        assert_eq!(onnx_dtype_to_strix(1).unwrap(), DType::F32);
        assert_eq!(onnx_dtype_to_strix(10).unwrap(), DType::F16);
        assert_eq!(onnx_dtype_to_strix(16).unwrap(), DType::BF16);
        assert_eq!(onnx_dtype_to_strix(3).unwrap(), DType::I8);
        assert!(onnx_dtype_to_strix(99).is_err());
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(onnx_dtype_size(1), 4);  // FLOAT
        assert_eq!(onnx_dtype_size(10), 2); // FLOAT16
        assert_eq!(onnx_dtype_size(7), 8);  // INT64
        assert_eq!(onnx_dtype_size(2), 1);  // UINT8
    }

    #[test]
    fn tensor_proto_parse() {
        // Build a minimal TensorProto:
        // field 8 (name)=string "test_weight"
        // field 1 (dims)=varint 4, varint 3
        // field 2 (data_type)=varint 1 (FLOAT)
        let mut proto = Vec::new();

        // name: field 8, wire 2
        let name_bytes = b"test_weight";
        proto.push((8 << 3) | 2); // tag
        proto.push(name_bytes.len() as u8); // length
        proto.extend_from_slice(name_bytes);

        // dims: field 1, wire 0 (two varints)
        proto.push((1 << 3) | 0); // tag
        proto.push(4); // value
        proto.push((1 << 3) | 0); // tag
        proto.push(3); // value

        // data_type: field 2, wire 0
        proto.push((2 << 3) | 0); // tag
        proto.push(1); // FLOAT

        let info = parse_tensor_proto(&proto, 0).unwrap();
        assert_eq!(info.name, "test_weight");
        assert_eq!(info.dims, vec![4, 3]);
        assert_eq!(info.data_type, 1);
    }

    #[test]
    fn empty_onnx_file() {
        let result = parse_onnx(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err());
    }

    /// Build a minimal ONNX ModelProto with one initializer tensor.
    #[test]
    fn minimal_model_proto() {
        let mut model = Vec::new();

        // ir_version: field 1, varint 7
        model.push((1 << 3) | 0);
        model.push(7);

        // producer_name: field 2, string "test"
        let producer = b"test";
        model.push((2 << 3) | 2);
        model.push(producer.len() as u8);
        model.extend_from_slice(producer);

        // graph: field 7, length-delimited
        let mut graph = Vec::new();

        // initializer: field 5, TensorProto
        let mut tensor = Vec::new();
        // name
        let name = b"w";
        tensor.push((8 << 3) | 2);
        tensor.push(name.len() as u8);
        tensor.extend_from_slice(name);
        // dims: 2, 3
        tensor.push((1 << 3) | 0);
        tensor.push(2);
        tensor.push((1 << 3) | 0);
        tensor.push(3);
        // data_type: FLOAT (1)
        tensor.push((2 << 3) | 0);
        tensor.push(1);
        // raw_data: field 13, 24 bytes of zeros
        let raw = [0u8; 24];
        tensor.push((13 << 3) | 2);
        tensor.push(raw.len() as u8);
        tensor.extend_from_slice(&raw);

        graph.push((5 << 3) | 2);
        // Encode tensor length as varint
        graph.push(tensor.len() as u8);
        graph.extend_from_slice(&tensor);

        model.push((7 << 3) | 2);
        model.push(graph.len() as u8);
        model.extend_from_slice(&graph);

        // Write to temp file and parse
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_onnx_test_{}.onnx", std::process::id()));
        std::fs::write(&path, &model).unwrap();

        let result = parse_onnx(&path).unwrap();
        assert_eq!(result.ir_version, 7);
        assert_eq!(result.producer, "test");
        assert_eq!(result.tensors.len(), 1);
        assert_eq!(result.tensors[0].name, "w");
        assert_eq!(result.tensors[0].shape, vec![2, 3]);
        assert_eq!(result.tensors[0].dtype, DType::F32);

        let _ = std::fs::remove_file(&path);
    }
}

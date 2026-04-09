//! SafeTensors format reader — STRIX Protocol §15.1.
//!
//! SafeTensors is a simple, safe format for storing tensors:
//! - 8-byte LE header size
//! - JSON header with tensor metadata (name → {dtype, shape, data_offsets})
//! - Raw tensor data (contiguous)
//!
//! Supports both single-file and sharded formats (with index.json).

use super::types::DType;
use std::path::{Path, PathBuf};

// ── Public Types ─────────────────────────────────────────────────────────

/// Per-tensor info extracted from a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Byte offset from start of the data section.
    pub data_offset: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Which file shard this tensor lives in (for sharded models).
    pub shard_path: PathBuf,
}

/// Parsed SafeTensors model (single or sharded).
#[derive(Debug, Clone)]
pub struct SafeTensorModel {
    /// All tensor infos across all shards.
    pub tensors: Vec<SafeTensorInfo>,
    /// Total number of shards.
    pub n_shards: usize,
}

/// Errors that can occur during SafeTensors parsing.
#[derive(Debug)]
pub enum SafeTensorError {
    /// I/O error reading file.
    Io(std::io::Error),
    /// File too small for header.
    FileTooSmall,
    /// Header size exceeds file size or sanity limit (100MB).
    HeaderTooLarge(u64),
    /// JSON header parse error.
    InvalidJson(String),
    /// Unknown dtype string in header.
    UnknownDtype(String),
    /// Shard index file parse error.
    InvalidIndex(String),
    /// Referenced shard file not found.
    ShardNotFound(PathBuf),
}

impl std::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "SafeTensors I/O error: {e}"),
            Self::FileTooSmall => write!(f, "SafeTensors file too small for header"),
            Self::HeaderTooLarge(n) => write!(f, "SafeTensors header too large: {n} bytes"),
            Self::InvalidJson(e) => write!(f, "SafeTensors invalid JSON header: {e}"),
            Self::UnknownDtype(s) => write!(f, "SafeTensors unknown dtype: {s}"),
            Self::InvalidIndex(e) => write!(f, "SafeTensors shard index error: {e}"),
            Self::ShardNotFound(p) => write!(f, "SafeTensors shard not found: {}", p.display()),
        }
    }
}

impl std::error::Error for SafeTensorError {}

impl From<std::io::Error> for SafeTensorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── JSON micro-parser ────────────────────────────────────────────────────
//
// We implement a minimal JSON parser to avoid adding a serde_json dependency.
// SafeTensors headers are simple flat objects: `{ "tensor_name": { "dtype": "F32", "shape": [N, M], "data_offsets": [start, end] }, ... }`

/// A minimal JSON value for SafeTensors header parsing.
#[derive(Debug, Clone)]
enum JsonValue {
    String(String),
    Number(f64),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
    Null,
}

/// Parse a JSON string (minimal, sufficient for SafeTensors headers).
fn parse_json(input: &str) -> Result<JsonValue, SafeTensorError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SafeTensorError::InvalidJson("empty input".into()));
    }
    let (val, _) = parse_json_value(trimmed, 0)?;
    Ok(val)
}

fn parse_json_value(s: &str, pos: usize) -> Result<(JsonValue, usize), SafeTensorError> {
    let pos = skip_whitespace(s, pos);
    if pos >= s.len() {
        return Err(SafeTensorError::InvalidJson("unexpected end".into()));
    }
    match s.as_bytes()[pos] {
        b'"' => {
            let (st, next) = parse_json_string(s, pos)?;
            Ok((JsonValue::String(st), next))
        }
        b'{' => parse_json_object(s, pos),
        b'[' => parse_json_array(s, pos),
        b'n' if s[pos..].starts_with("null") => Ok((JsonValue::Null, pos + 4)),
        b'-' | b'0'..=b'9' => parse_json_number(s, pos),
        c => Err(SafeTensorError::InvalidJson(format!("unexpected char '{}' at {pos}", c as char))),
    }
}

fn skip_whitespace(s: &str, mut pos: usize) -> usize {
    let bytes = s.as_bytes();
    while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\t' | b'\n' | b'\r') {
        pos += 1;
    }
    pos
}

fn parse_json_string(s: &str, pos: usize) -> Result<(String, usize), SafeTensorError> {
    if s.as_bytes()[pos] != b'"' {
        return Err(SafeTensorError::InvalidJson("expected '\"'".into()));
    }
    let mut result = String::new();
    let mut i = pos + 1;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'"' => { result.push('"'); i += 2; }
                b'\\' => { result.push('\\'); i += 2; }
                b'n' => { result.push('\n'); i += 2; }
                b't' => { result.push('\t'); i += 2; }
                b'/' => { result.push('/'); i += 2; }
                _ => { result.push(bytes[i + 1] as char); i += 2; }
            }
        } else if bytes[i] == b'"' {
            return Ok((result, i + 1));
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    Err(SafeTensorError::InvalidJson("unterminated string".into()))
}

fn parse_json_number(s: &str, pos: usize) -> Result<(JsonValue, usize), SafeTensorError> {
    let mut end = pos;
    let bytes = s.as_bytes();
    while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.' || bytes[end] == b'-' || bytes[end] == b'e' || bytes[end] == b'E' || bytes[end] == b'+') {
        end += 1;
    }
    let num_str = &s[pos..end];
    let val: f64 = num_str.parse().map_err(|_| SafeTensorError::InvalidJson(format!("bad number: {num_str}")))?;
    Ok((JsonValue::Number(val), end))
}

fn parse_json_array(s: &str, pos: usize) -> Result<(JsonValue, usize), SafeTensorError> {
    let mut items = Vec::new();
    let mut i = pos + 1; // skip '['
    loop {
        i = skip_whitespace(s, i);
        if i >= s.len() {
            return Err(SafeTensorError::InvalidJson("unterminated array".into()));
        }
        if s.as_bytes()[i] == b']' {
            return Ok((JsonValue::Array(items), i + 1));
        }
        if !items.is_empty() {
            if s.as_bytes()[i] != b',' {
                return Err(SafeTensorError::InvalidJson("expected ','".into()));
            }
            i += 1;
        }
        let (val, next) = parse_json_value(s, i)?;
        items.push(val);
        i = next;
    }
}

fn parse_json_object(s: &str, pos: usize) -> Result<(JsonValue, usize), SafeTensorError> {
    let mut entries = Vec::new();
    let mut i = pos + 1; // skip '{'
    loop {
        i = skip_whitespace(s, i);
        if i >= s.len() {
            return Err(SafeTensorError::InvalidJson("unterminated object".into()));
        }
        if s.as_bytes()[i] == b'}' {
            return Ok((JsonValue::Object(entries), i + 1));
        }
        if !entries.is_empty() {
            if s.as_bytes()[i] != b',' {
                return Err(SafeTensorError::InvalidJson("expected ','".into()));
            }
            i += 1;
            i = skip_whitespace(s, i);
        }
        let (key, next) = parse_json_string(s, i)?;
        i = skip_whitespace(s, next);
        if i >= s.len() || s.as_bytes()[i] != b':' {
            return Err(SafeTensorError::InvalidJson("expected ':'".into()));
        }
        i += 1;
        let (val, next) = parse_json_value(s, i)?;
        entries.push((key, val));
        i = next;
    }
}

// ── Dtype Mapping ────────────────────────────────────────────────────────

fn safetensor_dtype_to_strix(s: &str) -> Result<DType, SafeTensorError> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I32" => Ok(DType::I32),
        "I16" => Ok(DType::I16),
        "I8" => Ok(DType::I8),
        "U8" => Ok(DType::U8),
        "BOOL" => Ok(DType::U8), // SafeTensors BOOL stored as U8
        "F64" => Ok(DType::F32), // downcast -- STRIX doesn't use F64
        "I64" => Ok(DType::I32), // downcast
        other => Err(SafeTensorError::UnknownDtype(other.to_string())),
    }
}

// ── Public API ───────────────────────────────────────────────────────────

/// Parse a single SafeTensors file, returning tensor metadata.
///
/// Does NOT read tensor data — only the JSON header.
pub fn parse_safetensors(path: &Path) -> Result<SafeTensorModel, SafeTensorError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();

    // Read 8-byte header size
    if file_size < 8 {
        return Err(SafeTensorError::FileTooSmall);
    }
    let mut size_buf = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf);

    // Sanity check: header shouldn't exceed 100MB or file size
    if header_size > 100_000_000 || header_size + 8 > file_size {
        return Err(SafeTensorError::HeaderTooLarge(header_size));
    }

    // Read JSON header
    let mut header_buf = vec![0u8; header_size as usize];
    file.read_exact(&mut header_buf)?;
    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| SafeTensorError::InvalidJson(format!("non-UTF8 header: {e}")))?;

    let json = parse_json(header_str)?;

    // Data section starts at offset 8 + header_size
    let data_offset_base = 8 + header_size;

    let entries = match json {
        JsonValue::Object(entries) => entries,
        _ => return Err(SafeTensorError::InvalidJson("root must be object".into())),
    };

    let mut tensors = Vec::new();
    for (name, value) in entries {
        // Skip "__metadata__" key
        if name == "__metadata__" {
            continue;
        }

        let fields = match value {
            JsonValue::Object(f) => f,
            _ => continue,
        };

        let mut dtype_str = None;
        let mut shape = Vec::new();
        let mut data_offsets = (0u64, 0u64);

        for (key, val) in fields {
            match key.as_str() {
                "dtype" => {
                    if let JsonValue::String(s) = val {
                        dtype_str = Some(s);
                    }
                }
                "shape" => {
                    if let JsonValue::Array(arr) = val {
                        for item in arr {
                            if let JsonValue::Number(n) = item {
                                shape.push(n as usize);
                            }
                        }
                    }
                }
                "data_offsets" => {
                    if let JsonValue::Array(arr) = val {
                        if arr.len() >= 2 {
                            if let (JsonValue::Number(a), JsonValue::Number(b)) = (&arr[0], &arr[1]) {
                                data_offsets = (*a as u64, *b as u64);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let dtype = match dtype_str {
            Some(s) => safetensor_dtype_to_strix(&s)?,
            None => continue, // skip entries without dtype
        };

        let size = (data_offsets.1 - data_offsets.0) as usize;

        tensors.push(SafeTensorInfo {
            name,
            shape,
            dtype,
            data_offset: data_offset_base + data_offsets.0,
            size_bytes: size,
            shard_path: path.to_path_buf(),
        });
    }

    // Sort by offset for sequential reads
    tensors.sort_by_key(|t| t.data_offset);

    Ok(SafeTensorModel {
        tensors,
        n_shards: 1,
    })
}

/// Parse a sharded SafeTensors model from its index file.
///
/// Expects a `model.safetensors.index.json` path. Reads the index to find
/// all shard files, then parses each shard's header.
pub fn parse_safetensors_sharded(index_path: &Path) -> Result<SafeTensorModel, SafeTensorError> {
    let index_bytes = std::fs::read(index_path)?;
    let index_str = std::str::from_utf8(&index_bytes)
        .map_err(|e| SafeTensorError::InvalidIndex(format!("non-UTF8 index: {e}")))?;

    let json = parse_json(index_str)?;
    let root = match json {
        JsonValue::Object(entries) => entries,
        _ => return Err(SafeTensorError::InvalidIndex("root must be object".into())),
    };

    // Extract "weight_map": { "tensor_name": "shard_file.safetensors", ... }
    let weight_map = root.into_iter()
        .find(|(k, _)| k == "weight_map")
        .map(|(_, v)| v);

    let map_entries = match weight_map {
        Some(JsonValue::Object(entries)) => entries,
        _ => return Err(SafeTensorError::InvalidIndex("missing weight_map".into())),
    };

    // Collect unique shard files
    let parent = index_path.parent().unwrap_or(Path::new("."));
    let mut shard_files: Vec<PathBuf> = Vec::new();
    for (_, val) in &map_entries {
        if let JsonValue::String(shard_name) = val {
            let shard_path = parent.join(shard_name);
            if !shard_files.contains(&shard_path) {
                shard_files.push(shard_path);
            }
        }
    }

    // Parse each shard
    let mut all_tensors = Vec::new();
    for shard_path in &shard_files {
        if !shard_path.exists() {
            return Err(SafeTensorError::ShardNotFound(shard_path.clone()));
        }
        let model = parse_safetensors(shard_path)?;
        all_tensors.extend(model.tensors);
    }

    Ok(SafeTensorModel {
        n_shards: shard_files.len(),
        tensors: all_tensors,
    })
}

/// Auto-detect whether a path is a single SafeTensors file or a sharded index,
/// and parse accordingly.
pub fn parse_safetensors_auto(path: &Path) -> Result<SafeTensorModel, SafeTensorError> {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    if name.ends_with(".index.json") {
        parse_safetensors_sharded(path)
    } else {
        parse_safetensors(path)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid SafeTensors file in memory.
    fn make_safetensors_file(tensors: &[(&str, &str, &[usize])]) -> Vec<u8> {
        // Build the data section and track offsets
        let mut data = Vec::new();
        let mut entries = Vec::new();

        for (name, dtype, shape) in tensors {
            let elem_size = match *dtype {
                "F32" => 4usize,
                "F16" | "BF16" => 2,
                "I8" | "U8" => 1,
                _ => 4,
            };
            let n_elements: usize = shape.iter().product();
            let size = n_elements * elem_size;
            let start = data.len();
            data.resize(data.len() + size, 0xAB);
            let end = data.len();
            entries.push((
                name.to_string(),
                dtype.to_string(),
                shape.to_vec(),
                start,
                end,
            ));
        }

        // Build JSON header
        let mut json_parts = Vec::new();
        for (name, dtype, shape, start, end) in &entries {
            let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            json_parts.push(format!(
                "\"{name}\": {{\"dtype\": \"{dtype}\", \"shape\": [{}], \"data_offsets\": [{start}, {end}]}}",
                shape_str.join(", ")
            ));
        }
        let json = format!("{{{}}}", json_parts.join(", "));
        let json_bytes = json.as_bytes();

        // Assemble: 8-byte LE header size + JSON + data
        let mut out = Vec::new();
        out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(json_bytes);
        out.extend_from_slice(&data);
        out
    }

    fn write_temp(name: &str, data: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_st_test_{}_{name}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        path
    }

    #[test]
    fn parse_single_file() {
        let data = make_safetensors_file(&[
            ("weight_a", "F32", &[4, 4]),
            ("weight_b", "F16", &[8, 2]),
        ]);
        let path = write_temp("single.safetensors", &data);
        let model = parse_safetensors(&path).unwrap();
        assert_eq!(model.tensors.len(), 2);
        assert_eq!(model.tensors[0].name, "weight_a");
        assert_eq!(model.tensors[0].dtype, DType::F32);
        assert_eq!(model.tensors[0].shape, vec![4, 4]);
        assert_eq!(model.tensors[0].size_bytes, 64); // 16 * 4
        assert_eq!(model.tensors[1].name, "weight_b");
        assert_eq!(model.tensors[1].dtype, DType::F16);
        assert_eq!(model.tensors[1].size_bytes, 32); // 16 * 2
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_empty_tensors() {
        let data = make_safetensors_file(&[]);
        let path = write_temp("empty.safetensors", &data);
        let model = parse_safetensors(&path).unwrap();
        assert_eq!(model.tensors.len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_file_too_small() {
        let path = write_temp("tiny.safetensors", &[0, 1, 2]);
        let result = parse_safetensors(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_header_too_large() {
        let mut data = vec![0u8; 16];
        // Set header size to 1 billion
        data[..8].copy_from_slice(&(1_000_000_001u64).to_le_bytes());
        let path = write_temp("huge_header.safetensors", &data);
        let result = parse_safetensors(&path);
        assert!(matches!(result, Err(SafeTensorError::HeaderTooLarge(_))));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn metadata_key_skipped() {
        // Build a file with __metadata__ key — should be ignored
        let json = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        let json_bytes = json.as_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(json_bytes);
        out.extend_from_slice(&[0u8; 8]); // data
        let path = write_temp("meta.safetensors", &out);
        let model = parse_safetensors(&path).unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "w");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dtype_mapping() {
        assert_eq!(safetensor_dtype_to_strix("F32").unwrap(), DType::F32);
        assert_eq!(safetensor_dtype_to_strix("F16").unwrap(), DType::F16);
        assert_eq!(safetensor_dtype_to_strix("BF16").unwrap(), DType::BF16);
        assert_eq!(safetensor_dtype_to_strix("I8").unwrap(), DType::I8);
        assert!(safetensor_dtype_to_strix("UNKNOWN").is_err());
    }

    #[test]
    fn json_parser_basics() {
        let obj = parse_json(r#"{"a": 1, "b": "hello"}"#).unwrap();
        match obj {
            JsonValue::Object(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].0, "a");
                assert_eq!(entries[1].0, "b");
            }
            _ => panic!("expected object"),
        }
    }
}

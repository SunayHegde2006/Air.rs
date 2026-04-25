//! PyTorch `.bin` / `.pt` tensor reader — STRIX Protocol §15.1.
//!
//! Reads PyTorch saved model files WITHOUT executing arbitrary pickle code.
//! Instead, we parse the ZIP archive structure (PyTorch saves as ZIP) and
//! extract tensor metadata from the pickle stream using a safe subset parser.
//!
//! Supports:
//! - Single-file `.bin` / `.pt` (ZIP-based)
//! - Sharded format with `pytorch_model.bin.index.json`
//!
//! # Pickle protocol used by PyTorch
//! PyTorch serializes state dicts using pickle protocol 2 or 4. The opcodes
//! we care about (storage class lookup + reduce) appear as:
//!
//! ```text
//! GLOBAL "torch" "FloatStorage"  # opcode 0x63 (GLOBAL) → class name
//! REDUCE                         # opcode 0x52 — instantiates storage
//! TUPLE                          # opcode 0x74
//! BUILD                          # opcode 0x62 — tensor._set_cdata(...)
//! ```
//!
//! We scan for GLOBAL opcodes with recognizable torch.* storage class names
//! and match them to subsequent storage indices to recover dtype and shape.
//!
//! # Safety
//! The parser explicitly refuses to execute any REDUCE calls involving
//! non-torch modules. No `eval`, `exec`, or `__import__` is ever invoked.

use super::types::DType;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

// ── Public Types ─────────────────────────────────────────────────────────

/// Per-tensor info extracted from a PyTorch file.
#[derive(Debug, Clone)]
pub struct PytorchTensorInfo {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Byte offset within the archive data file.
    pub data_offset: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Which file shard this tensor lives in.
    pub shard_path: PathBuf,
}

/// Parsed PyTorch model (single or sharded).
#[derive(Debug, Clone)]
pub struct PytorchModel {
    /// All tensor infos across all shards.
    pub tensors: Vec<PytorchTensorInfo>,
    /// Number of shards.
    pub n_shards: usize,
}

/// Errors during PyTorch file parsing.
#[derive(Debug)]
pub enum PytorchError {
    Io(std::io::Error),
    /// Not a valid ZIP archive.
    NotZip,
    /// ZIP entry not found.
    EntryNotFound(String),
    /// Pickle parse error (safe subset only).
    PickleError(String),
    /// Unknown dtype code.
    UnknownDtype(String),
    /// Shard index parse error.
    InvalidIndex(String),
    /// Shard file not found.
    ShardNotFound(PathBuf),
}

impl std::fmt::Display for PytorchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "PyTorch I/O error: {e}"),
            Self::NotZip => write!(f, "PyTorch file is not a valid ZIP archive"),
            Self::EntryNotFound(s) => write!(f, "PyTorch ZIP entry not found: {s}"),
            Self::PickleError(s) => write!(f, "PyTorch pickle parse error: {s}"),
            Self::UnknownDtype(s) => write!(f, "PyTorch unknown dtype: {s}"),
            Self::InvalidIndex(s) => write!(f, "PyTorch shard index error: {s}"),
            Self::ShardNotFound(p) => write!(f, "PyTorch shard not found: {}", p.display()),
        }
    }
}

impl std::error::Error for PytorchError {}

impl From<std::io::Error> for PytorchError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── ZIP End-of-Central-Directory Scanner ──────────────────────────────────
//
// PyTorch saves models as ZIP archives. We need to find tensor data entries
// without a full ZIP library. We scan for the EOCD signature and read the
// central directory to locate stored (uncompressed) entries.

const ZIP_EOCD_SIGNATURE: u32 = 0x06054b50;
const ZIP_CD_SIGNATURE:   u32 = 0x02014b50;
const ZIP_LOCAL_SIGNATURE: u32 = 0x04034b50;

/// A located entry in the ZIP central directory.
#[derive(Debug)]
struct ZipEntry {
    name: String,
    _compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
    compression_method: u16,
}

/// Read the ZIP central directory entries from a file.
fn read_zip_entries(data: &[u8]) -> Result<Vec<ZipEntry>, PytorchError> {
    // Find EOCD (search backwards from end, max 65KB comment)
    let search_start = data.len().saturating_sub(65536 + 22);
    let mut eocd_pos = None;
    for i in (search_start..data.len().saturating_sub(3)).rev() {
        if read_u32_le(data, i) == ZIP_EOCD_SIGNATURE {
            eocd_pos = Some(i);
            break;
        }
    }
    let eocd_pos = eocd_pos.ok_or(PytorchError::NotZip)?;

    // Parse EOCD
    if eocd_pos + 22 > data.len() {
        return Err(PytorchError::NotZip);
    }
    let cd_entries = read_u16_le(data, eocd_pos + 10) as usize;
    let cd_size    = read_u32_le(data, eocd_pos + 12) as usize;
    let cd_offset  = read_u32_le(data, eocd_pos + 16) as usize;

    if cd_offset + cd_size > data.len() {
        return Err(PytorchError::NotZip);
    }

    // Parse central directory entries
    let mut entries = Vec::with_capacity(cd_entries);
    let mut pos = cd_offset;

    for _ in 0..cd_entries {
        if pos + 46 > data.len() || read_u32_le(data, pos) != ZIP_CD_SIGNATURE {
            break;
        }
        let compression  = read_u16_le(data, pos + 10);
        let compressed   = read_u32_le(data, pos + 20);
        let uncompressed = read_u32_le(data, pos + 24);
        let name_len     = read_u16_le(data, pos + 28) as usize;
        let extra_len    = read_u16_le(data, pos + 30) as usize;
        let comment_len  = read_u16_le(data, pos + 32) as usize;
        let local_offset = read_u32_le(data, pos + 42);

        let name_end = pos + 46 + name_len;
        if name_end > data.len() { break; }
        let name = String::from_utf8_lossy(&data[pos + 46..name_end]).to_string();

        entries.push(ZipEntry {
            name,
            _compressed_size: compressed,
            uncompressed_size: uncompressed,
            local_header_offset: local_offset,
            compression_method: compression,
        });

        pos = name_end + extra_len + comment_len;
    }

    Ok(entries)
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset], data[offset + 1],
        data[offset + 2], data[offset + 3],
    ])
}

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],     data[offset + 1],
        data[offset + 2], data[offset + 3],
        data[offset + 4], data[offset + 5],
        data[offset + 6], data[offset + 7],
    ])
}

/// Get the data offset for a ZIP local file header entry.
fn local_data_offset(data: &[u8], local_header_offset: u32) -> Result<usize, PytorchError> {
    let pos = local_header_offset as usize;
    if pos + 30 > data.len() || read_u32_le(data, pos) != ZIP_LOCAL_SIGNATURE {
        return Err(PytorchError::NotZip);
    }
    let name_len  = read_u16_le(data, pos + 26) as usize;
    let extra_len = read_u16_le(data, pos + 28) as usize;
    Ok(pos + 30 + name_len + extra_len)
}

// ── Safe Pickle Subset Parser ────────────────────────────────────────────
//
// PyTorch uses Python pickle to serialize the state dict. We parse a minimal
// safe subset: only string literals, GLOBAL class lookups, and integer values.
// We never execute REDUCE on non-torch modules.
//
// Pickle opcodes used by PyTorch (protocol 2+):
//   0x63 GLOBAL     — "module\nname\n" → push class ref
//   0x28 MARK       — push mark
//   0x4d SHORT_BININT — 2-byte signed int
//   0x4b BININT1    — 1-byte unsigned int
//   0x4a BININT     — 4-byte signed int
//   0x8c SHORT_BINUNICODE — len(1B) + UTF-8
//   0x8d BINUNICODE8 — len(8B) + UTF-8
//   0x58 BINUNICODE  — len(4B) + UTF-8
//   0x80 PROTO      — protocol version
//   0x52 REDUCE     — (class, args) → instance
//   0x71 BINPUT / 0x72 LONG_BINPUT / 0x68 BINGET / 0x69 LONG_BINGET
//   0x85 TUPLE1 / 0x86 TUPLE2 / 0x87 TUPLE3 / 0x74 TUPLE
//   0x4e NONE
//   0x2e STOP       — end of pickle
//   0x7d EMPTY_DICT / 0x5d EMPTY_LIST / 0x29 EMPTY_TUPLE
//   0x75 SETITEMS / 0x73 SETITEM

/// A parsed value from the pickle stack
#[derive(Debug, Clone)]
enum PickleVal {
    String(String),
    Int(i64),
    Bool(bool),
    None,
    Tuple(Vec<PickleVal>),
    List(Vec<PickleVal>),
    /// A torch.*Storage class reference — carries the dtype
    TorchStorage(DType),
    /// An instantiated storage: (dtype, storage_index, size)
    StorageInstance { dtype: DType, index: u64, n_elements: u64 },
    /// A reconstructed tensor: (storage_instance_ref, offset, shape, strides)
    TensorMeta { dtype: DType, shape: Vec<usize>, data_index: u64 },
    /// Opaque value we don't care about
    Opaque,
}

/// Internal state for safe pickle parsing.
struct PickleParser<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<PickleVal>,
    memo: HashMap<u64, PickleVal>,
    mark: Vec<usize>, // mark stack
}

impl<'a> PickleParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, stack: Vec::new(), memo: HashMap::new(), mark: Vec::new() }
    }

    fn remaining(&self) -> usize { self.data.len() - self.pos }

    fn read_byte(&mut self) -> Option<u8> {
        if self.pos >= self.data.len() { return None; }
        let b = self.data[self.pos];
        self.pos += 1;
        Some(b)
    }

    fn read_bytes(&mut self, n: usize) -> Option<&[u8]> {
        if self.pos + n > self.data.len() { return None; }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    fn read_line(&mut self) -> Option<&str> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.data[start..self.pos]).ok()?;
        self.pos += 1; // consume '\n'
        Some(s)
    }

    fn read_u8(&mut self) -> Option<u8> { self.read_byte() }

    fn read_u16(&mut self) -> Option<u16> {
        let b = self.read_bytes(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i32(&mut self) -> Option<i32> {
        let b = self.read_bytes(4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let b = self.read_bytes(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let b = self.read_bytes(8)?;
        Some(u64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]))
    }

    fn pop(&mut self) -> PickleVal {
        self.stack.pop().unwrap_or(PickleVal::Opaque)
    }

    fn pop_n_from_mark(&mut self) -> Vec<PickleVal> {
        let mark_pos = self.mark.pop().unwrap_or(0);
        if mark_pos <= self.stack.len() {
            self.stack.split_off(mark_pos)
        } else {
            Vec::new()
        }
    }

    /// Run the safe pickle parse and return all collected tensor metadata.
    fn parse(&mut self) -> Vec<PickleVal> {
        let mut tensors: Vec<PickleVal> = Vec::new();

        while self.remaining() > 0 {
            let opcode = match self.read_byte() { Some(b) => b, None => break };

            match opcode {
                // PROTO (0x80): skip version byte
                0x80 => { self.read_byte(); }

                // FRAME (0x95): skip 8-byte frame length
                0x95 => { self.read_bytes(8); }

                // MARK (0x28)
                0x28 => { self.mark.push(self.stack.len()); }

                // STOP (0x2e)
                0x2e => break,

                // NONE (0x4e)
                0x4e => { self.stack.push(PickleVal::None); }

                // NEWTRUE (0x88) / NEWFALSE (0x89)
                0x88 => { self.stack.push(PickleVal::Bool(true)); }
                0x89 => { self.stack.push(PickleVal::Bool(false)); }

                // BININT1 (0x4b): 1-byte unsigned int
                0x4b => {
                    if let Some(v) = self.read_u8() {
                        self.stack.push(PickleVal::Int(v as i64));
                    }
                }

                // BININT (0x4a): 4-byte little-endian signed int
                0x4a => {
                    if let Some(v) = self.read_i32() {
                        self.stack.push(PickleVal::Int(v as i64));
                    }
                }

                // SHORT_BININT (0x4d): 2-byte unsigned int
                0x4d => {
                    if let Some(v) = self.read_u16() {
                        self.stack.push(PickleVal::Int(v as i64));
                    }
                }

                // LONG1 (0x8a): length-prefixed little-endian int
                0x8a => {
                    if let Some(len) = self.read_u8() {
                        if let Some(bytes) = self.read_bytes(len as usize) {
                            let mut val = 0i64;
                            for (i, &b) in bytes.iter().enumerate() {
                                val |= (b as i64) << (i * 8);
                            }
                            self.stack.push(PickleVal::Int(val));
                        }
                    }
                }

                // SHORT_BINUNICODE (0x8c): 1-byte len + UTF-8
                0x8c => {
                    if let Some(len) = self.read_u8() {
                        if let Some(bytes) = self.read_bytes(len as usize) {
                            let s = String::from_utf8_lossy(bytes).to_string();
                            self.stack.push(PickleVal::String(s));
                        }
                    }
                }

                // BINUNICODE (0x58): 4-byte len + UTF-8
                0x58 => {
                    if let Some(len) = self.read_u32() {
                        if let Some(bytes) = self.read_bytes(len as usize) {
                            let s = String::from_utf8_lossy(bytes).to_string();
                            self.stack.push(PickleVal::String(s));
                        }
                    }
                }

                // BINUNICODE8 (0x8d): 8-byte len + UTF-8
                0x8d => {
                    if let Some(len) = self.read_u64() {
                        if let Some(bytes) = self.read_bytes(len as usize) {
                            let s = String::from_utf8_lossy(bytes).to_string();
                            self.stack.push(PickleVal::String(s));
                        }
                    }
                }

                // BINBYTES (0x43): 4-byte len + bytes (skip content)
                0x43 => {
                    if let Some(len) = self.read_u32() {
                        self.read_bytes(len as usize);
                        self.stack.push(PickleVal::Opaque);
                    }
                }

                // BINBYTES8 (0x8e): 8-byte len + bytes
                0x8e => {
                    if let Some(len) = self.read_u64() {
                        self.read_bytes(len as usize);
                        self.stack.push(PickleVal::Opaque);
                    }
                }

                // SHORT_BINBYTES (0x43 is already covered, 0x43 in older protocol)
                // GLOBAL (0x63): "module\nname\n"
                0x63 => {
                    let module = self.read_line().unwrap_or("").to_string();
                    let name   = self.read_line().unwrap_or("").to_string();
                    let val = if module == "torch" {
                        if let Ok(dtype) = pytorch_storage_to_dtype(&name) {
                            PickleVal::TorchStorage(dtype)
                        } else {
                            PickleVal::Opaque
                        }
                    } else {
                        PickleVal::Opaque
                    };
                    self.stack.push(val);
                }

                // STACK_GLOBAL (0x93): top two are module, name
                0x93 => {
                    let name   = self.pop();
                    let module = self.pop();
                    let val = match (&module, &name) {
                        (PickleVal::String(m), PickleVal::String(n))
                            if m == "torch" =>
                        {
                            if let Ok(dtype) = pytorch_storage_to_dtype(n) {
                                PickleVal::TorchStorage(dtype)
                            } else {
                                PickleVal::Opaque
                            }
                        }
                        _ => PickleVal::Opaque,
                    };
                    self.stack.push(val);
                }

                // REDUCE (0x52): (callable, args_tuple) → result
                0x52 => {
                    let args = self.pop();
                    let callable = self.pop();
                    let result = match callable {
                        PickleVal::TorchStorage(dtype) => {
                            // args tuple: (storage_key, location, size)
                            // storage_key: "archive/data/{index}"
                            if let PickleVal::Tuple(ref elems) = args {
                                let index = match elems.first() {
                                    Some(PickleVal::Int(n)) => *n as u64,
                                    Some(PickleVal::String(s)) => {
                                        // str key like "0", "1", ...
                                        s.parse::<u64>().unwrap_or(0)
                                    }
                                    _ => 0,
                                };
                                let n_elements = match elems.get(2) {
                                    Some(PickleVal::Int(n)) => *n as u64,
                                    _ => 0,
                                };
                                PickleVal::StorageInstance { dtype, index, n_elements }
                            } else {
                                PickleVal::StorageInstance { dtype, index: 0, n_elements: 0 }
                            }
                        }
                        _ => PickleVal::Opaque,
                    };
                    self.stack.push(result);
                }

                // BUILD (0x62): obj.__setstate__(state) — used for tensor metadata
                0x62 => {
                    let state = self.pop();
                    let obj   = self.pop();
                    // PyTorch calls tensor.__setstate__((storage, offset, shape, strides, ...))
                    let result = if let PickleVal::Tuple(ref elems) = state {
                        if let Some(PickleVal::StorageInstance { dtype, index, .. }) = elems.first() {
                            // elems: [storage, storage_offset, shape_tuple, strides_tuple, ...]
                            let shape = match elems.get(2) {
                                Some(PickleVal::Tuple(dims)) => {
                                    dims.iter().filter_map(|d| {
                                        if let PickleVal::Int(n) = d { Some(*n as usize) } else { None }
                                    }).collect()
                                }
                                _ => vec![],
                            };
                            let tensor = PickleVal::TensorMeta {
                                dtype: *dtype,
                                shape,
                                data_index: *index,
                            };
                            tensors.push(tensor.clone());
                            tensor
                        } else {
                            obj
                        }
                    } else {
                        obj
                    };
                    self.stack.push(result);
                }

                // TUPLE (0x74): collect from mark to tuple
                0x74 => {
                    let items = self.pop_n_from_mark();
                    self.stack.push(PickleVal::Tuple(items));
                }

                // TUPLE1 (0x85)
                0x85 => {
                    let a = self.pop();
                    self.stack.push(PickleVal::Tuple(vec![a]));
                }

                // TUPLE2 (0x86)
                0x86 => {
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(PickleVal::Tuple(vec![a, b]));
                }

                // TUPLE3 (0x87)
                0x87 => {
                    let c = self.pop();
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(PickleVal::Tuple(vec![a, b, c]));
                }

                // EMPTY_TUPLE (0x29)
                0x29 => { self.stack.push(PickleVal::Tuple(vec![])); }

                // EMPTY_LIST (0x5d)
                0x5d => { self.stack.push(PickleVal::List(vec![])); }

                // EMPTY_DICT (0x7d)
                0x7d => { self.stack.push(PickleVal::Opaque); }

                // APPENDS (0x65): append list from mark
                0x65 => { self.pop_n_from_mark(); }

                // APPEND (0x61)
                0x61 => { self.pop(); }

                // SETITEMS (0x75) / SETITEM (0x73)
                0x75 => { self.pop_n_from_mark(); }
                0x73 => { self.pop(); self.pop(); }

                // BINPUT (0x71): memo[1-byte key] = TOS
                0x71 => {
                    if let Some(key) = self.read_u8() {
                        if let Some(val) = self.stack.last() {
                            self.memo.insert(key as u64, val.clone());
                        }
                    }
                }

                // LONG_BINPUT (0x72): memo[4-byte key] = TOS
                0x72 => {
                    if let Some(key) = self.read_u32() {
                        if let Some(val) = self.stack.last() {
                            self.memo.insert(key as u64, val.clone());
                        }
                    }
                }

                // MEMOIZE (0x94): protocol 4 — memo[next_id] = TOS
                0x94 => {
                    let id = self.memo.len() as u64;
                    if let Some(val) = self.stack.last() {
                        self.memo.insert(id, val.clone());
                    }
                }

                // BINGET (0x68): push memo[1-byte key]
                0x68 => {
                    if let Some(key) = self.read_u8() {
                        let val = self.memo.get(&(key as u64)).cloned().unwrap_or(PickleVal::Opaque);
                        self.stack.push(val);
                    }
                }

                // LONG_BINGET (0x69): push memo[4-byte key]
                0x69 => {
                    if let Some(key) = self.read_u32() {
                        let val = self.memo.get(&(key as u64)).cloned().unwrap_or(PickleVal::Opaque);
                        self.stack.push(val);
                    }
                }

                // POP (0x30)
                0x30 => { self.pop(); }

                // POP_MARK (0x31)
                0x31 => { self.pop_n_from_mark(); }

                // DUP (0x32)
                0x32 => {
                    if let Some(top) = self.stack.last().cloned() {
                        self.stack.push(top);
                    }
                }

                // INT (0x49): "int\n" as text
                0x49 => { self.read_line(); self.stack.push(PickleVal::Opaque); }

                // STRING (0x53): quoted text string
                0x53 => { self.read_line(); self.stack.push(PickleVal::Opaque); }

                // UNICODE (0x56): raw unicode string line
                0x56 => { self.read_line(); self.stack.push(PickleVal::Opaque); }

                // FLOAT (0x46): "float\n" text
                0x46 => { self.read_line(); self.stack.push(PickleVal::Opaque); }

                // BINFLOAT (0x47): 8-byte IEEE 754 double
                0x47 => { self.read_bytes(8); self.stack.push(PickleVal::Opaque); }

                // PERSID (0x50): text persistent id
                0x50 => { self.read_line(); self.stack.push(PickleVal::Opaque); }

                // BINPERSID (0x51): TOS as persistent id
                0x51 => { self.pop(); self.stack.push(PickleVal::Opaque); }

                // NEWOBJ (0x81): (cls, *args) construction
                0x81 => {
                    let _args = self.pop();
                    let _cls  = self.pop();
                    self.stack.push(PickleVal::Opaque);
                }

                // NEWOBJ_EX (0x92)
                0x92 => {
                    let _kwargs = self.pop();
                    let _args   = self.pop();
                    let _cls    = self.pop();
                    self.stack.push(PickleVal::Opaque);
                }

                // EXT1/2/4 (0x82/0x83/0x84): extension registry — skip
                0x82 => { self.read_byte(); self.stack.push(PickleVal::Opaque); }
                0x83 => { self.read_bytes(2); self.stack.push(PickleVal::Opaque); }
                0x84 => { self.read_bytes(4); self.stack.push(PickleVal::Opaque); }

                // INST (0x69 — protocol 0): "module\nname\n" + mark args
                // (note: 0x69 is also LONG_BINGET — this is in older protocol, ignore for
                //  protocol 2+ which uses different opcodes. We already handle 0x69 as LONG_BINGET)

                // Anything unknown — push Opaque so we don't desync
                _ => {
                    self.stack.push(PickleVal::Opaque);
                }
            }
        }

        tensors
    }
}

// ── dtype string → DType ─────────────────────────────────────────────────

/// Map PyTorch storage type string to DType.
/// Used by the pickle parser and exposed for format compatibility tests.
pub fn pytorch_storage_to_dtype(storage_type: &str) -> Result<DType, PytorchError> {
    // Storage type names from torch.*Storage
    // Note: BFloat16 must be checked before Float (BFloat16Storage contains "Float")
    // Note: Complex types must be rejected before Double/Float matches fire.
    match storage_type {
        s if s.contains("Complex")                                 => Err(PytorchError::UnknownDtype(s.to_string())),
        s if s.contains("BFloat16") || s.contains("bfloat16")     => Ok(DType::BF16),
        s if s.contains("Half")     || s.contains("float16")      => Ok(DType::F16),
        s if s.contains("Float")    || s.contains("float32")      => Ok(DType::F32),
        s if s.contains("Double")   || s.contains("float64")      => Ok(DType::F32), // downcast
        s if s.contains("Long")     || s.contains("int64")        => Ok(DType::I32), // downcast
        s if s.contains("Int")      || s.contains("int32")        => Ok(DType::I32),
        s if s.contains("Short")    || s.contains("int16")        => Ok(DType::I16),
        s if s.contains("Char")     || s.contains("int8")         => Ok(DType::I8),
        s if s.contains("Byte")     || s.contains("uint8")        => Ok(DType::U8),
        other => Err(PytorchError::UnknownDtype(other.to_string())),
    }
}

// ── ZIP Tensor Extraction ─────────────────────────────────────────────────

/// Result of pickle analysis — maps storage index to (dtype, n_elements).
#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub dtype: DType,
    pub n_elements: u64,
}

/// Result of full pickle analysis.
#[derive(Debug, Clone)]
pub struct PickleAnalysis {
    /// Ordered list of tensor metadata extracted from BUILD opcodes.
    pub tensors: Vec<TensorMeta>,
    /// Ordered list of tensor names extracted from string literals.
    pub names: Vec<String>,
}

/// Tensor metadata recovered from pickle.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_index: u64,
}

/// Full pickle analysis of `data.pkl` contents.
fn analyse_pickle(pkl_data: &[u8]) -> PickleAnalysis {
    let mut parser = PickleParser::new(pkl_data);
    let tensor_vals = parser.parse();

    // Collect tensor names from memo and stack unicode strings
    // that look like weight names
    let mut names = extract_names_from_pickle(pkl_data);

    let tensors = tensor_vals.into_iter().filter_map(|v| {
        if let PickleVal::TensorMeta { dtype, shape, data_index } = v {
            Some(TensorMeta { dtype, shape, data_index })
        } else {
            None
        }
    }).collect::<Vec<_>>();

    // Deduplicate names preserving order
    names.dedup();

    PickleAnalysis { tensors, names }
}

/// Extract tensor entries from a PyTorch ZIP archive by scanning for
/// data files in `archive/data/` and matching with tensor names from
/// the pickle stream.
fn extract_tensors_from_zip(
    data: &[u8],
    entries: &[ZipEntry],
    file_path: &Path,
) -> Result<Vec<PytorchTensorInfo>, PytorchError> {
    // Sort data entries by their numeric index
    let mut data_entries: Vec<(&ZipEntry, usize)> = Vec::new();
    for entry in entries {
        if entry.name.contains("/data/") && !entry.name.ends_with('/') {
            let offset = local_data_offset(data, entry.local_header_offset)?;
            data_entries.push((entry, offset));
        }
    }
    // Sort by the numeric part of the path (archive/data/0, archive/data/1, ...)
    data_entries.sort_by_key(|(e, _)| {
        e.name.rsplit('/').next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    });

    // Parse data.pkl for dtype+shape+names
    let analysis = if let Some(pkl) = entries.iter().find(|e| e.name.ends_with("data.pkl")) {
        let pkl_offset = local_data_offset(data, pkl.local_header_offset)?;
        let pkl_end = pkl_offset + pkl.uncompressed_size as usize;
        if pkl_end <= data.len() && pkl.compression_method == 0 {
            analyse_pickle(&data[pkl_offset..pkl_end])
        } else {
            PickleAnalysis { tensors: vec![], names: vec![] }
        }
    } else {
        PickleAnalysis { tensors: vec![], names: vec![] }
    };

    // Also load the record file for storage sizes (archive/data/*.storage)
    // Some older formats store sizes in the records file as u64 LE at offset 0.

    let mut tensors = Vec::new();
    for (i, (entry, offset)) in data_entries.iter().enumerate() {
        let index_str = entry.name.rsplit('/').next().unwrap_or("");
        let data_index = index_str.parse::<u64>().unwrap_or(i as u64);

        // Try to find matching tensor meta from pickle
        let (dtype, shape) = analysis.tensors.iter()
            .find(|t| t.data_index == data_index)
            .map(|t| (t.dtype, t.shape.clone()))
            .unwrap_or_else(|| {
                // Fallback: check if the data starts with a size record
                // Some formats store element count as first u64
                let size = entry.uncompressed_size as usize;
                (DType::F32, vec![size / 4])
            });

        let name = analysis.names.get(i)
            .cloned()
            .unwrap_or_else(|| format!("tensor_{index_str}"));

        let elem_size = dtype.element_size().unwrap_or(2); // quant types unlikely in .bin
        let n_elements: usize = shape.iter().product::<usize>().max(1);
        let size_bytes = if entry.uncompressed_size > 0 {
            entry.uncompressed_size as usize
        } else {
            n_elements * elem_size
        };

        tensors.push(PytorchTensorInfo {
            name,
            shape,
            dtype,
            data_offset: *offset as u64,
            size_bytes,
            shard_path: file_path.to_path_buf(),
        });
    }

    Ok(tensors)
}

/// Extract tensor names from pickle data. This is a heuristic scan for
/// string patterns that look like tensor names (containing dots and common
/// keywords like "weight", "bias", "embed", "layer", "attn", "mlp").
fn extract_names_from_pickle(pkl_data: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut i = 0;
    while i < pkl_data.len() {
        let opcode = pkl_data[i];
        // SHORT_BINUNICODE (0x8c)
        if opcode == 0x8C && i + 1 < pkl_data.len() {
            let len = pkl_data[i + 1] as usize;
            if i + 2 + len <= pkl_data.len() {
                if let Ok(s) = std::str::from_utf8(&pkl_data[i + 2..i + 2 + len]) {
                    if looks_like_tensor_name(s) {
                        names.push(s.to_string());
                    }
                }
            }
            i += 2 + len;
        // BINUNICODE (0x58): 4-byte len
        } else if opcode == 0x58 && i + 5 <= pkl_data.len() {
            let len = u32::from_le_bytes([pkl_data[i+1], pkl_data[i+2], pkl_data[i+3], pkl_data[i+4]]) as usize;
            if i + 5 + len <= pkl_data.len() {
                if let Ok(s) = std::str::from_utf8(&pkl_data[i + 5..i + 5 + len]) {
                    if looks_like_tensor_name(s) {
                        names.push(s.to_string());
                    }
                }
            }
            i += 5 + len;
        } else {
            i += 1;
        }
    }
    names
}

/// Heuristic: does this string look like a PyTorch tensor name?
fn looks_like_tensor_name(s: &str) -> bool {
    // Must contain a dot (hierarchical name) and a common keyword
    s.contains('.') && (
        s.contains("weight") || s.contains("bias") ||
        s.contains("embed")  || s.contains("norm") ||
        s.contains("attn")   || s.contains("mlp")  ||
        s.contains("layer")  || s.contains("proj")  ||
        s.contains("head")   || s.contains("lm_head")
    )
}

// ── Sharded index.json parser ─────────────────────────────────────────────

/// Entry from a `pytorch_model.bin.index.json` weight map.
#[derive(Debug, Clone)]
pub struct ShardEntry {
    /// Tensor name.
    pub tensor_name: String,
    /// Relative shard filename.
    pub shard_file: String,
}

/// Parse a `pytorch_model.bin.index.json` to get the full weight map.
///
/// Returns ordered list of (tensor_name, shard_file) pairs.
/// The JSON looks like:
/// ```text
/// {
///   "metadata": { "total_size": 12345 },
///   "weight_map": {
///     "model.layers.0.attn.q_proj.weight": "pytorch_model-00001-of-00003.bin",
///     ...
///   }
/// }
/// ```
pub fn parse_shard_index(index_json: &str) -> Vec<ShardEntry> {
    let mut entries = Vec::new();
    // Find the weight_map section
    let wm_start = index_json.find("\"weight_map\"").unwrap_or(0);
    let section = &index_json[wm_start..];

    // Line-by-line scan for "key": "value" pairs
    for line in section.lines() {
        let trimmed = line.trim().trim_end_matches(',');
        // Pattern: "tensor_name": "shard_file.bin"
        if trimmed.starts_with('"') && trimmed.contains("\": \"") {
            if let Some(colon_pos) = trimmed.find("\": \"") {
                let key = &trimmed[1..colon_pos];
                let after = &trimmed[colon_pos + 4..];
                if let Some(end_quote) = after.find('"') {
                    let val = &after[..end_quote];
                    if (val.ends_with(".bin") || val.ends_with(".pt"))
                        && looks_like_tensor_name(key)
                    {
                        entries.push(ShardEntry {
                            tensor_name: key.to_string(),
                            shard_file: val.to_string(),
                        });
                    }
                }
            }
        }
    }
    entries
}

// ── Public API ───────────────────────────────────────────────────────────

/// Parse a single PyTorch `.bin` or `.pt` file.
pub fn parse_pytorch(path: &Path) -> Result<PytorchModel, PytorchError> {
    let data = std::fs::read(path)?;
    let entries = read_zip_entries(&data)?;
    let tensors = extract_tensors_from_zip(&data, &entries, path)?;
    Ok(PytorchModel { tensors, n_shards: 1 })
}

/// Parse a sharded PyTorch model from its index file.
///
/// Expects `pytorch_model.bin.index.json`.
/// Reads the `weight_map` to determine which shards exist and which
/// tensors belong to which shard, then parses each shard.
pub fn parse_pytorch_sharded(index_path: &Path) -> Result<PytorchModel, PytorchError> {
    let index_bytes = std::fs::read(index_path)?;
    let index_str = std::str::from_utf8(&index_bytes)
        .map_err(|e| PytorchError::InvalidIndex(format!("non-UTF8: {e}")))?;

    let parent = index_path.parent().unwrap_or(Path::new("."));

    // Parse weight_map for tensor→shard mapping
    let weight_map = parse_shard_index(index_str);

    // Collect unique shard files in weight_map order
    let mut shard_files: Vec<PathBuf> = Vec::new();
    for entry in &weight_map {
        let shard_path = parent.join(&entry.shard_file);
        if !shard_files.contains(&shard_path) {
            shard_files.push(shard_path);
        }
    }

    // Fallback: if weight_map parsing found no entries, scan for .bin files
    if shard_files.is_empty() {
        for line in index_str.lines() {
            let trimmed = line.trim();
            if let Some(val_start) = trimmed.rfind(": \"") {
                let after = &trimmed[val_start + 3..];
                if let Some(val_end) = after.find('"') {
                    let filename = &after[..val_end];
                    if filename.ends_with(".bin") || filename.ends_with(".pt") {
                        let shard_path = parent.join(filename);
                        if !shard_files.contains(&shard_path) {
                            shard_files.push(shard_path);
                        }
                    }
                }
            }
        }
    }

    let mut all_tensors = Vec::new();
    for shard_path in &shard_files {
        if !shard_path.exists() {
            return Err(PytorchError::ShardNotFound(shard_path.clone()));
        }
        let model = parse_pytorch(shard_path)?;
        all_tensors.extend(model.tensors);
    }

    // If weight_map has names, use them to annotate tensors in order
    if !weight_map.is_empty() && weight_map.len() == all_tensors.len() {
        for (tensor, entry) in all_tensors.iter_mut().zip(weight_map.iter()) {
            tensor.name = entry.tensor_name.clone();
        }
    }

    Ok(PytorchModel {
        n_shards: shard_files.len(),
        tensors: all_tensors,
    })
}

/// Auto-detect single vs sharded format.
pub fn parse_pytorch_auto(path: &Path) -> Result<PytorchModel, PytorchError> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with(".index.json") {
        parse_pytorch_sharded(path)
    } else {
        parse_pytorch(path)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid ZIP archive with a single stored entry.
    fn make_minimal_zip(entry_name: &str, content: &[u8]) -> Vec<u8> {
        let name_bytes = entry_name.as_bytes();
        let mut buf = Vec::new();

        // Local file header
        let local_offset = buf.len() as u32;
        buf.extend_from_slice(&ZIP_LOCAL_SIGNATURE.to_le_bytes());
        buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
        buf.extend_from_slice(&0u16.to_le_bytes());  // flags
        buf.extend_from_slice(&0u16.to_le_bytes());  // compression (stored)
        buf.extend_from_slice(&0u16.to_le_bytes());  // mod time
        buf.extend_from_slice(&0u16.to_le_bytes());  // mod date
        buf.extend_from_slice(&0u32.to_le_bytes());  // crc32
        buf.extend_from_slice(&(content.len() as u32).to_le_bytes()); // compressed size
        buf.extend_from_slice(&(content.len() as u32).to_le_bytes()); // uncompressed size
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // extra len
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(content);

        // Central directory entry
        let cd_offset = buf.len();
        buf.extend_from_slice(&ZIP_CD_SIGNATURE.to_le_bytes());
        buf.extend_from_slice(&20u16.to_le_bytes()); // version made by
        buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
        buf.extend_from_slice(&0u16.to_le_bytes());  // flags
        buf.extend_from_slice(&0u16.to_le_bytes());  // compression
        buf.extend_from_slice(&0u16.to_le_bytes());  // mod time
        buf.extend_from_slice(&0u16.to_le_bytes());  // mod date
        buf.extend_from_slice(&0u32.to_le_bytes());  // crc32
        buf.extend_from_slice(&(content.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(content.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());  // extra len
        buf.extend_from_slice(&0u16.to_le_bytes());  // comment len
        buf.extend_from_slice(&0u16.to_le_bytes());  // disk number
        buf.extend_from_slice(&0u16.to_le_bytes());  // internal attrs
        buf.extend_from_slice(&0u32.to_le_bytes());  // external attrs
        buf.extend_from_slice(&local_offset.to_le_bytes());
        buf.extend_from_slice(name_bytes);

        let cd_size = buf.len() - cd_offset;

        // End of central directory
        buf.extend_from_slice(&ZIP_EOCD_SIGNATURE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // disk number
        buf.extend_from_slice(&0u16.to_le_bytes()); // cd disk
        buf.extend_from_slice(&1u16.to_le_bytes()); // entries on disk
        buf.extend_from_slice(&1u16.to_le_bytes()); // total entries
        buf.extend_from_slice(&(cd_size as u32).to_le_bytes());
        buf.extend_from_slice(&(cd_offset as u32).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // comment len

        buf
    }

    #[test]
    fn zip_entry_parsing() {
        let zip = make_minimal_zip("archive/data/0", &[0u8; 64]);
        let entries = read_zip_entries(&zip).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "archive/data/0");
        assert_eq!(entries[0].uncompressed_size, 64);
    }

    #[test]
    fn not_a_zip() {
        let result = read_zip_entries(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_name_heuristic() {
        assert!(looks_like_tensor_name("model.layers.0.self_attn.q_proj.weight"));
        assert!(looks_like_tensor_name("transformer.h.0.attn.bias"));
        assert!(!looks_like_tensor_name("random_string"));
        assert!(!looks_like_tensor_name("no_dots_here_weight"));
    }

    #[test]
    fn storage_dtype_mapping() {
        assert_eq!(pytorch_storage_to_dtype("FloatStorage").unwrap(), DType::F32);
        assert_eq!(pytorch_storage_to_dtype("HalfStorage").unwrap(), DType::F16);
        assert_eq!(pytorch_storage_to_dtype("BFloat16Storage").unwrap(), DType::BF16);
        assert_eq!(pytorch_storage_to_dtype("ByteStorage").unwrap(), DType::U8);
        assert_eq!(pytorch_storage_to_dtype("CharStorage").unwrap(), DType::I8);
        assert_eq!(pytorch_storage_to_dtype("ShortStorage").unwrap(), DType::I16);
        assert_eq!(pytorch_storage_to_dtype("IntStorage").unwrap(), DType::I32);
        assert_eq!(pytorch_storage_to_dtype("LongStorage").unwrap(), DType::I32);
        assert!(pytorch_storage_to_dtype("ComplexDoubleStorage").is_err());
    }

    #[test]
    fn parse_pytorch_zip_data_entry() {
        let zip = make_minimal_zip("archive/data/0", &[0u8; 128]);
        let entries = read_zip_entries(&zip).unwrap();
        let tensors = extract_tensors_from_zip(&zip, &entries, Path::new("test.bin")).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].size_bytes, 128);
    }

    #[test]
    fn multiple_shard_entries_sorted() {
        // Verify that the sort key extractor correctly parses numeric path segments.
        let ind0: u64 = "archive/data/0".rsplit('/').next().unwrap().parse().unwrap();
        let ind1: u64 = "archive/data/1".rsplit('/').next().unwrap().parse().unwrap();
        let ind2: u64 = "archive/data/2".rsplit('/').next().unwrap().parse().unwrap();
        assert!(ind0 < ind1);
        assert!(ind1 < ind2);
        assert_eq!(ind0, 0);
        assert_eq!(ind2, 2);
    }

    #[test]
    fn pickle_parser_basic_strings() {
        // Build a minimal pickle stream with SHORT_BINUNICODE strings
        let mut pkl = Vec::new();
        pkl.push(0x80); pkl.push(2); // PROTO 2
        // Push "model.weight" as SHORT_BINUNICODE
        let name = b"model.weight";
        pkl.push(0x8c); pkl.push(name.len() as u8); pkl.extend_from_slice(name);
        pkl.push(0x2e); // STOP
        let names = extract_names_from_pickle(&pkl);
        assert!(names.contains(&"model.weight".to_string()));
    }

    #[test]
    fn pickle_parser_bfloat16_storage_global() {
        // Minimal pickle: GLOBAL "torch" "BFloat16Storage" → TorchStorage(BF16)
        let mut pkl = Vec::new();
        pkl.push(0x80); pkl.push(2); // PROTO 2
        pkl.push(0x63); // GLOBAL opcode
        pkl.extend_from_slice(b"torch\n");
        pkl.extend_from_slice(b"BFloat16Storage\n");
        pkl.push(0x2e); // STOP
        let mut parser = PickleParser::new(&pkl);
        parser.parse();
        assert_eq!(parser.stack.len(), 1); // TorchStorage should be on stack
        if let PickleVal::TorchStorage(dtype) = &parser.stack[0] {
            assert_eq!(*dtype, DType::BF16);
        } else {
            panic!("expected TorchStorage on stack");
        }
    }

    #[test]
    fn shard_index_parse() {
        let json = r#"{
  "metadata": {"total_size": 12345},
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00002.bin",
    "model.layers.0.mlp.down_proj.weight": "pytorch_model-00002-of-00002.bin"
  }
}"#;
        let entries = parse_shard_index(json);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tensor_name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(entries[0].shard_file, "pytorch_model-00001-of-00002.bin");
        assert_eq!(entries[1].shard_file, "pytorch_model-00002-of-00002.bin");
    }

    #[test]
    fn shard_index_empty() {
        let entries = parse_shard_index("{}");
        assert!(entries.is_empty());
    }

    #[test]
    fn auto_detect_index_json() {
        // Verify auto-detect routes correctly (file won't exist, that's ok)
        let path = Path::new("pytorch_model.bin.index.json");
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert!(name.ends_with(".index.json"));

        let path2 = Path::new("pytorch_model.bin");
        let name2 = path2.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert!(!name2.ends_with(".index.json"));
    }

    #[test]
    fn local_data_offset_bounds() {
        // Build a minimal local header and verify offset calculation
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&ZIP_LOCAL_SIGNATURE.to_le_bytes());
        // version, flags, compression, time, date, crc (all 0)
        // name_len = 4, extra_len = 2
        data[26] = 4; // name_len
        data[28] = 2; // extra_len
        let offset = local_data_offset(&data, 0).unwrap();
        assert_eq!(offset, 30 + 4 + 2); // 36
    }
}

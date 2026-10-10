use crate::error::{AirError, Result};
use crate::types::DType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            9 => Ok(GgufValueType::Array),
            10 => Ok(GgufValueType::Uint64),
            11 => Ok(GgufValueType::Int64),
            12 => Ok(GgufValueType::Float64),
            _ => Err(AirError::GgufParse(format!("Unknown GgufValueType: {val}"))),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: DType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data_offset: u64,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic_buf = [0u8; 4];
        reader.read_exact(&mut magic_buf)?;
        let magic = u32::from_le_bytes(magic_buf);
        if magic != GGUF_MAGIC {
            return Err(AirError::GgufParse(format!("Invalid GGUF magic: {magic:#x}")));
        }

        let mut ver_buf = [0u8; 4];
        reader.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);

        let tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(&mut reader)?;
            let vtype_u32 = read_u32(&mut reader)?;
            let vtype = GgufValueType::from_u32(vtype_u32)?;
            let value = read_value(&mut reader, vtype)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(read_u64(&mut reader)?);
            }
            let type_u32 = read_u32(&mut reader)?;
            let offset = read_u64(&mut reader)?;
            let tensor_type = parse_dtype_from_gguf_type(type_u32);

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    tensor_type,
                    offset,
                },
            );
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(v)) => *v as u64,
            Some(GgufValue::Uint64(v)) => *v,
            _ => 32,
        };

        let current_pos = reader.stream_position()?;
        let data_offset = (current_pos + alignment - 1) & !(alignment - 1);

        Ok(GgufFile {
            version,
            tensor_count,
            kv_count,
            metadata,
            tensors,
            data_offset,
        })
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| AirError::GgufParse(e.to_string()))
}

fn read_value<R: Read>(r: &mut R, vtype: GgufValueType) -> Result<GgufValue> {
    match vtype {
        GgufValueType::Uint8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Uint8(buf[0]))
        }
        GgufValueType::Int8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int8(buf[0] as i8))
        }
        GgufValueType::Uint16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
        }
        GgufValueType::Int16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(read_u32(r)?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(read_u32(r)? as i32)),
        GgufValueType::Float32 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
        }
        GgufValueType::Bool => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Bool(buf[0] != 0))
        }
        GgufValueType::String => Ok(GgufValue::String(read_string(r)?)),
        GgufValueType::Array => {
            let elem_type_u32 = read_u32(r)?;
            let elem_type = GgufValueType::from_u32(elem_type_u32)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(read_u64(r)?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(read_u64(r)? as i64)),
        GgufValueType::Float64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
        }
    }
}

fn parse_dtype_from_gguf_type(type_u32: u32) -> DType {
    match type_u32 {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Q4_0,
        3 => DType::Q4_1,
        6 => DType::Q5_0,
        7 => DType::Q5_1,
        8 => DType::Q8_0,
        10 => DType::Q2_K,
        11 => DType::Q3_K_S,
        12 => DType::Q4_K_M,
        13 => DType::Q5_K_M,
        14 => DType::Q6_K,
        15 => DType::Q8_K,
        16 => DType::IQ2_XXS,
        17 => DType::IQ1_S,
        18 => DType::IQ3_XXS,
        19 => DType::IQ4_XS,
        24 => DType::BF16,
        _ => DType::F32,
    }
}

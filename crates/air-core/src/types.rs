use serde::{Deserialize, Serialize};
use std::fmt;

pub type TokenId = u32;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
    Q4_0,
    Q4_1,
    Q4_K_M,
    Q4_K_S,
    Q4_K_L,
    Q5_0,
    Q5_1,
    Q5_K_M,
    Q6_K,
    Q8_0,
    Q8_K,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q5_K_S,
    Q5_K_L,
    IQ1_S,
    IQ2_XXS,
    IQ3_XXS,
    IQ4_XS,
    FP8_E4M3,
    FP8_E5M2,
    BitNetB158,
    Ternary,
    NF4,
}

impl DType {
    pub fn element_size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::FP8_E4M3 | DType::FP8_E5M2 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q4_K_M | DType::Q4_K_S | DType::Q4_K_L => 1,
            DType::Q8_0 | DType::Q8_K => 1,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    Cuda { compute_capability: u32 },
    Rocm,
    Metal,
    Wgpu,
    Cpu,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceKind::Cuda { compute_capability } => write!(f, "CUDA (sm_{compute_capability})"),
            DeviceKind::Rocm => write!(f, "ROCm"),
            DeviceKind::Metal => write!(f, "Metal"),
            DeviceKind::Wgpu => write!(f, "WebGPU"),
            DeviceKind::Cpu => write!(f, "CPU"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub name: String,
    pub total_vram_bytes: usize,
    pub free_vram_bytes: usize,
    pub compute_capability: u32,
    pub sm_count: usize,
    pub memory_bus_bits: u32,
    pub max_bandwidth_gbps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

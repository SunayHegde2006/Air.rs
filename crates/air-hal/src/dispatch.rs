use air_core::types::{DeviceKind, Phase};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKernel {
    FlashAttention3,
    FlashAttention2,
    FlashDecodingPlusPlus,
    FlashDecoding,
    MetalTiledAttention,
    MetalPagedAttention,
    WgpuAttention,
    CpuTiledAttention,
}

pub fn select_attention_kernel(phase: Phase, device_kind: DeviceKind) -> AttentionKernel {
    match (phase, device_kind) {
        (Phase::Prefill, DeviceKind::Cuda { compute_capability }) if compute_capability >= 90 => {
            AttentionKernel::FlashAttention3
        }
        (Phase::Prefill, DeviceKind::Cuda { compute_capability }) if compute_capability >= 75 => {
            AttentionKernel::FlashAttention2
        }
        (Phase::Decode, DeviceKind::Cuda { compute_capability }) if compute_capability >= 75 => {
            AttentionKernel::FlashDecodingPlusPlus
        }
        (Phase::Prefill, DeviceKind::Rocm) => AttentionKernel::FlashAttention2,
        (Phase::Decode, DeviceKind::Rocm) => AttentionKernel::FlashDecoding,
        (Phase::Prefill, DeviceKind::Metal) => AttentionKernel::MetalTiledAttention,
        (Phase::Decode, DeviceKind::Metal) => AttentionKernel::MetalPagedAttention,
        (_, DeviceKind::Cpu) => AttentionKernel::CpuTiledAttention,
        _ => AttentionKernel::WgpuAttention,
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKernel {
    WarpSpecializedGemm,
    StandardGemm,
    PrefillTensorCoreGemm,
    PrefillTensorCoreV2Gemm,
    WgpuGemm,
    CpuGemm,
}

pub fn select_gemm_kernel(phase: Phase, device_kind: DeviceKind) -> GemmKernel {
    match (phase, device_kind) {
        (Phase::Prefill, DeviceKind::Cuda { compute_capability }) if compute_capability >= 80 => {
            GemmKernel::PrefillTensorCoreV2Gemm
        }
        (Phase::Decode, DeviceKind::Cuda { compute_capability }) if compute_capability >= 80 => {
            GemmKernel::WarpSpecializedGemm
        }
        (_, DeviceKind::Cuda { .. }) => GemmKernel::StandardGemm,
        (_, DeviceKind::Cpu) => GemmKernel::CpuGemm,
        _ => GemmKernel::WgpuGemm,
    }
}

pub fn create_decode_gemm(compute_capability: (u32, u32)) -> GemmKernel {
    if compute_capability >= (8, 0) {
        GemmKernel::WarpSpecializedGemm
    } else {
        GemmKernel::StandardGemm
    }
}


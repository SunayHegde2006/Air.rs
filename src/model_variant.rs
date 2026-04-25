//! Model architecture variant detection and per-family configuration.
//!
//! Air.rs supports multiple LLM architecture families by detecting the
//! `general.architecture` field in GGUF metadata and selecting the correct
//! normalization, FFN activation, and attention variant for each family.
//!
//! # Supported Architectures
//!
//! | GGUF arch string | Family   | Norm        | FFN    | Special |
//! |------------------|----------|-------------|--------|---------|
//! | `llama`          | Llama    | RMSNorm     | SwiGLU | —       |
//! | `mistral`        | Mistral  | RMSNorm     | SwiGLU | Sliding window |
//! | `phi3`, `phi`    | Phi3     | RMSNorm     | SwiGLU | Partial RoPE, sliding window (even layers) |
//! | `phi4`           | Phi4     | RMSNorm     | SwiGLU | —       |
//! | `qwen2`, `qwen`  | Qwen2    | RMSNorm     | SwiGLU | QKV biases |
//! | `gemma`, `gemma2`, `gemma3` | Gemma | GemmaRMSNorm | GeGLU | — |
//! | `falcon`         | Falcon   | LayerNorm   | ReLU/GELU | — |
//! | anything else    | Unknown  | RMSNorm     | SwiGLU | — |
//!
//! # Compounding with OCS
//!
//! Architecture dispatch happens inside `transformer_block()` — the OCS stack
//! (FP4 SageAttention, KIMI linear attention, QJL compression, HERMES eviction)
//! is applied identically regardless of architecture variant. The arch variant
//! only selects *which* primitives to call for norm/FFN/RoPE.

use std::collections::HashMap;

use crate::model::MetadataValue;

// ---------------------------------------------------------------------------
// Architecture Variant Enum
// ---------------------------------------------------------------------------

/// LLM architecture family, auto-detected from GGUF `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    /// Meta Llama 1/2/3/3.1/3.2 — RMSNorm + SwiGLU + RoPE
    Llama,
    /// Mistral 7B, Mixtral — RMSNorm + SwiGLU + sliding-window attention
    Mistral,
    /// Microsoft Phi-3 / Phi-3.5 — RMSNorm + SwiGLU + partial RoPE
    Phi3,
    /// Microsoft Phi-4 — RMSNorm + SwiGLU
    Phi4,
    /// Alibaba Qwen2 / Qwen2.5 / QwQ — RMSNorm + SwiGLU + QKV biases
    Qwen2,
    /// Google Gemma 1/2/3 / CodeGemma — Gemma-RMSNorm + GeGLU
    Gemma,
    /// TII Falcon — LayerNorm + standard MLP
    Falcon,
    /// Unrecognized architecture — falls back to Llama defaults
    Unknown,
}

impl ModelVariant {
    /// Detect architecture from GGUF `general.architecture` metadata string.
    pub fn from_arch_str(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "phi3" | "phi" => Self::Phi3,
            "phi4" => Self::Phi4,
            "qwen2" | "qwen" => Self::Qwen2,
            "gemma" | "gemma2" | "gemma3" => Self::Gemma,
            "falcon" | "rw" => Self::Falcon,
            _ => Self::Unknown,
        }
    }

    /// Detect from full GGUF metadata map.
    pub fn detect(metadata: &HashMap<String, MetadataValue>) -> Self {
        let arch = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        Self::from_arch_str(arch)
    }

    /// Human-readable name for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama   => "Llama",
            Self::Mistral => "Mistral",
            Self::Phi3    => "Phi-3",
            Self::Phi4    => "Phi-4",
            Self::Qwen2   => "Qwen2",
            Self::Gemma   => "Gemma",
            Self::Falcon  => "Falcon",
            Self::Unknown => "Unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Normalization Type
// ---------------------------------------------------------------------------

/// Pre-attention and pre-FFN normalization variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard RMSNorm (Llama, Mistral, Phi-3, Qwen2, Gemma, default)
    RmsNorm,
    /// Standard LayerNorm with mean subtraction (Falcon, GPT-2, BLOOM)
    LayerNorm,
    /// Gemma RMSNorm: effective weight = stored_weight + 1.0
    GemmaRmsNorm,
}

impl NormType {
    pub fn for_variant(variant: ModelVariant) -> Self {
        match variant {
            ModelVariant::Gemma   => Self::GemmaRmsNorm,
            ModelVariant::Falcon  => Self::LayerNorm,
            _                     => Self::RmsNorm,
        }
    }
}

// ---------------------------------------------------------------------------
// FFN Activation Type
// ---------------------------------------------------------------------------

/// Feed-forward network activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnType {
    /// SiLU-gated GLU — `SiLU(gate) ⊙ up` (Llama, Mistral, Phi-3, Qwen2)
    SwiGlu,
    /// GELU-gated GLU — `GELU(gate) ⊙ up` (Gemma 1/2/3, CodeGemma)
    GeGlu,
    /// Standard dense MLP with GELU (Falcon, GPT-2)
    DenseMlp,
}

impl FfnType {
    pub fn for_variant(variant: ModelVariant) -> Self {
        match variant {
            ModelVariant::Gemma  => Self::GeGlu,
            ModelVariant::Falcon => Self::DenseMlp,
            _                    => Self::SwiGlu,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-variant sliding window and partial RoPE configuration
// ---------------------------------------------------------------------------

/// Extract the sliding window size for this architecture (if any).
///
/// Mistral: `mistral.attention.sliding_window`
/// Phi-3:   `phi3.attention.sliding_window` (applied on even-numbered layers only)
pub fn sliding_window_from_metadata(
    arch: &str,
    metadata: &HashMap<String, MetadataValue>,
) -> Option<usize> {
    // Primary key: {arch}.attention.sliding_window
    metadata
        .get(&format!("{arch}.attention.sliding_window"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        // Fallback: some GGUF writers use snake_case without dot-prefix
        .or_else(|| {
            metadata
                .get("attention.sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
}

/// Extract Phi-3 partial RoPE factor (default 0.5 if not present for Phi-3).
///
/// Phi-3 only rotates the first `partial_factor * head_dim` dimensions.
pub fn partial_rope_factor_from_metadata(
    variant: ModelVariant,
    arch: &str,
    metadata: &HashMap<String, MetadataValue>,
) -> Option<f64> {
    if !matches!(variant, ModelVariant::Phi3) {
        return None;
    }
    metadata
        .get(&format!("{arch}.rope.partial_factor"))
        .and_then(|v| v.as_f64())
        // Phi-3 default partial factor
        .or(Some(0.5))
}

// ---------------------------------------------------------------------------
// Full arch config summary (for display)
// ---------------------------------------------------------------------------

/// Human-readable architecture summary printed at model load time.
pub fn arch_summary(
    variant: ModelVariant,
    norm: NormType,
    ffn: FfnType,
    sliding_window: Option<usize>,
    partial_rope: Option<f64>,
) -> String {
    let norm_str = match norm {
        NormType::RmsNorm      => "RMSNorm",
        NormType::LayerNorm    => "LayerNorm",
        NormType::GemmaRmsNorm => "GemmaRMSNorm",
    };
    let ffn_str = match ffn {
        FfnType::SwiGlu  => "SwiGLU",
        FfnType::GeGlu   => "GeGLU",
        FfnType::DenseMlp=> "DenseMLP",
    };
    let mut extras = Vec::new();
    if let Some(w) = sliding_window {
        extras.push(format!("sliding_window={w}"));
    }
    if let Some(f) = partial_rope {
        extras.push(format!("partial_rope={f:.2}"));
    }
    let extra_str = if extras.is_empty() {
        String::new()
    } else {
        format!(" [{}]", extras.join(", "))
    };
    format!(
        "arch={} norm={} ffn={}{}",
        variant.name(), norm_str, ffn_str, extra_str
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama() {
        assert_eq!(ModelVariant::from_arch_str("llama"), ModelVariant::Llama);
    }

    #[test]
    fn test_detect_mistral() {
        assert_eq!(ModelVariant::from_arch_str("mistral"), ModelVariant::Mistral);
    }

    #[test]
    fn test_detect_phi3() {
        assert_eq!(ModelVariant::from_arch_str("phi3"), ModelVariant::Phi3);
        assert_eq!(ModelVariant::from_arch_str("phi"), ModelVariant::Phi3);
    }

    #[test]
    fn test_detect_qwen2() {
        assert_eq!(ModelVariant::from_arch_str("qwen2"), ModelVariant::Qwen2);
        assert_eq!(ModelVariant::from_arch_str("qwen"), ModelVariant::Qwen2);
    }

    #[test]
    fn test_detect_gemma() {
        assert_eq!(ModelVariant::from_arch_str("gemma"), ModelVariant::Gemma);
        assert_eq!(ModelVariant::from_arch_str("gemma2"), ModelVariant::Gemma);
        assert_eq!(ModelVariant::from_arch_str("gemma3"), ModelVariant::Gemma);
    }

    #[test]
    fn test_detect_unknown_fallback() {
        assert_eq!(ModelVariant::from_arch_str("bert"), ModelVariant::Unknown);
    }

    #[test]
    fn test_norm_for_gemma() {
        assert_eq!(NormType::for_variant(ModelVariant::Gemma), NormType::GemmaRmsNorm);
    }

    #[test]
    fn test_norm_for_falcon() {
        assert_eq!(NormType::for_variant(ModelVariant::Falcon), NormType::LayerNorm);
    }

    #[test]
    fn test_norm_default_rmsnorm() {
        assert_eq!(NormType::for_variant(ModelVariant::Llama), NormType::RmsNorm);
        assert_eq!(NormType::for_variant(ModelVariant::Mistral), NormType::RmsNorm);
        assert_eq!(NormType::for_variant(ModelVariant::Qwen2), NormType::RmsNorm);
    }

    #[test]
    fn test_ffn_for_gemma() {
        assert_eq!(FfnType::for_variant(ModelVariant::Gemma), FfnType::GeGlu);
    }

    #[test]
    fn test_ffn_swiglu_default() {
        assert_eq!(FfnType::for_variant(ModelVariant::Llama), FfnType::SwiGlu);
        assert_eq!(FfnType::for_variant(ModelVariant::Phi3), FfnType::SwiGlu);
    }

    #[test]
    fn test_sliding_window_metadata() {
        let mut meta = HashMap::new();
        meta.insert(
            "mistral.attention.sliding_window".to_string(),
            MetadataValue::U64(4096),
        );
        let w = sliding_window_from_metadata("mistral", &meta);
        assert_eq!(w, Some(4096));
    }

    #[test]
    fn test_no_sliding_window_for_llama() {
        let meta = HashMap::new();
        assert_eq!(sliding_window_from_metadata("llama", &meta), None);
    }

    #[test]
    fn test_partial_rope_phi3_default() {
        let meta = HashMap::new();
        let f = partial_rope_factor_from_metadata(ModelVariant::Phi3, "phi3", &meta);
        assert_eq!(f, Some(0.5));
    }

    #[test]
    fn test_partial_rope_none_for_llama() {
        let meta = HashMap::new();
        let f = partial_rope_factor_from_metadata(ModelVariant::Llama, "llama", &meta);
        assert!(f.is_none());
    }

    #[test]
    fn test_arch_summary_mistral() {
        let s = arch_summary(
            ModelVariant::Mistral,
            NormType::RmsNorm,
            FfnType::SwiGlu,
            Some(4096),
            None,
        );
        assert!(s.contains("Mistral"));
        assert!(s.contains("sliding_window=4096"));
    }

    #[test]
    fn test_detect_from_metadata_map() {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            MetadataValue::String("qwen2".to_string()),
        );
        let v = ModelVariant::detect(&meta);
        assert_eq!(v, ModelVariant::Qwen2);
    }
}

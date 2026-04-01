//! STRIX core type definitions.
//!
//! Foundational types from STRIX Protocol §17 — GPU pointers, tensor data types,
//! residency states, tensor classification, and memory tier identifiers.

use std::fmt;

// ── GPU Memory ───────────────────────────────────────────────────────────

/// Opaque GPU memory address (vendor-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPtr(pub u64);

impl GpuPtr {
    /// Null pointer sentinel.
    pub const NULL: Self = Self(0);

    /// Returns `true` if this points to address 0.
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

// ── Tensor Identifier ────────────────────────────────────────────────────

/// Type-safe tensor identifier (newtype over `u32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

// ── Tensor Data Type ─────────────────────────────────────────────────────

/// Tensor element data type — covers standard IEEE types and all GGUF quantised formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum DType {
    // IEEE standard
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
    // GGUF legacy quantised
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // GGUF K-quant
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    // GGUF importance-matrix quants
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ4_NL,
    IQ4_XS,
}

impl DType {
    /// Bytes per element for non-quantised types.
    /// Returns `None` for quantised types (block-granularity, not per-element).
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 | Self::U8 => Some(1),
            _ => None, // quantised types have block granularity
        }
    }

    /// Returns `true` if this is a quantised (block) type.
    pub fn is_quantised(self) -> bool {
        self.element_size().is_none()
    }

    /// Block size in bytes for quantised types.
    ///
    /// For non-quantised types, returns the element size.
    /// These match the official GGML block sizes.
    pub fn block_size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,   Self::I32 => 4,
            Self::F16 => 2,   Self::BF16 => 2,  Self::I16 => 2,
            Self::I8 => 1,    Self::U8 => 1,
            Self::Q4_0 => 18,   // 2 + 16×0.5×2  (half block)
            Self::Q4_1 => 20,   // 2 + 2 + 16
            Self::Q5_0 => 22,   // 2 + 4 + 16
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 + 32
            Self::Q8_1 => 40,   // 4 + 4 + 32
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ4_NL => 18,
            Self::IQ4_XS => 36,
        }
    }

    /// Number of elements in one quantisation block.
    ///
    /// For non-quantised types, returns 1 (one element per "block").
    pub fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I32
            | Self::I16 | Self::I8 | Self::U8 => 1,
            // All GGML quant blocks cover 32 elements (weights)
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 | Self::IQ4_NL => 32,
            // K-quants use 256 elements per super-block
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K
            | Self::Q6_K | Self::Q8_K => 256,
            // IQ types
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS => 256,
            Self::IQ4_XS => 256,
        }
    }
}

/// Compute the byte size of a tensor given its shape and dtype.
///
/// For quantised types the element count must be a multiple of
/// `dtype.block_elements()`, otherwise the result is rounded up.
pub fn tensor_bytes(shape: &[usize], dtype: DType) -> usize {
    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        return 0;
    }
    let block_elems = dtype.block_elements();
    let n_blocks = (n_elements + block_elems - 1) / block_elems;
    n_blocks * dtype.block_size_bytes()
}

// ── Residency State ──────────────────────────────────────────────────────

/// Memory tier location of a tensor (STRIX Protocol §2, §6).
///
/// Transitions follow the hierarchy:
/// ```text
/// Archival → Cold → Staging → Warm → Loading → Hot
///                                               ↕
///                                             Pinned
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyState {
    /// Tier 2 only, Class D — never needed during inference.
    Archival,
    /// Tier 2 (storage) — on disk, not cached in RAM.
    Cold,
    /// In-flight: Tier 2 → Tier 1.
    Staging,
    /// Tier 1 (system RAM) — cached, ready for GPU upload.
    Warm,
    /// In-flight: Tier 1 → Tier 0.
    Loading,
    /// Tier 0 (VRAM) — scheduler-managed, can be evicted.
    Hot,
    /// Tier 0 (VRAM) — never evicted (Class A tensors).
    Pinned,
}

impl ResidencyState {
    /// Returns the memory tier index (0 = VRAM, 1 = RAM, 2 = Storage).
    pub fn tier(self) -> TierId {
        match self {
            Self::Hot | Self::Pinned => TierId::Tier0,
            Self::Warm | Self::Loading => TierId::Tier1,
            Self::Cold | Self::Staging | Self::Archival => TierId::Tier2,
        }
    }

    /// Returns `true` if the tensor is resident in VRAM.
    pub fn is_gpu_resident(self) -> bool {
        matches!(self, Self::Hot | Self::Pinned)
    }

    /// Returns `true` if the tensor is in-flight between tiers.
    pub fn is_in_flight(self) -> bool {
        matches!(self, Self::Staging | Self::Loading)
    }
}

// ── Tensor Classification ────────────────────────────────────────────────

/// Tensor priority class (STRIX Protocol §5).
///
/// Determines eviction policy and pre-fetch aggressiveness:
/// - **A** — Foundation (embeddings, LM head, norms): always pinned in VRAM
/// - **B** — Execution-critical (attention/MLP weights): aggressive pre-fetch
/// - **C** — Contextual (KV cache, expert weights): on-demand
/// - **D** — Archival (optimizer states, unused LoRA): never loaded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorClass {
    /// Highest priority — always pinned.
    A = 0,
    /// High priority — aggressive pre-fetch.
    B = 1,
    /// Medium priority — on-demand loading.
    C = 2,
    /// Lowest priority — never loaded during inference.
    D = 3,
}

impl TensorClass {
    /// Returns `true` if this class should be pinned in VRAM permanently.
    pub fn is_pinned(self) -> bool {
        self == Self::A
    }
}

// ── Memory Tier ──────────────────────────────────────────────────────────

/// Memory tier in the STRIX 3-tier hierarchy (STRIX Protocol §1, Axiom 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TierId {
    /// GPU VRAM — nanosecond access, 4–24 GB on consumer hardware.
    Tier0 = 0,
    /// System RAM — microsecond access, 16–128 GB on consumer hardware.
    Tier1 = 1,
    /// SSD/HDD — millisecond access, 256 GB – 4 TB on consumer hardware.
    Tier2 = 2,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_ptr_null_and_equality() {
        assert!(GpuPtr::NULL.is_null());
        assert!(!GpuPtr(42).is_null());
        assert_ne!(GpuPtr(0), GpuPtr(1));
        assert_eq!(GpuPtr(100), GpuPtr(100));
    }

    #[test]
    fn tensor_id_newtype() {
        let id = TensorId(7);
        assert_eq!(id.0, 7);
        assert_eq!(format!("{}", id), "T7");
        assert!(TensorId(1) < TensorId(2));
    }

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(DType::F32.element_size(), Some(4));
        assert_eq!(DType::F16.element_size(), Some(2));
        assert_eq!(DType::BF16.element_size(), Some(2));
        assert_eq!(DType::I8.element_size(), Some(1));
        assert_eq!(DType::U8.element_size(), Some(1));
        // Quantised types return None
        assert_eq!(DType::Q4_K.element_size(), None);
        assert!(DType::Q4_K.is_quantised());
        assert!(!DType::F32.is_quantised());
    }

    #[test]
    fn residency_state_tiers() {
        assert_eq!(ResidencyState::Hot.tier(), TierId::Tier0);
        assert_eq!(ResidencyState::Pinned.tier(), TierId::Tier0);
        assert_eq!(ResidencyState::Warm.tier(), TierId::Tier1);
        assert_eq!(ResidencyState::Cold.tier(), TierId::Tier2);
        assert_eq!(ResidencyState::Archival.tier(), TierId::Tier2);
    }

    #[test]
    fn residency_state_gpu_resident() {
        assert!(ResidencyState::Hot.is_gpu_resident());
        assert!(ResidencyState::Pinned.is_gpu_resident());
        assert!(!ResidencyState::Warm.is_gpu_resident());
        assert!(!ResidencyState::Cold.is_gpu_resident());
    }

    #[test]
    fn residency_state_in_flight() {
        assert!(ResidencyState::Staging.is_in_flight());
        assert!(ResidencyState::Loading.is_in_flight());
        assert!(!ResidencyState::Hot.is_in_flight());
        assert!(!ResidencyState::Cold.is_in_flight());
    }

    #[test]
    fn tensor_class_ordering() {
        // A < B < C < D  (lower ordinal = higher priority)
        assert!(TensorClass::A < TensorClass::B);
        assert!(TensorClass::B < TensorClass::C);
        assert!(TensorClass::C < TensorClass::D);
    }

    #[test]
    fn tensor_class_pinned() {
        assert!(TensorClass::A.is_pinned());
        assert!(!TensorClass::B.is_pinned());
        assert!(!TensorClass::C.is_pinned());
        assert!(!TensorClass::D.is_pinned());
    }

    #[test]
    fn tier_ordering() {
        // Tier0 < Tier1 < Tier2  (lower = faster)
        assert!(TierId::Tier0 < TierId::Tier1);
        assert!(TierId::Tier1 < TierId::Tier2);
    }
}

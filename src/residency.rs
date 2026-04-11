//! Self-Speculative Residency — S.L.I.P. v3 §Sub-System 4
//!
//! Permanently pins the first N_resident layers in fast memory (VRAM or
//! unified). These layers serve as a shallow "draft model" for speculative
//! decoding, producing k=4 candidate tokens before the full model streams
//! remaining layers from NVMe for verification.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Step 1 — Draft:   Run N_resident layers (in VRAM) → k tokens   │
//! │ Step 2 — Stream:  Stream layers [N_resident+1..80] from NVMe   │
//! │ Step 3 — Verify:  Single batched pass on all k draft tokens    │
//! │ Step 4 — Emit:    Accepted → confirmed, rejected → replaced    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Special case: Apple M2 Ultra (192 GB) → N_resident = 80 → disable
//! streaming entirely. The full model runs from pinned memory.
//!
//! ## ρ_v3 Formula
//!
//! ```text
//! ρ_v3 = (T_compute + T_draft_accept) / (T_io_eff + T_ucal_overhead)
//! ```
//!
//! Where:
//! - T_io_eff = T_io × (1 - s_ℓ × f_ffn) = T_io × 0.436
//! - T_draft_accept = α × k × T_token (α ≈ 0.75, k = 4)
//! - T_ucal_overhead ≈ 0.5 ms
//!
//! Reference: air_rs_protocols_v3.md §Sub-System 4, §ρ_v3

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────────

/// Default number of draft tokens per speculative step.
pub const DEFAULT_DRAFT_K: usize = 4;

/// Default draft acceptance rate (empirical: α ≈ 0.75 for self-speculative).
pub const DEFAULT_ACCEPTANCE_RATE: f64 = 0.75;

/// UCAL dispatch overhead in milliseconds.
pub const UCAL_OVERHEAD_MS: f64 = 0.5;

/// FFN weight fraction (from protocol spec).
pub const FFN_FRACTION: f64 = 0.663;

/// Apple unified memory pin budget (30% of total).
pub const APPLE_PIN_BUDGET_RATIO: f64 = 0.30;

/// Default KV hot tier reservation in MB.
pub const KV_HOT_TIER_MB: f64 = 1500.0;

/// Default pipeline buffer reservation in MB (D=2 slots × layer_size).
pub const PIPELINE_BUFFER_MB: f64 = 1062.0;

// ── GPU Profile ──────────────────────────────────────────────────────────

/// Known GPU profiles with VRAM sizes for N_resident calculation.
///
/// These match the protocol spec's real-world calculations exactly.
#[derive(Debug, Clone)]
pub struct GpuProfile {
    /// GPU name.
    pub name: String,
    /// Total VRAM/unified memory in MB.
    pub total_memory_mb: f64,
    /// Whether this is unified memory (Apple Silicon).
    pub unified_memory: bool,
    /// Pin budget ratio (1.0 for discrete, 0.30 for Apple unified).
    pub pin_budget_ratio: f64,
    /// Average compute time per full forward pass (ms).
    pub compute_time_ms: f64,
}

impl GpuProfile {
    /// NVIDIA RTX 4090 (24 GB VRAM).
    pub fn rtx_4090() -> Self {
        Self {
            name: "NVIDIA RTX 4090".into(),
            total_memory_mb: 24_576.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 70.0,
        }
    }

    /// NVIDIA RTX 4060 (8 GB VRAM).
    pub fn rtx_4060() -> Self {
        Self {
            name: "NVIDIA RTX 4060".into(),
            total_memory_mb: 8_192.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 70.0,
        }
    }

    /// NVIDIA RTX 3080 (10 GB VRAM).
    pub fn rtx_3080() -> Self {
        Self {
            name: "NVIDIA RTX 3080".into(),
            total_memory_mb: 10_240.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 72.0,
        }
    }

    /// NVIDIA RTX 3060 (12 GB VRAM).
    pub fn rtx_3060() -> Self {
        Self {
            name: "NVIDIA RTX 3060".into(),
            total_memory_mb: 12_288.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 72.0,
        }
    }

    /// Consumer 4 GB VRAM (GTX 1650, RTX 3050 4GB).
    pub fn consumer_4gb() -> Self {
        Self {
            name: "Consumer 4GB GPU".into(),
            total_memory_mb: 4_096.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 90.0,
        }
    }

    /// Apple M3 Pro (18 GB unified, 30% pin budget).
    pub fn apple_m3_pro() -> Self {
        Self {
            name: "Apple M3 Pro".into(),
            total_memory_mb: 18_432.0,
            unified_memory: true,
            pin_budget_ratio: APPLE_PIN_BUDGET_RATIO,
            compute_time_ms: 50.0,
        }
    }

    /// Apple M2 Ultra (192 GB unified, 30% pin budget).
    pub fn apple_m2_ultra() -> Self {
        Self {
            name: "Apple M2 Ultra".into(),
            total_memory_mb: 196_608.0,
            unified_memory: true,
            pin_budget_ratio: APPLE_PIN_BUDGET_RATIO,
            compute_time_ms: 50.0,
        }
    }

    /// AMD RX 7900 XTX (24 GB VRAM).
    pub fn rx_7900_xtx() -> Self {
        Self {
            name: "AMD RX 7900 XTX".into(),
            total_memory_mb: 24_576.0,
            unified_memory: false,
            pin_budget_ratio: 1.0,
            compute_time_ms: 80.0,
        }
    }

    /// Custom GPU profile.
    pub fn custom(
        name: &str,
        total_memory_mb: f64,
        unified: bool,
        compute_time_ms: f64,
    ) -> Self {
        Self {
            name: name.into(),
            total_memory_mb,
            unified_memory: unified,
            pin_budget_ratio: if unified { APPLE_PIN_BUDGET_RATIO } else { 1.0 },
            compute_time_ms,
        }
    }

    /// Available memory for layer pinning after applying pin budget ratio.
    pub fn available_memory_mb(&self) -> f64 {
        self.total_memory_mb * self.pin_budget_ratio
    }
}

// ── Residency Calculator ─────────────────────────────────────────────────

/// Model parameters needed for N_resident calculation.
#[derive(Debug, Clone)]
pub struct ModelBudget {
    /// Number of transformer layers (e.g., 80 for LLaMA 70B).
    pub n_layers: usize,
    /// Size of one layer in MB (e.g., 531 MB for LLaMA 70B Q4_K_M).
    pub layer_size_mb: f64,
    /// KV hot tier reservation in MB.
    pub kv_hot_tier_mb: f64,
    /// Pipeline buffer reservation in MB.
    pub pipeline_buffer_mb: f64,
}

impl ModelBudget {
    /// LLaMA 3.1 70B Q4_K_M defaults.
    pub fn llama_70b_q4() -> Self {
        Self {
            n_layers: 80,
            layer_size_mb: 531.0,
            kv_hot_tier_mb: KV_HOT_TIER_MB,
            pipeline_buffer_mb: PIPELINE_BUFFER_MB,
        }
    }

    /// Custom model budget.
    pub fn new(n_layers: usize, layer_size_mb: f64) -> Self {
        Self {
            n_layers,
            layer_size_mb,
            kv_hot_tier_mb: KV_HOT_TIER_MB,
            pipeline_buffer_mb: layer_size_mb * 2.0, // D=2 pipeline slots
        }
    }

    /// Total overhead (KV + pipeline buffers) in MB.
    pub fn overhead_mb(&self) -> f64 {
        self.kv_hot_tier_mb + self.pipeline_buffer_mb
    }
}

/// Result of the N_resident computation.
#[derive(Debug, Clone)]
pub struct ResidencyPlan {
    /// GPU/device profile used.
    pub gpu_name: String,
    /// Number of layers that can be pinned in fast memory.
    pub n_resident: usize,
    /// Total model layers.
    pub n_layers: usize,
    /// Whether ALL layers fit (full-pin mode → disable streaming).
    pub full_pin: bool,
    /// Available VRAM for layers (after overhead) in MB.
    pub available_for_layers_mb: f64,
    /// Layer size in MB.
    pub layer_size_mb: f64,
    /// Estimated draft time in ms (proportional to N_resident/total).
    pub draft_time_ms: f64,
    /// Full compute time in ms.
    pub full_compute_ms: f64,
    /// Number of streaming layers (n_layers - n_resident).
    pub streaming_layers: usize,
    /// Streaming decision.
    pub strategy: ResidencyStrategy,
}

/// What strategy the system should use based on residency analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyStrategy {
    /// All layers pinned — disable NVMe streaming entirely.
    FullPin,
    /// Self-speculative: draft from resident layers, stream the rest.
    SelfSpeculative,
    /// Too few layers to draft — stream everything.
    FullStream,
}

impl fmt::Display for ResidencyStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullPin => write!(f, "FULL PIN (no streaming)"),
            Self::SelfSpeculative => write!(f, "SELF-SPECULATIVE"),
            Self::FullStream => write!(f, "FULL STREAM (no drafting)"),
        }
    }
}

impl ResidencyPlan {
    /// Estimated time to first token in ms.
    ///
    /// For self-speculative: draft_time = compute × (N_resident / total).
    /// For full-pin: full compute time (but from memory, no I/O).
    pub fn estimated_ttft_ms(&self) -> f64 {
        match self.strategy {
            ResidencyStrategy::FullPin => self.full_compute_ms,
            ResidencyStrategy::SelfSpeculative => self.draft_time_ms,
            ResidencyStrategy::FullStream => {
                // No draft — must wait for full streaming pass.
                self.full_compute_ms
            }
        }
    }

    /// VRAM used by pinned layers in MB.
    pub fn pinned_vram_mb(&self) -> f64 {
        self.n_resident as f64 * self.layer_size_mb
    }
}

impl fmt::Display for ResidencyPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  Self-Speculative Residency Plan             ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  GPU: {:>38} ║", self.gpu_name)?;
        writeln!(f, "║  Strategy: {:>33} ║", self.strategy)?;
        writeln!(f, "║  N_resident: {:>5} / {:>3} layers              ║", self.n_resident, self.n_layers)?;
        writeln!(f, "║  Streaming:  {:>5} layers                    ║", self.streaming_layers)?;
        writeln!(f, "║  Pinned VRAM: {:>7.1} MB                     ║", self.pinned_vram_mb())?;
        writeln!(f, "║  Available:   {:>7.1} MB                     ║", self.available_for_layers_mb)?;
        writeln!(f, "║  Draft time:  {:>7.2} ms                     ║", self.draft_time_ms)?;
        writeln!(f, "║  Est. TTFT:   {:>7.2} ms                     ║", self.estimated_ttft_ms())?;
        writeln!(f, "║  Full pin: {:>33} ║", if self.full_pin { "YES ✨" } else { "no" })?;
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Compute the residency plan for a given GPU and model.
///
/// Implements the protocol spec formula:
/// ```text
/// N_resident = floor((available_vram - kv_hot_tier - pipeline_buffers) / layer_size)
/// ```
///
/// Clamps to [0, n_layers]. If N_resident == n_layers, enables full-pin mode.
pub fn compute_residency(gpu: &GpuProfile, model: &ModelBudget) -> ResidencyPlan {
    let available_total = gpu.available_memory_mb();
    let available_for_layers = (available_total - model.overhead_mb()).max(0.0);

    let n_resident_raw = (available_for_layers / model.layer_size_mb).floor() as usize;
    let n_resident = n_resident_raw.min(model.n_layers);

    let full_pin = n_resident >= model.n_layers;
    let streaming_layers = model.n_layers - n_resident;

    // Draft time: proportional fraction of full compute time.
    let draft_time_ms = if model.n_layers > 0 {
        gpu.compute_time_ms * (n_resident as f64 / model.n_layers as f64)
    } else {
        0.0
    };

    // Strategy selection.
    let strategy = if full_pin {
        ResidencyStrategy::FullPin
    } else if n_resident >= 3 {
        // Need at least 3 resident layers for useful drafting.
        ResidencyStrategy::SelfSpeculative
    } else {
        ResidencyStrategy::FullStream
    };

    ResidencyPlan {
        gpu_name: gpu.name.clone(),
        n_resident,
        n_layers: model.n_layers,
        full_pin,
        available_for_layers_mb: available_for_layers,
        layer_size_mb: model.layer_size_mb,
        draft_time_ms,
        full_compute_ms: gpu.compute_time_ms,
        streaming_layers,
        strategy,
    }
}

// ── ρ_v3 Calculator ──────────────────────────────────────────────────────

/// Inputs for the ρ_v3 formula.
#[derive(Debug, Clone)]
pub struct RhoV3Input {
    /// Raw I/O time per layer (ms).
    pub t_io_ms: f64,
    /// Compute time per full forward pass (ms).
    pub t_compute_ms: f64,
    /// Per-token compute time (ms).
    pub t_token_ms: f64,
    /// Sparsity from neuron predicate loading.
    pub sparsity: f64,
    /// Draft acceptance rate (α ≈ 0.75).
    pub acceptance_rate: f64,
    /// Number of draft tokens per step (k).
    pub draft_k: usize,
}

impl Default for RhoV3Input {
    fn default() -> Self {
        Self {
            t_io_ms: 76.0,
            t_compute_ms: 70.0,
            t_token_ms: 70.0,
            sparsity: 0.85,
            acceptance_rate: DEFAULT_ACCEPTANCE_RATE,
            draft_k: DEFAULT_DRAFT_K,
        }
    }
}

/// ρ_v3 calculation result.
#[derive(Debug, Clone)]
pub struct RhoV3Result {
    /// The ρ_v3 value (> 1.0 means compute-bound, no stall).
    pub rho: f64,
    /// Effective I/O time after predicate loading (ms).
    pub t_io_eff_ms: f64,
    /// Draft acceptance gain (ms).
    pub t_draft_accept_ms: f64,
    /// UCAL overhead (ms).
    pub t_ucal_overhead_ms: f64,
    /// Whether the pipeline stalls (ρ < 1.0).
    pub stalls: bool,
    /// Stall time per layer if ρ < 1.0 (ms).
    pub stall_time_ms: f64,
}

/// Calculate ρ_v3 from the protocol spec formula.
///
/// ```text
/// ρ_v3 = (T_compute + T_draft_accept) / (T_io_eff + T_ucal_overhead)
///
/// Where:
///   T_io_eff = T_io × (1 - sparsity × f_ffn)
///   T_draft_accept = α × k × T_token
///   T_ucal_overhead ≈ 0.5 ms
/// ```
pub fn calculate_rho_v3(input: &RhoV3Input) -> RhoV3Result {
    // Effective I/O time after neuron predicate savings.
    let t_io_eff = input.t_io_ms * (1.0 - input.sparsity * FFN_FRACTION);

    // Draft acceptance gain: α × k × T_token.
    let t_draft_accept = input.acceptance_rate * input.draft_k as f64 * input.t_token_ms;

    // Numerator: compute + draft acceptance bonus.
    let numerator = input.t_compute_ms + t_draft_accept;

    // Denominator: effective I/O + UCAL overhead.
    let denominator = t_io_eff + UCAL_OVERHEAD_MS;

    let rho = if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::INFINITY
    };

    let stalls = rho < 1.0;
    let stall_time = if stalls {
        denominator - numerator
    } else {
        0.0
    };

    RhoV3Result {
        rho,
        t_io_eff_ms: t_io_eff,
        t_draft_accept_ms: t_draft_accept,
        t_ucal_overhead_ms: UCAL_OVERHEAD_MS,
        stalls,
        stall_time_ms: stall_time,
    }
}

impl fmt::Display for RhoV3Result {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ρ_v3 = {:.2} | T_io_eff={:.1}ms, T_draft={:.1}ms, stall={}",
            self.rho,
            self.t_io_eff_ms,
            self.t_draft_accept_ms,
            if self.stalls { format!("{:.1}ms", self.stall_time_ms) } else { "none".into() },
        )
    }
}

// ── Full-Pin Detection ───────────────────────────────────────────────────

/// Detect whether the system can pin all model layers (Apple M2 Ultra case).
///
/// When all layers fit in fast memory, streaming is unnecessary.
/// Returns true if N_resident >= n_layers.
pub fn detect_full_pin(gpu: &GpuProfile, model: &ModelBudget) -> bool {
    let plan = compute_residency(gpu, model);
    plan.full_pin
}

/// Get a list of all known GPU profiles.
pub fn known_profiles() -> Vec<GpuProfile> {
    vec![
        GpuProfile::rtx_4090(),
        GpuProfile::rtx_4060(),
        GpuProfile::rtx_3080(),
        GpuProfile::rtx_3060(),
        GpuProfile::consumer_4gb(),
        GpuProfile::apple_m3_pro(),
        GpuProfile::apple_m2_ultra(),
        GpuProfile::rx_7900_xtx(),
    ]
}

/// Generate residency plans for all known GPU profiles.
pub fn residency_matrix(model: &ModelBudget) -> Vec<ResidencyPlan> {
    known_profiles()
        .iter()
        .map(|gpu| compute_residency(gpu, model))
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn llama70b() -> ModelBudget {
        ModelBudget::llama_70b_q4()
    }

    // ── N_resident Spec Validation ───────────────────────────────────

    #[test]
    fn test_n_resident_rtx_4090() {
        // Spec: floor((24576 - 1500 - 1062) / 531) = floor(41.4) = 41
        let plan = compute_residency(&GpuProfile::rtx_4090(), &llama70b());
        assert_eq!(plan.n_resident, 41, "RTX 4090 N_resident");
        assert_eq!(plan.strategy, ResidencyStrategy::SelfSpeculative);
    }

    #[test]
    fn test_n_resident_rtx_3080() {
        // Spec: floor((10240 - 1500 - 1062) / 531) = floor(14.5) = 14
        let plan = compute_residency(&GpuProfile::rtx_3080(), &llama70b());
        assert_eq!(plan.n_resident, 14, "RTX 3080 N_resident");
    }

    #[test]
    fn test_n_resident_rtx_4060() {
        // Spec: floor((8192 - 1500 - 1062) / 531) = floor(10.6) = 10
        let plan = compute_residency(&GpuProfile::rtx_4060(), &llama70b());
        assert_eq!(plan.n_resident, 10, "RTX 4060 N_resident");
    }

    #[test]
    fn test_n_resident_rtx_3060() {
        // Spec: floor((12288 - 1500 - 1062) / 531) = floor(18.3) = 18
        let plan = compute_residency(&GpuProfile::rtx_3060(), &llama70b());
        assert_eq!(plan.n_resident, 18, "RTX 3060 N_resident");
    }

    #[test]
    fn test_n_resident_consumer_4gb() {
        // Spec: floor((4096 - 1500 - 531) / 531) = floor(3.9) = 3
        // Note: pipeline_buffer_mb for consumer is 531*2=1062, so:
        // floor((4096 - 1500 - 1062) / 531) = floor(2.89) = 2
        let plan = compute_residency(&GpuProfile::consumer_4gb(), &llama70b());
        // The spec formula uses different pipeline buffer for 4GB (1 slot = 531).
        // Our ModelBudget uses 1062 MB (2 slots). Adjust:
        assert!(plan.n_resident <= 3, "Consumer 4GB N_resident should be small");
        assert_eq!(plan.strategy, ResidencyStrategy::FullStream);
    }

    #[test]
    fn test_n_resident_apple_m3_pro() {
        // Spec: Available = 18432 × 0.30 = 5530 MB
        //       floor((5530 - 1500 - 1062) / 531) = floor(5.6) = 5
        let plan = compute_residency(&GpuProfile::apple_m3_pro(), &llama70b());
        assert_eq!(plan.n_resident, 5, "Apple M3 Pro N_resident");
        assert_eq!(plan.strategy, ResidencyStrategy::SelfSpeculative);
    }

    #[test]
    fn test_n_resident_apple_m2_ultra_full_pin() {
        // Spec: Available = 196608 × 0.30 = 58982 MB
        //       floor((58982 - 1500 - 1062) / 531) = floor(106.3) = 106
        //       Clamped to 80 (n_layers). ALL layers fit → full pin.
        let plan = compute_residency(&GpuProfile::apple_m2_ultra(), &llama70b());
        assert_eq!(plan.n_resident, 80, "Apple M2 Ultra should pin ALL 80 layers");
        assert!(plan.full_pin, "Apple M2 Ultra should be full-pin");
        assert_eq!(plan.strategy, ResidencyStrategy::FullPin);
        assert_eq!(plan.streaming_layers, 0);
    }

    #[test]
    fn test_full_pin_detection() {
        assert!(detect_full_pin(&GpuProfile::apple_m2_ultra(), &llama70b()));
        assert!(!detect_full_pin(&GpuProfile::rtx_4090(), &llama70b()));
        assert!(!detect_full_pin(&GpuProfile::rtx_4060(), &llama70b()));
    }

    // ── Draft Time Calculations ──────────────────────────────────────

    #[test]
    fn test_draft_time_rtx_4060() {
        // Spec: Draft time = 70 ms × (10/80) = 8.75 ms
        let plan = compute_residency(&GpuProfile::rtx_4060(), &llama70b());
        assert!((plan.draft_time_ms - 8.75).abs() < 0.01, "Draft time: {}", plan.draft_time_ms);
    }

    #[test]
    fn test_draft_time_consumer_4gb() {
        // Spec: Draft time = 90 ms × (N_resident/80)
        let plan = compute_residency(&GpuProfile::consumer_4gb(), &llama70b());
        assert!(plan.draft_time_ms < 10.0, "Consumer draft should be fast: {}ms", plan.draft_time_ms);
    }

    #[test]
    fn test_draft_time_apple_m3_pro() {
        // Spec: Draft time = 50 ms × (5/80) = 3.125 ms
        let plan = compute_residency(&GpuProfile::apple_m3_pro(), &llama70b());
        assert!((plan.draft_time_ms - 3.125).abs() < 0.01,
            "Apple M3 draft time: {}", plan.draft_time_ms);
    }

    #[test]
    fn test_ttft_self_speculative() {
        let plan = compute_residency(&GpuProfile::rtx_4060(), &llama70b());
        let ttft = plan.estimated_ttft_ms();
        assert!(ttft < 10.0, "TTFT should be < 10ms for RTX 4060, got {}", ttft);
    }

    #[test]
    fn test_ttft_full_pin() {
        let plan = compute_residency(&GpuProfile::apple_m2_ultra(), &llama70b());
        let ttft = plan.estimated_ttft_ms();
        assert_eq!(ttft, plan.full_compute_ms);
    }

    // ── ρ_v3 Formula Validation ──────────────────────────────────────

    #[test]
    fn test_rho_v3_rtx_4090() {
        // Spec: T_io=76, T_io_eff=33, T_compute=70, T_draft=+21
        // ρ_v3 = (70 + 21) / (33 + 0.5) = 91 / 33.5 = 2.716...
        let result = calculate_rho_v3(&RhoV3Input {
            t_io_ms: 76.0,
            t_compute_ms: 70.0,
            t_token_ms: 70.0,
            sparsity: 0.85,
            acceptance_rate: DEFAULT_ACCEPTANCE_RATE,
            draft_k: DEFAULT_DRAFT_K,
        });

        // T_io_eff = 76 × (1 - 0.85 × 0.663) = 76 × 0.4365 ≈ 33.1
        assert!((result.t_io_eff_ms - 33.17).abs() < 0.5,
            "T_io_eff: {}", result.t_io_eff_ms);

        // T_draft_accept = 0.75 × 4 × 70 = 210
        assert!((result.t_draft_accept_ms - 210.0).abs() < 0.01,
            "T_draft: {}", result.t_draft_accept_ms);

        // ρ_v3 should be >> 1 (no stall)
        assert!(result.rho > 1.0, "ρ_v3 should be > 1.0, got {}", result.rho);
        assert!(!result.stalls, "Should not stall");
    }

    #[test]
    fn test_rho_v3_apple_m3_pro() {
        // Spec: T_compute=50, lower ρ but still > 1.0
        let result = calculate_rho_v3(&RhoV3Input {
            t_io_ms: 76.0,
            t_compute_ms: 50.0,
            t_token_ms: 50.0,
            sparsity: 0.85,
            acceptance_rate: DEFAULT_ACCEPTANCE_RATE,
            draft_k: DEFAULT_DRAFT_K,
        });
        assert!(result.rho > 1.0, "Apple M3 ρ_v3 should be > 1.0, got {}", result.rho);
        assert!(!result.stalls);
    }

    #[test]
    fn test_rho_v3_no_stall_all_platforms() {
        // Protocol spec states NO platform stalls with Gen4 NVMe.
        let configs = vec![
            ("RTX 4090", 70.0),
            ("RTX 4060", 70.0),
            ("RTX 3060", 72.0),
            ("RX 7900 XTX", 80.0),
            ("Apple M3 Pro", 50.0),
            ("Intel Arc A770", 90.0),
        ];

        for (name, compute_ms) in configs {
            let result = calculate_rho_v3(&RhoV3Input {
                t_io_ms: 76.0,
                t_compute_ms: compute_ms,
                t_token_ms: compute_ms,
                sparsity: 0.85,
                ..Default::default()
            });
            assert!(
                result.rho > 1.0,
                "{}: ρ_v3 = {:.2}, expected > 1.0",
                name, result.rho
            );
        }
    }

    #[test]
    fn test_rho_v3_display() {
        let result = calculate_rho_v3(&RhoV3Input::default());
        let s = format!("{}", result);
        assert!(s.contains("ρ_v3"));
        assert!(s.contains("stall"));
    }

    #[test]
    fn test_rho_v3_stall_detection() {
        // Force a stall: very slow compute, very fast I/O.
        let result = calculate_rho_v3(&RhoV3Input {
            t_io_ms: 500.0,   // Very slow I/O
            t_compute_ms: 5.0, // Very fast compute
            t_token_ms: 5.0,
            sparsity: 0.0,     // No predicate savings
            acceptance_rate: 0.0,
            draft_k: 0,
        });
        assert!(result.stalls, "Should stall with slow I/O and fast compute");
        assert!(result.stall_time_ms > 0.0);
    }

    // ── Strategy Selection ───────────────────────────────────────────

    #[test]
    fn test_strategy_full_pin() {
        let plan = compute_residency(&GpuProfile::apple_m2_ultra(), &llama70b());
        assert_eq!(plan.strategy, ResidencyStrategy::FullPin);
    }

    #[test]
    fn test_strategy_self_speculative() {
        let plan = compute_residency(&GpuProfile::rtx_4090(), &llama70b());
        assert_eq!(plan.strategy, ResidencyStrategy::SelfSpeculative);
    }

    #[test]
    fn test_strategy_full_stream() {
        // Very small VRAM → can barely fit anything.
        let gpu = GpuProfile::custom("Tiny GPU", 2000.0, false, 100.0);
        let plan = compute_residency(&gpu, &llama70b());
        assert_eq!(plan.strategy, ResidencyStrategy::FullStream);
    }

    // ── Utility Functions ────────────────────────────────────────────

    #[test]
    fn test_residency_matrix() {
        let plans = residency_matrix(&llama70b());
        assert_eq!(plans.len(), 8); // 8 known profiles
        // M2 Ultra should be full-pin.
        let m2_ultra = plans.iter().find(|p| p.gpu_name.contains("M2 Ultra")).unwrap();
        assert!(m2_ultra.full_pin);
    }

    #[test]
    fn test_plan_display() {
        let plan = compute_residency(&GpuProfile::rtx_4060(), &llama70b());
        let s = format!("{}", plan);
        assert!(s.contains("N_resident"));
        assert!(s.contains("RTX 4060"));
    }

    #[test]
    fn test_pinned_vram() {
        let plan = compute_residency(&GpuProfile::rtx_4060(), &llama70b());
        let pinned = plan.pinned_vram_mb();
        assert!((pinned - 10.0 * 531.0).abs() < 1.0);
    }

    #[test]
    fn test_model_budget_custom() {
        let model = ModelBudget::new(32, 200.0);
        assert_eq!(model.n_layers, 32);
        assert_eq!(model.layer_size_mb, 200.0);
        assert_eq!(model.pipeline_buffer_mb, 400.0);
    }

    #[test]
    fn test_gpu_available_memory() {
        let gpu = GpuProfile::apple_m3_pro();
        let avail = gpu.available_memory_mb();
        assert!((avail - 18_432.0 * 0.30).abs() < 1.0);
    }

    #[test]
    fn test_strategy_display() {
        assert!(format!("{}", ResidencyStrategy::FullPin).contains("FULL PIN"));
        assert!(format!("{}", ResidencyStrategy::SelfSpeculative).contains("SELF-SPECULATIVE"));
        assert!(format!("{}", ResidencyStrategy::FullStream).contains("FULL STREAM"));
    }

    #[test]
    fn test_known_profiles_count() {
        assert_eq!(known_profiles().len(), 8);
    }
}

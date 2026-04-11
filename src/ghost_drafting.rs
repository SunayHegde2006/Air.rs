//! Ghost Drafting — M.I.S.T. v3 §Sub-System 3
//!
//! Bridges user-perceived latency on slow storage by running a tiny "ghost"
//! model (Llama 3.2 1B/3B) that produces draft tokens while the main 70B
//! model's first layer is still loading from HDD.
//!
//! ```text
//! t = 0 ms      User submits prompt
//! t = 5 ms      Ghost Model begins forward pass
//! t = 85 ms     Ghost produces first draft token → displayed as "thinking..."
//! t = 4,827 ms  70B model finishes loading first layer from HDD
//! t = 4,897 ms  70B begins verifying Ghost's draft batch (k=4 tokens)
//! t = 4,967 ms  Verification complete — confirmed tokens shown
//!
//! User-perceived TTFT: ~85 ms (via Ghost)
//! Ghost acceptance rate: ~75% → ~3 of 4 draft tokens confirmed
//! ```
//!
//! ## Ghost Model Selection
//!
//! ```text
//! available_ram >= 4,000 MB → Llama 3.2 3B (1.9 GB Q4_K_M)
//! available_ram >= 2,000 MB → Llama 3.2 1B (0.7 GB Q4_K_M)
//! available_ram <  2,000 MB → None (skip drafting)
//! ```
//!
//! ## Feasibility
//!
//! Ghost is useful iff Ghost TTFT < T_io(first_layer). For all M.I.S.T.
//! targets (HDD/USB/SATA < 3 GB/s) this is always true: 80ms ≪ 1,062ms.
//!
//! Reference: air_rs_protocols_v3.md §Sub-System 3

use std::fmt;

// ── Ghost Model Definitions ─────────────────────────────────────────────

/// Available ghost models for speculative drafting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostModel {
    /// Llama 3.2 3B Q4_K_M — 1.9 GB, higher quality drafts.
    Llama32_3B,
    /// Llama 3.2 1B Q4_K_M — 0.7 GB, minimal RAM usage.
    Llama32_1B,
}

impl GhostModel {
    /// Model file size in MB.
    pub fn size_mb(&self) -> f64 {
        match self {
            Self::Llama32_3B => 1_900.0,
            Self::Llama32_1B => 700.0,
        }
    }

    /// Model name string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama32_3B => "Llama 3.2 3B Q4_K_M",
            Self::Llama32_1B => "Llama 3.2 1B Q4_K_M",
        }
    }

    /// Parameter count (billions).
    pub fn params_b(&self) -> f64 {
        match self {
            Self::Llama32_3B => 3.0,
            Self::Llama32_1B => 1.0,
        }
    }
}

impl fmt::Display for GhostModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.1} GB)", self.name(), self.size_mb() / 1024.0)
    }
}

// ── Ghost Model Selection ────────────────────────────────────────────────

/// Select the appropriate ghost model based on available RAM.
///
/// ```text
/// >= 4,000 MB → Llama 3.2 3B (1.9 GB Q4_K_M)
/// >= 2,000 MB → Llama 3.2 1B (0.7 GB Q4_K_M)
/// <  2,000 MB → None (skip drafting)
/// ```
///
/// `available_ram_mb` is computed after subtracting pipeline buffers and
/// KV Pinned tier from total system RAM.
pub fn select_ghost_model(available_ram_mb: usize) -> Option<GhostModel> {
    match available_ram_mb {
        r if r >= 4_000 => Some(GhostModel::Llama32_3B),
        r if r >= 2_000 => Some(GhostModel::Llama32_1B),
        _ => None,
    }
}

// ── Platform TTFT Profiles ───────────────────────────────────────────────

/// Ghost TTFT (time to first token) by compute platform.
#[derive(Debug, Clone)]
pub struct GhostPlatformProfile {
    /// Platform name.
    pub name: String,
    /// Ghost TTFT in ms (time to produce first draft token).
    pub ttft_ms: f64,
    /// Which ghost model is used.
    pub model: GhostModel,
    /// Compute method (CUDA, ROCm, Metal, Vulkan, CPU).
    pub method: String,
    /// Whether this is CPU-only (no drafting on CPU if CPU IS the main model).
    pub cpu_only: bool,
}

impl GhostPlatformProfile {
    /// NVIDIA GPU: ~80 ms TTFT (CUDA).
    pub fn nvidia_gpu() -> Self {
        Self {
            name: "NVIDIA GPU".into(),
            ttft_ms: 80.0,
            model: GhostModel::Llama32_3B,
            method: "CUDA kernel".into(),
            cpu_only: false,
        }
    }

    /// AMD GPU: ~100 ms TTFT (ROCm/Vulkan).
    pub fn amd_gpu() -> Self {
        Self {
            name: "AMD GPU".into(),
            ttft_ms: 100.0,
            model: GhostModel::Llama32_3B,
            method: "ROCm / Vulkan".into(),
            cpu_only: false,
        }
    }

    /// Apple Silicon: ~60 ms TTFT (Metal, UMA advantage).
    pub fn apple_silicon() -> Self {
        Self {
            name: "Apple Silicon".into(),
            ttft_ms: 60.0,
            model: GhostModel::Llama32_3B,
            method: "Metal (UMA)".into(),
            cpu_only: false,
        }
    }

    /// Intel Arc: ~110 ms TTFT (Vulkan).
    pub fn intel_arc() -> Self {
        Self {
            name: "Intel Arc".into(),
            ttft_ms: 110.0,
            model: GhostModel::Llama32_3B,
            method: "Vulkan".into(),
            cpu_only: false,
        }
    }

    /// CPU Ryzen 5 7600: ~400 ms TTFT (OpenBLAS).
    pub fn cpu_ryzen_5() -> Self {
        Self {
            name: "Ryzen 5 7600 (CPU)".into(),
            ttft_ms: 400.0,
            model: GhostModel::Llama32_1B,
            method: "OpenBLAS (all cores)".into(),
            cpu_only: true,
        }
    }

    /// CPU Intel i5-12600K: ~350 ms TTFT (OpenBLAS).
    pub fn cpu_i5_12600k() -> Self {
        Self {
            name: "Intel i5-12600K (CPU)".into(),
            ttft_ms: 350.0,
            model: GhostModel::Llama32_1B,
            method: "OpenBLAS (all cores)".into(),
            cpu_only: true,
        }
    }

    /// Custom platform.
    pub fn custom(name: &str, ttft_ms: f64, model: GhostModel, method: &str, cpu_only: bool) -> Self {
        Self {
            name: name.into(),
            ttft_ms,
            model,
            method: method.into(),
            cpu_only,
        }
    }
}

impl fmt::Display for GhostPlatformProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}ms TTFT via {} ({})", self.name, self.ttft_ms, self.method, self.model)
    }
}

// ── Feasibility Check ────────────────────────────────────────────────────

/// Result of Ghost vs Disk feasibility analysis.
#[derive(Debug, Clone)]
pub struct FeasibilityResult {
    /// Storage description.
    pub storage_name: String,
    /// T_io for first layer in ms.
    pub t_io_first_layer_ms: f64,
    /// Ghost TTFT in ms.
    pub ghost_ttft_ms: f64,
    /// Speedup ratio: T_io / Ghost TTFT.
    pub speedup: f64,
    /// Whether Ghost Drafting is useful.
    pub feasible: bool,
    /// Whether Ghost is applicable (not applicable on CPU-only main model).
    pub applicable: bool,
}

impl fmt::Display for FeasibilityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if !self.applicable {
            "N/A (CPU is the ghost)"
        } else if self.feasible {
            "✓ YES"
        } else {
            "✗ NO"
        };

        write!(
            f,
            "Ghost feasible: {} | T_io={:.0}ms, Ghost={:.0}ms, speedup={:.0}×",
            status, self.t_io_first_layer_ms, self.ghost_ttft_ms, self.speedup
        )
    }
}

/// Check if Ghost Drafting is feasible for a given storage+platform combo.
///
/// Ghost is useful iff Ghost TTFT < T_io(first_layer).
/// Ghost is NOT applicable if the main model also runs on CPU
/// (CPU IS the ghost — no separate device to draft on).
pub fn check_feasibility(
    t_io_first_layer_ms: f64,
    ghost_profile: &GhostPlatformProfile,
    main_model_cpu_only: bool,
) -> FeasibilityResult {
    // CPU-only main model: Ghost is not applicable because
    // the CPU IS the ghost — there's no separate device.
    let applicable = !main_model_cpu_only;

    let speedup = if ghost_profile.ttft_ms > 0.0 {
        t_io_first_layer_ms / ghost_profile.ttft_ms
    } else {
        f64::INFINITY
    };

    let feasible = applicable && ghost_profile.ttft_ms < t_io_first_layer_ms;

    FeasibilityResult {
        storage_name: String::new(),
        t_io_first_layer_ms,
        ghost_ttft_ms: ghost_profile.ttft_ms,
        speedup,
        feasible,
        applicable,
    }
}

/// Run feasibility analysis across all M.I.S.T. storage targets.
///
/// Returns a matrix of (storage_name, T_io, feasibility) for the given platform.
pub fn feasibility_matrix(
    ghost_profile: &GhostPlatformProfile,
    layer_size_mb: f64,
) -> Vec<FeasibilityResult> {
    let storages = [
        ("USB 3.0 HDD", 80.0_f64),
        ("5400 RPM HDD", 110.0),
        ("7200 RPM HDD", 160.0),
        ("SATA SSD", 500.0),
    ];

    storages
        .iter()
        .map(|(name, speed)| {
            let t_io = (layer_size_mb / speed) * 1000.0;
            let mut result = check_feasibility(t_io, ghost_profile, false);
            result.storage_name = name.to_string();
            result
        })
        .collect()
}

// ── Acceptance Rate Tracker ──────────────────────────────────────────────

/// Default number of draft tokens per Ghost step.
pub const DEFAULT_DRAFT_K: usize = 4;

/// Default expected acceptance rate.
pub const DEFAULT_ACCEPTANCE_RATE: f64 = 0.75;

/// Tracks Ghost draft acceptance statistics in real-time.
///
/// The acceptance rate α directly impacts the M.I.S.T. ρ_v3 numerator:
/// T_ghost_accept = α × k × T_token
#[derive(Debug)]
pub struct AcceptanceTracker {
    /// Total draft tokens proposed.
    total_drafted: u64,
    /// Total draft tokens accepted by the main model.
    total_accepted: u64,
    /// Draft batch size (k).
    draft_k: usize,
    /// Running EMA of acceptance rate.
    ema_rate: f64,
    /// EMA smoothing factor.
    ema_alpha: f64,
    /// Per-batch history: (drafted, accepted).
    history: Vec<(usize, usize)>,
    /// Consecutive rejection streak (for adaptive k).
    rejection_streak: usize,
}

impl AcceptanceTracker {
    /// Create a new tracker with default parameters.
    pub fn new() -> Self {
        Self {
            total_drafted: 0,
            total_accepted: 0,
            draft_k: DEFAULT_DRAFT_K,
            ema_rate: DEFAULT_ACCEPTANCE_RATE,
            ema_alpha: 0.2,
            history: Vec::new(),
            rejection_streak: 0,
        }
    }

    /// Create with custom k and initial rate.
    pub fn with_params(draft_k: usize, initial_rate: f64) -> Self {
        Self {
            total_drafted: 0,
            total_accepted: 0,
            draft_k,
            ema_rate: initial_rate,
            ema_alpha: 0.2,
            history: Vec::new(),
            rejection_streak: 0,
        }
    }

    /// Report a batch of k draft tokens where `accepted` were confirmed.
    pub fn report_batch(&mut self, drafted: usize, accepted: usize) {
        let accepted = accepted.min(drafted);

        self.total_drafted += drafted as u64;
        self.total_accepted += accepted as u64;
        self.history.push((drafted, accepted));

        // Update EMA.
        let batch_rate = if drafted > 0 {
            accepted as f64 / drafted as f64
        } else {
            0.0
        };
        self.ema_rate = self.ema_alpha * batch_rate + (1.0 - self.ema_alpha) * self.ema_rate;

        // Track rejection streak.
        if accepted == 0 {
            self.rejection_streak += 1;
        } else {
            self.rejection_streak = 0;
        }
    }

    /// Overall acceptance rate (lifetime).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            return DEFAULT_ACCEPTANCE_RATE;
        }
        self.total_accepted as f64 / self.total_drafted as f64
    }

    /// EMA acceptance rate (recent trend).
    pub fn ema_rate(&self) -> f64 {
        self.ema_rate
    }

    /// Current draft batch size.
    pub fn draft_k(&self) -> usize {
        self.draft_k
    }

    /// Effective tokens gained per draft step: α × k.
    pub fn effective_tokens(&self) -> f64 {
        self.ema_rate * self.draft_k as f64
    }

    /// Ghost acceptance time contribution to ρ_v3 numerator.
    ///
    /// T_ghost_accept = α × k × T_token
    pub fn ghost_accept_time_ms(&self, t_token_ms: f64) -> f64 {
        self.ema_rate * self.draft_k as f64 * t_token_ms
    }

    /// Net ghost contribution after overhead.
    ///
    /// Spec: contribution to numerator = 158 ms net after overhead
    /// for α=0.75, k=4, T_token=70ms: 0.75 × 4 × 70 = 210ms raw,
    /// ~158ms net after Ghost model overhead.
    pub fn net_ghost_contribution_ms(&self, t_token_ms: f64) -> f64 {
        let raw = self.ghost_accept_time_ms(t_token_ms);
        // Subtract ~25% overhead for Ghost model inference bookkeeping.
        raw * 0.75
    }

    /// Number of completed draft batches.
    pub fn batch_count(&self) -> usize {
        self.history.len()
    }

    /// Total tokens drafted.
    pub fn total_drafted(&self) -> u64 {
        self.total_drafted
    }

    /// Total tokens accepted.
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Consecutive batches with 0 acceptances.
    pub fn rejection_streak(&self) -> usize {
        self.rejection_streak
    }

    /// Whether Ghost Drafting should be disabled due to poor acceptance.
    ///
    /// If 5+ consecutive batches are fully rejected, the Ghost model
    /// is wasting compute and should be paused.
    pub fn should_disable(&self) -> bool {
        self.rejection_streak >= 5
    }

    /// Suggest adaptive k based on acceptance trend.
    ///
    /// High acceptance → increase k for more throughput.
    /// Low acceptance → decrease k to avoid wasted verification.
    pub fn suggest_k(&self) -> usize {
        if self.ema_rate >= 0.85 {
            (self.draft_k + 2).min(8) // Increase aggressively
        } else if self.ema_rate >= 0.60 {
            self.draft_k // Keep current
        } else if self.ema_rate >= 0.30 {
            (self.draft_k.saturating_sub(1)).max(1) // Decrease
        } else {
            1 // Minimal drafting
        }
    }
}

impl Default for AcceptanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AcceptanceTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ghost: α={:.1}% (EMA={:.1}%), k={}, {}/{} accepted, {} batches",
            self.acceptance_rate() * 100.0,
            self.ema_rate * 100.0,
            self.draft_k,
            self.total_accepted,
            self.total_drafted,
            self.history.len(),
        )
    }
}

// ── Ghost Drafting Session ───────────────────────────────────────────────

/// Full Ghost Drafting session combining model selection, feasibility, and tracking.
#[derive(Debug)]
pub struct GhostSession {
    /// Selected ghost model (None if RAM too low).
    pub model: Option<GhostModel>,
    /// Platform profile.
    pub platform: GhostPlatformProfile,
    /// Acceptance tracker.
    pub tracker: AcceptanceTracker,
    /// Whether Ghost is active (enabled and feasible).
    pub active: bool,
    /// Available RAM at session start.
    pub available_ram_mb: usize,
    /// T_io for first layer (ms).
    pub t_io_first_layer_ms: f64,
    /// Main model on CPU only.
    pub main_model_cpu_only: bool,
}

impl GhostSession {
    /// Create a Ghost Drafting session, auto-selecting model and checking feasibility.
    pub fn new(
        available_ram_mb: usize,
        platform: GhostPlatformProfile,
        t_io_first_layer_ms: f64,
        main_model_cpu_only: bool,
    ) -> Self {
        let model = select_ghost_model(available_ram_mb);
        let feasibility = check_feasibility(t_io_first_layer_ms, &platform, main_model_cpu_only);

        let active = model.is_some() && feasibility.feasible;

        Self {
            model,
            platform,
            tracker: AcceptanceTracker::new(),
            active,
            available_ram_mb,
            t_io_first_layer_ms,
            main_model_cpu_only,
        }
    }

    /// Report a draft verification result.
    pub fn report_verification(&mut self, drafted: usize, accepted: usize) {
        self.tracker.report_batch(drafted, accepted);

        // Auto-disable if acceptance is terrible.
        if self.tracker.should_disable() {
            self.active = false;
        }
    }

    /// User-perceived TTFT in ms.
    ///
    /// With Ghost: Ghost TTFT (~80ms).
    /// Without Ghost: T_io for first layer (seconds on HDD).
    pub fn perceived_ttft_ms(&self) -> f64 {
        if self.active {
            // 5ms prompt processing + Ghost TTFT.
            5.0 + self.platform.ttft_ms
        } else {
            self.t_io_first_layer_ms
        }
    }

    /// Speedup over no-Ghost baseline.
    pub fn ttft_speedup(&self) -> f64 {
        if self.active && self.platform.ttft_ms > 0.0 {
            self.t_io_first_layer_ms / self.perceived_ttft_ms()
        } else {
            1.0
        }
    }

    /// ρ_v3 contribution from Ghost acceptance.
    pub fn rho_contribution(&self, t_token_ms: f64) -> f64 {
        if self.active {
            self.tracker.net_ghost_contribution_ms(t_token_ms)
        } else {
            0.0
        }
    }
}

impl fmt::Display for GhostSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  M.I.S.T. v3 — Ghost Drafting Session        ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        match &self.model {
            Some(m) => writeln!(f, "║  Model:    {:>33} ║", m.name())?,
            None => writeln!(f, "║  Model:    {:>33} ║", "None (RAM too low)")?,
        }
        writeln!(f, "║  Platform: {:>33} ║", self.platform.name)?;
        writeln!(f, "║  Active:   {:>33} ║", if self.active { "YES" } else { "NO" })?;
        writeln!(f, "║  RAM:      {:>29} MB ║", self.available_ram_mb)?;
        writeln!(f, "║  TTFT:     {:>29.0} ms ║", self.perceived_ttft_ms())?;
        writeln!(f, "║  Speedup:  {:>29.0}× ║", self.ttft_speedup())?;
        writeln!(f, "║  {} ║", format!("{:>44}", self.tracker))?;
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

// ── Tiered Eviction Cascade ──────────────────────────────────────────────

/// Eviction index entry: maps token position to cold log file offset.
#[derive(Debug, Clone)]
pub struct EvictionEntry {
    /// Token position in the sequence.
    pub token_position: usize,
    /// Byte offset in the cold log file.
    pub file_offset: u64,
    /// Size of the compressed entry in bytes.
    pub entry_size: usize,
    /// Layer index (or ALL_LAYERS sentinel).
    pub layer: usize,
}

/// Sentinel for "all layers" in eviction entries.
pub const ALL_LAYERS: usize = usize::MAX;

/// Append-only cold log manager for evicted KV entries.
///
/// The KV cold log is the M.I.S.T. extension of Sequential Magnetism:
/// ```text
/// Model GGUF file:  ←——— read sequentially, never seek ———→
/// KV cold log file: ←——— append-only writes, read on recall ——→
/// ```
///
/// Both files treated as linear tapes. HDD head stays in two predictable tracks.
#[derive(Debug)]
pub struct ColdLog {
    /// Current write offset (append position).
    write_offset: u64,
    /// Eviction index: token_position → EvictionEntry.
    index: Vec<EvictionEntry>,
    /// Total bytes written.
    total_bytes_written: u64,
    /// Total entries evicted.
    total_evictions: u64,
    /// Total entries recalled.
    total_recalls: u64,
}

impl ColdLog {
    /// Create a new cold log.
    pub fn new() -> Self {
        Self {
            write_offset: 0,
            index: Vec::new(),
            total_bytes_written: 0,
            total_evictions: 0,
            total_recalls: 0,
        }
    }

    /// Evict a compressed KV entry to the cold log.
    ///
    /// Returns the file offset where the entry was written.
    /// In production, this would write to an actual file; here we
    /// track the offset for the sequential I/O pattern.
    pub fn evict(&mut self, token_position: usize, entry_size: usize, layer: usize) -> u64 {
        let offset = self.write_offset;

        self.index.push(EvictionEntry {
            token_position,
            file_offset: offset,
            entry_size,
            layer,
        });

        self.write_offset += entry_size as u64;
        self.total_bytes_written += entry_size as u64;
        self.total_evictions += 1;

        offset
    }

    /// Look up the file offset for a recalled entry.
    pub fn lookup(&self, token_position: usize) -> Option<&EvictionEntry> {
        self.index.iter().rev().find(|e| e.token_position == token_position)
    }

    /// Record a recall (read from cold log).
    pub fn record_recall(&mut self) {
        self.total_recalls += 1;
    }

    /// Total evictions performed.
    pub fn total_evictions(&self) -> u64 {
        self.total_evictions
    }

    /// Total recalls performed.
    pub fn total_recalls(&self) -> u64 {
        self.total_recalls
    }

    /// Total bytes in the cold log.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes_written
    }

    /// Number of entries in the index.
    pub fn index_size(&self) -> usize {
        self.index.len()
    }

    /// Miss rate: recalls / evictions.
    pub fn miss_rate(&self) -> f64 {
        if self.total_evictions == 0 {
            return 0.0;
        }
        self.total_recalls as f64 / self.total_evictions as f64
    }

    /// Clear the cold log (new generation).
    pub fn clear(&mut self) {
        self.write_offset = 0;
        self.index.clear();
        self.total_bytes_written = 0;
        self.total_evictions = 0;
        self.total_recalls = 0;
    }
}

impl Default for ColdLog {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ColdLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ColdLog: {} entries, {:.1} MB written, {} recalls, miss_rate={:.1}%",
            self.index.len(),
            self.total_bytes_written as f64 / (1024.0 * 1024.0),
            self.total_recalls,
            self.miss_rate() * 100.0,
        )
    }
}

// ── Prefetch Prediction ──────────────────────────────────────────────────

/// Default attention window for P_recall computation.
pub const PREFETCH_WINDOW: usize = 16;

/// Default attention threshold.
pub const PREFETCH_THRESHOLD: f64 = 0.1;

/// Default lookahead steps for prefetch.
pub const PREFETCH_DELTA: usize = 3;

/// Compute prefetch recall probability for an evicted entry.
///
/// P_recall(k_i, t+δ) = (1/W) × Σ_{j=t-W}^{t} 𝟙[A_{ij} > θ]
///
/// If historical attention probability over W=16 steps exceeds θ=0.1,
/// the entry should be prefetched δ=3 steps before expected use.
pub fn prefetch_probability(
    attention_history: &[f64],
    threshold: f64,
) -> f64 {
    if attention_history.is_empty() {
        return 0.0;
    }

    let window = attention_history.len().min(PREFETCH_WINDOW);
    let recent = &attention_history[attention_history.len() - window..];

    let count = recent.iter().filter(|&&a| a > threshold).count();
    count as f64 / window as f64
}

/// Decide whether to prefetch an evicted entry.
pub fn should_prefetch(
    attention_history: &[f64],
    threshold: f64,
) -> bool {
    prefetch_probability(attention_history, threshold) > PREFETCH_THRESHOLD
}

/// Estimate read time for a prefetched entry from cold storage.
///
/// Entry size (one token, all layers, Q8): 2 × 80 × 8 × 128 × 1 = 163,840 bytes = 160 KB
/// At 160 MB/s (7200 RPM): 0.001 ms — negligible.
pub fn prefetch_read_time_ms(entry_size_bytes: usize, disk_speed_mbps: f64) -> f64 {
    if disk_speed_mbps <= 0.0 {
        return f64::INFINITY;
    }
    let size_mb = entry_size_bytes as f64 / (1024.0 * 1024.0);
    (size_mb / disk_speed_mbps) * 1000.0
}

/// Default entry size for all-layer Q8 KV per token.
///
/// 2 × 80 × 8 × 128 × 1 byte = 163,840 bytes
pub fn default_entry_size() -> usize {
    2 * 80 * 8 * 128 * 1
}

// ── Sequential Magnetism (Dual-File) ─────────────────────────────────────

/// Sequential access enforcer for M.I.S.T. dual-file pattern.
///
/// Ensures both the model GGUF file and KV cold log file are accessed
/// sequentially. Any non-sequential access returns a hard error — this
/// is a protocol violation, not a warning.
#[derive(Debug)]
pub struct SequentialEnforcer {
    /// Last model layer accessed.
    last_model_layer: Option<usize>,
    /// Total model layer accesses.
    model_accesses: u64,
    /// Total sequential violations (should be 0).
    violations: u64,
}

impl SequentialEnforcer {
    /// Create a new enforcer.
    pub fn new() -> Self {
        Self {
            last_model_layer: None,
            model_accesses: 0,
            violations: 0,
        }
    }

    /// Assert sequential model layer access.
    ///
    /// Returns Ok(()) if sequential, Err with expected vs actual if not.
    pub fn assert_sequential_model(&mut self, requested_layer: usize) -> Result<(), SequentialViolation> {
        let result = match self.last_model_layer {
            None => Ok(()), // First access is always valid.
            Some(last) => {
                if requested_layer == last + 1 || requested_layer == 0 {
                    // Sequential or wrap-around (new pass).
                    Ok(())
                } else {
                    self.violations += 1;
                    Err(SequentialViolation {
                        expected: last + 1,
                        requested: requested_layer,
                    })
                }
            }
        };

        if result.is_ok() {
            self.last_model_layer = Some(requested_layer);
            self.model_accesses += 1;
        }

        result
    }

    /// Reset for a new forward pass.
    pub fn reset(&mut self) {
        self.last_model_layer = None;
    }

    /// Total violations (should be 0 in correct operation).
    pub fn violations(&self) -> u64 {
        self.violations
    }
}

impl Default for SequentialEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// A sequential access violation error.
#[derive(Debug, Clone)]
pub struct SequentialViolation {
    /// Expected layer index.
    pub expected: usize,
    /// Actually requested layer index.
    pub requested: usize,
}

impl fmt::Display for SequentialViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "M.I.S.T. Sequential Magnetism violation: expected layer {}, got {}",
            self.expected, self.requested
        )
    }
}

impl std::error::Error for SequentialViolation {}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ghost Model Selection ────────────────────────────────────────

    #[test]
    fn test_select_ghost_4gb_ram() {
        // Spec: >= 4,000 MB → Llama 3.2 3B
        assert_eq!(select_ghost_model(4_000), Some(GhostModel::Llama32_3B));
        assert_eq!(select_ghost_model(8_000), Some(GhostModel::Llama32_3B));
    }

    #[test]
    fn test_select_ghost_2gb_ram() {
        // Spec: >= 2,000 MB → Llama 3.2 1B
        assert_eq!(select_ghost_model(2_000), Some(GhostModel::Llama32_1B));
        assert_eq!(select_ghost_model(3_999), Some(GhostModel::Llama32_1B));
    }

    #[test]
    fn test_select_ghost_low_ram() {
        // Spec: < 2,000 MB → None
        assert_eq!(select_ghost_model(1_999), None);
        assert_eq!(select_ghost_model(500), None);
        assert_eq!(select_ghost_model(0), None);
    }

    #[test]
    fn test_ghost_model_sizes() {
        assert!((GhostModel::Llama32_3B.size_mb() - 1900.0).abs() < 1.0);
        assert!((GhostModel::Llama32_1B.size_mb() - 700.0).abs() < 1.0);
    }

    #[test]
    fn test_ghost_model_display() {
        let s = format!("{}", GhostModel::Llama32_3B);
        assert!(s.contains("3B"));
        assert!(s.contains("GB"));
    }

    // ── Platform TTFT Profiles ───────────────────────────────────────

    #[test]
    fn test_nvidia_ttft() {
        let p = GhostPlatformProfile::nvidia_gpu();
        assert!((p.ttft_ms - 80.0).abs() < 0.1);
        assert!(!p.cpu_only);
    }

    #[test]
    fn test_apple_ttft() {
        let p = GhostPlatformProfile::apple_silicon();
        assert!((p.ttft_ms - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_cpu_ttft() {
        let p = GhostPlatformProfile::cpu_ryzen_5();
        assert!((p.ttft_ms - 400.0).abs() < 0.1);
        assert!(p.cpu_only);
    }

    #[test]
    fn test_platform_display() {
        let p = GhostPlatformProfile::nvidia_gpu();
        let s = format!("{}", p);
        assert!(s.contains("NVIDIA"));
        assert!(s.contains("80"));
    }

    // ── Feasibility Checks ───────────────────────────────────────────

    #[test]
    fn test_feasibility_usb3_gpu() {
        // Spec: USB 3.0 HDD, T_io=6,638ms, Ghost=80ms → 83× faster
        let gpu = GhostPlatformProfile::nvidia_gpu();
        let result = check_feasibility(6_638.0, &gpu, false);
        assert!(result.feasible);
        assert!((result.speedup - 83.0).abs() < 1.0, "Speedup: {:.0}", result.speedup);
    }

    #[test]
    fn test_feasibility_5400_gpu() {
        // Spec: 5400 RPM, T_io=4,827ms, Ghost=80ms → 60× faster
        let gpu = GhostPlatformProfile::nvidia_gpu();
        let result = check_feasibility(4_827.0, &gpu, false);
        assert!(result.feasible);
        assert!((result.speedup - 60.0).abs() < 1.0, "Speedup: {:.0}", result.speedup);
    }

    #[test]
    fn test_feasibility_7200_gpu() {
        // Spec: 7200 RPM, T_io=3,319ms, Ghost=80ms → 41× faster
        let gpu = GhostPlatformProfile::nvidia_gpu();
        let result = check_feasibility(3_319.0, &gpu, false);
        assert!(result.feasible);
        assert!((result.speedup - 41.0).abs() < 1.0, "Speedup: {:.0}", result.speedup);
    }

    #[test]
    fn test_feasibility_sata_gpu() {
        // Spec: SATA SSD, T_io=1,062ms, Ghost=80ms → 13× faster
        let gpu = GhostPlatformProfile::nvidia_gpu();
        let result = check_feasibility(1_062.0, &gpu, false);
        assert!(result.feasible);
        assert!((result.speedup - 13.0).abs() < 1.0, "Speedup: {:.0}", result.speedup);
    }

    #[test]
    fn test_feasibility_cpu_not_applicable() {
        // CPU-only: Ghost is not applicable (CPU IS the ghost).
        let cpu = GhostPlatformProfile::cpu_ryzen_5();
        let result = check_feasibility(4_827.0, &cpu, true);
        assert!(!result.applicable);
        assert!(!result.feasible);
    }

    #[test]
    fn test_feasibility_matrix() {
        let gpu = GhostPlatformProfile::nvidia_gpu();
        let matrix = feasibility_matrix(&gpu, 531.0);
        assert_eq!(matrix.len(), 4);
        // All M.I.S.T. targets should be feasible.
        for result in &matrix {
            assert!(result.feasible, "Should be feasible for {}", result.storage_name);
        }
    }

    // ── Acceptance Rate Tracker ──────────────────────────────────────

    #[test]
    fn test_tracker_default() {
        let tracker = AcceptanceTracker::new();
        assert_eq!(tracker.draft_k(), 4);
        assert!((tracker.acceptance_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_tracker_report_batch() {
        let mut tracker = AcceptanceTracker::new();
        tracker.report_batch(4, 3); // 3/4 = 75%
        assert_eq!(tracker.total_drafted(), 4);
        assert_eq!(tracker.total_accepted(), 3);
        assert_eq!(tracker.batch_count(), 1);
    }

    #[test]
    fn test_tracker_acceptance_rate() {
        let mut tracker = AcceptanceTracker::new();
        tracker.report_batch(4, 3); // 75%
        tracker.report_batch(4, 4); // 100%
        tracker.report_batch(4, 2); // 50%
        // Overall: 9/12 = 75%
        assert!((tracker.acceptance_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_tracker_ema_trends() {
        let mut tracker = AcceptanceTracker::new();
        // Start with high acceptance.
        for _ in 0..10 {
            tracker.report_batch(4, 4);
        }
        let high_ema = tracker.ema_rate();

        // Then low acceptance.
        for _ in 0..10 {
            tracker.report_batch(4, 0);
        }
        let low_ema = tracker.ema_rate();

        assert!(high_ema > low_ema, "EMA should decrease: {:.2} > {:.2}", high_ema, low_ema);
    }

    #[test]
    fn test_tracker_rejection_streak() {
        let mut tracker = AcceptanceTracker::new();
        tracker.report_batch(4, 3);
        assert_eq!(tracker.rejection_streak(), 0);

        tracker.report_batch(4, 0);
        assert_eq!(tracker.rejection_streak(), 1);

        tracker.report_batch(4, 0);
        assert_eq!(tracker.rejection_streak(), 2);

        tracker.report_batch(4, 1); // Resets streak.
        assert_eq!(tracker.rejection_streak(), 0);
    }

    #[test]
    fn test_tracker_should_disable() {
        let mut tracker = AcceptanceTracker::new();
        assert!(!tracker.should_disable());

        for _ in 0..5 {
            tracker.report_batch(4, 0);
        }
        assert!(tracker.should_disable());
    }

    #[test]
    fn test_tracker_suggest_k() {
        let mut tracker = AcceptanceTracker::with_params(4, 0.90);
        assert!(tracker.suggest_k() >= 5); // High rate → increase

        tracker = AcceptanceTracker::with_params(4, 0.70);
        assert_eq!(tracker.suggest_k(), 4); // Medium → keep

        tracker = AcceptanceTracker::with_params(4, 0.40);
        assert!(tracker.suggest_k() <= 3); // Low → decrease

        tracker = AcceptanceTracker::with_params(4, 0.10);
        assert_eq!(tracker.suggest_k(), 1); // Very low → minimal
    }

    #[test]
    fn test_tracker_ghost_accept_time() {
        let tracker = AcceptanceTracker::with_params(4, 0.75);
        // Spec: α × k × T_token = 0.75 × 4 × 70 = 210 ms
        let t = tracker.ghost_accept_time_ms(70.0);
        assert!((t - 210.0).abs() < 0.1, "Ghost accept time: {:.0}", t);
    }

    #[test]
    fn test_tracker_net_contribution() {
        let tracker = AcceptanceTracker::with_params(4, 0.75);
        // Spec: 210ms raw → ~158ms net (75% after overhead)
        let net = tracker.net_ghost_contribution_ms(70.0);
        assert!((net - 157.5).abs() < 1.0, "Net contribution: {:.0}", net);
    }

    #[test]
    fn test_tracker_display() {
        let tracker = AcceptanceTracker::new();
        let s = format!("{}", tracker);
        assert!(s.contains("Ghost"));
        assert!(s.contains("k=4"));
    }

    // ── Ghost Session ────────────────────────────────────────────────

    #[test]
    fn test_session_gpu_with_ram() {
        let session = GhostSession::new(
            8_000,
            GhostPlatformProfile::nvidia_gpu(),
            4_827.0,
            false,
        );
        assert!(session.active);
        assert_eq!(session.model, Some(GhostModel::Llama32_3B));
        assert!((session.perceived_ttft_ms() - 85.0).abs() < 0.1);
    }

    #[test]
    fn test_session_ttft_speedup() {
        // Spec: 5400 RPM HDD, GPU: 85ms vs 4,827ms → 57× speedup
        let session = GhostSession::new(
            8_000,
            GhostPlatformProfile::nvidia_gpu(),
            4_827.0,
            false,
        );
        let speedup = session.ttft_speedup();
        assert!(speedup > 50.0, "TTFT speedup: {:.0}×", speedup);
    }

    #[test]
    fn test_session_low_ram() {
        let session = GhostSession::new(
            500, // Too low for any ghost model
            GhostPlatformProfile::nvidia_gpu(),
            4_827.0,
            false,
        );
        assert!(!session.active);
        assert_eq!(session.model, None);
        // TTFT falls back to T_io.
        assert!((session.perceived_ttft_ms() - 4_827.0).abs() < 1.0);
    }

    #[test]
    fn test_session_cpu_only() {
        let session = GhostSession::new(
            8_000,
            GhostPlatformProfile::cpu_ryzen_5(),
            4_827.0,
            true, // CPU main model
        );
        assert!(!session.active, "CPU-only should not draft (CPU IS the ghost)");
    }

    #[test]
    fn test_session_auto_disable() {
        let mut session = GhostSession::new(
            8_000,
            GhostPlatformProfile::nvidia_gpu(),
            4_827.0,
            false,
        );
        assert!(session.active);

        // 5 consecutive full rejections.
        for _ in 0..5 {
            session.report_verification(4, 0);
        }
        assert!(!session.active, "Should auto-disable after 5 rejections");
    }

    #[test]
    fn test_session_display() {
        let session = GhostSession::new(
            8_000,
            GhostPlatformProfile::nvidia_gpu(),
            4_827.0,
            false,
        );
        let s = format!("{}", session);
        assert!(s.contains("Ghost Drafting"));
        assert!(s.contains("NVIDIA"));
    }

    // ── Cold Log ─────────────────────────────────────────────────────

    #[test]
    fn test_cold_log_evict() {
        let mut log = ColdLog::new();
        let offset = log.evict(42, 163_840, ALL_LAYERS);
        assert_eq!(offset, 0);
        assert_eq!(log.total_evictions(), 1);
        assert_eq!(log.total_bytes(), 163_840);
    }

    #[test]
    fn test_cold_log_sequential_offsets() {
        let mut log = ColdLog::new();
        let o1 = log.evict(0, 1000, 0);
        let o2 = log.evict(1, 2000, 0);
        let o3 = log.evict(2, 1500, 0);
        assert_eq!(o1, 0);
        assert_eq!(o2, 1000);
        assert_eq!(o3, 3000);
    }

    #[test]
    fn test_cold_log_lookup() {
        let mut log = ColdLog::new();
        log.evict(42, 163_840, ALL_LAYERS);
        log.evict(99, 163_840, ALL_LAYERS);

        let entry = log.lookup(42).unwrap();
        assert_eq!(entry.file_offset, 0);
        assert_eq!(entry.entry_size, 163_840);

        let entry = log.lookup(99).unwrap();
        assert_eq!(entry.file_offset, 163_840);

        assert!(log.lookup(999).is_none());
    }

    #[test]
    fn test_cold_log_miss_rate() {
        let mut log = ColdLog::new();
        log.evict(0, 1000, 0);
        log.evict(1, 1000, 0);
        log.record_recall();
        // 1 recall / 2 evictions = 50% miss rate.
        assert!((log.miss_rate() - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_cold_log_clear() {
        let mut log = ColdLog::new();
        log.evict(0, 1000, 0);
        log.record_recall();
        log.clear();
        assert_eq!(log.total_evictions(), 0);
        assert_eq!(log.total_recalls(), 0);
        assert_eq!(log.index_size(), 0);
    }

    #[test]
    fn test_cold_log_display() {
        let log = ColdLog::new();
        let s = format!("{}", log);
        assert!(s.contains("ColdLog"));
    }

    // ── Prefetch Prediction ──────────────────────────────────────────

    #[test]
    fn test_prefetch_probability_high() {
        // Token often attended → high P_recall.
        let history = vec![0.5, 0.3, 0.2, 0.4, 0.5, 0.3, 0.2, 0.4];
        let p = prefetch_probability(&history, PREFETCH_THRESHOLD);
        assert!(p > 0.9, "Highly attended → high P: {:.2}", p);
    }

    #[test]
    fn test_prefetch_probability_low() {
        // Token rarely attended → low P_recall.
        let history = vec![0.01, 0.02, 0.005, 0.0, 0.01, 0.0, 0.0, 0.005];
        let p = prefetch_probability(&history, PREFETCH_THRESHOLD);
        assert!(p < 0.1, "Rarely attended → low P: {:.2}", p);
    }

    #[test]
    fn test_prefetch_empty() {
        assert_eq!(prefetch_probability(&[], 0.1), 0.0);
    }

    #[test]
    fn test_should_prefetch() {
        let high = vec![0.5; 16];
        let low = vec![0.01; 16];
        assert!(should_prefetch(&high, 0.1));
        assert!(!should_prefetch(&low, 0.1));
    }

    #[test]
    fn test_prefetch_read_time() {
        // Spec: 160 KB at 160 MB/s = 0.001 ms
        let entry_size = default_entry_size(); // 163,840 bytes
        assert_eq!(entry_size, 163_840);

        let time = prefetch_read_time_ms(entry_size, 160.0);
        assert!(time < 1.0, "Prefetch read time: {:.4} ms", time);
    }

    // ── Sequential Enforcer ──────────────────────────────────────────

    #[test]
    fn test_sequential_access_ok() {
        let mut enforcer = SequentialEnforcer::new();
        assert!(enforcer.assert_sequential_model(0).is_ok());
        assert!(enforcer.assert_sequential_model(1).is_ok());
        assert!(enforcer.assert_sequential_model(2).is_ok());
    }

    #[test]
    fn test_sequential_violation() {
        let mut enforcer = SequentialEnforcer::new();
        assert!(enforcer.assert_sequential_model(0).is_ok());
        assert!(enforcer.assert_sequential_model(1).is_ok());
        // Skip layer 2 → violation.
        let err = enforcer.assert_sequential_model(5);
        assert!(err.is_err());
        let violation = err.unwrap_err();
        assert_eq!(violation.expected, 2);
        assert_eq!(violation.requested, 5);
    }

    #[test]
    fn test_sequential_wrap_around() {
        let mut enforcer = SequentialEnforcer::new();
        enforcer.assert_sequential_model(0).unwrap();
        enforcer.assert_sequential_model(1).unwrap();
        // Layer 0 = wrap-around (new forward pass) → allowed.
        assert!(enforcer.assert_sequential_model(0).is_ok());
    }

    #[test]
    fn test_sequential_reset() {
        let mut enforcer = SequentialEnforcer::new();
        enforcer.assert_sequential_model(0).unwrap();
        enforcer.assert_sequential_model(1).unwrap();
        enforcer.reset();
        // After reset, any layer is valid.
        assert!(enforcer.assert_sequential_model(50).is_ok());
    }

    #[test]
    fn test_sequential_violation_count() {
        let mut enforcer = SequentialEnforcer::new();
        enforcer.assert_sequential_model(0).unwrap();
        let _ = enforcer.assert_sequential_model(5); // violation
        let _ = enforcer.assert_sequential_model(10); // another (state unchanged after violation)
        assert_eq!(enforcer.violations(), 2);
    }

    #[test]
    fn test_sequential_violation_display() {
        let v = SequentialViolation { expected: 5, requested: 10 };
        let s = format!("{}", v);
        assert!(s.contains("Sequential Magnetism"));
        assert!(s.contains("5"));
        assert!(s.contains("10"));
    }
}

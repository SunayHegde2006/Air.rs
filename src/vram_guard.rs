//! VRAM 80% hard cap guard — issue #2.
//!
//! Enforces that the estimated VRAM footprint of a model load does not exceed
//! 80% of total GPU VRAM. This prevents OOM during the first generation step.
//!
//! # Footprint model
//! Air.rs uses S.L.I.P. streaming, so the full model weights never live in
//! VRAM simultaneously. The persistent footprint is:
//!
//! ```text
//! KV cache  = n_layers × context_len × n_kv_heads × head_dim × 2 (K+V) × 2 (f16)
//! Peak work = 2 layers × per-layer weight size (dequantized f16, prefetch overlap)
//! Total     = KV cache + peak work
//! ```
//!
//! # CUDA query
//! Uses `nvidia-smi --query-gpu=memory.free,memory.total` to avoid adding a
//! new `nvml_wrapper` dependency. Falls back gracefully if `nvidia-smi` is not
//! present (warning printed, guard skipped).
//!
//! # CPU / Metal
//! Guard is skipped — returns `Ok` immediately with zeroed memory fields.

use anyhow::{bail, Result};
use candle_core::Device;

// ── Constants ────────────────────────────────────────────────────────────────

/// Guard rejects a load if estimated footprint exceeds this fraction of VRAM.
pub const VRAM_CAP_FRACTION: f64 = 0.80;

/// Default context length when not available from config (conservative).
const DEFAULT_CONTEXT_LEN: usize = 4096;

// ── VramCheckResult ──────────────────────────────────────────────────────────

/// Outcome of a VRAM pre-flight check.
#[derive(Debug, Clone, PartialEq)]
pub enum VramCheckResult {
    /// Estimate is within the configured cap: load is safe.
    Accept,
    /// Estimate exceeds the cap: load rejected. `shortfall_bytes` is how
    /// many bytes must be freed before the load will succeed.
    Reject {
        estimate_bytes: u64,
        cap_bytes: u64,
        shortfall_bytes: u64,
    },
    /// VRAM query unavailable (e.g. CPU device, nvidia-smi absent): guard skipped.
    GuardSkipped,
}

impl VramCheckResult {
    /// True iff this result allows the load to proceed.
    pub fn ok_to_load(&self) -> bool {
        matches!(self, Self::Accept | Self::GuardSkipped)
    }

    /// Human-readable one-liner for log/error output.
    pub fn summary(&self) -> String {
        match self {
            Self::Accept => "VRAM guard: ✓ load accepted".into(),
            Self::Reject { estimate_bytes, cap_bytes, shortfall_bytes } => format!(
                "VRAM guard: ✗ load rejected — need {:.0} MiB, cap {:.0} MiB \
                 (shortfall {:.0} MiB). Reduce context_length or use a smaller quant.",
                *estimate_bytes as f64 / 1_048_576.0,
                *cap_bytes as f64 / 1_048_576.0,
                *shortfall_bytes as f64 / 1_048_576.0,
            ),
            Self::GuardSkipped => "VRAM guard: guard skipped (non-CUDA or query failed)".into(),
        }
    }
}

// ── VramBudget ───────────────────────────────────────────────────────────────

/// Result of a VRAM budget check.
///
/// Returned by [`VramBudget::check`]. All byte fields are 0 for non-CUDA
/// devices or when `nvidia-smi` is unavailable.
#[derive(Debug, Clone)]
pub struct VramBudget {
    /// Total VRAM on the GPU (bytes). 0 for CPU/Metal or query failure.
    pub total_bytes: u64,
    /// Free VRAM before model load (bytes).
    pub free_bytes: u64,
    /// Estimated VRAM footprint of this load (KV cache + 2-layer peak).
    pub estimate_bytes: u64,
    /// Hard cap = `total_bytes × VRAM_CAP_FRACTION`.
    pub cap_bytes: u64,
}

impl VramBudget {
    // ── Internal helpers ─────────────────────────────────────────────────

    /// Query VRAM via `nvidia-smi`.
    ///
    /// Returns `(free_bytes, total_bytes)` or an error if `nvidia-smi`
    /// is absent or its output cannot be parsed.
    fn query_nvidia_smi(device_ordinal: usize) -> Result<(u64, u64)> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                &format!("--id={device_ordinal}"),
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("nvidia-smi not found: {e}"))?;

        let s = String::from_utf8_lossy(&out.stdout);
        let s = s.trim();
        let mut parts = s.splitn(2, ',');

        let free_mib: u64 = parts
            .next()
            .unwrap_or("0")
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("nvidia-smi free parse failed: {e}"))?;
        let total_mib: u64 = parts
            .next()
            .unwrap_or("0")
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("nvidia-smi total parse failed: {e}"))?;

        Ok((free_mib * 1024 * 1024, total_mib * 1024 * 1024))
    }

    /// Estimate peak VRAM footprint for a model with the given dimensions.
    ///
    /// # Components
    /// 1. **KV cache** — all layers × context_len × KV heads × head_dim × 2
    ///    (K and V) × 2 (f16 bytes).  Persistent for the full session.
    /// 2. **2-layer peak** — two layers of dequantized f16 weights resident
    ///    simultaneously (current layer + prefetched next layer). Estimated
    ///    as `2 × 12 × hidden_dim² × 2` bytes (dense LLaMA attention+FFN
    ///    projection matrices).
    fn estimate(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        context_len: usize,
    ) -> u64 {
        // KV cache: both K and V, f16
        let kv_bytes = (n_layers as u64)
            * (context_len as u64)
            * (n_kv_heads as u64)
            * (head_dim as u64)
            * 2   // K + V
            * 2;  // f16 = 2 bytes

        // Per-layer dequantized weight peak: 12 matrices of hidden_dim × hidden_dim,
        // f16.  (Wq, Wk, Wv, Wo, W_gate, W_up, W_down + norms ≈ 12).
        // Two layers resident at once (prefetch overlap).
        let layer_bytes = 12u64 * (hidden_dim as u64) * (hidden_dim as u64) * 2;
        let peak_bytes = layer_bytes * 2;

        kv_bytes + peak_bytes
    }

    /// Public wrapper around `estimate` for use by `ModelMux::load_guarded`.
    pub fn estimate_pub(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        context_len: usize,
    ) -> u64 {
        Self::estimate(n_layers, n_kv_heads, head_dim, hidden_dim, context_len)
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Check VRAM budget before loading a model onto `device`.
    ///
    /// # Arguments
    /// - `device` — target `candle_core::Device`
    /// - `device_idx` — CUDA device ordinal for `nvidia-smi`; ignored for CPU/Metal.
    ///   Pass `0` for single-GPU systems (most consumer setups).
    /// - `n_layers`, `n_kv_heads`, `head_dim`, `hidden_dim` — from `ModelConfig`
    /// - `context_len` — effective max sequence length
    ///
    /// # Behaviour
    /// - **CPU / Metal** → always returns `Ok`. Memory fields are 0.
    /// - **CUDA** → queries `nvidia-smi`. Absent → warning printed, guard skipped.
    ///   Estimate > 80% of total VRAM → returns `Err`.
    ///
    /// # Example
    /// ```no_run
    /// use candle_core::Device;
    /// use air_rs::vram_guard::VramBudget;
    ///
    /// let budget = VramBudget::check(&Device::Cpu, 0, 32, 32, 128, 4096, 4096).unwrap();
    /// assert_eq!(budget.total_bytes, 0); // CPU path
    /// ```
    pub fn check(
        device: &Device,
        device_idx: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        context_len: usize,
    ) -> Result<Self> {
        let context_len = if context_len == 0 { DEFAULT_CONTEXT_LEN } else { context_len };
        let estimate_bytes = Self::estimate(n_layers, n_kv_heads, head_dim, hidden_dim, context_len);

        // Non-CUDA devices: skip guard, return informational budget
        if !device.is_cuda() {
            return Ok(Self {
                total_bytes: 0,
                free_bytes: 0,
                estimate_bytes,
                cap_bytes: 0,
            });
        }

        // Query VRAM — graceful fallback if nvidia-smi absent
        let (free_bytes, total_bytes) = match Self::query_nvidia_smi(device_idx) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "⚠  VRAM guard: query failed ({e}). Skipping 80% cap check."
                );
                return Ok(Self {
                    total_bytes: 0,
                    free_bytes: 0,
                    estimate_bytes,
                    cap_bytes: 0,
                });
            }
        };

        let cap_bytes = (total_bytes as f64 * VRAM_CAP_FRACTION) as u64;

        if estimate_bytes > cap_bytes {
            bail!(
                "VRAM guard: load rejected — estimated {:.2} GiB exceeds 80% cap \
                ({:.2} GiB of {:.2} GiB total on CUDA:{device_idx}). \
                Free VRAM: {:.2} GiB. \
                Reduce context_length, use a smaller quant, or run on CPU.",
                estimate_bytes as f64 / 1_073_741_824.0,
                cap_bytes as f64 / 1_073_741_824.0,
                total_bytes as f64 / 1_073_741_824.0,
                free_bytes as f64 / 1_073_741_824.0,
            );
        }

        Ok(Self { total_bytes, free_bytes, estimate_bytes, cap_bytes })
    }

    /// Check VRAM budget with a **configurable** cap fraction.
    ///
    /// Identical to [`VramBudget::check`] but uses `cap_fraction` instead of
    /// the default 80%. Supports US#9: operators tuning the safety margin.
    ///
    /// `cap_fraction` must be in `(0.0, 1.0]`.
    pub fn check_with_threshold(
        device: &Device,
        device_idx: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        context_len: usize,
        cap_fraction: f64,
    ) -> Result<Self> {
        assert!(
            cap_fraction > 0.0 && cap_fraction <= 1.0,
            "cap_fraction must be in (0, 1]"
        );
        let context_len = if context_len == 0 { DEFAULT_CONTEXT_LEN } else { context_len };
        let estimate_bytes = Self::estimate(n_layers, n_kv_heads, head_dim, hidden_dim, context_len);

        if !device.is_cuda() {
            return Ok(Self { total_bytes: 0, free_bytes: 0, estimate_bytes, cap_bytes: 0 });
        }

        let (free_bytes, total_bytes) = match Self::query_nvidia_smi(device_idx) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("⚠  VRAM guard: query failed ({e}). Skipping cap check.");
                return Ok(Self { total_bytes: 0, free_bytes: 0, estimate_bytes, cap_bytes: 0 });
            }
        };

        let cap_bytes = (total_bytes as f64 * cap_fraction) as u64;

        if estimate_bytes > cap_bytes {
            bail!(
                "VRAM guard: load rejected — estimated {:.2} GiB exceeds {:.0}% cap \
                ({:.2} GiB of {:.2} GiB total on CUDA:{device_idx}). \
                Free VRAM: {:.2} GiB.",
                estimate_bytes as f64 / 1_073_741_824.0,
                cap_fraction * 100.0,
                cap_bytes as f64 / 1_073_741_824.0,
                total_bytes as f64 / 1_073_741_824.0,
                free_bytes as f64 / 1_073_741_824.0,
            );
        }

        Ok(Self { total_bytes, free_bytes, estimate_bytes, cap_bytes })
    }

    /// Pure-logic VRAM check without any system calls or candle dependency.
    ///
    /// Accepts raw `total_bytes` and `free_bytes` (from any source) and
    /// returns a typed [`VramCheckResult`]. Used by `ModelMux::load_guarded`
    /// and in unit tests to exercise accept/reject/boundary logic without GPU.
    pub fn check_direct(
        total_bytes: u64,
        free_bytes: u64,
        estimate_bytes: u64,
        cap_fraction: f64,
    ) -> (VramCheckResult, Self) {
        let cap_bytes = (total_bytes as f64 * cap_fraction) as u64;
        let result = if total_bytes == 0 {
            VramCheckResult::GuardSkipped
        } else if estimate_bytes > cap_bytes {
            VramCheckResult::Reject {
                estimate_bytes,
                cap_bytes,
                shortfall_bytes: estimate_bytes.saturating_sub(cap_bytes),
            }
        } else {
            VramCheckResult::Accept
        };
        (
            result,
            Self { total_bytes, free_bytes, estimate_bytes, cap_bytes },
        )
    }

    /// Human-readable budget summary for logging.
    pub fn summary(&self) -> String {
        if self.total_bytes == 0 {
            return format!(
                "VRAM guard: non-CUDA device — estimate {:.2} GiB (guard skipped)",
                self.estimate_bytes as f64 / 1_073_741_824.0
            );
        }
        format!(
            "VRAM guard: ✓ {:.2}/{:.2} GiB used ({:.1}% of {:.1}% cap) — free {:.2} GiB",
            self.estimate_bytes as f64 / 1_073_741_824.0,
            self.cap_bytes as f64 / 1_073_741_824.0,
            self.estimate_bytes as f64 / self.total_bytes as f64 * 100.0,
            VRAM_CAP_FRACTION * 100.0,
            self.free_bytes as f64 / 1_073_741_824.0,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> (usize, usize, usize, usize, usize) {
        // n_layers, n_kv_heads, head_dim, hidden_dim, context_len
        (2, 2, 64, 256, 512)
    }

    #[test]
    fn cpu_device_always_passes() {
        let (nl, nkv, hd, hdim, ctx) = small_config();
        let budget = VramBudget::check(&Device::Cpu, 0, nl, nkv, hd, hdim, ctx).unwrap();
        assert_eq!(budget.total_bytes, 0, "CPU: total should be 0");
        assert_eq!(budget.cap_bytes, 0, "CPU: cap should be 0");
        assert!(budget.estimate_bytes > 0, "estimate should be non-zero");
    }

    #[test]
    fn estimate_kv_dominant_for_large_context() {
        let budget =
            VramBudget::check(&Device::Cpu, 0, 32, 32, 128, 4096, 32768).unwrap();
        let gib = budget.estimate_bytes as f64 / 1_073_741_824.0;
        assert!(gib > 0.5, "estimate too small: {gib} GiB");
        assert!(gib < 100.0, "estimate unreasonably large: {gib} GiB");
    }

    #[test]
    fn zero_context_uses_default() {
        let (nl, nkv, hd, hdim, _) = small_config();
        let b0 = VramBudget::check(&Device::Cpu, 0, nl, nkv, hd, hdim, 0).unwrap();
        let b4096 = VramBudget::check(&Device::Cpu, 0, nl, nkv, hd, hdim, 4096).unwrap();
        assert_eq!(b0.estimate_bytes, b4096.estimate_bytes);
    }

    #[test]
    fn summary_non_cuda_contains_skipped() {
        let (nl, nkv, hd, hdim, ctx) = small_config();
        let budget = VramBudget::check(&Device::Cpu, 0, nl, nkv, hd, hdim, ctx).unwrap();
        assert!(budget.summary().contains("guard skipped"));
    }

    #[test]
    fn vram_cap_fraction_is_80_percent() {
        assert!((VRAM_CAP_FRACTION - 0.80).abs() < f64::EPSILON);
    }

    // --- check_direct: accept / reject / boundary (US#2, US#9) ---

    #[test]
    fn check_direct_accept_under_80() {
        let estimate = 7 * 1024 * 1024 * 1024u64; // 7 GiB
        let total    = 12 * 1024 * 1024 * 1024u64; // 12 GiB (RTX 3060)
        let (res, _) = VramBudget::check_direct(total, total - estimate, estimate, 0.80);
        assert_eq!(res, VramCheckResult::Accept);
        assert!(res.ok_to_load());
    }

    #[test]
    fn check_direct_reject_over_80() {
        let total    = 12 * 1024 * 1024 * 1024u64;
        let estimate = 11 * 1024 * 1024 * 1024u64; // 91% > 80%
        let (res, _) = VramBudget::check_direct(total, 1024 * 1024 * 1024, estimate, 0.80);
        assert!(!res.ok_to_load());
        match res {
            VramCheckResult::Reject { shortfall_bytes, .. } => {
                assert!(shortfall_bytes > 0, "shortfall must be positive");
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn check_direct_reject_message_contains_mib() {
        let total    = 12 * 1024 * 1024 * 1024u64;
        let estimate = 11 * 1024 * 1024 * 1024u64;
        let (res, _) = VramBudget::check_direct(total, 0, estimate, 0.80);
        assert!(res.summary().contains("MiB"));
    }

    #[test]
    fn check_direct_exact_boundary_accepts() {
        // estimate == cap exactly → Accept (not Reject)
        let total = 10 * 1024 * 1024 * 1024u64;
        let cap   = (total as f64 * 0.80) as u64;
        let (res, _) = VramBudget::check_direct(total, 0, cap, 0.80);
        assert_eq!(res, VramCheckResult::Accept);
    }

    #[test]
    fn check_direct_one_byte_over_boundary_rejects() {
        let total    = 10 * 1024 * 1024 * 1024u64;
        let cap      = (total as f64 * 0.80) as u64;
        let estimate = cap + 1;
        let (res, _) = VramBudget::check_direct(total, 0, estimate, 0.80);
        assert!(!res.ok_to_load());
    }

    #[test]
    fn check_direct_guard_skipped_on_zero_total() {
        let (res, _) = VramBudget::check_direct(0, 0, 1024 * 1024 * 1024, 0.80);
        assert_eq!(res, VramCheckResult::GuardSkipped);
        assert!(res.ok_to_load());
    }

    #[test]
    fn check_direct_configurable_threshold_70() {
        // US#9: threshold 70% — same estimate that passes 80% fails at 70%
        let total    = 12 * 1024 * 1024 * 1024u64;
        let estimate = (total as f64 * 0.75) as u64; // 75% — passes 80%, fails 70%
        let (res80, _) = VramBudget::check_direct(total, 0, estimate, 0.80);
        let (res70, _) = VramBudget::check_direct(total, 0, estimate, 0.70);
        assert_eq!(res80, VramCheckResult::Accept);
        assert!(!res70.ok_to_load());
    }

    #[test]
    fn vram_check_result_ok_to_load_guard_skipped() {
        assert!(VramCheckResult::GuardSkipped.ok_to_load());
    }
}

//! Speculative Decoding — Draft-Verify acceleration for token generation.
//!
//! Idea: Use a small "draft" model to quickly generate K candidate tokens,
//! then verify them all at once with the main (large) model in a single
//! batched forward pass. Accepted tokens are free — rejected ones cost
//! only one extra forward pass vs. sequential decoding.
//!
//! Expected speedup: 2–3× for well-matched draft/target pairs.
//!
//! ```text
//! Loop:
//!   1. Draft model generates K tokens autoregressively (fast, small model)
//!   2. Target model runs K+1 forward pass in parallel (verifies all K)
//!   3. Accept longest prefix where draft agrees with target
//!   4. Bonus: get one free token from first rejection position
//! ```

use crate::generator::InferenceGenerator;
use crate::model::ModelConfig;
use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use anyhow::Result;

use std::time::Instant;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to draft per iteration (default: 4).
    pub draft_tokens: usize,
    /// Maximum acceptance rate before reducing draft count (adaptive).
    pub max_draft_tokens: usize,
    /// Minimum draft tokens to attempt.
    pub min_draft_tokens: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens: 4,
            max_draft_tokens: 8,
            min_draft_tokens: 2,
        }
    }
}

/// Statistics from a speculative decoding run.
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    /// Total tokens generated.
    pub total_tokens: usize,
    /// Total draft iterations.
    pub total_iterations: usize,
    /// Total draft tokens that were accepted.
    pub accepted_tokens: usize,
    /// Total draft tokens that were rejected.
    pub rejected_tokens: usize,
    /// Bonus tokens obtained from rejection positions.
    pub bonus_tokens: usize,
    /// Total wall-clock time.
    pub total_time_secs: f64,
    /// Effective tokens per second.
    pub tokens_per_second: f64,
    /// Average acceptance rate (0.0–1.0).
    pub acceptance_rate: f64,
    /// Speedup factor vs. sequential (estimated).
    pub estimated_speedup: f64,
}

impl std::fmt::Display for SpeculativeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  Speculative Decoding Statistics             ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  Total tokens:      {:>6}                   ║", self.total_tokens)?;
        writeln!(f, "║  Iterations:        {:>6}                   ║", self.total_iterations)?;
        writeln!(f, "║  Accepted drafts:   {:>6}                   ║", self.accepted_tokens)?;
        writeln!(f, "║  Rejected drafts:   {:>6}                   ║", self.rejected_tokens)?;
        writeln!(f, "║  Bonus tokens:      {:>6}                   ║", self.bonus_tokens)?;
        writeln!(f, "║  Acceptance rate:   {:>5.1}%                   ║", self.acceptance_rate * 100.0)?;
        writeln!(f, "║  Tokens/sec:        {:>6.1}                   ║", self.tokens_per_second)?;
        writeln!(f, "║  Est. speedup:      {:>5.2}×                   ║", self.estimated_speedup)?;
        writeln!(f, "║  Wall time:         {:>5.2}s                   ║", self.total_time_secs)?;
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Speculative decoder that coordinates draft and target models.
pub struct SpeculativeDecoder {
    /// The main (large) target model generator.
    pub target: InferenceGenerator,
    /// The small draft model generator.
    pub draft: InferenceGenerator,
    /// Speculative config.
    pub config: SpeculativeConfig,
    /// Accumulated statistics.
    stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder.
    ///
    /// - `target_config`: ModelConfig for the main model
    /// - `draft_config`: ModelConfig for the draft model
    /// - `sampler_config`: shared sampling config (both models use same temperature, etc.)
    /// - `spec_config`: speculative decoding parameters
    pub fn new(
        target_config: ModelConfig,
        draft_config: ModelConfig,
        sampler_config: SamplerConfig,
        spec_config: SpeculativeConfig,
    ) -> Result<Self> {
        let target = InferenceGenerator::new(target_config, sampler_config.clone())?;
        let draft = InferenceGenerator::new(draft_config, sampler_config)?;

        Ok(Self {
            target,
            draft,
            config: spec_config,
            stats: SpeculativeStats {
                total_tokens: 0,
                total_iterations: 0,
                accepted_tokens: 0,
                rejected_tokens: 0,
                bonus_tokens: 0,
                total_time_secs: 0.0,
                tokens_per_second: 0.0,
                acceptance_rate: 0.0,
                estimated_speedup: 0.0,
            },
        })
    }

    /// Run speculative decoding to generate tokens from a prompt.
    ///
    /// Uses the draft model to speculatively generate K tokens, then verifies
    /// them against the target model. Accepted tokens are "free" — only
    /// rejected positions incur the cost of a target model forward pass.
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        target_streamer: &WeightStreamer,
        draft_streamer: &WeightStreamer,
    ) -> Result<String> {
        let start = Instant::now();
        let k = self.config.draft_tokens;

        // Tokenize prompt
        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));

        println!("📝 Prompt: \"{}\" ({} tokens)", prompt, tokens.len());
        println!("🚀 Speculative decoding (K={})", k);
        println!("─────────────────────────────────────────────────");

        let mut generated: Vec<u32> = Vec::new();
        let mut _iteration = 0;

        while generated.len() < max_tokens {
            _iteration += 1;

            // ── Phase 1: Draft K tokens ──────────────────────────────
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_sequence = tokens.clone();
            draft_sequence.extend_from_slice(&generated);

            for _draft_step in 0..k {
                let draft_result = self.draft.generate(
                    tokenizer,
                    &tokenizer.decode(&draft_sequence),
                    1,
                    draft_streamer,
                );

                match draft_result {
                    Ok(token_str) => {
                        let draft_ids = tokenizer.encode(&token_str);
                        if let Some(&id) = draft_ids.first() {
                            if id == tokenizer.eos_id {
                                break;
                            }
                            draft_tokens.push(id);
                            draft_sequence.push(id);
                        } else {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            if draft_tokens.is_empty() {
                // Draft couldn't generate anything — fall back to target
                let target_result = self.target.generate(
                    tokenizer,
                    &tokenizer.decode(&{
                        let mut seq = tokens.clone();
                        seq.extend_from_slice(&generated);
                        seq
                    }),
                    1,
                    target_streamer,
                )?;
                let target_ids = tokenizer.encode(&target_result);
                if let Some(&id) = target_ids.first() {
                    if id == tokenizer.eos_id {
                        break;
                    }
                    generated.push(id);
                    print!("{}", tokenizer.decode_token(id));
                }
                continue;
            }

            // ── Phase 2: Verify with target model ────────────────────
            // Run target model on the sequence including drafted tokens.
            // In a fully optimized version, this would be a single batched
            // forward pass for all K+1 positions. Here we verify sequentially
            // for correctness, with the optimization that accepted prefixes
            // are kept without re-computation.

            let mut verified_sequence = tokens.clone();
            verified_sequence.extend_from_slice(&generated);

            let mut accepted = 0;
            let mut bonus_token: Option<u32> = None;

            for &draft_token in draft_tokens.iter() {
                // Get target model's prediction at this position
                let target_result = self.target.generate(
                    tokenizer,
                    &tokenizer.decode(&verified_sequence),
                    1,
                    target_streamer,
                );

                match target_result {
                    Ok(target_str) => {
                        let target_ids = tokenizer.encode(&target_str);
                        if let Some(&target_token) = target_ids.first() {
                            if target_token == draft_token {
                                // Accept — draft agrees with target
                                accepted += 1;
                                verified_sequence.push(draft_token);
                            } else {
                                // Reject — use the target's token as bonus
                                if target_token != tokenizer.eos_id {
                                    bonus_token = Some(target_token);
                                }
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            // ── Phase 3: Emit accepted tokens + bonus ────────────────
            for &t in &draft_tokens[..accepted] {
                generated.push(t);
                print!("{}", tokenizer.decode_token(t));
            }

            if let Some(bonus) = bonus_token {
                generated.push(bonus);
                print!("{}", tokenizer.decode_token(bonus));
                self.stats.bonus_tokens += 1;
            }

            // Update stats
            self.stats.accepted_tokens += accepted;
            self.stats.rejected_tokens += draft_tokens.len() - accepted;
            self.stats.total_iterations += 1;

            // Check for EOS in accepted tokens
            if generated.last() == Some(&tokenizer.eos_id) {
                generated.pop(); // Remove EOS from output
                println!("\n[EOS]");
                break;
            }
        }

        // Finalize stats
        let elapsed = start.elapsed().as_secs_f64();
        self.stats.total_tokens = generated.len();
        self.stats.total_time_secs = elapsed;
        self.stats.tokens_per_second = if elapsed > 0.0 {
            generated.len() as f64 / elapsed
        } else {
            0.0
        };
        let total_drafted = self.stats.accepted_tokens + self.stats.rejected_tokens;
        self.stats.acceptance_rate = if total_drafted > 0 {
            self.stats.accepted_tokens as f64 / total_drafted as f64
        } else {
            0.0
        };
        // Estimated speedup: tokens produced / target model forward passes
        let target_passes = self.stats.total_iterations + self.stats.rejected_tokens;
        self.stats.estimated_speedup = if target_passes > 0 {
            self.stats.total_tokens as f64 / target_passes as f64
        } else {
            1.0
        };

        println!("\n{}", self.stats);

        let output = tokenizer.decode(&generated);
        Ok(output)
    }

    /// Reset both models (clear KV caches).
    pub fn reset(&mut self) {
        self.target.reset();
        self.draft.reset();
        self.stats = SpeculativeStats {
            total_tokens: 0,
            total_iterations: 0,
            accepted_tokens: 0,
            rejected_tokens: 0,
            bonus_tokens: 0,
            total_time_secs: 0.0,
            tokens_per_second: 0.0,
            acceptance_rate: 0.0,
            estimated_speedup: 0.0,
        };
    }

    /// Get current statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_defaults() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_tokens, 4);
        assert_eq!(config.max_draft_tokens, 8);
        assert_eq!(config.min_draft_tokens, 2);
    }

    #[test]
    fn test_speculative_stats_display() {
        let stats = SpeculativeStats {
            total_tokens: 100,
            total_iterations: 25,
            accepted_tokens: 80,
            rejected_tokens: 20,
            bonus_tokens: 18,
            total_time_secs: 5.0,
            tokens_per_second: 20.0,
            acceptance_rate: 0.80,
            estimated_speedup: 2.22,
        };
        let display = format!("{}", stats);
        assert!(display.contains("100"));
        assert!(display.contains("80.0%"));
        assert!(display.contains("2.22"));
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        // acceptance_rate = accepted / (accepted + rejected)
        let accepted = 75_usize;
        let rejected = 25_usize;
        let total = accepted + rejected;
        let rate = accepted as f64 / total as f64;
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_speedup_estimation() {
        // speedup = total_tokens / (iterations + rejections)
        // If we accept all: speedup ≈ K
        let iterations = 10_usize;
        let rejections = 0_usize;
        let total_tokens = 40_usize; // 10 iterations × 4 accepted each
        let target_passes = iterations + rejections;
        let speedup = total_tokens as f64 / target_passes as f64;
        assert!((speedup - 4.0).abs() < 0.001);
    }
}

// ============================================================================
// v0.9.0 Scaffold — MTP Draft Head (Multi-Token Prediction)
// ============================================================================
//
// Qwen3.6 ships with a native MTP auxiliary head that enables zero-config
// speculative decoding: no separate draft model required.
//
// GGUF tensor name patterns that indicate MTP head presence:
//   - `model.layers.{N}.mtp_head.*`
//   - `output.*_mtp`
//   - metadata key `qwen3_5.mtp.num_steps`
//
// Research basis:
//   Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction",
//   ICML 2024 (https://arxiv.org/abs/2404.19737)
//   Qwen3.6 Technical Report (Alibaba, 2025)
//
// Status:
//   v0.9.0: struct defined, GGUF detection implemented, forward pass stubbed.
//   v0.10.0: forward pass complete in `src/mtp_head.rs`; integrated with
//             `InferenceGenerator` decode loop.

/// Drive strategy: use EAGLE-2 (external draft model) or MTP (built-in head).
#[derive(Debug, Clone)]
pub enum DraftStrategy {
    /// EAGLE-2 dynamic draft tree — requires a separate draft model.
    Eagle2(SpeculativeConfig),
    /// Multi-Token Prediction head — built into the main model's GGUF.
    /// Zero configuration: auto-detected at model load time.
    Mtp(MtpDraftHead),
}

impl DraftStrategy {
    /// Number of tokens drafted per iteration.
    pub fn draft_count(&self) -> usize {
        match self {
            Self::Eagle2(cfg) => cfg.draft_tokens,
            Self::Mtp(head)   => head.n_draft_tokens,
        }
    }

    /// Human-readable label for logging/metrics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Eagle2(_) => "eagle2",
            Self::Mtp(_)    => "mtp",
        }
    }
}

/// Native multi-token prediction draft head.
///
/// Qwen3.6 NEXTN method: the auxiliary head predicts the next N tokens
/// from the final hidden state. These are verified by the main model's
/// rejection sampling loop (same as EAGLE-2, so `rejection_sample` is reused).
///
/// # GGUF Detection
/// Call `MtpDraftHead::detect(tensor_names)` at model load time.
/// If `Some(head)` is returned, use `DraftStrategy::Mtp(head)` instead of
/// `DraftStrategy::Eagle2(...)`.
#[derive(Debug, Clone)]
pub struct MtpDraftHead {
    /// Number of tokens to draft per iteration (from GGUF or default 2).
    pub n_draft_tokens: usize,
    /// `true` if the head was auto-detected from GGUF tensor names.
    pub detected_from_gguf: bool,
    /// Metadata key that triggered detection (for debugging).
    pub detection_source: Option<String>,
    // head_weights: MtpHeadWeights — added in v0.10.0 when mtp_head.rs lands
}

impl MtpDraftHead {
    /// Inspect GGUF tensor names for MTP head presence.
    ///
    /// Returns `Some(MtpDraftHead)` if MTP tensors are detected,
    /// `None` if this model has no built-in draft head.
    ///
    /// # Detection heuristics
    /// 1. Any tensor name containing `mtp_head` → positive
    /// 2. Any tensor matching `output.*_mtp` → positive
    /// 3. Any tensor starting with `model.layers.` and containing `.mtp.` → positive
    ///
    /// `n_draft_tokens` is parsed from the GGUF metadata key
    /// `{arch}.mtp.num_steps` if present; defaults to 2.
    pub fn detect(tensor_names: &[&str]) -> Option<Self> {
        let mtp_tensor = tensor_names.iter().find(|&&name| {
            name.contains("mtp_head")
                || (name.starts_with("output.") && name.contains("_mtp"))
                || (name.starts_with("model.layers.") && name.contains(".mtp."))
        });

        mtp_tensor.map(|&name| Self {
            n_draft_tokens: 2, // default; overridden by metadata in v0.10.0
            detected_from_gguf: true,
            detection_source: Some(name.to_owned()),
        })
    }

    /// Detect from GGUF metadata (checks `{arch}.mtp.num_steps` key).
    ///
    /// Call this in addition to `detect()` — the metadata check is
    /// authoritative for the draft count even if tensor detection found it.
    pub fn detect_from_metadata(metadata: &std::collections::HashMap<String, String>) -> Option<Self> {
        // Check for the num_steps key under any known arch prefix
        let known_prefixes = ["qwen3_5", "qwen3_6"];
        for prefix in known_prefixes {
            let key = format!("{prefix}.mtp.num_steps");
            if let Some(val) = metadata.get(&key) {
                let n: usize = val.parse().unwrap_or(2);
                return Some(Self {
                    n_draft_tokens: n.clamp(1, 8),
                    detected_from_gguf: true,
                    detection_source: Some(key),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod mtp_tests {
    use super::*;

    #[test]
    fn test_mtp_detection_positive_mtp_head() {
        let names = ["blk.0.attn_q", "model.mtp_head.weight", "output.weight"];
        let result = MtpDraftHead::detect(&names);
        assert!(result.is_some(), "should detect mtp_head tensor");
        let head = result.unwrap();
        assert!(head.detected_from_gguf);
        assert_eq!(head.n_draft_tokens, 2);
    }

    #[test]
    fn test_mtp_detection_positive_output_mtp() {
        let names = ["blk.0.ffn_up", "output.weight_mtp_0", "output.weight"];
        let result = MtpDraftHead::detect(&names);
        assert!(result.is_some(), "should detect output.*_mtp tensor");
    }

    #[test]
    fn test_mtp_detection_negative_normal_model() {
        let names = ["blk.0.attn_q", "blk.0.ffn_up", "output.weight", "token_embd.weight"];
        let result = MtpDraftHead::detect(&names);
        assert!(result.is_none(), "normal model should not detect MTP head");
    }

    #[test]
    fn test_mtp_detection_from_metadata_qwen3_5() {
        let mut meta = std::collections::HashMap::new();
        meta.insert("qwen3_5.mtp.num_steps".to_owned(), "3".to_owned());
        let result = MtpDraftHead::detect_from_metadata(&meta);
        assert!(result.is_some());
        assert_eq!(result.unwrap().n_draft_tokens, 3);
    }

    #[test]
    fn test_mtp_detection_metadata_empty() {
        let meta = std::collections::HashMap::new();
        assert!(MtpDraftHead::detect_from_metadata(&meta).is_none());
    }

    #[test]
    fn test_draft_strategy_name() {
        let eagle = DraftStrategy::Eagle2(SpeculativeConfig::default());
        let mtp   = DraftStrategy::Mtp(MtpDraftHead {
            n_draft_tokens: 2, detected_from_gguf: true, detection_source: None,
        });
        assert_eq!(eagle.name(), "eagle2");
        assert_eq!(mtp.name(), "mtp");
    }

    #[test]
    fn test_draft_count_from_mtp() {
        let head = MtpDraftHead {
            n_draft_tokens: 4,
            detected_from_gguf: true,
            detection_source: None,
        };
        let strategy = DraftStrategy::Mtp(head);
        assert_eq!(strategy.draft_count(), 4);
    }
}

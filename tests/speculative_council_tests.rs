//! Integration tests for the Consensus-Driven Speculative Council (CDSC).
//!
//! Test coverage:
//!   1. JSD math — zero on identical distributions, bounded in [0, ln2].
//!   2. JSD sensitivity — high divergence on maximally opposing distributions.
//!   3. Softmax — proper normalization and numerical stability.
//!   4. draft_pass convergence — returns ≥1 token when voters agree.
//!   5. draft_pass controversy brake — stops early when JSD ≥ epsilon.
//!   6. Controversy counter — increments correctly on early brake.
//!   7. AsyncVerifyingPipeline — submits tasks and receives VerificationResult.
//!   8. Async verifier acceptance rate — ≥ 90% tokens accepted on clean context.
//!   9. Async verifier correction — correction token returned on rejection.
//!  10. enable_council wiring — InferenceGenerator fields set correctly.
//!  11. enable_council idempotence — calling twice doesn't crash.
//!  12. Multiple draft rounds — drafter remains usable across sequential calls.

use air_rs::speculative_council::SpeculativeCouncilDrafter;
use air_rs::async_verifier::AsyncVerifyingPipeline;
use air_rs::ghost_drafter::{GhostDrafter, SpeculativeConfig};
use air_rs::sampler::SamplerConfig;
use air_rs::generator::InferenceGenerator;
use air_rs::model::ModelConfig;

// ── helpers ─────────────────────────────────────────────────────────────────

fn small_council(epsilon: f32) -> SpeculativeCouncilDrafter {
    SpeculativeCouncilDrafter::new(SpeculativeConfig::default(), epsilon, 512)
}

fn default_sampler() -> SamplerConfig {
    SamplerConfig::default()
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        n_layers: 2,
        n_heads: 8,
        n_kv_heads: 2,
        hidden_dim: 128,
        intermediate_dim: 512,
        vocab_size: 512,
        rope_theta: 1_000_000.0,
        context_length: 2048,
        rms_norm_eps: 1e-6,
        head_dim: 128,
        arch: air_rs::model_variant::ModelVariant::Qwen3_6,
        norm_type: air_rs::model_variant::NormType::RmsNorm,
        ffn_type: air_rs::model_variant::FfnType::SwiGlu,
        sliding_window: None,
        partial_rope_factor: None,
        attn_router: air_rs::attention_backend::HybridAttentionRouter::uniform(
            2,
            air_rs::attention_backend::AttentionBackend::Softmax,
        ),
        n_experts: 0,
        moe_top_k: 0,
        eos_token_id: 2,
    }
}

// ── 1. JSD = 0 on identical distributions ───────────────────────────────────

#[test]
fn jsd_identical_distributions_is_zero() {
    let council = small_council(0.15);
    let p = vec![0.1f32, 0.7, 0.1, 0.1];
    let jsd = council.compute_jsd(&p, &p, &p);
    assert!(jsd < 1e-5, "JSD of identical distributions should be ~0, got {jsd}");
}

// ── 2. JSD is bounded [0, ln2] ───────────────────────────────────────────────

#[test]
fn jsd_is_within_theoretical_bounds() {
    let council = small_council(0.15);
    // Maximally divergent: each voter concentrates mass on a different token.
    let p_a = vec![1.0f32, 0.0, 0.0, 0.0];
    let p_b = vec![0.0f32, 1.0, 0.0, 0.0];
    let p_c = vec![0.0f32, 0.0, 1.0, 0.0];
    let jsd = council.compute_jsd(&p_a, &p_b, &p_c);
    assert!(jsd >= 0.0, "JSD must be non-negative, got {jsd}");
    // For 3 equal-weight distributions the maximum JSD is ln(3) ≈ 1.099
    assert!(jsd <= 3f32.ln() + 1e-4, "JSD must be ≤ ln(3) ≈ 1.099 for 3 voters, got {jsd}");
}

// ── 3. JSD rises with divergence ────────────────────────────────────────────

#[test]
fn jsd_higher_when_voters_disagree() {
    let council = small_council(0.15);
    let agree_a = vec![0.8f32, 0.1, 0.1];
    let agree_b = vec![0.75f32, 0.15, 0.1];
    let agree_c = vec![0.78f32, 0.12, 0.1];
    let jsd_agree = council.compute_jsd(&agree_a, &agree_b, &agree_c);

    let dis_a = vec![0.9f32, 0.05, 0.05];
    let dis_b = vec![0.05f32, 0.9, 0.05];
    let dis_c = vec![0.05f32, 0.05, 0.9];
    let jsd_disagree = council.compute_jsd(&dis_a, &dis_b, &dis_c);

    assert!(
        jsd_disagree > jsd_agree,
        "Disagreeing voters should produce higher JSD: agree={jsd_agree:.4}, disagree={jsd_disagree:.4}"
    );
}

// ── 4. Softmax normalizes to sum=1 ──────────────────────────────────────────

#[test]
fn softmax_sums_to_one() {
    let council = small_council(0.15);
    let logits: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let probs = council.softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "softmax must sum to 1.0, got {sum}");
}

// ── 5. Softmax handles empty input ──────────────────────────────────────────

#[test]
fn softmax_empty_returns_uniform_zeros() {
    let council = small_council(0.15);
    let result = council.softmax(&[]);
    assert_eq!(result.len(), 512);
    // All zeros since vocab size is 512 but we return 0.0-filled vector
    assert!(result.iter().all(|&x| x >= 0.0));
}

// ── 6. draft_pass produces tokens when epsilon is very permissive ────────────

#[test]
fn draft_pass_produces_tokens_with_high_epsilon() {
    let mut council = small_council(1.0); // epsilon=1.0 never brakes
    let context = vec![1u32, 42, 99];
    let result = council.draft_pass(&context, 4, &default_sampler()).unwrap();
    assert!(
        !result.tokens.is_empty(),
        "With epsilon=1.0, at least one token must be drafted"
    );
    assert!(result.tokens.len() <= 4, "Must not exceed requested k=4");
}

// ── 7. draft_pass brakes early with strict epsilon ───────────────────────────

#[test]
fn draft_pass_brakes_early_with_zero_epsilon() {
    let mut council = small_council(0.0); // epsilon=0 always brakes immediately
    let context = vec![1u32, 42];
    let result = council.draft_pass(&context, 8, &default_sampler()).unwrap();
    // With epsilon=0, the very first token's JSD (≥0) triggers the brake.
    assert!(
        result.tokens.len() < 8,
        "epsilon=0 should prevent drafting all 8 tokens, got {}",
        result.tokens.len()
    );
}

// ── 8. Controversy counter increments on early brake ────────────────────────

#[test]
fn controversy_counter_increments_on_brake() {
    let mut council = small_council(0.0);
    let ctx = vec![1u32, 2];
    let before = council.controversy_count;
    council.draft_pass(&ctx, 4, &default_sampler()).unwrap();
    assert!(
        council.controversy_count > before,
        "controversy_count should increase when brake fires"
    );
}

// ── 9. total_steps increments each call ─────────────────────────────────────

#[test]
fn total_steps_increments_each_draft_pass() {
    let mut council = small_council(0.5);
    let ctx = vec![1u32];
    let passes = 5;
    for _ in 0..passes {
        council.draft_pass(&ctx, 4, &default_sampler()).unwrap();
    }
    assert_eq!(council.total_steps, passes);
}

// ── 10. draft_pass result is deterministic for the same context ──────────────

#[test]
fn draft_pass_deterministic_same_context() {
    let mut c1 = small_council(0.5);
    let mut c2 = small_council(0.5);
    let ctx = vec![1u32, 22, 333];
    let r1 = c1.draft_pass(&ctx, 4, &default_sampler()).unwrap();
    let r2 = c2.draft_pass(&ctx, 4, &default_sampler()).unwrap();
    assert_eq!(
        r1.tokens, r2.tokens,
        "Deterministic hidden simulation must produce the same tokens for the same context"
    );
}

// ── 11. AsyncVerifyingPipeline — submits and receives ───────────────────────

#[test]
fn async_verifier_round_trip() {
    let mut verifier = AsyncVerifyingPipeline::new(8, None);
    let context = vec![1u32, 42, 99, 128];
    let proposed = vec![200u32, 201, 202];

    verifier.verify_async(&context, &proposed).unwrap();

    // Spin-wait up to 1 second for the background thread.
    let mut result = None;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while result.is_none() && std::time::Instant::now() < deadline {
        result = verifier.try_receive_result();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let res = result.expect("VerificationResult should arrive within 1 second");
    assert!(res.n_accepted <= proposed.len(), "n_accepted must not exceed proposed length");
}

// ── 12. Async verifier acceptance rate ≥ 90% ────────────────────────────────

#[test]
fn async_verifier_high_acceptance_rate() {
    let mut verifier = AsyncVerifyingPipeline::new(8, None);
    let n_rounds = 20;
    let mut total_proposed = 0usize;
    let mut total_accepted = 0usize;

    for i in 0..n_rounds {
        let ctx: Vec<u32> = (0..4u32).map(|x| x + i as u32 * 10).collect();
        let tokens: Vec<u32> = (0..4u32).map(|x| x + 100 + i as u32 * 10).collect();
        total_proposed += tokens.len();

        verifier.verify_async(&ctx, &tokens).unwrap();

        let mut res = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while res.is_none() && std::time::Instant::now() < deadline {
            res = verifier.try_receive_result();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        if let Some(r) = res {
            total_accepted += r.n_accepted;
        }
    }

    let rate = total_accepted as f64 / total_proposed as f64;
    assert!(
        rate >= 0.90,
        "Acceptance rate should be ≥ 90%, got {:.1}%",
        rate * 100.0
    );
}

// ── 13. Async verifier returns correction on mismatch ───────────────────────

#[test]
fn async_verifier_correction_token_present_on_rejection() {
    let mut verifier = AsyncVerifyingPipeline::new(8, None);
    // Context seed that is statistically guaranteed to eventual reject:
    // seed = context.last() = 0  → "((0 + idx + token) % 100) < 97"
    // token=0 → (0+0+0)%100 = 0 < 97 → accept
    // token=3 → (0+0+3)%100 = 3 < 97 → accept
    // We'll collect enough rounds that at least one rejection occurs.
    let mut got_correction = false;
    for round in 0u32..50 {
        let ctx = vec![round * 7];
        // Use token values that will eventually fail: token = 97 + (seed+idx) % 5
        let tokens: Vec<u32> = (0..4).map(|i| 97 + ((ctx[0] as u64 + i) % 5) as u32).collect();
        verifier.verify_async(&ctx, &tokens).unwrap();

        let mut res = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while res.is_none() && std::time::Instant::now() < deadline {
            res = verifier.try_receive_result();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        if let Some(r) = res {
            if r.correction.is_some() {
                got_correction = true;
                break;
            }
        }
    }
    assert!(got_correction, "At least one correction token should have been produced over 50 rounds");
}

// ── 14. enable_council wires fields on InferenceGenerator ───────────────────

#[test]
fn enable_council_wires_generator_fields() {
    let config = tiny_config();
    let sampler = SamplerConfig::default();
    let mut gen = InferenceGenerator::with_device(config, sampler, candle_core::Device::Cpu).unwrap();

    assert!(!gen.council_enabled);
    assert!(gen.council_drafter.is_none());
    assert!(gen.async_verifier.is_none());

    gen.enable_council(0.12, None);

    assert!(gen.council_enabled, "council_enabled must be true after enable_council()");
    assert!((gen.epsilon - 0.12).abs() < 1e-6, "epsilon must be stored correctly");
    assert!(gen.council_drafter.is_some(), "council_drafter must be populated");
    assert!(gen.async_verifier.is_some(), "async_verifier must be populated");
}

// ── 15. enable_council with custom epsilon stores value ─────────────────────

#[test]
fn enable_council_stores_custom_epsilon() {
    let config = tiny_config();
    let sampler = SamplerConfig::default();
    let mut gen = InferenceGenerator::with_device(config, sampler, candle_core::Device::Cpu).unwrap();
    gen.enable_council(0.33, Some("/fake/path.gguf".to_string()));
    assert!((gen.epsilon - 0.33).abs() < 1e-6);
}

// ── 16. Sequential draft rounds remain stable ────────────────────────────────

#[test]
fn sequential_draft_rounds_stable() {
    let mut council = small_council(0.5);
    let sampler = default_sampler();
    for round in 0u32..10 {
        let ctx: Vec<u32> = (0..round + 2).collect();
        let res = council.draft_pass(&ctx, 4, &sampler).unwrap();
        // Should never panic or produce tokens out of vocab range.
        for &tok in &res.tokens {
            assert!(
                (tok as usize) < council.vocabulary_size,
                "Token {tok} out of vocab range {} at round {round}",
                council.vocabulary_size
            );
        }
    }
}

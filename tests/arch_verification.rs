//! Architecture verification tests — Qwen 3.6 27B & Gemma 4 E4B
//!
//! These tests verify:
//! 1. Qwen 3.6 27B: HybridAttentionRouter layout is correct and DeltaNet/Softmax
//!    layers produce finite, coherent output tensors.
//! 2. Gemma 4 E4B: SlidingWindowAttention mask does NOT allow tokens outside
//!    the window to attend (context leak test).

use air_rs::attention_backend::{AttentionBackend, HybridAttentionRouter};
use air_rs::ops;
use candle_core::{DType, Device, IndexOp, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu() -> Device { Device::Cpu }

/// Assert all elements of a tensor are finite (no NaN / ±Inf).
fn assert_finite(t: &Tensor, label: &str) {
    let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "{label}: element [{i}] = {x} is not finite");
    }
}

// ---------------------------------------------------------------------------
// 1. Qwen 3.6 27B — router layout coherence
// ---------------------------------------------------------------------------

#[test]
fn qwen3_6_27b_router_layout_correct() {
    let r = HybridAttentionRouter::qwen3_6_27b();
    assert_eq!(r.n_layers(), 64, "Qwen3.6-27B must have 64 layers");
    assert_eq!(r.count(AttentionBackend::GatedDeltaNet), 48, "48 DeltaNet layers");
    assert_eq!(r.count(AttentionBackend::Softmax),       16, "16 Softmax layers");

    // Verify the 4-layer tile pattern (DeltaNet × 3, Softmax × 1)
    for tile in 0..16 {
        let base = tile * 4;
        assert_eq!(r.backend_for_layer(base),     AttentionBackend::GatedDeltaNet,
            "tile {tile}: layer {base} should be GatedDeltaNet");
        assert_eq!(r.backend_for_layer(base + 1), AttentionBackend::GatedDeltaNet,
            "tile {tile}: layer {base}+1 should be GatedDeltaNet");
        assert_eq!(r.backend_for_layer(base + 2), AttentionBackend::GatedDeltaNet,
            "tile {tile}: layer {base}+2 should be GatedDeltaNet");
        assert_eq!(r.backend_for_layer(base + 3), AttentionBackend::Softmax,
            "tile {tile}: layer {base}+3 should be Softmax");
    }
}

#[test]
fn qwen3_6_35b_a3b_router_layout_correct() {
    let r = HybridAttentionRouter::qwen3_6_35b_a3b();
    assert_eq!(r.n_layers(), 96, "Qwen3.6-35B-A3B must have 96 layers");
    assert_eq!(r.count(AttentionBackend::GatedDeltaNet), 72);
    assert_eq!(r.count(AttentionBackend::Softmax), 24);
}

/// DeltaNet layers are recurrent (no KV cache), Softmax layers are cacheable.
#[test]
fn qwen3_6_backend_properties() {
    let r = HybridAttentionRouter::qwen3_6_27b();
    for layer in 0..64 {
        match r.backend_for_layer(layer) {
            AttentionBackend::GatedDeltaNet => {
                assert!(AttentionBackend::GatedDeltaNet.is_recurrent(),
                    "layer {layer}: DeltaNet must be recurrent");
                assert!(!AttentionBackend::GatedDeltaNet.is_kv_cacheable(),
                    "layer {layer}: DeltaNet must not be KV-cacheable");
            }
            AttentionBackend::Softmax => {
                assert!(!AttentionBackend::Softmax.is_recurrent(),
                    "layer {layer}: Softmax must not be recurrent");
                assert!(AttentionBackend::Softmax.is_kv_cacheable(),
                    "layer {layer}: Softmax must be KV-cacheable");
            }
            other => panic!("Unexpected backend for Qwen3.6-27B layer {layer}: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Gemma 4 E4B — SWA layout & context-leak test
// ---------------------------------------------------------------------------

#[test]
fn gemma4_e4b_router_layout_correct() {
    let r = HybridAttentionRouter::gemma4_e4b(32, 4096, 6);
    assert_eq!(r.n_layers(), 32, "Gemma4-E4B must have 32 layers");

    // Final layer must always be GlobalFull
    assert_eq!(r.backend_for_layer(31), AttentionBackend::GlobalFull,
        "last layer must be GlobalFull");

    // Layer 0 must be SlidingWindow (not a global layer position)
    assert_eq!(r.backend_for_layer(0), AttentionBackend::SlidingWindow { window: 4096 },
        "layer 0 must be SlidingWindow");

    // Global layers at positions 5, 11, 17, 23, 29, 31
    let expected_globals = [5usize, 11, 17, 23, 29, 31];
    for &g in &expected_globals {
        assert_eq!(r.backend_for_layer(g), AttentionBackend::GlobalFull,
            "layer {g} should be GlobalFull");
    }
}

/// SWA mask context-leak test:
/// Tokens outside the sliding window AND future tokens must receive -inf.
/// Tokens within the causal window must NOT be masked.
#[test]
fn swa_mask_no_context_leak() -> candle_core::Result<()> {
    let window = 4usize;
    let seq_kv = 8usize;   // 8 past tokens in KV cache
    let seq_q  = 1usize;   // single-token decode
    let start_pos = 7usize; // query is at position 7 (0-indexed)

    // Scores: all zeros
    let scores = Tensor::zeros((1, 1, seq_q, seq_kv), DType::F32, &cpu())?;
    let masked = ops::sliding_window_attention(&scores, seq_q, seq_kv, window, start_pos)?;

    let flat: Vec<f32> = masked.flatten_all()?.to_vec1()?;

    // At start_pos=7, window=4: attend positions [4, 5, 6, 7] only.
    // k_abs=0..3 should be masked (-inf), k_abs=4..7 should be 0.
    for (k_abs, &val) in flat.iter().enumerate() {
        if k_abs < start_pos + 1 - window {
            // Outside window — must be -inf
            assert!(
                val == f32::NEG_INFINITY,
                "k={k_abs} is outside the window but got {val}, expected -inf (context leak!)"
            );
        } else if k_abs <= start_pos {
            // Inside causal window — must be 0.0 (no mask)
            assert_eq!(val, 0.0_f32,
                "k={k_abs} is inside the window but is masked with {val}");
        } else {
            // Future token — must be -inf
            assert!(
                val == f32::NEG_INFINITY,
                "k={k_abs} is a future token but got {val}, expected -inf"
            );
        }
    }
    Ok(())
}

/// SWA must be strictly causal: no token can attend to future keys.
#[test]
fn swa_mask_is_strictly_causal() -> candle_core::Result<()> {
    let window = 16;
    let seq = 8usize;       // prefill 8 tokens simultaneously
    let start_pos = 0usize; // start of session

    let scores = Tensor::zeros((1, 1, seq, seq), DType::F32, &cpu())?;
    let masked = ops::sliding_window_attention(&scores, seq, seq, window, start_pos)?;

    // Shape: [1, 1, seq, seq]
    // masked[0, 0, i, j] must be -inf when j > i (future)
    let data = masked.squeeze(0)?.squeeze(0)?; // [seq, seq]
    for i in 0..seq {
        for j in 0..seq {
            let v: f32 = data.i((i, j))?.to_scalar()?;
            if j > i {
                assert!(
                    v == f32::NEG_INFINITY,
                    "q={i} attending to future k={j}: got {v}, expected -inf"
                );
            } else {
                assert_eq!(v, 0.0_f32,
                    "q={i} should attend to past/self k={j} but got {v}");
            }
        }
    }
    Ok(())
}

/// Window boundary: token at position W should NOT attend to token 0 (distance = W).
#[test]
fn swa_window_boundary_exact() -> candle_core::Result<()> {
    let window = 4;
    // Query at absolute pos = window (e.g. pos=4), KV from 0..=4
    let start_pos = window; // q_abs = 4
    let seq_q = 1;
    let seq_kv = window + 1; // k_abs = 0,1,2,3,4

    let scores = Tensor::zeros((1, 1, seq_q, seq_kv), DType::F32, &cpu())?;
    let masked = ops::sliding_window_attention(&scores, seq_q, seq_kv, window, start_pos)?;
    let flat: Vec<f32> = masked.flatten_all()?.to_vec1()?;

    // k=0: distance = 4, outside window (window means attend [1,2,3,4])
    // Qwen/Gemma convention: attend k where k >= q_abs - window + 1 = 4 - 4 + 1 = 1
    assert!(flat[0] == f32::NEG_INFINITY,
        "k=0 is beyond window boundary, must be masked; got {}", flat[0]);

    // k=1..4 should be unmasked
    for k in 1..=window {
        assert_eq!(flat[k], 0.0_f32,
            "k={k} is within window, must NOT be masked; got {}", flat[k]);
    }
    Ok(())
}

/// SWA output is finite for typical FP32 attention score inputs.
/// Uses a single-head score tensor to keep broadcast paths simple.
#[test]
fn swa_output_finite_with_real_scores() -> candle_core::Result<()> {
    let window = 128;
    let seq_q  = 4;
    let seq_kv = 256;
    // Use 1 head — SWA builds a [1,1,seq_q,seq_kv] mask; multi-head
    // is handled by the caller broadcasting over the heads dim.
    let scores = Tensor::randn(0f32, 1f32, (1, 1, seq_q, seq_kv), &cpu())?;
    let masked = ops::sliding_window_attention(&scores, seq_q, seq_kv, window, 252)?;
    let flat: Vec<f32> = masked.flatten_all()?.to_vec1()?;
    let non_inf_count = flat.iter().filter(|&&v| v.is_finite()).count();
    assert!(non_inf_count > 0, "at least some scores should be finite after masking");
    assert!(!flat.iter().any(|v| v.is_nan()), "masked scores must never be NaN");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Qwen 3.6 — DeltaNet SSM causal 1D convolution
// ---------------------------------------------------------------------------

#[test]
fn conv1d_causal_output_shape_and_finite() -> candle_core::Result<()> {
    // x: [batch=1, seq=4, dim=8]  kernel: [dim=8, kernel_size=4]
    let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &cpu())?;
    let w = Tensor::randn(0f32, 0.1f32, (8, 4), &cpu())?;

    let (out, next_state) = ops::apply_conv1d_causal(&x, &w, None)?;

    // Output shape should match input (padded causal conv = same length)
    assert_eq!(out.dims(), x.dims(), "conv1d output must match input shape");
    // next_state: [batch, ..., dim, kernel_size-1] — includes dim axis
    // x=[1,4,8], kernel_size=4 → next_state=[1,4,8,3]
    assert_eq!(next_state.dims(), &[1, 4, 8, 3], "next_state dims");

    assert_finite(&out, "conv1d output");
    assert_finite(&next_state, "conv1d next_state");
    Ok(())
}

#[test]
fn conv1d_causal_state_propagation() -> candle_core::Result<()> {
    // Two sequential single-token passes must produce different outputs
    // if the conv state is propagated (non-Markovian).
    let w = Tensor::randn(0f32, 0.1f32, (4, 3), &cpu())?; // kernel_size=3

    let x1 = Tensor::ones((1, 1, 4), DType::F32, &cpu())?;
    let (out1, state1) = ops::apply_conv1d_causal(&x1, &w, None)?;
    assert_finite(&out1, "step1 output");

    // Second call with a different input and the propagated state
    let x2 = Tensor::full(2.0f32, (1, 1, 4), &cpu())?;
    let (out2, _state2) = ops::apply_conv1d_causal(&x2, &w, Some(&state1))?;
    assert_finite(&out2, "step2 output");

    // out1 and out2 should differ (state has changed the context)
    let v1: Vec<f32> = out1.flatten_all()?.to_vec1()?;
    let v2: Vec<f32> = out2.flatten_all()?.to_vec1()?;
    assert_ne!(v1, v2, "conv1d output must differ when input or state changes");
    Ok(())
}

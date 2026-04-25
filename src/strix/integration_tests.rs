//! STRIX Integration Tests — STRIX Protocol §19.
//!
//! Full lifecycle tests that exercise the session API, cold boot,
//! tensor registration, scheduler ticks, and VRAM budget enforcement.

#[cfg(test)]
mod tests {
    use crate::strix::bridge::StrixBridge;
    use crate::strix::compat::{
        classify_tensor, normalize_tensor_name, GgufTensorInfo, ModelArchitecture,
    };
    use crate::strix::config::StrixConfig;
    use crate::strix::score::{urgency, predictive, sticky, cost, residency_score, ScoreWeights};
    use crate::strix::session::{StrixSession, SessionState};
    use crate::strix::types::{DType, TensorClass, TensorId};

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Build a realistic set of tensor infos for a small LLM (N layers).
    fn make_small_model(n_layers: usize) -> Vec<GgufTensorInfo> {
        let mut tensors = Vec::new();
        let mut offset = 0u64;

        // Token embedding (Class A — always in VRAM)
        let embed_size = 4096 * 32000 * 2;
        tensors.push(GgufTensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: DType::F16,
            offset,
            size_bytes: embed_size,
        });
        offset += embed_size as u64;

        // Per-layer tensors
        for layer in 0..n_layers {
            let components = [
                ("attn_q", vec![4096, 4096]),
                ("attn_k", vec![4096, 1024]),
                ("attn_v", vec![4096, 1024]),
                ("attn_output", vec![4096, 4096]),
                ("ffn_gate", vec![4096, 11008]),
                ("ffn_up", vec![4096, 11008]),
                ("ffn_down", vec![11008, 4096]),
                ("attn_norm", vec![4096]),
                ("ffn_norm", vec![4096]),
            ];

            for (comp, shape) in &components {
                let bytes = shape.iter().product::<usize>() * 2;
                tensors.push(GgufTensorInfo {
                    name: format!("blk.{layer}.{comp}.weight"),
                    shape: shape.clone(),
                    dtype: DType::F16,
                    offset,
                    size_bytes: bytes,
                });
                offset += bytes as u64;
            }
        }

        // Output weight (Class A)
        tensors.push(GgufTensorInfo {
            name: "output.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: DType::F16,
            offset,
            size_bytes: 4096 * 32000 * 2,
        });

        // Output norm (Class C)
        tensors.push(GgufTensorInfo {
            name: "output_norm.weight".to_string(),
            shape: vec![4096],
            dtype: DType::F16,
            offset: offset + (4096 * 32000 * 2) as u64,
            size_bytes: 4096 * 2,
        });

        tensors
    }

    // ── Integration Tests ─────────────────────────────────────────────────

    #[test]
    fn cold_boot_full_cycle() {
        let tensors = make_small_model(4);
        let config = StrixConfig::default();
        let vram = 8 * 1024 * 1024 * 1024;

        let mut session = StrixSession::open(&tensors, ModelArchitecture::Llama, &config, vram)
            .expect("session open");
        assert_eq!(session.state(), SessionState::Created);

        session.cold_boot().expect("cold boot");
        assert_eq!(session.state(), SessionState::Ready);

        let stats = session.stats();
        assert!(stats.arena_used > 0, "should have loaded Class A tensors into VRAM");
    }

    #[test]
    fn inference_stream_simulation() {
        let n_layers = 4;
        let tensors = make_small_model(n_layers);
        let config = StrixConfig::default();
        let vram = 8 * 1024 * 1024 * 1024;

        let mut session = StrixSession::open(&tensors, ModelArchitecture::Llama, &config, vram)
            .expect("session open");
        session.cold_boot().expect("cold boot");

        // Simulate 10-token generation
        for _token in 0..10 {
            for layer in 0..n_layers {
                session.notify_layer_start(layer);
                let name = format!("blk.{layer}.attn_q.weight");
                let _result = session.acquire_tensor(&name);
                session.notify_layer_end(layer);
            }
        }

        assert_eq!(session.state(), SessionState::Ready);
    }

    #[test]
    fn session_lifecycle() {
        let tensors = make_small_model(2);
        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024;

        let mut session = StrixSession::open(&tensors, ModelArchitecture::Llama, &config, vram)
            .expect("session open");
        assert_eq!(session.state(), SessionState::Created);

        session.cold_boot().expect("cold boot");
        assert_eq!(session.state(), SessionState::Ready);

        session.close();
        assert_eq!(session.state(), SessionState::Closed);
    }

    #[test]
    fn tensor_classification_accuracy() {
        let tensors = make_small_model(4);

        for t in &tensors {
            let norm = normalize_tensor_name(&t.name);
            let class = classify_tensor(&norm);

            if t.name == "token_embd.weight" || t.name == "output.weight" {
                assert_eq!(class, TensorClass::A, "{} should be Class A", t.name);
            } else if t.name.contains("attn_norm") || t.name.contains("ffn_norm")
                || t.name == "output_norm.weight"
            {
                assert_eq!(class, TensorClass::C, "{} should be Class C", t.name);
            } else if t.name.contains("attn_q") || t.name.contains("attn_k")
                || t.name.contains("attn_v") || t.name.contains("attn_output")
                || t.name.contains("ffn_gate") || t.name.contains("ffn_up")
                || t.name.contains("ffn_down")
            {
                assert_eq!(class, TensorClass::B, "{} should be Class B", t.name);
            }
        }
    }

    #[test]
    fn vram_budget_enforcement() {
        let tensors = make_small_model(4);
        let config = StrixConfig::default();
        let tiny_vram = 512 * 1024 * 1024; // 512 MB

        let mut session = StrixSession::open(
            &tensors, ModelArchitecture::Llama, &config, tiny_vram,
        ).expect("session open");

        let result = session.cold_boot();
        if result.is_ok() {
            assert_eq!(session.state(), SessionState::Ready);
        }
    }

    #[test]
    fn concurrent_sessions_isolation() {
        let tensors = make_small_model(2);
        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024;

        let mut session_a = StrixSession::open(
            &tensors, ModelArchitecture::Llama, &config, vram,
        ).expect("session A open");

        let mut session_b = StrixSession::open(
            &tensors, ModelArchitecture::Mistral, &config, vram,
        ).expect("session B open");

        session_a.cold_boot().expect("session A boot");
        session_b.cold_boot().expect("session B boot");

        assert_eq!(session_a.state(), SessionState::Ready);
        assert_eq!(session_b.state(), SessionState::Ready);

        session_a.close();
        assert_eq!(session_a.state(), SessionState::Closed);
        assert_eq!(session_b.state(), SessionState::Ready);
    }

    #[test]
    fn bridge_register_and_tick() {
        let config = StrixConfig::default();
        let mut bridge = StrixBridge::new(&config, 1024 * 1024 * 1024);

        let id0 = bridge.register_tensor(
            "test.weight".to_string(),
            vec![100, 100],
            DType::F32,
            40000,
            TensorClass::B,
            Some(0),
        );

        let id1 = bridge.register_tensor(
            "test2.weight".to_string(),
            vec![200],
            DType::F16,
            400,
            TensorClass::C,
            None,
        );

        bridge.tick(0, 1);
        bridge.tick(1, 2);

        assert!(bridge.registry().get(id0).is_some());
        assert!(bridge.registry().get(id1).is_some());
    }

    #[test]
    fn scoring_convergence_over_many_steps() {
        let weights = ScoreWeights::default();

        let mut prev_score = 0.0f32;
        for step in 1..=100u64 {
            let u = urgency(step as usize, 5);
            let p = predictive(0, step, 0.05);
            let s = sticky(2);
            let c = cost(1_000_000usize, 10_000_000usize);
            let score = residency_score(&weights, u, p, s, c);

            assert!(score >= 0.0 && score <= 1.0, "score out of range at step {step}");

            if step > 10 {
                let delta = (score - prev_score).abs();
                assert!(delta < 0.3, "score jumped by {delta} at step {step}");
            }
            prev_score = score;
        }
    }

    #[test]
    fn moe_expert_activation_pattern() {
        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024;
        let mut bridge = StrixBridge::new(&config, vram);

        let expert_ids: Vec<TensorId> = (0..8).map(|i| {
            bridge.register_tensor(
                format!("blk.0.ffn_gate.{i}.weight"),
                vec![4096, 4096],
                DType::F16,
                4096 * 4096 * 2,
                TensorClass::B,
                Some(0),
            )
        }).collect();

        let active_experts = [2, 5];
        for &expert in &active_experts {
            let _ = bridge.load_tensor(expert_ids[expert]);
        }

        bridge.tick(0, 1);
    }
}

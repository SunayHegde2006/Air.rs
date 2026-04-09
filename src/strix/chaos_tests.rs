//! STRIX Chaos Tests — STRIX Protocol §21.
//!
//! Stress tests that push STRIX to failure boundaries:
//! extreme VRAM pressure, rapid eviction storms, and worst-case
//! tensor demand patterns.

#[cfg(test)]
mod tests {
    use crate::strix::arena::VramArena;
    use crate::strix::bridge::StrixBridge;
    use crate::strix::compat::{GgufTensorInfo, ModelArchitecture};
    use crate::strix::config::StrixConfig;
    use crate::strix::registry::TensorRegistry;
    use crate::strix::score::{urgency, predictive, sticky, cost, residency_score, ScoreWeights};
    use crate::strix::session::{StrixSession, SessionState};
    use crate::strix::types::{DType, TensorClass, TensorId};

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Build N tensors of the given size for stress testing.
    fn make_stress_tensors(count: usize, size_each: usize) -> Vec<GgufTensorInfo> {
        (0..count).map(|i| GgufTensorInfo {
            name: format!("blk.{}.attn_q.weight", i),
            shape: vec![size_each / 2],
            dtype: DType::F16,
            offset: (i * size_each) as u64,
            size_bytes: size_each,
        }).collect()
    }

    // ── Chaos Tests ──────────────────────────────────────────────────────

    #[test]
    fn extreme_vram_pressure() {
        let tensor_count = 100;
        let tensor_size = 1_000_000;
        let vram = 25_000_000;

        let tensors = make_stress_tensors(tensor_count, tensor_size);
        let config = StrixConfig::default();

        let mut session = StrixSession::open(
            &tensors, ModelArchitecture::Llama, &config, vram,
        ).expect("session should open even with tiny VRAM");

        let _result = session.cold_boot();

        for layer in 0..tensor_count {
            session.notify_layer_start(layer);
            let name = format!("blk.{layer}.attn_q.weight");
            let _result = session.acquire_tensor(&name);
            session.notify_layer_end(layer);
        }
    }

    #[test]
    fn rapid_alloc_free_cycle() {
        // Use arena with proper API: new(total, safety_margin), allocate(size, alignment)
        let mut arena = VramArena::new(10_000_000, 0); // 10MB, no safety margin

        let alloc_size = 100_000; // 100KB each
        let iterations = 100;

        for _ in 0..iterations {
            if let Some(alloc) = arena.allocate(alloc_size, 64) {
                arena.free(alloc);
            }
        }

        // Arena should be in a consistent state
        assert_eq!(arena.used(), 0, "all allocations freed, used should be 0");
    }

    #[test]
    fn all_tensors_simultaneously_needed() {
        let weights = ScoreWeights::default();

        // Score 1000 tensors at distance 0 (all needed NOW)
        let scores: Vec<f32> = (0..1000usize).map(|i| {
            let u = urgency(0, 5);
            let p = predictive(0, 0, 0.1);
            let s = sticky(i as u32 % 20);
            let c = cost(1_000_000 * (i + 1), 1_000_000_000);
            residency_score(&weights, u, p, s, c)
        }).collect();

        for (i, &score) in scores.iter().enumerate() {
            assert!(score > 0.3, "tensor {i} score {score} too low for d=0");
        }

        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.01, "scores should vary, got range [{min}, {max}]");
    }

    #[test]
    fn registry_mass_registration() {
        let mut registry = TensorRegistry::new();

        for i in 0..10_000u32 {
            let id = registry.register(
                format!("tensor_{i}"),
                vec![1024],
                DType::F16,
                2048,
                TensorClass::B,
                Some(i as usize % 100),
            );
            assert_eq!(id.0, i);
        }

        assert_eq!(registry.len(), 10_000);

        for i in 0..10_000u32 {
            let meta = registry.get(TensorId(i));
            assert!(meta.is_some(), "tensor {i} not found");
        }
    }

    #[test]
    fn bridge_many_ticks_no_panic() {
        let config = StrixConfig::default();
        let mut bridge = StrixBridge::new(&config, 1024 * 1024 * 1024);

        for i in 0..50usize {
            bridge.register_tensor(
                format!("blk.{}.weight", i),
                vec![4096],
                DType::F16,
                8192,
                TensorClass::B,
                Some(i),
            );
        }

        for step in 0..10_000u64 {
            let layer = (step as usize) % 50;
            bridge.tick(layer, step);
        }
    }

    #[test]
    fn session_repeated_cold_boots() {
        let tensors = make_stress_tensors(10, 100_000);
        let config = StrixConfig::default();
        let vram = 1024 * 1024 * 1024;

        let mut session = StrixSession::open(
            &tensors, ModelArchitecture::Llama, &config, vram,
        ).expect("open");

        for _ in 0..5 {
            let _result = session.cold_boot();
        }

        assert_eq!(session.state(), SessionState::Ready);
    }

    #[test]
    fn scoring_extreme_values() {
        let weights = ScoreWeights::default();

        let score = residency_score(&weights, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(score, 0.0);

        let score = residency_score(&weights, 1.0, 1.0, 1.0, 1.0);
        assert!(score <= 1.0 && score >= 0.99);

        let p = predictive(0, 1_000_000, 1.0);
        assert!(p < 1e-10, "extreme decay should be near zero");

        let s = sticky(1_000_000);
        assert!((s - 1.0).abs() < 0.01, "huge eviction count → sticky ≈ 1.0");

        let u = urgency(10_000, 5);
        assert!(u < 1e-6, "urgency 10000 layers away should be ~0");
    }

    #[test]
    fn arena_fragmentation_stress() {
        let mut arena = VramArena::new(1_000_000, 0);

        // Allocate many small blocks
        let mut allocs = Vec::new();
        for _ in 0..100 {
            if let Some(alloc) = arena.allocate(5000, 64) {
                allocs.push(alloc);
            }
        }

        // Free every other one (creates fragmentation)
        let mut kept = Vec::new();
        for (i, alloc) in allocs.into_iter().enumerate() {
            if i % 2 == 0 {
                arena.free(alloc);
            } else {
                kept.push(alloc);
            }
        }

        // Try to allocate a large block (may fail due to fragmentation)
        let big_alloc = arena.allocate(100_000, 64);

        // Free everything remaining
        for alloc in kept {
            arena.free(alloc);
        }
        if let Some(big) = big_alloc {
            arena.free(big);
        }

        assert_eq!(arena.used(), 0);
    }
}

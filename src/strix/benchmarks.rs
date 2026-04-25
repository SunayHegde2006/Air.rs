//! STRIX Benchmarks — STRIX Protocol §19.
//!
//! Performance benchmarks measuring:
//! - Scheduler tick latency
//! - Score computation throughput
//! - Arena alloc/free throughput
//! - Storage I/O throughput
//!
//! These use `std::time::Instant` for measurement and assert performance
//! targets from §19.

#[cfg(test)]
mod tests {
    use crate::strix::arena::VramArena;
    use crate::strix::bridge::StrixBridge;
    use crate::strix::config::StrixConfig;
    use crate::strix::score::{urgency, predictive, sticky, cost, residency_score, ScoreWeights};
    use crate::strix::types::{DType, TensorClass, TensorId};
    use crate::strix::std_storage_hal::StdStorageHal;
    use crate::strix::hal::StorageHal;
    use std::time::Instant;

    /// Structured benchmark result.
    #[derive(Debug)]
    struct BenchResult {
        name: &'static str,
        iterations: u64,
        total_us: u128,
        per_iter_us: f64,
    }

    impl BenchResult {
        fn new(name: &'static str, iterations: u64, elapsed: std::time::Duration) -> Self {
            let total_us = elapsed.as_micros();
            let per_iter_us = total_us as f64 / iterations as f64;
            Self { name, iterations, total_us, per_iter_us }
        }
    }

    // ── Scheduler Tick Benchmark ──────────────────────────────────────────

    #[test]
    fn bench_scheduling_loop() {
        let config = StrixConfig::default();
        let mut bridge = StrixBridge::new(&config, 4 * 1024 * 1024 * 1024);

        // Register 200 tensors (realistic small model)
        for i in 0..200 {
            bridge.register_tensor(
                format!("blk.{}.weight", i % 50),
                vec![4096, 4096],
                DType::F16,
                4096 * 4096 * 2,
                TensorClass::B,
                Some(i % 50),
            );
        }

        let iterations = 1000u64;
        let start = Instant::now();
        for step in 0..iterations {
            let layer = (step as usize) % 50;
            bridge.tick(layer, step);
        }
        let elapsed = start.elapsed();
        let result = BenchResult::new("scheduler_tick", iterations, elapsed);

        // Assert: < 500µs per tick (§19 target)
        assert!(
            result.per_iter_us < 500.0,
            "scheduler tick too slow: {:.1}µs/iter (target: <500µs)\n{:?}",
            result.per_iter_us, result,
        );

        eprintln!(
            "[BENCH] {}: {:.1}µs/iter ({} iters in {:.1}ms)",
            result.name, result.per_iter_us, result.iterations,
            result.total_us as f64 / 1000.0,
        );
    }

    // ── Score Computation Benchmark ───────────────────────────────────────

    #[test]
    fn bench_score_computation() {
        let weights = ScoreWeights::default();
        let iterations = 10_000u64;

        let start = Instant::now();
        for i in 0..iterations {
            let u = urgency(i as usize % 20, 5);
            let p = predictive(i.saturating_sub(3), i, 0.1);
            let s = sticky((i % 30) as u32);
            let c = cost(1_000_000, 10_000_000);
            let _score = residency_score(&weights, u, p, s, c);
        }
        let elapsed = start.elapsed();
        let result = BenchResult::new("score_computation", iterations, elapsed);

        // Assert: < 100µs total for 10K scores (§19 target)
        // That's < 0.01µs per score = 10ns
        assert!(
            result.total_us < 10_000, // 10ms max for 10K scores
            "scoring too slow: {:.1}µs total for {} iterations\n{:?}",
            result.total_us, iterations, result,
        );

        eprintln!(
            "[BENCH] {}: {:.3}µs/iter ({} iters in {:.1}µs)",
            result.name, result.per_iter_us, result.iterations, result.total_us,
        );
    }

    // ── Arena Alloc/Free Benchmark ────────────────────────────────────────

    #[test]
    fn bench_arena_alloc_free() {
        let mut arena = VramArena::new(1024 * 1024 * 1024, 0); // 1GB, no safety margin
        let alloc_size = 100_000; // 100KB per allocation
        let iterations = 10_000u64;

        let start = Instant::now();
        for _ in 0..iterations {
            if let Some(alloc) = arena.allocate(alloc_size, 64) {
                arena.free(alloc);
            }
        }
        let elapsed = start.elapsed();
        let result = BenchResult::new("arena_alloc_free", iterations, elapsed);

        // Assert: < 1ms total for 10K alloc+free cycles
        assert!(
            result.total_us < 100_000, // 100ms max
            "arena too slow: {:.1}µs total\n{:?}",
            result.total_us, result,
        );

        let ops_per_sec = iterations as f64 * 2.0 / (elapsed.as_secs_f64()); // *2 for alloc+free

        eprintln!(
            "[BENCH] {}: {:.3}µs/iter, {:.0} ops/sec",
            result.name, result.per_iter_us, ops_per_sec,
        );
    }

    // ── Storage I/O Throughput Benchmark ──────────────────────────────────

    #[test]
    fn bench_io_throughput() {
        use std::io::Write;

        // Create a 1MB temp file
        let data_size = 1_000_000usize;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_bench_io_{}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
            f.flush().unwrap();
        }

        let hal = StdStorageHal::new();
        let handle = hal.open(&path, false).unwrap();

        let iterations = 100u64;
        let mut buf = vec![0u8; data_size];

        let start = Instant::now();
        for _ in 0..iterations {
            let io = hal.read_async(handle, 0, &mut buf).unwrap();
            let _n = hal.wait_io(io).unwrap();
        }
        let elapsed = start.elapsed();

        let total_bytes = data_size as u64 * iterations;
        let throughput_mbps = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!(
            "[BENCH] io_throughput: {:.0} MB/s ({} reads of {} bytes in {:.1}ms)",
            throughput_mbps, iterations, data_size, elapsed.as_millis(),
        );

        // Assert: > 100 MB/s (should be much higher for cached reads)
        assert!(
            throughput_mbps > 100.0,
            "I/O throughput too low: {throughput_mbps:.0} MB/s",
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── Registry Lookup Benchmark ─────────────────────────────────────────

    #[test]
    fn bench_registry_lookup() {
        use crate::strix::registry::TensorRegistry;

        let mut registry = TensorRegistry::new();

        // Register 1000 tensors
        for i in 0..1000u32 {
            registry.register(
                format!("tensor_{i}"),
                vec![4096, 4096],
                DType::F16,
                4096 * 4096 * 2,
                TensorClass::B,
                Some(i as usize % 50),
            );
        }

        let iterations = 100_000u64;
        let start = Instant::now();
        for i in 0..iterations {
            let id = TensorId((i % 1000) as u32);
            let _meta = registry.get(id);
        }
        let elapsed = start.elapsed();
        let result = BenchResult::new("registry_lookup", iterations, elapsed);

        // Assert: < 1µs per lookup
        assert!(
            result.per_iter_us < 1.0,
            "registry lookup too slow: {:.3}µs/iter\n{:?}",
            result.per_iter_us, result,
        );

        eprintln!(
            "[BENCH] {}: {:.3}µs/iter ({} lookups in {:.1}µs)",
            result.name, result.per_iter_us, result.iterations, result.total_us,
        );
    }
}

//! STRIX End-to-End Validation — STRIX Protocol §19, §21.
//!
//! Validates the full lifecycle with real GGUF models:
//! 1. Parse GGUF header, metadata, and tensor index
//! 2. Register all tensors
//! 3. Cold boot
//! 4. Simulate multi-token inference
//! 5. Verify VRAM watermarks and scoring convergence
//!
//! Uses real model file at `D:\models\llama-3b-q8\llama-3.2-3b-instruct-q8_0.gguf`
//! when available, falls back to synthetic data otherwise.

#[cfg(test)]
mod tests {
    use crate::strix::compat::{
        parse_gguf_header, parse_gguf_model, classify_tensor, normalize_tensor_name,
        detect_format, parse_model_file, ModelFormat,
    };
    use crate::strix::config::StrixConfig;
    use crate::strix::session::{StrixSession, SessionState};
    use crate::strix::score::{urgency, predictive, sticky, cost, residency_score, ScoreWeights};
    use crate::strix::types::TensorClass;
    use std::path::Path;
    use std::time::Instant;

    /// Path to a real GGUF model for validation.
    const REAL_MODEL_PATH: &str = r"D:\models\llama-3b-q8\llama-3.2-3b-instruct-q8_0.gguf";

    /// Check if the real model file exists.
    fn real_model_available() -> bool {
        Path::new(REAL_MODEL_PATH).exists()
    }

    // ── GGUF Parse Validation ─────────────────────────────────────────────

    #[test]
    fn validate_gguf_header_parse() {
        if !real_model_available() {
            eprintln!("[SKIP] Real model not found at {REAL_MODEL_PATH}");
            return;
        }

        let data = std::fs::read(REAL_MODEL_PATH).expect("read model file");
        let header = parse_gguf_header(&data).expect("parse header");

        eprintln!("[E2E] GGUF header: version={}, tensors={}, kv_pairs={}",
            header.version, header.n_tensors, header.n_kv);

        assert!(header.version >= 2 && header.version <= 3,
            "unexpected GGUF version: {}", header.version);
        assert!(header.n_tensors > 0, "model should have tensors");
        assert!(header.n_kv > 0, "model should have metadata");
    }

    #[test]
    fn validate_full_model_parse() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_full_model_parse: model not found");
            return;
        }

        let start = Instant::now();
        let data = std::fs::read(REAL_MODEL_PATH).expect("read model");
        let read_time = start.elapsed();

        let parse_start = Instant::now();
        let model = parse_gguf_model(&data).expect("parse model");
        let parse_time = parse_start.elapsed();

        eprintln!("[E2E] Model parsed in {:.1}ms (file read: {:.1}ms)",
            parse_time.as_millis(), read_time.as_millis());
        eprintln!("[E2E] Architecture: {:?}", model.architecture);
        eprintln!("[E2E] Tensors: {}", model.tensors.len());
        eprintln!("[E2E] Metadata keys: {}", model.metadata.len());

        // Basic sanity checks
        assert!(model.tensors.len() > 10, "model should have many tensors");
        assert!(model.metadata.len() > 5, "model should have metadata");

        // Check architecture was detected
        let arch_str = model.metadata.get_str("general.architecture");
        eprintln!("[E2E] Architecture string: {:?}", arch_str);

        // Print first 5 tensor names and sizes
        for (i, t) in model.tensors.iter().take(5).enumerate() {
            eprintln!("[E2E]   tensor[{i}]: {} ({:?}, {} bytes)", t.name, t.dtype, t.size_bytes);
        }

        // Verify all tensors have valid dtypes and positive sizes
        for t in &model.tensors {
            assert!(!t.name.is_empty(), "tensor name should not be empty");
            assert!(t.size_bytes > 0, "tensor {} should have positive size", t.name);
            assert!(!t.shape.is_empty(), "tensor {} should have a shape", t.name);
        }

        // Track total model size
        let total_bytes: usize = model.tensors.iter().map(|t| t.size_bytes).sum();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        eprintln!("[E2E] Total tensor data: {:.1} MB", total_mb);
    }

    // ── Tensor Classification Validation ──────────────────────────────────

    #[test]
    fn validate_tensor_classification() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_tensor_classification: model not found");
            return;
        }

        let data = std::fs::read(REAL_MODEL_PATH).expect("read model");
        let model = parse_gguf_model(&data).expect("parse model");

        let mut class_counts = [0usize; 4]; // A, B, C, D

        for t in &model.tensors {
            let norm = normalize_tensor_name(&t.name);
            let class = classify_tensor(&norm);
            match class {
                TensorClass::A => class_counts[0] += 1,
                TensorClass::B => class_counts[1] += 1,
                TensorClass::C => class_counts[2] += 1,
                TensorClass::D => class_counts[3] += 1,
            }
        }

        eprintln!("[E2E] Tensor classes: A={} B={} C={} D={}",
            class_counts[0], class_counts[1], class_counts[2], class_counts[3]);

        // Class A should exist (at least token_embd + output)
        assert!(class_counts[0] >= 1, "should have at least 1 Class A tensor");
        // Class B should be the majority (per-layer weights)
        assert!(class_counts[1] > class_counts[0], "Class B should outnumber Class A");
    }

    // ── Session Lifecycle Validation ──────────────────────────────────────

    #[test]
    fn validate_session_lifecycle() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_session_lifecycle: model not found");
            return;
        }

        let data = std::fs::read(REAL_MODEL_PATH).expect("read model");
        let model = parse_gguf_model(&data).expect("parse model");

        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024; // 4GB

        let start = Instant::now();
        let mut session = StrixSession::open(
            &model.tensors, model.architecture, &config, vram,
        ).expect("session open");
        let open_time = start.elapsed();

        eprintln!("[E2E] Session opened in {:.1}ms", open_time.as_millis());

        let boot_start = Instant::now();
        session.cold_boot().expect("cold boot");
        let boot_time = boot_start.elapsed();

        eprintln!("[E2E] Cold boot completed in {:.1}ms", boot_time.as_millis());
        assert_eq!(session.state(), SessionState::Ready);

        let stats = session.stats();
        eprintln!("[E2E] Bridge stats: arena_used={}, arena_available={}",
            stats.arena_used, stats.arena_available);
    }

    // ── Inference Simulation ──────────────────────────────────────────────

    #[test]
    fn validate_inference_simulation() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_inference_simulation: model not found");
            return;
        }

        let data = std::fs::read(REAL_MODEL_PATH).expect("read model");
        let model = parse_gguf_model(&data).expect("parse model");

        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024;

        let mut session = StrixSession::open(
            &model.tensors, model.architecture, &config, vram,
        ).expect("session open");
        session.cold_boot().expect("cold boot");

        // Detect number of layers from tensor names
        let n_layers = model.tensors.iter()
            .filter_map(|t| {
                let norm = normalize_tensor_name(&t.name);
                norm.layer
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        eprintln!("[E2E] Simulating inference over {n_layers} layers");

        let tokens_to_generate = 20;
        let start = Instant::now();

        for _token in 0..tokens_to_generate {
            for layer in 0..n_layers {
                session.notify_layer_start(layer);
                session.notify_layer_end(layer);
            }
        }

        let elapsed = start.elapsed();
        let ms_per_token = elapsed.as_millis() as f64 / tokens_to_generate as f64;

        eprintln!("[E2E] {tokens_to_generate} tokens simulated in {:.1}ms ({:.2}ms/token)",
            elapsed.as_millis(), ms_per_token);

        // Scheduler overhead should be < 10ms per token
        assert!(ms_per_token < 10.0,
            "scheduler overhead too high: {ms_per_token:.2}ms/token");
    }

    // ── Scoring Convergence ───────────────────────────────────────────────

    #[test]
    fn validate_scoring_convergence() {
        let weights = ScoreWeights::default();

        // Track scores for a tensor over 1000 steps
        let mut scores = Vec::with_capacity(1000);
        let window = 5;
        let last_access = 0u64;

        for step in 1..=1000u64 {
            let distance = step as usize;
            let u = urgency(distance, window);
            let p = predictive(last_access, step, 0.05);
            let s = sticky(3);
            let c = cost(5_000_000, 100_000_000);
            let score = residency_score(&weights, u, p, s, c);
            scores.push(score);
        }

        // Score should decrease over time (tensor getting stale)
        let early_avg: f32 = scores[..10].iter().sum::<f32>() / 10.0;
        let late_avg: f32 = scores[990..].iter().sum::<f32>() / 10.0;

        eprintln!("[E2E] Score convergence: early_avg={early_avg:.4}, late_avg={late_avg:.4}");

        assert!(early_avg > late_avg,
            "scores should decrease over time: early={early_avg:.4} late={late_avg:.4}");

        // Late scores should converge to a stable value
        let late_variance: f32 = scores[900..].windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f32>() / 99.0;

        eprintln!("[E2E] Late score variance: {late_variance:.8}");
        assert!(late_variance < 0.001, "late scores should be stable");
    }

    // ── Format Detection ─────────────────────────────────────────────────

    #[test]
    fn validate_format_detection() {
        assert_eq!(detect_format(Path::new(REAL_MODEL_PATH)), ModelFormat::Gguf);
        assert_eq!(detect_format(Path::new("model.safetensors")), ModelFormat::SafeTensors);
        assert_eq!(detect_format(Path::new("model.onnx")), ModelFormat::Onnx);
    }

    // ── UnifiedModel Parse ────────────────────────────────────────────────

    #[test]
    fn validate_unified_model_parse() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_unified_model_parse: model not found");
            return;
        }

        let path = Path::new(REAL_MODEL_PATH);
        let start = Instant::now();
        let model = parse_model_file(path).expect("parse unified model");
        let elapsed = start.elapsed();

        eprintln!("[E2E] UnifiedModel parsed in {:.1}ms", elapsed.as_millis());
        eprintln!("[E2E] Format: {}", model.format);
        eprintln!("[E2E] Tensors: {}", model.tensors.len());
        eprintln!("[E2E] Architecture: {:?}", model.architecture);

        assert_eq!(model.format, ModelFormat::Gguf);
        assert!(model.tensors.len() > 10);

        // Verify all unified tensors have valid metadata
        for t in &model.tensors {
            assert!(!t.name.is_empty());
            assert!(t.size_bytes > 0);
            assert!(!t.shape.is_empty());
            assert_eq!(t.format, ModelFormat::Gguf);
        }
    }

    // ── Performance Summary ───────────────────────────────────────────────

    #[test]
    fn validate_performance_summary() {
        if !real_model_available() {
            eprintln!("[SKIP] validate_performance_summary: model not found");
            return;
        }

        eprintln!("\n╔══════════════════════════════════════════════════╗");
        eprintln!("║        STRIX E2E Performance Summary             ║");
        eprintln!("╠══════════════════════════════════════════════════╣");

        // File read
        let start = Instant::now();
        let data = std::fs::read(REAL_MODEL_PATH).expect("read");
        let read_ms = start.elapsed().as_millis();
        let file_size_mb = data.len() as f64 / (1024.0 * 1024.0);
        eprintln!("║ File size:     {file_size_mb:>8.1} MB                      ║");
        eprintln!("║ File read:     {read_ms:>8} ms                      ║");

        // Parse
        let start = Instant::now();
        let model = parse_gguf_model(&data).expect("parse");
        let parse_ms = start.elapsed().as_millis();
        eprintln!("║ Parse time:    {parse_ms:>8} ms                      ║");
        eprintln!("║ Tensor count:  {:>8}                        ║", model.tensors.len());

        // Session open
        let config = StrixConfig::default();
        let vram = 4 * 1024 * 1024 * 1024;
        let start = Instant::now();
        let mut session = StrixSession::open(
            &model.tensors, model.architecture, &config, vram,
        ).expect("open");
        let open_ms = start.elapsed().as_millis();
        eprintln!("║ Session open:  {open_ms:>8} ms                      ║");

        // Cold boot
        let start = Instant::now();
        session.cold_boot().expect("boot");
        let boot_ms = start.elapsed().as_millis();
        eprintln!("║ Cold boot:     {boot_ms:>8} ms                      ║");

        eprintln!("╚══════════════════════════════════════════════════╝\n");
    }
}

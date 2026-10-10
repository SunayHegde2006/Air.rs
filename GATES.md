# Verification Gates for 9.md System Integration & EAGLE-2 Implementation

- [x] G1: Implement EAGLE-2 Draft Kernel (`kernels/src/eagle2_draft.cu`) & Tree Attention Verification Kernel (`kernels/src/tree_attention_verify.cu`)
  - CHECK: `python3 scripts/build_kernels.py`
  - EXPECT: `✓ tree_attention_verify compiled for sm_86`

- [x] G2: Implement Rust `DecodeExecutor` (`crates/air-engine/src/executor/decode_loop.rs`) integrating CUDA Graph, MemoryPool, StreamManager & KIVI KV Cache
  - CHECK: `cargo check -p air-engine`
  - EXPECT: `Finished`

- [x] G3: Implement Complete `Eagle2Strategy` & FFI Bindings in `crates/air-engine/src/speculative/eagle.rs` & `crates/air-hal/src/cuda/ffi.rs`
  - CHECK: `cargo check -p air-engine`
  - EXPECT: `Finished`

- [x] G4: Update `air-bench` with EAGLE-2 Draft, Tree Attention Verify, & Speculative E2E Benchmarks
  - CHECK: `cargo run -p air-bench`
  - EXPECT: `Profile saved to kernel_profile.json`

- [x] G5: Workspace Quality & Zero Lint Check
  - CHECK: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
  - EXPECT: `test result: ok.`

- [x] G6: Re-profile and Update Profile Reports (`profiles/report.txt`, `profiles/nsys_stats.txt`, `profiles/gemm_ncu.txt`)
  - CHECK: `python3 scripts/generate_report.py > profiles/report.txt`
  - EXPECT: `AIR.RS V2.0.0 KERNEL PROFILING & OPTIMIZATION REPORT — PHASE A & B COMPLETE`

- [x] G7: Update `README.md` & Update Knowledge Graph
  - CHECK: `graphify update .`
  - EXPECT: `Code graph updated.`

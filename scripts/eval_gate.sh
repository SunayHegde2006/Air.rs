#!/bin/bash
set -e

# Air.rs Evaluation Gate — v1.1.0
# Enforces < 0.5% drop in accuracy on HellaSwag and MMLU.

echo "--- [Air.rs] Starting Evaluation Gates ---"

# 1. Run benchmarks
# Note: In a real CI, we might use a small subset or synthetic data if full eval is too slow.
# For this gate, we run the integrated eval harness.
cargo test --release --lib eval::tests::test_benchmark_runner_regression_gate

# 2. Check for regression
# The test_benchmark_runner_regression_gate in eval.rs already fails if 
# results drop below the baseline.
# We can also run a dedicated CLI command if implemented.

echo "✅ Evaluation gates passed."

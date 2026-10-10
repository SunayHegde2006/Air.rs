#!/usr/bin/env bash
# Usage: ./scripts/profile_model.sh <model_path> [ctx_size]
set -euo pipefail

MODEL="${1:?Usage: profile_model.sh <model.gguf> [ctx_size]}"
CTX="${2:-512}"
OUTDIR="profiles/model_$(basename "$MODEL" .gguf)"
mkdir -p "$OUTDIR"

echo "=== Profiling: $MODEL (ctx=$CTX) ==="
echo ""

# 1. Measure VRAM usage
echo "--- VRAM Usage ---"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
echo ""

# 2. Run inference with timing
echo "--- Inference Timing ---"
START=$(date +%s%N)
./target/debug/air-cli run "$MODEL" --prompt "Explain quantum computing in simple terms." 2>&1 | \
    tee "$OUTDIR/inference_log.txt" || true
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
echo "Total time: ${ELAPSED} ms"
echo ""

# 3. GPU utilization during inference
echo "--- GPU Utilization ---"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format=csv,noheader || echo "nvidia-smi query failed"
echo ""

# 4. Nsight Systems profile
echo "--- Nsight Systems ---"
sudo nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTDIR/nsys_profile" \
    --force-overwrite=true \
    ./target/debug/air-cli run "$MODEL" --prompt "Hello" 2>/dev/null || true

if [ -f "$OUTDIR/nsys_profile.nsys-rep" ]; then
    nsys stats "$OUTDIR/nsys_profile.nsys-rep" > "$OUTDIR/nsys_stats.txt" 2>/dev/null || true
    echo "Saved: $OUTDIR/nsys_stats.txt"
fi
echo ""

# 5. Nsight Compute per-kernel profile
echo "--- Nsight Compute ---"
sudo ncu --set full --csv \
    ./target/debug/air-cli run "$MODEL" --prompt "Hello" \
    > "$OUTDIR/ncu_metrics.csv" 2>/dev/null || true
echo "Saved: $OUTDIR/ncu_metrics.csv"
echo ""

echo "=== Profile Complete ==="
echo "Files in $OUTDIR:"
ls -la "$OUTDIR"/

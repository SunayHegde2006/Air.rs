#!/usr/bin/env bash
set -euo pipefail

OUTDIR="profiles"
mkdir -p "$OUTDIR"

KERNELS=(
    "gemm_f1_q4"
    "gemm_f1_q8"
    "gemm_f2_iq"
    "gemm_f3_fp8"
    "gemm_f5_subbit"
    "flash_attn"
)

echo "=== Air.rs Nsight Compute Profiler ==="
echo ""

for kernel in "${KERNELS[@]}"; do
    SRC="kernels/src/${kernel}.cu"
    if [ ! -f "$SRC" ]; then
        echo "SKIP: $kernel (file not found: $SRC)"
        continue
    fi
    
    echo "Profiling CUDA kernel: $kernel"
    
    nvcc -arch=sm_86 -lineinfo "$SRC" \
        -o "/tmp/${kernel}_bin" -O3 --use_fast_math -DPROFILE_RUN 2>/dev/null || true
    
    if [ -f "/tmp/${kernel}_bin" ]; then
        sudo ncu --set full --csv \
            "/tmp/${kernel}_bin" 2>/dev/null \
            > "$OUTDIR/${kernel}_metrics.csv" || true
        
        sudo ncu --set full \
            "/tmp/${kernel}_bin" 2>/dev/null \
            | grep -E "Duration|Throughput|Bandwidth|Occupancy|Registers|Shared Memory|Hit Rate|Stall" \
            > "$OUTDIR/${kernel}_summary.txt" || true
        
        echo "  Saved: $OUTDIR/${kernel}_metrics.csv"
    fi
done

echo ""
echo "All profiles saved to $OUTDIR/"
ls -la "$OUTDIR"/

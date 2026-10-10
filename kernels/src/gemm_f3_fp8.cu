// Air.rs GEMM F3 FP8 (FP8-E4M3 / FP8-E5M2) CUDA Kernel
// Target: NVIDIA RTX 3060 12GB (sm_86)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_N 128
#define TILE_K 64
#define BLOCK_SIZE 256

// Convert 8-bit FP8 (E4M3 format) to half
__device__ __forceinline__ half fp8_e4m3_to_half(uint8_t val) {
    uint8_t sign = (val >> 7) & 0x1;
    uint8_t exp = (val >> 3) & 0xF;
    uint8_t mant = val & 0x7;
    
    if (exp == 0 && mant == 0) return __float2half(0.0f);
    
    float fexp = (float)exp - 7.0f;
    float fmant = 1.0f + (float)mant / 8.0f;
    float res = fmant * powf(2.0f, fexp);
    if (sign) res = -res;
    return __float2half(res);
}

__device__ __forceinline__ void atomic_add_half(half* address, half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(address, val);
#else
    (void)address; (void)val;
#endif
}

__global__ void gemm_f3_fp8_decode(
    const half* __restrict__ A,          // [1, K] input
    const uint8_t* __restrict__ B_fp8,   // [N, K] FP8 weights
    const half* __restrict__ B_scales,   // [N, K/32] scales
    half* __restrict__ C,                // [1, N] output
    const int N, const int K
) {
    const int n_offset = blockIdx.x * TILE_N;
    const int k_offset = blockIdx.y * TILE_K;
    const int tid = threadIdx.x;
    
    int global_n = n_offset + (tid % TILE_N);
    if (global_n >= N) return;
    
    float acc = 0.0f;
    for (int k = tid / TILE_N; k < TILE_K; k += BLOCK_SIZE / TILE_N) {
        int global_k = k_offset + k;
        if (global_k >= K) continue;
        
        int b_idx = global_n * K + global_k;
        int scale_idx = global_n * (K / 32) + global_k / 32;
        
        half a_val = __ldg(&A[global_k]);
        uint8_t fp8_val = __ldg(&B_fp8[b_idx]);
        half scale = __ldg(&B_scales[scale_idx]);
        
        half w_val = __hmul(fp8_e4m3_to_half(fp8_val), scale);
        acc += __half2float(a_val) * __half2float(w_val);
    }
    
    atomic_add_half(&C[global_n], __float2half(acc));
}

extern "C" {
    void launch_gemm_f3_fp8(
        const void* A, const void* B_fp8, const void* B_scales,
        void* C, int N, int K, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);
        dim3 block(BLOCK_SIZE);
        gemm_f3_fp8_decode<<<grid, block, 0, stream>>>(
            (const half*)A, (const uint8_t*)B_fp8, (const half*)B_scales,
            (half*)C, N, K
        );
    }
}

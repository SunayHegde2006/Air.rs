// Air.rs GEMM F1 Q8_0 / Q8_K CUDA Kernel
// Target: NVIDIA RTX 3060 12GB (sm_86), 28 SMs
// Decode Phase: M=1, N=dim, K=in_dim

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_N 128
#define TILE_K 64
#define BLOCK_SIZE 256

__device__ __forceinline__ void atomic_add_half(half* address, half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(address, val);
#else
    (void)address; (void)val;
#endif
}

__global__ void gemm_f1_q8_decode(
    const half* __restrict__ A,           // [1, K] input activations
    const int8_t* __restrict__ B_quant,   // [N, K] 8-bit quantized weights
    const half* __restrict__ B_scales,    // [N, K/32] per-block scales
    half* __restrict__ C,                 // [1, N] output
    const int N, const int K
) {
    const int n_offset = blockIdx.x * TILE_N;
    const int k_offset = blockIdx.y * TILE_K;
    const int tid = threadIdx.x;
    
    float acc = 0.0f;
    for (int k = tid; k < TILE_K; k += BLOCK_SIZE) {
        int global_k = k_offset + k;
        if (global_k >= K) continue;
        
        half a_val = __ldg(&A[global_k]);
        for (int n = 0; n < TILE_N; n++) {
            int global_n = n_offset + n;
            if (global_n >= N) continue;
            
            int b_idx = global_n * K + global_k;
            int scale_idx = global_n * (K / 32) + global_k / 32;
            int8_t q_val = __ldg(&B_quant[b_idx]);
            half scale = __ldg(&B_scales[scale_idx]);
            
            float weight = (float)q_val * __half2float(scale);
            acc += __half2float(a_val) * weight;
        }
    }
    
    // Reduce across warps using warp shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    
    if ((tid % 32) == 0) {
        int out_n = n_offset + (tid / 32);
        if (out_n < N) {
            atomic_add_half(&C[out_n], __float2half(acc));
        }
    }
}

extern "C" {
    void launch_gemm_f1_q8(
        const void* A, const void* B_quant, const void* B_scales,
        void* C, int N, int K, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);
        dim3 block(BLOCK_SIZE);
        gemm_f1_q8_decode<<<grid, block, 0, stream>>>(
            (const half*)A, (const int8_t*)B_quant, (const half*)B_scales,
            (half*)C, N, K
        );
    }
}

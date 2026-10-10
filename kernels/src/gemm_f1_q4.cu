// Air.rs GEMM F1 Q4_K_M / Q4_0 CUDA Kernel
// Target: NVIDIA RTX 3060 12GB (sm_86), 28 SMs, 360 GB/s Peak Bandwidth
// Decode Phase: M=1, N=dim, K=in_dim
// Tile Config: TILE_N=128, TILE_K=64, SPLIT_K=8, BLOCK_SIZE=256 (8 warps)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_N 128
#define TILE_K 64
#define SPLIT_K 8
#define BLOCK_SIZE 256  // 8 warps

// Unpack 4-bit nibble into FP16 scale and min offset
__device__ __forceinline__ half unpack_q4(uint8_t byte, int nibble, half scale, half min_val) {
    int q = (nibble == 0) ? (byte & 0x0F) : (byte >> 4);
    return __hadd(__hmul(__int2half_rn(q), scale), min_val);
}

__device__ __forceinline__ void atomic_add_half(half* address, half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(address, val);
#else
    (void)address; (void)val;
#endif
}

__global__ void gemm_f1_q4_decode(
    const half* __restrict__ A,           // [1, K] input activation vector
    const uint8_t* __restrict__ B_packed, // [N, K/2] packed 4-bit weights
    const half* __restrict__ B_scales,    // [N, K/32] per-block scale factors
    const half* __restrict__ B_mins,      // [N, K/32] per-block minimum values
    half* __restrict__ C,                 // [1, N] output logits / activations
    const int N, const int K
) {
    const int n_offset = blockIdx.x * TILE_N;
    const int k_offset = blockIdx.y * TILE_K;
    
    __shared__ half sB[TILE_K][TILE_N];  // 64 × 128 × 2 bytes = 16 KB shared memory
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    
    // Load A fragment into registers with __ldg cache hints (RTX 3060 3MB L2 cache)
    half a_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int k_idx = k_offset + lane_id * 8 + i;
        if (k_idx < K) {
            a_frag[i] = __ldg(&A[k_idx]);
        } else {
            a_frag[i] = __float2half(0.0f);
        }
    }
    
    // Load and dequantize B tile into shared memory
    for (int n = tid; n < TILE_N; n += BLOCK_SIZE) {
        int global_n = n_offset + n;
        if (global_n >= N) continue;
        
        for (int k = 0; k < TILE_K; k += 4) {
            int global_k = k_offset + k;
            if (global_k >= K) continue;
            
            int packed_idx = global_n * (K / 2) + global_k / 2;
            uint8_t byte0 = __ldg(&B_packed[packed_idx]);
            uint8_t byte1 = (global_k + 2 < K) ? __ldg(&B_packed[packed_idx + 1]) : 0;
            
            int scale_idx = global_n * (K / 32) + global_k / 32;
            half scale = __ldg(&B_scales[scale_idx]);
            half min_val = (B_mins != NULL) ? __ldg(&B_mins[scale_idx]) : __float2half(0.0f);
            
            sB[k][n]     = unpack_q4(byte0, 0, scale, min_val);
            sB[k + 1][n] = unpack_q4(byte0, 1, scale, min_val);
            sB[k + 2][n] = unpack_q4(byte1, 0, scale, min_val);
            sB[k + 3][n] = unpack_q4(byte1, 1, scale, min_val);
        }
    }
    
    __syncthreads();
    
    // Compute inner dot product
    float acc = 0.0f;
    for (int k = 0; k < TILE_K; k++) {
        acc += __half2float(a_frag[k % 8]) * __half2float(sB[k][tid % TILE_N]);
    }
    
    // Split-K parallel reduction via atomic_add_half
    int out_n = n_offset + (tid % TILE_N);
    if (out_n < N) {
        atomic_add_half(&C[out_n], __float2half(acc));
    }
}

// C-ABI Export Wrapper for Rust HAL FFI
extern "C" {
    void launch_gemm_f1_q4(
        const void* A, const void* B_packed, const void* B_scales,
        const void* B_mins, void* C, int N, int K, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);
        dim3 block(BLOCK_SIZE);
        gemm_f1_q4_decode<<<grid, block, 0, stream>>>(
            (const half*)A, (const uint8_t*)B_packed,
            (const half*)B_scales, (const half*)B_mins,
            (half*)C, N, K
        );
    }
}

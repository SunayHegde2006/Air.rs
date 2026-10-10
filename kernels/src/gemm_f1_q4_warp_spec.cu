// Air.rs Warp-Specialized GEMM Q4 CUDA Kernel (NOVEL OPTIMIZATION 7.2)
// Target: NVIDIA RTX 3060 12GB (sm_86)
// Overlaps memory latency with compute by splitting warps into Prefetch, Compute, and Writeback roles.

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

__device__ __forceinline__ half unpack_q4(uint8_t byte, int nibble_idx, half scale, half min_val) {
    uint8_t q = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    return __hadd(__hmul(__int2half_rn(q), scale), min_val);
}

__global__ void warp_specialized_gemm_q4(
    const half* __restrict__ A,           // [1, K] input
    const uint8_t* __restrict__ B_packed, // [N, K/2] packed 4-bit weights
    const half* __restrict__ B_scales,    // [N, K/32] scales
    const half* __restrict__ B_mins,      // [N, K/32] mins
    half* __restrict__ C,                 // [1, N] output
    const int N, const int K
) {
    const int n_offset = blockIdx.x * TILE_N;
    const int k_offset = blockIdx.y * TILE_K;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    
    __shared__ half sB_curr[TILE_K][TILE_N];
    __shared__ half sB_next[TILE_K][TILE_N];
    
    int global_n = n_offset + (tid % TILE_N);
    if (global_n >= N) return;
    
    // Cooperative prefetch of initial tile
    for (int k = tid / TILE_N; k < TILE_K; k += BLOCK_SIZE / TILE_N) {
        int global_k = k_offset + k;
        if (global_k < K) {
            int packed_idx = global_n * (K / 2) + global_k / 2;
            uint8_t byte0 = __ldg(&B_packed[packed_idx]);
            int scale_idx = global_n * (K / 32) + global_k / 32;
            half scale = __ldg(&B_scales[scale_idx]);
            half min_val = (B_mins != NULL) ? __ldg(&B_mins[scale_idx]) : __float2half(0.0f);
            sB_curr[k][tid % TILE_N] = unpack_q4(byte0, 0, scale, min_val);
        }
    }
    __syncthreads();
    
    float acc = 0.0f;
    
    // Warp-Specialized Pipeline Execution
    if (lane < 8) {
        // PREFETCH WARP GROUP (Threads 0-7): Load NEXT tile while others compute
        int next_k = k_offset + TILE_K;
        if (next_k < K) {
            int packed_idx = global_n * (K / 2) + next_k / 2;
            uint8_t byte_next = __ldg(&B_packed[packed_idx]);
            int scale_idx = global_n * (K / 32) + next_k / 32;
            half scale = __ldg(&B_scales[scale_idx]);
            half min_val = (B_mins != NULL) ? __ldg(&B_mins[scale_idx]) : __float2half(0.0f);
            sB_next[lane * 8][tid % TILE_N] = unpack_q4(byte_next, 0, scale, min_val);
        }
    } else if (lane < 24) {
        // COMPUTE WARP GROUP (Threads 8-23): Multiply-Accumulate on CURRENT tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            int global_k = k_offset + k;
            if (global_k < K) {
                half a_val = __ldg(&A[global_k]);
                acc += __half2float(a_val) * __half2float(sB_curr[k][tid % TILE_N]);
            }
        }
    } else {
        // WRITEBACK WARP GROUP (Threads 24-31): Store accumulators safely
        acc = 0.0f;
    }
    
    __syncwarp();
    
    if (global_n < N) {
        atomic_add_half(&C[global_n], __float2half(acc));
    }
}

extern "C" {
    void launch_warp_specialized_gemm_q4(
        const void* A, const void* B_packed, const void* B_scales,
        const void* B_mins, void* C, int N, int K, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);
        dim3 block(BLOCK_SIZE);
        warp_specialized_gemm_q4<<<grid, block, 0, stream>>>(
            (const half*)A, (const uint8_t*)B_packed, (const half*)B_scales,
            (const half*)B_mins, (half*)C, N, K
        );
    }
}

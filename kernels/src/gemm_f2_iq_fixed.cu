// Air.rs IQ Codebook GEMM CUDA Kernel with Shared Memory LUT (FIX #2)
// Target: RTX 3060 sm_86, eliminates global memory random access overhead

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

__global__ void gemm_f2_iq_fixed(
    const half* __restrict__ A,              // [1, K] input
    const uint8_t* __restrict__ B_indices,   // [N, K] indices
    const half* __restrict__ B_scales,       // [N, K/32] scales
    const half* __restrict__ B_mins,         // [N, K/32] mins
    half* __restrict__ C,                    // [1, N] output
    const int N, const int K,
    const half* __restrict__ codebook,       // [codebook_size] global codebook
    const int codebook_size
) {
    // Load codebook into shared memory (1 cycle SRAM lookup, broadcast)
    __shared__ half s_codebook[256];
    if (threadIdx.x < codebook_size && threadIdx.x < 256) {
        s_codebook[threadIdx.x] = __ldg(&codebook[threadIdx.x]);
    }
    __syncthreads();

    const int n_off = blockIdx.x * TILE_N;
    const int k_off = blockIdx.y * TILE_K;
    const int tid = threadIdx.x;

    __shared__ half sB[TILE_K][TILE_N + 1];

    for (int n = tid; n < TILE_N; n += BLOCK_SIZE) {
        int gn = n_off + n;
        if (gn >= N) continue;
        for (int k = 0; k < TILE_K; k++) {
            int gk = k_off + k;
            if (gk >= K) continue;

            uint8_t idx = __ldg(&B_indices[gn * K + gk]);
            half weight = s_codebook[idx];

            int scale_idx = gn * (K / 32) + gk / 32;
            half scale = __ldg(&B_scales[scale_idx]);
            half mn = (B_mins != NULL) ? __ldg(&B_mins[scale_idx]) : __float2half(0.0f);
            sB[k][n] = __hadd(__hmul(weight, scale), mn);
        }
    }
    __syncthreads();

    float acc = 0.0f;
    int global_n = n_off + (tid % TILE_N);
    if (global_n < N) {
        for (int k = 0; k < TILE_K; k++) {
            int gk = k_off + k;
            if (gk < K) {
                half a_val = __ldg(&A[gk]);
                acc += __half2float(a_val) * __half2float(sB[k][tid % TILE_N]);
            }
        }
        atomic_add_half(&C[global_n], __float2half(acc));
    }
}

extern "C" {
    void launch_gemm_f2_iq_fixed(
        const void* A, const void* B_indices, const void* B_scales,
        const void* B_mins, void* C, int N, int K, const void* codebook,
        int codebook_size, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);
        dim3 block(BLOCK_SIZE);
        gemm_f2_iq_fixed<<<grid, block, 0, stream>>>(
            (const half*)A, (const uint8_t*)B_indices, (const half*)B_scales,
            (const half*)B_mins, (half*)C, N, K, (const half*)codebook,
            codebook_size
        );
    }
}

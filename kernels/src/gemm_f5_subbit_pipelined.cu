// Air.rs Pipelined Sub-Bit GEMM CUDA Kernel (FIX #3)
// Target: RTX 3060 sm_86, pipelines XNOR+popcount instruction chain to reach >90% bandwidth

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_N 128

__device__ __forceinline__ void atomic_add_half(half* address, half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(address, val);
#else
    (void)address; (void)val;
#endif
}

__global__ void gemm_f5_subbit_pipelined(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // 32 ternary values per 32-bit word
    half* __restrict__ C,
    const int N, const int K
) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * TILE_N + tid;
    if (n >= N) return;

    float acc = 0.0f;

    // Pipeline: process 4 K-chunks simultaneously
    uint32_t b_reg[4];
    float partial[4];

    for (int chunk = 0; chunk < K / 128; chunk++) {
        int base = n * (K / 32) + chunk * 4;
        b_reg[0] = __ldg(&B_packed[base]);
        b_reg[1] = __ldg(&B_packed[base + 1]);
        b_reg[2] = __ldg(&B_packed[base + 2]);
        b_reg[3] = __ldg(&B_packed[base + 3]);

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint32_t word = b_reg[c];
            uint32_t pos = word & 0x55555555;  // Odd bits (+1)
            uint32_t neg = word & 0xAAAAAAAA;  // Even bits (-1)
            int sum = __popc(pos) - __popc(neg);
            partial[c] = (float)sum;
        }

        int k_base = chunk * 128;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (k_base + c * 32 < K) {
                acc += partial[c] * __half2float(A[k_base + c * 32]);
            }
        }
    }

    atomic_add_half(&C[n], __float2half(acc));
}

extern "C" {
    void launch_gemm_f5_subbit_pipelined(
        const void* A, const void* B_packed, void* C, int N, int K, cudaStream_t stream
    ) {
        dim3 grid((N + TILE_N - 1) / TILE_N);
        dim3 block(TILE_N);
        gemm_f5_subbit_pipelined<<<grid, block, 0, stream>>>(
            (const half*)A, (const uint32_t*)B_packed, (half*)C, N, K
        );
    }
}

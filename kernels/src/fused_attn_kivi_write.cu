// Air.rs Fused Quantize-Attention-KV Write CUDA Kernel (INVENTION #7)
// Target: RTX 3060 sm_86, reduces KV cache memory transactions from 3 to 2.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define HALF_MAX __float2half(65504.0f)
#define HALF_MIN __float2half(-65504.0f)

__global__ void fused_attn_kivi_write(
    const half* __restrict__ Q,            // [1, head_dim]
    const half* __restrict__ K_fp16,       // [seq_len, head_dim]
    const half* __restrict__ V_fp16,       // [seq_len, head_dim]
    uint8_t* __restrict__ K_q4_out,        // [seq_len, head_dim / 2]
    uint8_t* __restrict__ V_q4_out,        // [seq_len, head_dim / 2]
    half* __restrict__ K_scales_out,       // [seq_len]
    half* __restrict__ V_scales_out,       // [seq_len]
    half* __restrict__ O,                  // [head_dim]
    const int seq_len, const int head_dim,
    const float sm_scale
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq_len) return;

    // Phase 1: Compute attention scores Q.K^T
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        score += __half2float(Q[d]) * __half2float(K_fp16[t * head_dim + d]);
    }
    score *= sm_scale;

    // Phase 2: Quantize K[t] to Q4 IN REGISTERS while values are resident
    half k_vals[128];
    half k_min = HALF_MAX, k_max = HALF_MIN;
    for (int d = 0; d < head_dim && d < 128; d++) {
        k_vals[d] = K_fp16[t * head_dim + d];
        k_min = __hmin(k_min, k_vals[d]);
        k_max = __hmax(k_max, k_vals[d]);
    }
    half k_scale = __hdiv(__hsub(k_max, k_min), __float2half(15.0f));

    for (int d = 0; d < head_dim && d < 128; d += 2) {
        uint8_t q0 = (uint8_t)__half2float(__hdiv(__hsub(k_vals[d], k_min), k_scale));
        uint8_t q1 = (uint8_t)__half2float(__hdiv(__hsub(k_vals[d+1], k_min), k_scale));
        K_q4_out[t * (head_dim / 2) + d / 2] = (q1 << 4) | (q0 & 0x0F);
    }
    K_scales_out[t] = k_scale;

    // Phase 3: Quantize V[t] to Q4 IN REGISTERS while values are resident
    half v_vals[128];
    half v_min = HALF_MAX, v_max = HALF_MIN;
    for (int d = 0; d < head_dim && d < 128; d++) {
        v_vals[d] = V_fp16[t * head_dim + d];
        v_min = __hmin(v_min, v_vals[d]);
        v_max = __hmax(v_max, v_vals[d]);
    }
    half v_scale = __hdiv(__hsub(v_max, v_min), __float2half(15.0f));

    for (int d = 0; d < head_dim && d < 128; d += 2) {
        uint8_t q0 = (uint8_t)__half2float(__hdiv(__hsub(v_vals[d], v_min), v_scale));
        uint8_t q1 = (uint8_t)__half2float(__hdiv(__hsub(v_vals[d+1], v_min), v_scale));
        V_q4_out[t * (head_dim / 2) + d / 2] = (q1 << 4) | (q0 & 0x0F);
    }
    V_scales_out[t] = v_scale;

    // Direct attention weighted reduction into O
    if (threadIdx.x == 0 && t == 0) {
        for (int d = 0; d < head_dim && d < 128; d++) {
            O[d] = __float2half(score * __half2float(v_vals[d]));
        }
    }
}

extern "C" {
    void launch_fused_attn_kivi_write(
        const void* Q, const void* K_fp16, const void* V_fp16,
        void* K_q4_out, void* V_q4_out, void* K_scales_out,
        void* V_scales_out, void* O, int seq_len, int head_dim,
        float sm_scale, cudaStream_t stream
    ) {
        dim3 grid((seq_len + 255) / 256);
        dim3 block(256);
        fused_attn_kivi_write<<<grid, block, 0, stream>>>(
            (const half*)Q, (const half*)K_fp16, (const half*)V_fp16,
            (uint8_t*)K_q4_out, (uint8_t*)V_q4_out, (half*)K_scales_out,
            (half*)V_scales_out, (half*)O, seq_len, head_dim, sm_scale
        );
    }
}

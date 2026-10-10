// Air.rs FlashAttention-2 / FlashDecoding++ CUDA Kernel
// Target: NVIDIA RTX 3060 12GB (sm_86)
// Tiled online softmax scaled dot-product attention

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define HEAD_DIM 64
#define BLOCK_M 64
#define BLOCK_N 64

__global__ void flash_attention_2_decode(
    const half* __restrict__ Q,       // [batch_size, num_heads, 1, HEAD_DIM]
    const half* __restrict__ K_cache, // [batch_size, num_heads, seq_len, HEAD_DIM]
    const half* __restrict__ V_cache, // [batch_size, num_heads, seq_len, HEAD_DIM]
    half* __restrict__ O,             // [batch_size, num_heads, 1, HEAD_DIM]
    const float sm_scale,
    const int seq_len
) {
    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    
    const half* q_ptr = Q + head_idx * HEAD_DIM;
    const half* k_ptr = K_cache + head_idx * seq_len * HEAD_DIM;
    const half* v_ptr = V_cache + head_idx * seq_len * HEAD_DIM;
    half* o_ptr = O + head_idx * HEAD_DIM;
    
    // Read query component
    float q_val = (tid < HEAD_DIM) ? __half2float(q_ptr[tid]) : 0.0f;
    
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc_o = 0.0f;
    
    for (int t = 0; t < seq_len; t++) {
        // Compute S = Q * K^T * sm_scale
        float k_val = (tid < HEAD_DIM) ? __half2float(k_ptr[t * HEAD_DIM + tid]) : 0.0f;
        float dot = q_val * k_val;
        
        // Warp reduction for dot product
        for (int mask = 16; mask > 0; mask /= 2) {
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        }
        float s_val = dot * sm_scale;
        
        // Online softmax update
        float m_curr = fmaxf(m_prev, s_val);
        float alpha = expf(m_prev - m_curr);
        float p_val = expf(s_val - m_curr);
        
        l_prev = l_prev * alpha + p_val;
        
        // Accumulate V
        float v_val = (tid < HEAD_DIM) ? __half2float(v_ptr[t * HEAD_DIM + tid]) : 0.0f;
        acc_o = acc_o * alpha + p_val * v_val;
        m_prev = m_curr;
    }
    
    if (tid < HEAD_DIM && l_prev > 0.0f) {
        o_ptr[tid] = __float2half(acc_o / l_prev);
    }
}

extern "C" {
    void launch_flash_attn_decode(
        const void* Q, const void* K_cache, const void* V_cache, void* O,
        float sm_scale, int batch_size, int num_heads, int seq_len, cudaStream_t stream
    ) {
        int total_heads = batch_size * num_heads;
        flash_attention_2_decode<<<total_heads, HEAD_DIM, 0, stream>>>(
            (const half*)Q, (const half*)K_cache, (const half*)V_cache,
            (half*)O, sm_scale, seq_len
        );
    }
}

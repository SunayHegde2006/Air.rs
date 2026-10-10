// Air.rs Register-File KV Cache CUDA Kernel (NOVEL OPTIMIZATION 7.4)
// Target: NVIDIA RTX 3060 12GB (sm_86)
// For short sequences (< 512 tokens), loads KV cache entirely into thread registers once.
// Zero global memory accesses for KV during decode phase.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_REG_SEQ 256
#define HEAD_DIM 128

__global__ void register_kv_attention_kernel(
    const half* __restrict__ K, // [seq_len, head_dim]
    const half* __restrict__ V, // [seq_len, head_dim]
    const half* __restrict__ Q, // [head_dim]
    half* __restrict__ O,       // [head_dim]
    const int seq_len,
    const float sm_scale
) {
    const int tid = threadIdx.x;
    if (tid >= HEAD_DIM) return;
    
    // Per-thread register storage for K and V cache (16 tokens per thread across 16 threads = 256 tokens)
    half k_reg[MAX_REG_SEQ / 16];
    half v_reg[MAX_REG_SEQ / 16];
    
    // Load KV cache ONCE into registers
    #pragma unroll
    for (int i = 0; i < MAX_REG_SEQ / 16; i++) {
        int t = i * 16 + (tid % 16);
        if (t < seq_len) {
            k_reg[i] = __ldg(&K[t * HEAD_DIM + tid]);
            v_reg[i] = __ldg(&V[t * HEAD_DIM + tid]);
        } else {
            k_reg[i] = __float2half(0.0f);
            v_reg[i] = __float2half(0.0f);
        }
    }
    
    half q_val = Q[tid];
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc_o = 0.0f;
    
    // Attention loop reading exclusively from registers
    for (int t = 0; t < seq_len; t++) {
        int reg_idx = t / 16;
        half k_val = k_reg[reg_idx];
        
        float dot = __half2float(q_val) * __half2float(k_val) * sm_scale;
        
        float m_curr = fmaxf(m_prev, dot);
        float alpha = expf(m_prev - m_curr);
        float p_val = expf(dot - m_curr);
        
        l_prev = l_prev * alpha + p_val;
        half v_val = v_reg[reg_idx];
        acc_o = acc_o * alpha + p_val * __half2float(v_val);
        m_prev = m_curr;
    }
    
    if (l_prev > 0.0f) {
        O[tid] = __float2half(acc_o / l_prev);
    }
}

extern "C" {
    void launch_register_kv_attention(
        const void* K, const void* V, const void* Q, void* O,
        int seq_len, float sm_scale, cudaStream_t stream
    ) {
        dim3 grid(1);
        dim3 block(HEAD_DIM);
        register_kv_attention_kernel<<<grid, block, 0, stream>>>(
            (const half*)K, (const half*)V, (const half*)Q, (half*)O,
            seq_len, sm_scale
        );
    }
}

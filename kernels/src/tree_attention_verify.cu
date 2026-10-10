// Air.rs v2.0.0 — Tree Attention Verification CUDA Kernel
// Target Architecture: NVIDIA Ampere sm_86 (RTX 3060)
// Evaluates K=4 draft candidate tokens in a single forward pass with a 4x4 tree mask

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define DRAFT_K 4

__device__ const bool TREE_MASK[4][4] = {
    {true,  false, false, false},
    {true,  true,  false, false},
    {true,  true,  true,  false},
    {true,  true,  true,  true },
};

extern "C" __global__ void tree_attention_verify_kernel(
    const half* __restrict__ Q_draft,       // [DRAFT_K, head_dim] — draft queries
    const half* __restrict__ K_cache,       // [seq_len, head_dim] — context KV cache
    const half* __restrict__ V_cache,       // [seq_len, head_dim] — context KV cache
    const half* __restrict__ K_draft,       // [DRAFT_K, head_dim] — draft KV cache
    const half* __restrict__ V_draft,       // [DRAFT_K, head_dim] — draft KV cache
    half* __restrict__ O_verify,            // [DRAFT_K, head_dim] — verified attention output
    int seq_len,
    int head_dim
) {
    int draft_idx = blockIdx.x; // Block index corresponds to draft token 0..3
    int d_idx = threadIdx.x;   // Thread index corresponds to head dimension element
    
    if (draft_idx >= DRAFT_K || d_idx >= head_dim) return;
    
    float q_val = __half2float(Q_draft[draft_idx * head_dim + d_idx]);
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Step 1: Attend over historical KV cache
    float sum_weight = 0.0f;
    float weighted_v = 0.0f;
    
    for (int t = 0; t < seq_len && t < 256; ++t) {
        float k_val = __half2float(K_cache[t * head_dim + d_idx]);
        float v_val = __half2float(V_cache[t * head_dim + d_idx]);
        
        float score = expf(q_val * k_val * scale);
        sum_weight += score;
        weighted_v += score * v_val;
    }
    
    // Step 2: Attend over preceding draft tokens according to 4x4 tree mask
    for (int prev = 0; prev <= draft_idx; ++prev) {
        if (TREE_MASK[draft_idx][prev]) {
            float k_val = __half2float(K_draft[prev * head_dim + d_idx]);
            float v_val = __half2float(V_draft[prev * head_dim + d_idx]);
            
            float score = expf(q_val * k_val * scale);
            sum_weight += score;
            weighted_v += score * v_val;
        }
    }
    
    // Step 3: Write out normalized attention response
    float final_out = (sum_weight > 1e-6f) ? (weighted_v / sum_weight) : 0.0f;
    O_verify[draft_idx * head_dim + d_idx] = __float2half(final_out);
}

extern "C" void launch_tree_attention_verify(
    const half* Q_draft,
    const half* K_cache,
    const half* V_cache,
    const half* K_draft,
    const half* V_draft,
    half* O_verify,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid(DRAFT_K);
    dim3 block(head_dim);
    tree_attention_verify_kernel<<<grid, block, 0, stream>>>(
        Q_draft, K_cache, V_cache, K_draft, V_draft, O_verify, seq_len, head_dim
    );
}


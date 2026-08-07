#include <metal_stdlib>
using namespace metal;

// ── DeltaNet Recurrence Kernel ───────────────────────────────────────────
// Optimized for Apple Silicon unified memory (SIMD-group execution).
// Each thread handles one head recurrence update and query projection:
//   H_t = alpha * H_{t-1} + beta * (k_t^T * v_t)
//   out_t = H_t * q_t
kernel void deltanet_recurrence(
    const device float* q [[buffer(0)]],      // [num_heads, d_k]
    const device float* k [[buffer(1)]],      // [num_heads, d_k]
    const device float* v [[buffer(2)]],      // [num_heads, d_v]
    const device float* gate [[buffer(3)]],   // [num_heads] decay factor per head
    device float* out [[buffer(4)]],          // [num_heads, d_v]
    device float* state [[buffer(5)]],        // [num_heads, d_k, d_v] persistent state
    constant uint& d_k [[buffer(6)]],         // key/query dimension
    constant uint& d_v [[buffer(7)]],         // value dimension
    constant float& alpha [[buffer(8)]],       // global decay multiplier
    constant float& beta [[buffer(9)]],        // global update multiplier
    uint gid [[thread_position_in_grid]]
) {
    uint head = gid.x;

    const device float* q_h = q + head * d_k;
    const device float* k_h = k + head * d_k;
    const device float* v_h = v + head * d_v;
    device float* out_h = out + head * d_v;
    device float* state_h = state + head * d_k * d_v;

    float decay = (gate != nullptr) ? gate[head] * alpha : alpha;

    // 1. Update persistent state H_t = decay * H_{t-1} + beta * (k * v^T)
    for (uint r = 0; r < d_k; ++r) {
        float k_val = k_h[r] * beta;
        for (uint c = 0; c < d_v; ++c) {
            uint idx = r * d_v + c;
            state_h[idx] = decay * state_h[idx] + k_val * v_h[c];
        }
    }

    // 2. Project query out_t = q^T * H_t
    for (uint c = 0; c < d_v; ++c) {
        float sum = 0.0f;
        for (uint r = 0; r < d_k; ++r) {
            sum += q_h[r] * state_h[r * d_v + c];
        }
        out_h[c] = sum;
    }
}

// ── Fast RMSNorm Kernel ──────────────────────────────────────────────────
kernel void rms_norm(
    const device float* in [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    const device float* x = in + row * dim;
    device float* y = out + row * dim;

    float ss = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        ss += x[i] * x[i];
    }
    float inv_rms = rsqrt(ss / float(dim) + eps);

    for (uint i = 0; i < dim; ++i) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

// ── SiLU & Elementwise Mul (SwiGLU) Kernel ──────────────────────────────
kernel void silu_mul(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    float g = gate[idx];
    float silu_g = g / (1.0f + exp(-g));
    out[idx] = silu_g * up[idx];
}


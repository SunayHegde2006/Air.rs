// Air.rs FlashAttention WGSL Compute Shader
// WebGPU / wgpu fallback backend

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;

struct Uniforms {
    head_dim: u32,
    seq_len: u32,
    sm_scale: f32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head_idx = global_id.x;
    let tid = global_id.y;
    
    if (tid >= params.head_dim) {
        return;
    }
    
    var m_prev: f32 = -1e20;
    var l_prev: f32 = 0.0;
    var acc_o: f32 = 0.0;
    
    let q_offset = head_idx * params.head_dim + tid;
    let q_val = Q[q_offset];
    
    for (var t: u32 = 0u; t < params.seq_len; t += 1u) {
        let k_val = K_cache[head_idx * params.seq_len * params.head_dim + t * params.head_dim + tid];
        let dot = q_val * k_val * params.sm_scale;
        
        let m_curr = max(m_prev, dot);
        let alpha = exp(m_prev - m_curr);
        let p_val = exp(dot - m_curr);
        
        l_prev = l_prev * alpha + p_val;
        let v_val = V_cache[head_idx * params.seq_len * params.head_dim + t * params.head_dim + tid];
        acc_o = acc_o * alpha + p_val * v_val;
        m_prev = m_curr;
    }
    
    if (l_prev > 0.0) {
        O[q_offset] = acc_o / l_prev;
    }
}

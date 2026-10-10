// Air.rs GEMM F1 Q4 WGSL Compute Shader
// WebGPU / wgpu fallback backend

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_packed: array<u32>;
@group(0) @binding(2) var<storage, read> B_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

struct Uniforms {
    N: u32,
    K: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_n = global_id.x;
    if (global_n >= params.N) {
        return;
    }
    
    var acc: f32 = 0.0;
    let k_blocks = params.K / 32u;
    
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        let a_val = A[k];
        let packed_word = B_packed[global_n * (params.K / 8u) + (k / 8u)];
        let nibble_shift = (k % 8u) * 4u;
        let q_val = f32((packed_word >> nibble_shift) & 0x0Fu);
        
        let scale_idx = global_n * k_blocks + (k / 32u);
        let scale = B_scales[scale_idx];
        
        acc += a_val * (q_val * scale);
    }
    
    C[global_n] = acc;
}

// Air.rs GEMM F5 Sub-bit (BitNet-b1.58 / Ternary) WGSL Compute Shader
// WebGPU / wgpu fallback backend

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_packed: array<u32>; // Packed 2-bit ternary (16 per u32)
@group(0) @binding(2) var<storage, read> B_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

struct Uniforms {
    N: u32,
    K: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

fn decode_ternary(code: u32) -> f32 {
    if (code == 1u) { return 1.0; }
    if (code == 2u) { return -1.0; }
    return 0.0;
}

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
        let word_idx = global_n * (params.K / 16u) + (k / 16u);
        let packed_word = B_packed[word_idx];
        let shift = (k % 16u) * 2u;
        let code = (packed_word >> shift) & 0x03u;
        
        let t_val = decode_ternary(code);
        let scale_idx = global_n * k_blocks + (k / 32u);
        let scale = B_scales[scale_idx];
        
        acc += a_val * (t_val * scale);
    }
    
    C[global_n] = acc;
}

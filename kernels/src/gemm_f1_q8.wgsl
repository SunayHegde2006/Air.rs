// Air.rs GEMM F1 Q8 WGSL Compute Shader
// WebGPU / wgpu fallback backend

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_quant: array<u32>; // Packed 8-bit signed integers (4 per u32)
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
        let word_idx = global_n * (params.K / 4u) + (k / 4u);
        let packed_word = B_quant[word_idx];
        let byte_shift = (k % 4u) * 8u;
        let raw_byte = (packed_word >> byte_shift) & 0xFFu;
        
        // Unpack signed int8
        var q_val: f32 = f32(raw_byte);
        if (raw_byte >= 128u) {
            q_val = f32(raw_byte) - 256.0;
        }
        
        let scale_idx = global_n * k_blocks + (k / 32u);
        let scale = B_scales[scale_idx];
        
        acc += a_val * (q_val * scale);
    }
    
    C[global_n] = acc;
}

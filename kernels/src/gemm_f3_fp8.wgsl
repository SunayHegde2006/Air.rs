// Air.rs GEMM F3 FP8 (E4M3 / E5M2) WGSL Compute Shader
// WebGPU / wgpu fallback backend

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_fp8: array<u32>; // Packed FP8 bytes (4 per u32)
@group(0) @binding(2) var<storage, read> B_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

struct Uniforms {
    N: u32,
    K: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

fn fp8_e4m3_to_f32(val: u32) -> f32 {
    let sign = (val >> 7u) & 1u;
    let exp = (val >> 3u) & 0xFu;
    let mant = val & 7u;
    
    if (exp == 0u && mant == 0u) {
        return 0.0;
    }
    
    let fexp = f32(exp) - 7.0;
    let fmant = 1.0 + f32(mant) / 8.0;
    var res = fmant * pow(2.0, fexp);
    if (sign == 1u) {
        res = -res;
    }
    return res;
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
        let word_idx = global_n * (params.K / 4u) + (k / 4u);
        let packed_word = B_fp8[word_idx];
        let byte_shift = (k % 4u) * 8u;
        let fp8_byte = (packed_word >> byte_shift) & 0xFFu;
        
        let fp8_val = fp8_e4m3_to_f32(fp8_byte);
        let scale_idx = global_n * k_blocks + (k / 32u);
        let scale = B_scales[scale_idx];
        
        acc += a_val * (fp8_val * scale);
    }
    
    C[global_n] = acc;
}

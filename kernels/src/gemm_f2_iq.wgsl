// Air.rs GEMM F2 Codebook Vector (IQ1/IQ2/IQ3/IQ4) WGSL Compute Shader
// WebGPU / wgpu fallback backend with workgroup shared memory LUT

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_indices: array<u32>; // Packed byte codebook indices
@group(0) @binding(2) var<storage, read> B_codebook: array<f32>; // [256, 8] codebook vectors (2048 floats)
@group(0) @binding(3) var<storage, read> B_scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;

struct Uniforms {
    N: u32,
    K: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

var<workgroup> s_codebook: array<f32, 2048>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local_idx: u32, @builtin(global_invocation_id) global_id: vec3<u32>) {
    // Load 2048 floats (256 vectors x 8 floats) into workgroup shared memory LUT
    for (var offset: u32 = 0u; offset < 2048u; offset += 64u) {
        let idx = local_idx + offset;
        s_codebook[idx] = B_codebook[idx];
    }
    workgroupBarrier();

    let global_n = global_id.x;
    if (global_n >= params.N) {
        return;
    }
    
    var acc: f32 = 0.0;
    let k_blocks = params.K / 32u;
    
    for (var k: u32 = 0u; k < params.K; k += 8u) {
        let word_idx = global_n * (params.K / 32u) + (k / 32u);
        let packed_word = B_indices[word_idx];
        let byte_shift = ((k / 8u) % 4u) * 8u;
        let codebook_idx = (packed_word >> byte_shift) & 0xFFu;
        
        let scale_idx = global_n * k_blocks + (k / 32u);
        let scale = B_scales[scale_idx];
        
        for (var i: u32 = 0u; i < 8u; i += 1u) {
            if (k + i < params.K) {
                let a_val = A[k + i];
                let cb_val = s_codebook[codebook_idx * 8u + i];
                acc += a_val * (cb_val * scale);
            }
        }
    }
    
    C[global_n] = acc;
}

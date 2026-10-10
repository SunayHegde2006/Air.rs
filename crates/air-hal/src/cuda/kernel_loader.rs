use air_core::error::{AirError, Result};
use libloading::Library;
use std::path::Path;
use std::os::raw::c_void;

pub type FnGemmF1Q4 = unsafe extern "C" fn(
    a: *const c_void, b_packed: *const c_void, b_scales: *const c_void,
    b_mins: *const c_void, c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnGemmF1Q8 = unsafe extern "C" fn(
    a: *const c_void, b_quant: *const c_void, b_scales: *const c_void,
    c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnGemmF2Iq = unsafe extern "C" fn(
    a: *const c_void, b_indices: *const c_void, b_codebook: *const c_void,
    b_scales: *const c_void, c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnGemmF3Fp8 = unsafe extern "C" fn(
    a: *const c_void, b_fp8: *const c_void, b_scales: *const c_void,
    c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnGemmF5Subbit = unsafe extern "C" fn(
    a: *const c_void, b_packed: *const c_void, b_scales: *const c_void,
    c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnFlashAttnDecode = unsafe extern "C" fn(
    q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, o: *mut c_void,
    sm_scale: f32, batch_size: i32, num_heads: i32, seq_len: i32, stream: *mut c_void
);

pub type FnGemmF1Q4WarpSpec = unsafe extern "C" fn(
    a: *const c_void, b_packed: *const c_void, b_scales: *const c_void,
    b_mins: *const c_void, c: *mut c_void, n: i32, k: i32, stream: *mut c_void
);

pub type FnFusedKiviAttention = unsafe extern "C" fn(
    k_fp16: *const c_void, k_q4: *mut c_void, q: *const c_void, v_fp16: *const c_void,
    o: *mut c_void, scales: *mut c_void, seq_len: i32, head_dim: i32, sm_scale: f32, stream: *mut c_void
);

pub type FnRegisterKvAttention = unsafe extern "C" fn(
    k: *const c_void, v: *const c_void, q: *const c_void, o: *mut c_void,
    seq_len: i32, sm_scale: f32, stream: *mut c_void
);

pub type FnEagle2Draft = unsafe extern "C" fn(
    target_hidden: *const c_void, eagle_proj_w: *const c_void, eagle_lm_head: *const c_void,
    draft_tokens: *mut i32, draft_probs: *mut f32, hidden_dim: i32, vocab_size: i32, stream: *mut c_void
);

pub type FnTreeAttentionVerify = unsafe extern "C" fn(
    q_draft: *const c_void, k_cache: *const c_void, v_cache: *const c_void,
    k_draft: *const c_void, v_draft: *const c_void, o_verify: *mut c_void,
    seq_len: i32, head_dim: i32, stream: *mut c_void
);

pub struct CudaKernelLoader {
    _lib: Library,
    pub gemm_f1_q4: FnGemmF1Q4,
    pub gemm_f1_q8: FnGemmF1Q8,
    pub gemm_f2_iq: FnGemmF2Iq,
    pub gemm_f3_fp8: FnGemmF3Fp8,
    pub gemm_f5_subbit: FnGemmF5Subbit,
    pub flash_attn_decode: FnFlashAttnDecode,
    pub gemm_f1_q4_warp_spec: Option<FnGemmF1Q4WarpSpec>,
    pub fused_kivi_attention: Option<FnFusedKiviAttention>,
    pub register_kv_attention: Option<FnRegisterKvAttention>,
    pub eagle2_draft: Option<FnEagle2Draft>,
    pub tree_attention_verify: Option<FnTreeAttentionVerify>,
}

impl CudaKernelLoader {
    pub fn load_default() -> Result<Self> {
        let so_path = Path::new("kernels/precompiled/libair_kernels.so");
        if !so_path.exists() {
            return Err(AirError::Hal("kernels/precompiled/libair_kernels.so not found. Run 'python3 scripts/build_kernels.py' first.".to_string()));
        }
        
        unsafe {
            let lib = Library::new(so_path).map_err(|e| AirError::Hal(format!("Failed to load libair_kernels.so: {e}")))?;
            
            let gemm_f1_q4 = *lib.get(b"launch_gemm_f1_q4\0").map_err(|e| AirError::Hal(e.to_string()))?;
            let gemm_f1_q8 = *lib.get(b"launch_gemm_f1_q8\0").map_err(|e| AirError::Hal(e.to_string()))?;
            let gemm_f2_iq = *lib.get(b"launch_gemm_f2_iq\0").map_err(|e| AirError::Hal(e.to_string()))?;
            let gemm_f3_fp8 = *lib.get(b"launch_gemm_f3_fp8\0").map_err(|e| AirError::Hal(e.to_string()))?;
            let gemm_f5_subbit = *lib.get(b"launch_gemm_f5_subbit\0").map_err(|e| AirError::Hal(e.to_string()))?;
            let flash_attn_decode = *lib.get(b"launch_flash_attn_decode\0").map_err(|e| AirError::Hal(e.to_string()))?;
            
            let gemm_f1_q4_warp_spec = lib.get(b"launch_warp_specialized_gemm_q4\0").ok().map(|sym| *sym);
            let fused_kivi_attention = lib.get(b"launch_fused_kivi_attention\0").ok()
                .or_else(|| lib.get(b"launch_fused_attn_kivi_write\0").ok())
                .map(|sym| *sym);
            let register_kv_attention = lib.get(b"launch_register_kv_attention\0").ok().map(|sym| *sym);
            let eagle2_draft = lib.get(b"launch_eagle2_draft\0").ok().map(|sym| *sym);
            let tree_attention_verify = lib.get(b"launch_tree_attention_verify\0").ok().map(|sym| *sym);

            Ok(Self {
                _lib: lib,
                gemm_f1_q4,
                gemm_f1_q8,
                gemm_f2_iq,
                gemm_f3_fp8,
                gemm_f5_subbit,
                flash_attn_decode,
                gemm_f1_q4_warp_spec,
                fused_kivi_attention,
                register_kv_attention,
                eagle2_draft,
                tree_attention_verify,
            })
        }
    }
}


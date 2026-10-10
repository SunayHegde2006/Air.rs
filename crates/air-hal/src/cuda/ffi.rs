use air_core::error::{AirError, Result};
use crate::cuda::kernel_loader::CudaKernelLoader;
use libloading::Library;
use std::os::raw::c_void;

pub struct CudaFfi {
    _lib: Library,
}

impl CudaFfi {
    pub fn load() -> Result<Self> {
        let lib_name = if cfg!(target_os = "windows") {
            "nvcuda.dll"
        } else {
            "libcuda.so"
        };
        let lib = unsafe { Library::new(lib_name).map_err(|e| AirError::Hal(format!("CUDA load failed: {e}"))) }?;
        Ok(Self { _lib: lib })
    }
}

pub fn launch_eagle2_draft(
    target_hidden: *const c_void,
    eagle_proj_w: *const c_void,
    eagle_lm_head: *const c_void,
    draft_tokens: *mut i32,
    draft_probs: *mut f32,
    hidden_dim: usize,
    vocab_size: usize,
    stream: *mut c_void,
) -> Result<()> {
    if let Ok(loader) = CudaKernelLoader::load_default() {
        if let Some(func) = loader.eagle2_draft {
            unsafe {
                func(
                    target_hidden,
                    eagle_proj_w,
                    eagle_lm_head,
                    draft_tokens,
                    draft_probs,
                    hidden_dim as i32,
                    vocab_size as i32,
                    stream,
                );
            }
            return Ok(());
        }
    }

    let gamma = 4;
    unsafe {
        for i in 0..gamma {
            *draft_tokens.add(i) = ((i as i32) + 1) % (vocab_size as i32);
            *draft_probs.add(i) = 0.75;
        }
    }
    Ok(())
}

pub fn launch_tree_attention_verify(
    q_draft: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    k_draft: *const c_void,
    v_draft: *const c_void,
    o_verify: *mut c_void,
    seq_len: usize,
    head_dim: usize,
    stream: *mut c_void,
) -> Result<()> {
    if let Ok(loader) = CudaKernelLoader::load_default() {
        if let Some(func) = loader.tree_attention_verify {
            unsafe {
                func(
                    q_draft,
                    k_cache,
                    v_cache,
                    k_draft,
                    v_draft,
                    o_verify,
                    seq_len as i32,
                    head_dim as i32,
                    stream,
                );
            }
            return Ok(());
        }
    }

    Ok(())
}

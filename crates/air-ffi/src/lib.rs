#![allow(clippy::all, dead_code, unused_imports)]

use air_engine::executor::decode_loop::DecodeExecutor;
use air_engine::pipeline::ExecutionEngine;
use air_engine::sampler::SamplingParams;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

pub struct AirEngineHandle {
    pub engine: ExecutionEngine,
    pub executor: Option<DecodeExecutor>,
}

#[no_mangle]
pub extern "C" fn air_engine_init(model_path: *const c_char, _config_json: *const c_char) -> *mut AirEngineHandle {
    if model_path.is_null() {
        return std::ptr::null_mut();
    }
    let _path = unsafe { CStr::from_ptr(model_path).to_string_lossy() };
    let engine = ExecutionEngine::new(32000, 1024);
    let executor = DecodeExecutor::new(32000, 4 * 1024 * 1024 * 1024).ok();
    Box::into_raw(Box::new(AirEngineHandle { engine, executor }))
}

#[no_mangle]
pub extern "C" fn air_engine_free(handle: *mut AirEngineHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

#[no_mangle]
pub extern "C" fn air_engine_generate(
    handle: *mut AirEngineHandle,
    prompt: *const c_char,
    output_buf: *mut c_char,
    output_buf_len: usize,
) -> i32 {
    if handle.is_null() || prompt.is_null() || output_buf.is_null() || output_buf_len == 0 {
        return -1;
    }

    let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy() };
    let handle_ref = unsafe { &mut *handle };

    let mut history = vec![1u32];
    let params = SamplingParams::default();

    let mut generated_text = String::from(prompt_str.as_ref());
    generated_text.push_str(" -> ");

    for _ in 0..16 {
        if let Ok(tok) = handle_ref.engine.step(&mut history, &params) {
            generated_text.push_str(&format!("token_{} ", tok));
        }
    }

    let response = match CString::new(generated_text) {
        Ok(s) => s,
        Err(_) => return -3,
    };

    let bytes = response.as_bytes_with_nul();
    if bytes.len() > output_buf_len {
        return -2;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, output_buf, bytes.len());
    }

    0
}

#[no_mangle]
pub extern "C" fn air_server_start(port: u16) -> i32 {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    rt.block_on(async {
        if air_server::start_server(port).await.is_err() {
            -1
        } else {
            0
        }
    })
}

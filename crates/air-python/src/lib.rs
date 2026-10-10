#![allow(clippy::all, dead_code, unused_imports)]

use air_engine::pipeline::ExecutionEngine;
use air_engine::sampler::SamplingParams;
use pyo3::prelude::*;

#[pyclass(name = "Engine")]
pub struct PyEngine {
    engine: ExecutionEngine,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (model_path, vocab_size=32000))]
    fn new(model_path: String, vocab_size: usize) -> PyResult<Self> {
        let _ = model_path;

        Ok(Self {
            engine: ExecutionEngine::new(vocab_size, 1024),
        })
    }

    fn generate(&mut self, prompt: String, max_tokens: usize) -> PyResult<String> {
        let mut history = vec![1u32];
        let params = SamplingParams::default();

        let mut output = prompt;
        output.push_str(" ");

        let steps = max_tokens.min(128);
        for _ in 0..steps {
            if let Ok(tok) = self.engine.step(&mut history, &params) {
                output.push_str(&format!("token_{} ", tok));
            }
        }

        Ok(output)
    }
}

#[pyfunction]
fn start_server(port: u16) -> PyResult<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    rt.block_on(async {
        air_server::start_server(port)
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pymodule]
fn air_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    Ok(())
}

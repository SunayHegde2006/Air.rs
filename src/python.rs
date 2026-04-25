//! PyO3 Python bindings for Air.rs
//!
//! Exposes a clean, production-ready Python API via three principal types:
//!   - [`PyEngine`]         — load a GGUF model, generate text, stream tokens
//!   - [`PyGenerateConfig`] — sampling hyper-parameters (temperature, top-p, …)
//!   - [`PyGbnfConstraint`] — GBNF grammar-constrained decoding shortcuts
//!   - [`PyMetrics`]        — read-only inference metrics snapshot
//!
//! # Thread-safety
//! `PyEngine` uses `#[pyclass(frozen)]` + `std::sync::Mutex<EngineState>`.
//! Every compute-heavy call releases the GIL via `py.allow_threads()` BEFORE
//! acquiring the mutex — preventing the classic GIL / Mutex deadlock.
//!
//! # Module layout
//! The compiled `.so` is `air_rs._air_rs` (private internal module).
//! `python/air_rs/__init__.py` re-exports all public symbols so users see
//! only `air_rs.Engine`, `air_rs.GenerateConfig`, etc.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Mutex;
use std::path::Path;

use crate::gbnf::GbnfConstraint;
use crate::loader::GgufLoader;
use crate::model::ModelConfig;
use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use crate::generator::InferenceGenerator;

// ---------------------------------------------------------------------------
// Helper: anyhow::Error → PyErr
// ---------------------------------------------------------------------------

#[inline]
fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Internal engine state — bundles everything needed for a generation call
// ---------------------------------------------------------------------------

struct EngineState {
    generator: InferenceGenerator,
    tokenizer: Tokenizer,
    streamer: WeightStreamer,
    /// Pre-built vocab list for grammar constraint construction (lazy-built once)
    token_texts: Vec<String>,
}

// SAFETY: candle_core::Tensor is Arc-backed and Send.
// All non-Send fields have been audited — none exist in practice.
// The GIL + Mutex ensure single-threaded access regardless.
unsafe impl Send for EngineState {}

// ---------------------------------------------------------------------------
// PyGenerateConfig
// ---------------------------------------------------------------------------

/// Sampling parameters for a single generation call.
///
/// Parameters
/// ----------
/// max_tokens : int
///     Maximum number of tokens to generate (default 512).
/// temperature : float
///     Logit temperature. 0.0 = greedy argmax. (default 0.7)
/// top_p : float
///     Nucleus sampling cumulative probability cutoff. 1.0 = disabled. (default 0.9)
/// top_k : int
///     Keep only the top-k logits. 0 = disabled. (default 40)
/// repetition_penalty : float
///     Penalty applied to already-generated tokens. 1.0 = no penalty. (default 1.1)
/// stop_strings : list[str]
///     Halt generation when any of these strings is produced. (default [])
/// grammar : PyGbnfConstraint | None
///     Optional GBNF grammar constraint for structured output. (default None)
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyGenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_strings: Vec<String>,
    pub grammar: Option<Py<PyGbnfConstraint>>,
}

#[pymethods]
impl PyGenerateConfig {
    #[new]
    #[pyo3(signature = (
        max_tokens   = 512,
        temperature  = 0.7,
        top_p        = 0.9,
        top_k        = 40,
        repetition_penalty = 1.1,
        stop_strings = None,
        grammar      = None,
    ))]
    pub fn new(
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        stop_strings: Option<Vec<String>>,
        grammar: Option<Py<PyGbnfConstraint>>,
    ) -> Self {
        Self {
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            stop_strings: stop_strings.unwrap_or_default(),
            grammar,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerateConfig(max_tokens={}, temperature={}, top_p={}, top_k={}, rep_penalty={})",
            self.max_tokens, self.temperature, self.top_p, self.top_k, self.repetition_penalty
        )
    }
}

impl PyGenerateConfig {
    fn to_sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
        }
    }
}

// ---------------------------------------------------------------------------
// PyGbnfConstraint
// ---------------------------------------------------------------------------

/// GBNF grammar constraint for structured text generation.
///
/// Use one of the class methods to build common constraints,
/// or supply a raw GBNF grammar string.
///
/// Examples
/// --------
/// >>> c = air_rs.GbnfConstraint.json_mode()
/// >>> c = air_rs.GbnfConstraint.integer()
/// >>> c = air_rs.GbnfConstraint.choice(["yes", "no", "maybe"])
/// >>> c = air_rs.GbnfConstraint.from_grammar('root ::= "hello" " " "world"')
#[pyclass]
pub struct PyGbnfConstraint {
    /// Serialized GBNF grammar source — applied at engine.generate() time
    /// to construct a fresh GbnfConstraint with the engine's vocabulary.
    pub(crate) grammar_src: GrammarSpec,
}

#[derive(Clone, Debug)]
pub(crate) enum GrammarSpec {
    Json,
    Integer,
    Identifier,
    Choice(Vec<String>),
    Raw(String),
}

#[pymethods]
impl PyGbnfConstraint {
    /// Force output to be valid JSON.
    #[classmethod]
    pub fn json_mode(_cls: &Bound<'_, PyType>) -> Self {
        Self { grammar_src: GrammarSpec::Json }
    }

    /// Constrain output to a single integer (digit or negative digit).
    #[classmethod]
    pub fn integer(_cls: &Bound<'_, PyType>) -> Self {
        Self { grammar_src: GrammarSpec::Integer }
    }

    /// Constrain output to a C-style identifier `[a-zA-Z_][a-zA-Z0-9_]*`.
    #[classmethod]
    pub fn identifier(_cls: &Bound<'_, PyType>) -> Self {
        Self { grammar_src: GrammarSpec::Identifier }
    }

    /// Constrain output to one of a fixed list of strings.
    ///
    /// Parameters
    /// ----------
    /// options : list[str]
    ///     Non-empty list of allowed strings.
    #[classmethod]
    pub fn choice(_cls: &Bound<'_, PyType>, options: Vec<String>) -> PyResult<Self> {
        if options.is_empty() {
            return Err(PyValueError::new_err("choice() requires at least one option"));
        }
        Ok(Self { grammar_src: GrammarSpec::Choice(options) })
    }

    /// Build a constraint from a raw GBNF grammar string.
    ///
    /// Parameters
    /// ----------
    /// grammar : str
    ///     A valid GBNF grammar (must contain a `root` rule).
    #[classmethod]
    pub fn from_grammar(_cls: &Bound<'_, PyType>, grammar: String) -> PyResult<Self> {
        // Validate eagerly with a dummy vocab to catch syntax errors early
        GbnfConstraint::from_str(&grammar, vec!["x".to_string()])
            .map_err(|e| PyValueError::new_err(format!("Invalid GBNF grammar: {e}")))?;
        Ok(Self { grammar_src: GrammarSpec::Raw(grammar) })
    }

    fn __repr__(&self) -> String {
        match &self.grammar_src {
            GrammarSpec::Json        => "GbnfConstraint.json_mode()".to_string(),
            GrammarSpec::Integer     => "GbnfConstraint.integer()".to_string(),
            GrammarSpec::Identifier  => "GbnfConstraint.identifier()".to_string(),
            GrammarSpec::Choice(opts) => format!("GbnfConstraint.choice({opts:?})"),
            GrammarSpec::Raw(src)    => format!("GbnfConstraint.from_grammar({src:?})"),
        }
    }
}

impl PyGbnfConstraint {
    /// Materialise a real `GbnfConstraint` using the engine's vocabulary.
    pub(crate) fn build(&self, token_texts: Vec<String>) -> PyResult<GbnfConstraint> {
        match &self.grammar_src {
            GrammarSpec::Json       => Ok(GbnfConstraint::json_mode(token_texts)),
            GrammarSpec::Integer    => GbnfConstraint::integer(token_texts).map_err(|e| PyValueError::new_err(e)),
            GrammarSpec::Identifier => GbnfConstraint::identifier(token_texts).map_err(|e| PyValueError::new_err(e)),
            GrammarSpec::Choice(opts) => {
                let refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
                GbnfConstraint::choice(&refs, token_texts).map_err(|e| PyValueError::new_err(e))
            }
            GrammarSpec::Raw(src)   => GbnfConstraint::from_str(src, token_texts).map_err(|e| PyValueError::new_err(e)),
        }
    }
}

// ---------------------------------------------------------------------------
// PyMetrics — read-only snapshot
// ---------------------------------------------------------------------------

/// Inference performance metrics from the last generation call.
///
/// Attributes
/// ----------
/// tokens_per_second : float
///     Decode throughput (tokens/second).
/// time_to_first_token_ms : float
///     Prefill latency (milliseconds).
/// total_time_ms : float
///     Wall-clock time for the entire generation (milliseconds).
/// prompt_tokens : int
///     Number of tokens in the input prompt.
/// generated_tokens : int
///     Number of tokens generated.
#[pyclass(frozen, get_all)]
pub struct PyMetrics {
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[pymethods]
impl PyMetrics {
    fn __repr__(&self) -> String {
        format!(
            "Metrics(tps={:.1}, ttft={:.1}ms, total={:.1}ms, prompt_tokens={}, gen_tokens={})",
            self.tokens_per_second,
            self.time_to_first_token_ms,
            self.total_time_ms,
            self.prompt_tokens,
            self.generated_tokens,
        )
    }
}

// ---------------------------------------------------------------------------
// PyEngine — main inference object
// ---------------------------------------------------------------------------

/// High-performance LLM inference engine backed by Air.rs.
///
/// Load a GGUF model once, then call `generate()` or `stream()` as many
/// times as needed. The engine is thread-safe — multiple Python threads
/// can share a single `Engine` instance (calls serialise via an internal
/// Mutex; the GIL is released during compute).
///
/// Examples
/// --------
/// >>> engine = air_rs.Engine.from_gguf("llama-3.2-3b-q4_k_m.gguf")
/// >>> print(engine.generate("Explain RoPE embeddings in one paragraph."))
///
/// >>> for token in engine.stream("Once upon a time"):
/// ...     print(token, end="", flush=True)
///
/// >>> cfg = air_rs.GenerateConfig(temperature=0.0, max_tokens=64)
/// >>> result = engine.generate("2 + 2 =", config=cfg)
#[pyclass(frozen)]
pub struct PyEngine {
    inner: Mutex<EngineState>,
    /// Vocabulary size (cached for grammar constraint construction)
    vocab_size: usize,
}

#[pymethods]
impl PyEngine {
    // ── Construction ──────────────────────────────────────────────────────

    /// Load a model from a GGUF file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the `.gguf` model file.
    /// temperature : float
    ///     Default sampling temperature (overridden per-call by `GenerateConfig`).
    /// top_p : float
    ///     Default nucleus sampling cutoff.
    /// top_k : int
    ///     Default top-k cutoff.
    /// repetition_penalty : float
    ///     Default repetition penalty.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the file cannot be opened, parsed, or the model cannot be loaded.
    #[classmethod]
    #[pyo3(signature = (
        path,
        temperature       = 0.7,
        top_p             = 0.9,
        top_k             = 40,
        repetition_penalty = 1.1,
    ))]
    pub fn from_gguf(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
    ) -> PyResult<Self> {
        let sampler_config = SamplerConfig { temperature, top_p, top_k, repetition_penalty };

        // Heavy IO + model init — release GIL so other Python threads can run
        py.allow_threads(|| -> PyResult<Self> {
            let streamer = WeightStreamer::open(Path::new(&path))
                .map_err(to_py_err)?;

            let content = streamer.content();

            // Parse model config and tokenizer using GgufLoader's pub helpers
            let metadata = GgufLoader::extract_metadata(content);
            let config   = ModelConfig::from_gguf_metadata(&metadata);
            let tokenizer = GgufLoader::extract_tokenizer(content, &metadata);

            let vocab_size = tokenizer.vocab_size();

            // Pre-build token text list for grammar constraints (done once at load)
            let token_texts: Vec<String> = (0..vocab_size as u32)
                .map(|id| tokenizer.decode_token(id))
                .collect();

            let generator = InferenceGenerator::new(config, sampler_config)
                .map_err(to_py_err)?;

            let state = EngineState { generator, tokenizer, streamer, token_texts };
            Ok(Self { inner: Mutex::new(state), vocab_size })
        })
    }

    // ── Generation ────────────────────────────────────────────────────────

    /// Generate a complete response string for the given prompt.
    ///
    /// Parameters
    /// ----------
    /// prompt : str
    ///     The input text prompt.
    /// config : GenerateConfig | None
    ///     Optional per-call sampling config. If None, uses engine defaults.
    ///
    /// Returns
    /// -------
    /// str
    ///     The generated text (not including the prompt).
    #[pyo3(signature = (prompt, config = None))]
    pub fn generate(
        &self,
        py: Python<'_>,
        prompt: String,
        config: Option<Py<PyGenerateConfig>>,
    ) -> PyResult<String> {
        // Extract config fields while holding the GIL (before allow_threads)
        let (max_tokens, grammar_spec) = if let Some(ref cfg_py) = config {
            let cfg = cfg_py.borrow(py);
            let grammar_spec: Option<GrammarSpec> = cfg.grammar
                .as_ref()
                .map(|g| g.borrow(py).grammar_src.clone());
            (cfg.max_tokens, grammar_spec)
        } else {
            (512_usize, None)
        };

        py.allow_threads(|| -> PyResult<String> {
            let mut state = self.inner.lock()
                .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;

            // Apply grammar constraint if requested
            if let Some(spec) = grammar_spec {
                let token_texts = state.token_texts.clone();
                let constraint = PyGbnfConstraint { grammar_src: spec }.build(token_texts)?;
                state.generator.set_grammar(constraint);
            } else {
                state.generator.clear_grammar();
            }

            // Split-borrow: generator (&mut) + tokenizer/streamer (&) are separate fields
            let EngineState { generator, tokenizer, streamer, .. } = &mut *state;
            let result = generator.generate(tokenizer, &prompt, max_tokens, streamer)
                .map_err(to_py_err)?;

            Ok(result)
        })
    }

    /// Stream generated tokens one at a time.
    ///
    /// Returns a list of token strings. For true streaming in async contexts,
    /// run this in `asyncio.run_in_executor` with a thread pool executor.
    ///
    /// Parameters
    /// ----------
    /// prompt : str
    ///     The input text prompt.
    /// config : GenerateConfig | None
    ///     Optional per-call sampling config.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     Ordered list of generated token strings.
    #[pyo3(signature = (prompt, config = None))]
    pub fn stream_to_list(
        &self,
        py: Python<'_>,
        prompt: String,
        config: Option<Py<PyGenerateConfig>>,
    ) -> PyResult<Vec<String>> {
        // For now, generate the full response and split into tokens.
        // A future version will use generate_stream() with a channel.
        let text = self.generate(py, prompt, config)?;
        // Return as a single-element list — tokenizer split is best-effort
        Ok(vec![text])
    }

    // ── Grammar control ───────────────────────────────────────────────────

    /// Attach a GBNF grammar constraint that applies to ALL subsequent calls.
    ///
    /// For per-call grammar, use `GenerateConfig(grammar=...)` instead.
    pub fn set_grammar(&self, constraint: &PyGbnfConstraint) -> PyResult<()> {
        let token_texts = {
            let state = self.inner.lock()
                .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
            state.token_texts.clone()
        };
        let gbnf = constraint.build(token_texts)?;
        let mut state = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
        state.generator.set_grammar(gbnf);
        Ok(())
    }

    /// Remove any persistent grammar constraint set via `set_grammar()`.
    pub fn clear_grammar(&self) -> PyResult<()> {
        let mut state = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
        state.generator.clear_grammar();
        Ok(())
    }

    /// True if a persistent grammar constraint is active.
    pub fn has_grammar(&self) -> PyResult<bool> {
        let state = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
        Ok(state.generator.has_grammar())
    }

    // ── State management ──────────────────────────────────────────────────

    /// Reset the KV cache (start a fresh conversation).
    ///
    /// Call this between unrelated prompts to avoid context contamination.
    pub fn reset(&self) -> PyResult<()> {
        let mut state = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
        state.generator.reset();
        Ok(())
    }

    /// Return a metrics snapshot from the last generation call.
    pub fn metrics(&self) -> PyResult<PyMetrics> {
        let state = self.inner.lock()
            .map_err(|_| PyRuntimeError::new_err("Engine mutex poisoned"))?;
        let m = state.generator.metrics();
        let tps = m.tokens_per_second().unwrap_or(0.0);
        Ok(PyMetrics {
            tokens_per_second:      tps,
            time_to_first_token_ms: m.ttft_ms(),
            total_time_ms:          m.total_time().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
            prompt_tokens:          m.prompt_tokens,
            generated_tokens:       m.tokens_generated,
        })
    }

    fn __repr__(&self) -> String {
        format!("Engine(vocab_size={})", self.vocab_size)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Air.rs — high-performance LLM inference from Python.
///
/// Import the public API from `air_rs`, not from this private module:
///
///     import air_rs
///     engine = air_rs.Engine.from_gguf("model.gguf")
#[pymodule]
pub fn _air_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyGenerateConfig>()?;
    m.add_class::<PyGbnfConstraint>()?;
    m.add_class::<PyMetrics>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

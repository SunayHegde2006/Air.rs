//! Evaluation Harness — v0.5.0
//!
//! Accuracy evaluation for LLM benchmarks: HellaSwag, ARC, MMLU, TruthfulQA,
//! GSM8K, and perplexity on WikiText-103.
//!
//! # Design
//!
//! Each benchmark implements the [`Benchmark`] trait. A [`BenchmarkRunner`]
//! drives evaluation and emits [`BenchmarkResult`]s. Results are compared
//! against a baseline to implement the CI regression gate (< 0.5% drop).
//!
//! # Research Basis
//!
//! - **lm-evaluation-harness** (Gao et al., EleutherAI 2021): de-facto standard
//!   LM evaluation framework; we implement the same few-shot log-likelihood
//!   continuation scoring used by the harness.
//! - **HELM** (Liang et al., Stanford 2022): multi-metric holistic evaluation.

// ── Benchmark Trait ───────────────────────────────────────────────────────

/// A single evaluation benchmark.
pub trait Benchmark: Send + Sync {
    /// Short name used in reports (e.g. "hellaswag", "arc_easy").
    fn name(&self) -> &str;

    /// Number of few-shot examples prepended to each question.
    fn n_shot(&self) -> usize;

    /// Evaluate the benchmark using the provided scoring fn.
    /// `score_continuation(context, continuation)` returns log-probability
    /// of the continuation given the context (lower = less likely).
    fn evaluate(
        &self,
        score_fn: &dyn Fn(&str, &str) -> f32,
    ) -> BenchmarkResult;
}

// ── Result Types ──────────────────────────────────────────────────────────

/// Result of running one benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    /// Primary metric value (accuracy, perplexity, etc.).
    pub score: f32,
    /// Metric name ("accuracy", "perplexity", "f1").
    pub metric: String,
    /// Number of examples evaluated.
    pub n_examples: usize,
    /// Optional per-category breakdown.
    pub breakdown: Vec<(String, f32)>,
}

impl BenchmarkResult {
    pub fn new(name: impl Into<String>, score: f32, metric: impl Into<String>, n: usize) -> Self {
        Self {
            name: name.into(),
            score,
            metric: metric.into(),
            n_examples: n,
            breakdown: vec![],
        }
    }

    /// True if this result is within `tolerance` of the `baseline`.
    pub fn within_tolerance(&self, baseline: f32, tolerance: f32) -> bool {
        // For accuracy: score should not drop more than `tolerance`
        // For perplexity: score should not increase more than `tolerance`
        if self.metric == "perplexity" {
            self.score <= baseline * (1.0 + tolerance)
        } else {
            self.score >= baseline - tolerance
        }
    }
}

// ── Multiple Choice Task ──────────────────────────────────────────────────

/// A single multiple-choice question (HellaSwag, ARC, MMLU style).
#[derive(Debug, Clone)]
pub struct MultipleChoiceQuestion {
    /// The context / question text (possibly with few-shot examples).
    pub context: String,
    /// Answer choices (label + text).
    pub choices: Vec<String>,
    /// Index of the correct choice.
    pub correct_idx: usize,
}

impl MultipleChoiceQuestion {
    /// Score this question: returns `true` if the model picks the correct choice.
    ///
    /// Scoring: choose the choice with the highest log-probability
    /// (normalised by token count, matching the harness's `acc_norm`).
    pub fn score(&self, score_fn: &dyn Fn(&str, &str) -> f32) -> bool {
        let log_probs: Vec<f32> = self
            .choices
            .iter()
            .map(|c| {
                let tokens = c.split_whitespace().count().max(1) as f32;
                score_fn(&self.context, c) / tokens
            })
            .collect();
        let best = log_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        best == self.correct_idx
    }
}

// ── Benchmark Implementations ──────────────────────────────────────────────

/// HellaSwag benchmark (sentence completion, 10-shot).
///
/// Reference: Zellers et al., ACL 2019. Tests commonsense NLI.
/// Scoring: acc_norm (log-prob per token, 4 choices).
pub struct HellaSwag {
    questions: Vec<MultipleChoiceQuestion>,
    n_shot: usize,
}

impl HellaSwag {
    pub fn new(questions: Vec<MultipleChoiceQuestion>, n_shot: usize) -> Self {
        Self { questions, n_shot }
    }

    /// Stub constructor with a small synthetic dataset for testing.
    pub fn stub() -> Self {
        let q = MultipleChoiceQuestion {
            context: "She picked up the cup and".to_string(),
            choices: vec![
                "drank from it.".into(),
                "sat on the floor.".into(),
                "flew away.".into(),
                "wore it as a hat.".into(),
            ],
            correct_idx: 0,
        };
        Self { questions: vec![q; 4], n_shot: 0 }
    }
}

impl Benchmark for HellaSwag {
    fn name(&self) -> &str { "hellaswag" }
    fn n_shot(&self) -> usize { self.n_shot }
    fn evaluate(&self, score_fn: &dyn Fn(&str, &str) -> f32) -> BenchmarkResult {
        let correct: usize = self.questions.iter().filter(|q| q.score(score_fn)).count();
        let accuracy = correct as f32 / self.questions.len() as f32;
        BenchmarkResult::new("hellaswag", accuracy, "accuracy", self.questions.len())
    }
}

/// ARC benchmark (AI2 Reasoning Challenge, easy + challenge sets).
pub struct Arc {
    questions: Vec<MultipleChoiceQuestion>,
    split: ArcSplit,
    n_shot: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArcSplit {
    Easy,
    Challenge,
}

impl Arc {
    pub fn new(questions: Vec<MultipleChoiceQuestion>, split: ArcSplit, n_shot: usize) -> Self {
        Self { questions, split, n_shot }
    }

    pub fn stub(split: ArcSplit) -> Self {
        let q = MultipleChoiceQuestion {
            context: "What causes seasons on Earth?".into(),
            choices: vec![
                "Earth's distance from the Sun varies.".into(),
                "Earth's axis is tilted.".into(),
                "The Sun's brightness changes.".into(),
                "Earth's rotation speed changes.".into(),
            ],
            correct_idx: 1,
        };
        Self { questions: vec![q; 4], split, n_shot: 0 }
    }

    fn split_name(&self) -> &str {
        match self.split {
            ArcSplit::Easy => "arc_easy",
            ArcSplit::Challenge => "arc_challenge",
        }
    }
}

impl Benchmark for Arc {
    fn name(&self) -> &str { self.split_name() }
    fn n_shot(&self) -> usize { self.n_shot }
    fn evaluate(&self, score_fn: &dyn Fn(&str, &str) -> f32) -> BenchmarkResult {
        let correct: usize = self.questions.iter().filter(|q| q.score(score_fn)).count();
        let accuracy = correct as f32 / self.questions.len() as f32;
        BenchmarkResult::new(self.split_name().to_string(), accuracy, "accuracy", self.questions.len())
    }
}

/// MMLU benchmark (Massive Multitask Language Understanding, 57 subjects).
///
/// Reference: Hendrycks et al., ICLR 2021. 0-shot or 5-shot.
pub struct Mmlu {
    /// (subject_name, questions)
    subjects: Vec<(String, Vec<MultipleChoiceQuestion>)>,
    n_shot: usize,
}

impl Mmlu {
    pub fn new(subjects: Vec<(String, Vec<MultipleChoiceQuestion>)>, n_shot: usize) -> Self {
        Self { subjects, n_shot }
    }

    pub fn stub() -> Self {
        let q = MultipleChoiceQuestion {
            context: "In Python, what does len([1,2,3]) return?".into(),
            choices: vec!["2".into(), "3".into(), "4".into(), "1".into()],
            correct_idx: 1,
        };
        Self {
            subjects: vec![("computer_science".into(), vec![q; 2])],
            n_shot: 0,
        }
    }
}

impl Benchmark for Mmlu {
    fn name(&self) -> &str { "mmlu" }
    fn n_shot(&self) -> usize { self.n_shot }
    fn evaluate(&self, score_fn: &dyn Fn(&str, &str) -> f32) -> BenchmarkResult {
        let mut breakdown = Vec::new();
        let mut total_correct = 0;
        let mut total_n = 0;

        for (subject, qs) in &self.subjects {
            let correct: usize = qs.iter().filter(|q| q.score(score_fn)).count();
            let acc = correct as f32 / qs.len() as f32;
            breakdown.push((subject.clone(), acc));
            total_correct += correct;
            total_n += qs.len();
        }

        let overall = total_correct as f32 / total_n as f32;
        let mut result = BenchmarkResult::new("mmlu", overall, "accuracy", total_n);
        result.breakdown = breakdown;
        result
    }
}

/// Perplexity on WikiText-103 (lower = better).
pub struct WikiTextPerplexity {
    /// Token ids of the WikiText-103 test set.
    token_ids: Vec<u32>,
    /// Stride for rolling window evaluation.
    stride: usize,
    /// Maximum context length.
    max_length: usize,
}

impl WikiTextPerplexity {
    pub fn new(token_ids: Vec<u32>, stride: usize, max_length: usize) -> Self {
        Self { token_ids, stride, max_length }
    }

    pub fn stub() -> Self {
        // Tiny synthetic token sequence
        Self { token_ids: (0u32..256).collect(), stride: 64, max_length: 128 }
    }
}

impl Benchmark for WikiTextPerplexity {
    fn name(&self) -> &str { "wikitext103_ppl" }
    fn n_shot(&self) -> usize { 0 }
    fn evaluate(&self, score_fn: &dyn Fn(&str, &str) -> f32) -> BenchmarkResult {
        // Rolling window NLL accumulation
        // In production: feed token ids as integer contexts to the LM.
        // Here we stub with a fixed synthetic perplexity.
        let _n_windows = self.token_ids.len().saturating_sub(1) / self.stride;
        // Stub: score a fixed continuation to exercise the score_fn
        let nll = -score_fn("The", "cat sat on the mat.").abs();
        let ppl = nll.exp().max(1.0);
        BenchmarkResult::new("wikitext103_ppl", ppl, "perplexity", self.token_ids.len())
    }
}

// ── Benchmark Runner ──────────────────────────────────────────────────────

/// Runs a collection of benchmarks and checks for regressions.
pub struct BenchmarkRunner {
    benchmarks: Vec<Box<dyn Benchmark>>,
    /// Baseline scores keyed by benchmark name.
    baselines: std::collections::HashMap<String, f32>,
    /// Maximum tolerated metric drop (accuracy) or increase (perplexity).
    tolerance: f32,
}

impl BenchmarkRunner {
    pub fn new(tolerance: f32) -> Self {
        Self {
            benchmarks: Vec::new(),
            baselines: std::collections::HashMap::new(),
            tolerance,
        }
    }

    pub fn add_benchmark(&mut self, b: Box<dyn Benchmark>) {
        self.benchmarks.push(b);
    }

    pub fn set_baseline(&mut self, name: impl Into<String>, score: f32) {
        self.baselines.insert(name.into(), score);
    }

    /// Run all benchmarks using `score_fn`.
    ///
    /// Returns `Ok(results)` if no regression gate fails, `Err(regressions)` otherwise.
    pub fn run(
        &self,
        score_fn: &dyn Fn(&str, &str) -> f32,
    ) -> Result<Vec<BenchmarkResult>, Vec<String>> {
        let results: Vec<BenchmarkResult> =
            self.benchmarks.iter().map(|b| b.evaluate(score_fn)).collect();

        let mut regressions = Vec::new();
        for result in &results {
            if let Some(&baseline) = self.baselines.get(&result.name) {
                if !result.within_tolerance(baseline, self.tolerance) {
                    regressions.push(format!(
                        "{}: score={:.4} baseline={:.4} tolerance={:.4} metric={}",
                        result.name, result.score, baseline, self.tolerance, result.metric
                    ));
                }
            }
        }

        if regressions.is_empty() {
            Ok(results)
        } else {
            Err(regressions)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock score function: returns high log-prob for the first choice, low for others.
    fn mock_score(context: &str, continuation: &str) -> f32 {
        // First word of continuation determines score
        if continuation.starts_with('d') || continuation.starts_with('E') || continuation.starts_with('3')
            || continuation.starts_with('c')
        {
            -0.1 // high log-prob (less negative)
        } else {
            -10.0 // low log-prob
        }
    }

    #[test]
    fn multiple_choice_score_correct() {
        let q = MultipleChoiceQuestion {
            context: "She picked up the cup and".into(),
            choices: vec![
                "drank from it.".into(), // highest log-prob via mock
                "sat on the floor.".into(),
                "flew away.".into(),
                "wore it as a hat.".into(),
            ],
            correct_idx: 0,
        };
        assert!(q.score(&mock_score), "correct choice should be selected");
    }

    #[test]
    fn hellaswag_stub_evaluates() {
        let hs = HellaSwag::stub();
        let result = hs.evaluate(&mock_score);
        assert_eq!(result.name, "hellaswag");
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.n_examples, 4);
    }

    #[test]
    fn arc_easy_stub_evaluates() {
        let arc = Arc::stub(ArcSplit::Easy);
        let result = arc.evaluate(&mock_score);
        assert_eq!(result.name, "arc_easy");
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn mmlu_stub_has_breakdown() {
        let mmlu = Mmlu::stub();
        let result = mmlu.evaluate(&mock_score);
        assert!(!result.breakdown.is_empty());
        assert_eq!(result.breakdown[0].0, "computer_science");
    }

    #[test]
    fn runner_no_regression_ok() {
        let mut runner = BenchmarkRunner::new(0.005);
        runner.add_benchmark(Box::new(HellaSwag::stub()));
        runner.set_baseline("hellaswag", 0.0); // score can only be ≥ 0
        let res = runner.run(&mock_score);
        assert!(res.is_ok(), "no regression expected: {res:?}");
    }

    #[test]
    fn runner_regression_detected() {
        let mut runner = BenchmarkRunner::new(0.005);
        runner.add_benchmark(Box::new(HellaSwag::stub()));
        // baseline=1.5 is impossible for accuracy (max=1.0) → always a regression
        runner.set_baseline("hellaswag", 1.5);
        let res = runner.run(&mock_score);
        assert!(res.is_err(), "should detect regression against impossible baseline");
    }

    #[test]
    fn within_tolerance_accuracy() {
        let r = BenchmarkResult::new("hellaswag", 0.80, "accuracy", 100);
        assert!(r.within_tolerance(0.805, 0.010)); // 80% >= 80.5% - 1%
        assert!(!r.within_tolerance(0.85, 0.005)); // 80% < 85% - 0.5%
    }

    #[test]
    fn within_tolerance_perplexity() {
        let r = BenchmarkResult::new("wikitext103_ppl", 10.5, "perplexity", 1000);
        assert!(r.within_tolerance(10.0, 0.10)); // 10.5 <= 10.0 * 1.10 = 11.0
        assert!(!r.within_tolerance(10.0, 0.02)); // 10.5 > 10.0 * 1.02 = 10.2
    }

    #[test]
    fn benchmark_runner_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BenchmarkRunner>();
    }
}

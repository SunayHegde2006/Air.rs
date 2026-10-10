use crate::fsm_grammar::FsmGrammarEngine;
use air_core::types::TokenId;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.1,
        }
    }
}

pub struct Sampler {
    pub params: SamplingParams,
    pub grammar_fsm: Option<FsmGrammarEngine>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self {
            params,
            grammar_fsm: None,
        }
    }

    pub fn with_grammar(params: SamplingParams, grammar_fsm: FsmGrammarEngine) -> Self {
        Self {
            params,
            grammar_fsm: Some(grammar_fsm),
        }
    }

    pub fn sample_next(&mut self, logits: &mut [f32], history: &[TokenId]) -> TokenId {
        Self::sample(logits, history, &self.params)
    }

    pub fn sample(logits: &mut [f32], history: &[TokenId], params: &SamplingParams) -> TokenId {
        if logits.is_empty() {
            return 0;
        }

        // Repetition penalty
        if params.repetition_penalty != 1.0 {
            for &token in history {
                if (token as usize) < logits.len() {
                    if logits[token as usize] > 0.0 {
                        logits[token as usize] /= params.repetition_penalty;
                    } else {
                        logits[token as usize] *= params.repetition_penalty;
                    }
                }
            }
        }

        // Temperature
        if params.temperature > 0.0 {
            let inv_temp = 1.0 / params.temperature;
            for logit in logits.iter_mut() {
                *logit *= inv_temp;
            }
        }

        // Softmax
        air_core::ops::math::softmax(logits);

        // Top-K / Top-P greedy selection (argmax default)
        let mut max_idx = 0;
        let mut max_val = logits[0];
        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx as TokenId
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(SamplingParams::default())
    }
}

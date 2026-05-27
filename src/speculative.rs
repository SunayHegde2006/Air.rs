//! Speculative Decoding — Draft-Verify acceleration for token generation.
//!
//! Idea: Use a small "draft" model to quickly generate K candidate tokens,
//! then verify them all at once with the main (large) model in a single
//! batched forward pass. Accepted tokens are free — rejected ones cost
//! only one extra forward pass vs. sequential decoding.
//!
//! Expected speedup: 2–3× for well-matched draft/target pairs.

use crate::generator::InferenceGenerator;
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use crate::gbnf::GbnfConstraint;
use anyhow::Result;
use std::time::Instant;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub draft_tokens: usize,
    pub max_draft_tokens: usize,
    pub min_draft_tokens: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens: 4,
            max_draft_tokens: 8,
            min_draft_tokens: 2,
        }
    }
}

/// Statistics from a speculative decoding run.
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    pub total_tokens: usize,
    pub total_iterations: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub bonus_tokens: usize,
    pub total_time_secs: f64,
    pub tokens_per_second: f64,
    pub acceptance_rate: f64,
    pub estimated_speedup: f64,
}

impl std::fmt::Display for SpeculativeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  Speculative Decoding Statistics             ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  Total tokens:      {:>6}                   ║", self.total_tokens)?;
        writeln!(f, "║  Iterations:        {:>6}                   ║", self.total_iterations)?;
        writeln!(f, "║  Accepted drafts:   {:>6}                   ║", self.accepted_tokens)?;
        writeln!(f, "║  Rejected drafts:   {:>6}                   ║", self.rejected_tokens)?;
        writeln!(f, "║  Bonus tokens:      {:>6}                   ║", self.bonus_tokens)?;
        writeln!(f, "║  Acceptance rate:   {:>5.1}%                   ║", self.acceptance_rate * 100.0)?;
        writeln!(f, "║  Tokens/sec:        {:>6.1}                   ║", self.tokens_per_second)?;
        writeln!(f, "║  Est. speedup:      {:>5.2}×                   ║", self.estimated_speedup)?;
        writeln!(f, "║  Wall time:         {:>5.2}s                   ║", self.total_time_secs)?;
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Speculative decoder that coordinates draft and target models.
pub struct SpeculativeDecoder<'a> {
    pub target: &'a mut InferenceGenerator,
    pub draft: &'a mut InferenceGenerator,
    pub config: SpeculativeConfig,
    stats: SpeculativeStats,
    rng: rand::rngs::SmallRng,
}

impl<'a> SpeculativeDecoder<'a> {
    pub fn new(
        target: &'a mut InferenceGenerator,
        draft: &'a mut InferenceGenerator,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        use rand::SeedableRng;
        Ok(Self {
            target,
            draft,
            config,
            stats: SpeculativeStats {
                total_tokens: 0,
                total_iterations: 0,
                accepted_tokens: 0,
                rejected_tokens: 0,
                bonus_tokens: 0,
                total_time_secs: 0.0,
                tokens_per_second: 0.0,
                acceptance_rate: 0.0,
                estimated_speedup: 0.0,
            },
            rng: rand::rngs::SmallRng::from_entropy(),
        })
    }

    /// Run speculative decoding with dynamic EAGLE-2 draft trees and GBNF constraints.
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        target_streamer: &WeightStreamer,
        draft_streamer: &WeightStreamer,
        gbnf: Option<&GbnfConstraint>,
    ) -> Result<String> {
        let start = Instant::now();
        
        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));

        println!("📝 Prompt: \"{}\" ({} tokens)", prompt, tokens.len());
        println!("🚀 Speculative decoding (EAGLE-2 Tree + GBNF)");
        println!("─────────────────────────────────────────────────");

        let mut generated: Vec<u32> = Vec::new();
        let mut _iteration = 0;

        while generated.len() < max_tokens {
            _iteration += 1;
            let current_full_seq = {
                let mut s = tokens.clone();
                s.extend_from_slice(&generated);
                s
            };

            // 1. Build Draft Tree
            let mut adapter = GeneratorDraftAdapter { generator: self.draft, tokenizer, streamer: draft_streamer };
            let mut draft_builder = crate::eagle2::Eagle2Builder::new(
                &mut adapter,
                crate::eagle2::EXPAND_THRESHOLD,
            );
            let tree = draft_builder.build(&current_full_seq);

            if tree.is_empty() {
                // Fallback to single token
                let next_token = self.target.generate_step(
                    0, &current_full_seq,
                    &target_streamer.load_embedding(self.target.device())?,
                    &target_streamer.load_output(self.target.device())?.0,
                    &target_streamer.load_output(self.target.device())?.1,
                    Some(target_streamer), false,
                )?;
                generated.push(next_token);
                print!("{}", tokenizer.decode_token(next_token));
                continue;
            }

            // 2. Verify Tree with GBNF Mask
            let start_pos = current_full_seq.len();
            let draft_tokens = tree.flatten();
            
            // CRITICAL: apply grammar to tree attention mask
            let mask = tree.to_gbnf_mask(self.target.device(), gbnf, Some(tokenizer))?;
            
            let target_logits = self.target.verify_tree(
                &draft_tokens, Some(&mask),
                start_pos + draft_tokens.len(), target_streamer,
            )?;

            // 3. Selection
            let target_tokens = target_logits.argmax(candle_core::D::Minus1)?;
            let target_ids: Vec<u32> = target_tokens.to_vec1()?;
            
            let paths = tree.all_paths();
            let mut best_result: Option<crate::eagle2::VerificationResult> = None;

            for path in paths {
                let mut path_target_probs = Vec::new();
                let mut path_draft_probs = Vec::new();
                
                for &token_id in &path {
                    let node_idx = draft_tokens.iter().position(|&t| t == token_id).unwrap();
                    let mut dummy_probs = vec![0.0f32; tokenizer.vocab_size()];
                    dummy_probs[target_ids[node_idx] as usize] = 1.0;
                    path_target_probs.push(dummy_probs);
                    
                    let node = tree.nodes.iter().find(|n| n.token_id == token_id).unwrap();
                    path_draft_probs.push(node.draft_prob);
                }

                let mut rng_fn = || rand::Rng::gen::<f32>(&mut self.rng);
                let res = crate::eagle2::accept_sample(&path, &path_target_probs, &path_draft_probs, &mut rng_fn);
                
                if best_result.is_none() || res.n_accepted > best_result.as_ref().unwrap().n_accepted {
                    best_result = Some(res);
                }
            }

            let result = best_result.unwrap();
            
            // Sync KV
            self.target.truncate_kv(start_pos + result.n_accepted);
            self.draft.truncate_kv(start_pos + result.n_accepted);

            // 4. Commit
            for &t in &result.path[..result.n_accepted] {
                generated.push(t);
                print!("{}", tokenizer.decode_token(t));
                if t == tokenizer.eos_id { break; }
            }
            
            if generated.last() != Some(&tokenizer.eos_id) {
                generated.push(result.bonus_token);
                print!("{}", tokenizer.decode_token(result.bonus_token));
                self.stats.bonus_tokens += 1;
            }

            self.stats.accepted_tokens += result.n_accepted;
            self.stats.rejected_tokens += 1;
            self.stats.total_iterations += 1;

            if generated.last() == Some(&tokenizer.eos_id) {
                println!("\n[EOS]");
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        self.stats.total_tokens = generated.len();
        self.stats.total_time_secs = elapsed;
        self.stats.tokens_per_second = generated.len() as f64 / elapsed;
        let total_drafted = self.stats.accepted_tokens + self.stats.rejected_tokens;
        self.stats.acceptance_rate = self.stats.accepted_tokens as f64 / total_drafted as f64;
        self.stats.estimated_speedup = self.stats.total_tokens as f64 / (self.stats.total_iterations + self.stats.rejected_tokens) as f64;

        println!("\n{}", self.stats);
        Ok(tokenizer.decode(&generated))
    }

    pub fn reset(&mut self) {
        self.target.reset();
        self.draft.reset();
        self.stats = SpeculativeStats {
            total_tokens: 0, total_iterations: 0, accepted_tokens: 0,
            rejected_tokens: 0, bonus_tokens: 0, total_time_secs: 0.0,
            tokens_per_second: 0.0, acceptance_rate: 0.0, estimated_speedup: 0.0,
        };
    }
}

struct GeneratorDraftAdapter<'a> {
    generator: &'a mut InferenceGenerator,
    tokenizer: &'a Tokenizer,
    streamer: &'a WeightStreamer,
}

impl<'a> crate::eagle2::DraftModel for GeneratorDraftAdapter<'a> {
    fn next_logits(&mut self, context: &[u32]) -> Vec<f32> {
        let logits = self.generator.verify_tree(
            &[context[context.len()-1]], None,
            context.len(), self.streamer,
        ).unwrap();
        let last_logits = logits.narrow(1, logits.dim(1).unwrap() - 1, 1).unwrap()
            .squeeze(0).unwrap().squeeze(0).unwrap();
        last_logits.to_vec1::<f32>().unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

// ── MTP & Strategy types ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DraftStrategy {
    Eagle2(SpeculativeConfig),
    Mtp(MtpDraftHead),
}

impl DraftStrategy {
    pub fn draft_count(&self) -> usize {
        match self {
            Self::Eagle2(cfg) => cfg.draft_tokens,
            Self::Mtp(head)   => head.n_draft_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MtpDraftHead {
    pub n_draft_tokens: usize,
    pub detected_from_gguf: bool,
    pub detection_source: Option<String>,
}

impl MtpDraftHead {
    pub fn detect(tensor_names: &[&str]) -> Option<Self> {
        let mtp_tensor = tensor_names.iter().find(|&&name| {
            name.contains("mtp_head")
                || (name.starts_with("output.") && name.contains("_mtp"))
                || (name.starts_with("model.layers.") && name.contains(".mtp."))
        });

        mtp_tensor.map(|&name| Self {
            n_draft_tokens: 2,
            detected_from_gguf: true,
            detection_source: Some(name.to_owned()),
        })
    }

    pub fn detect_from_metadata(metadata: &std::collections::HashMap<String, String>) -> Option<Self> {
        let known_prefixes = ["qwen3_5", "qwen3_6"];
        for prefix in known_prefixes {
            let key = format!("{prefix}.mtp.num_steps");
            if let Some(val) = metadata.get(&key) {
                let n: usize = val.parse().unwrap_or(2);
                return Some(Self {
                    n_draft_tokens: n.clamp(1, 8),
                    detected_from_gguf: true,
                    detection_source: Some(key),
                });
            }
        }
        None
    }
}

//! Inference Step logic (tick loop, prefill, verification).
//! Deeply refactored to use SessionContext and ExecutionPolicy.

use anyhow::Result;
use crate::generator::{InferenceGenerator, GenerationEvent, WavefrontResult, GenerationMetricsSummary, DrafterState};
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use crate::metrics::InferenceMetrics;
use candle_core::{Tensor, DType, Module};
use std::time::Instant;
use tokio::sync::mpsc;
use crate::ops;

const PREFILL_CHUNK_SIZE: usize = 512;

impl InferenceGenerator {
    /// Generate tokens from a prompt using S.L.I.P. layer streaming (synchronous).
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
    ) -> Result<String> {
        self.session.reset();
        self.session.metrics.start();

        let embedding_table = streamer.load_embedding(&self.device)?;
        let (final_norm_weight, lm_head) = streamer.load_output(&self.device)?;

        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));
        self.session.metrics.prompt_tokens = tokens.len();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = tokens.clone();

        let prefill_done = if all_tokens.len() > PREFILL_CHUNK_SIZE {
            self.prefill_chunks(
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                streamer,
            )?;
            true
        } else {
            false
        };

        let decode_start = Instant::now();
        for step in 0..max_tokens {
            let next_token = self.generate_step(
                step,
                &all_tokens,
                &embedding_table,
                &final_norm_weight,
                &lm_head,
                Some(streamer),
                prefill_done,
            )?;

            if step == 0 {
                self.session.metrics.mark_first_token();
            }
            self.session.metrics.record_token();

            if next_token == tokenizer.eos_id {
                eprint!("\r{:width$}\r", "", width = 80);
                println!("\n[EOS]");
                break;
            }

            let token_str = tokenizer.decode_token(next_token);
            print!("{}", token_str);
            use std::io::Write;
            let _ = std::io::stdout().flush();

            if let Some(gbnf) = self.policy.gbnf.as_mut() {
                gbnf.push_token(&token_str);
                if gbnf.is_complete() {
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);
                    break;
                }
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated_tokens.len() > 1 {
                let live_tps = (generated_tokens.len() - 1) as f64 / decode_elapsed;
                eprint!(
                    "\r  ⚡ {:.1} tok/s │ {} tokens │ {:.1}s",
                    live_tps,
                    generated_tokens.len(),
                    decode_elapsed,
                );
                let _ = std::io::stderr().flush();
            }
        }

        eprint!("\r{:width$}\r", "", width = 80);
        println!();
        println!("{}", self.session.metrics.summary());

        Ok(tokenizer.decode(&generated_tokens))
    }

    pub fn generate_stream(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
    ) -> mpsc::Receiver<GenerationEvent> {
        let (tx, rx) = mpsc::channel(32);
        let result = self.generate_stream_inner(tokenizer, prompt, max_tokens, streamer, &tx);
        if let Err(e) = result {
            let _ = tx.try_send(GenerationEvent::Error(e.to_string()));
        }
        rx
    }

    pub fn verify_tree(
        &mut self,
        draft_tokens: &[u32],
        mask: Option<&Tensor>,
        all_tokens_len: usize,
        streamer: &WeightStreamer,
    ) -> Result<Tensor> {
        let embedding_table = streamer.load_embedding(&self.device)?;
        let (final_norm_weight, _) = streamer.load_output(&self.device)?;
        
        let tokens_tensor = Tensor::new(draft_tokens, &self.device)?;
        let mut x = embedding_table.index_select(&tokens_tensor, 0)?;

        let lm_head_matrix = self.session.lm_head_tensor.as_ref().cloned().unwrap_or_else(|| {
             Tensor::zeros((self.config.vocab_size, self.config.hidden_dim), DType::F16, &self.device).unwrap()
        });

        let base_pos = all_tokens_len - draft_tokens.len();

        for layer_id in 0..self.config.n_layers {
            let cache = self.session.kv_cache.load(layer_id)?;
            
            let (new_x, next_cache) = self.dispatcher.forward_layer(
                layer_id,
                &x,
                Some(&cache),
                base_pos,
                Some(&self.rope_cache),
                self.session.dual_rope.as_ref(),
                mask,
                Some(&self.session.tp_config),
            )?;

            self.session.kv_cache.save(layer_id, next_cache)?;
            x = new_x;
            streamer.release_layer(layer_id);
        }

        let x = crate::ops::rms_norm(&x, &final_norm_weight, self.config.rms_norm_eps)?;
        x.matmul(&lm_head_matrix.t()?).map_err(anyhow::Error::from)
    }

    pub fn truncate_kv(&mut self, pos: usize) {
        self.session.kv_cache.truncate_to(pos);
    }

    fn generate_stream_inner(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        streamer: &WeightStreamer,
        tx: &mpsc::Sender<GenerationEvent>,
    ) -> Result<()> {
        self.session.reset();
        self.session.metrics.start();

        let embedding_table = streamer.load_embedding(&self.device)?;
        let (final_norm_weight, lm_head) = streamer.load_output(&self.device)?;

        if self.config.arch == crate::model_variant::ModelVariant::Qwen3_6 && self.policy.medusa_heads.is_none() {
            let _ = self.enable_wavefront(4, false, streamer);
        }

        let mut tokens: Vec<u32> = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt));
        self.session.metrics.prompt_tokens = tokens.len();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = tokens.clone();

        let prefill_done = if all_tokens.len() > PREFILL_CHUNK_SIZE {
            self.prefill_chunks(&all_tokens, &embedding_table, &final_norm_weight, &lm_head, streamer)?;
            true
        } else {
            false
        };

        let decode_start = Instant::now();
        for step in 0..max_tokens {
            if self.policy.medusa_heads.is_some() && step > 0 {
                let wavefront_result = self.generate_wavefront(&all_tokens, &embedding_table, &final_norm_weight, &lm_head, streamer)?;
                for next_token in wavefront_result.accepted_tokens {
                    let token_str = tokenizer.decode_token(next_token).to_string();
                    if tx.try_send(GenerationEvent::Token(token_str)).is_err() { break; }
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);
                }
                if wavefront_result.eos_reached { break; }
                continue;
            }

            let next_token = self.generate_step(step, &all_tokens, &embedding_table, &final_norm_weight, &lm_head, Some(streamer), prefill_done)?;

            if step == 0 { self.session.metrics.mark_first_token(); }
            self.session.metrics.record_token();

            if next_token == tokenizer.eos_id { break; }

            let token_str = tokenizer.decode_token(next_token).to_string();
            if tx.try_send(GenerationEvent::Token(token_str.clone())).is_err() { break; }

            if let Some(gbnf) = self.policy.gbnf.as_mut() {
                gbnf.push_token(&token_str);
                if gbnf.is_complete() {
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);
                    break;
                }
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated_tokens.len() >= 1 {
                let live_tps = generated_tokens.len() as f64 / decode_elapsed;
                eprint!("\r  ⚡ {:.2} tok/s │ {} tokens │ {:.1}s", live_tps, generated_tokens.len(), decode_elapsed);
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
        }
        eprint!("\r{:width$}\r", "", width = 80);

        let elapsed = self.session.metrics.elapsed_secs();
        let _ = tx.try_send(GenerationEvent::Done(GenerationMetricsSummary {
            prompt_tokens: self.session.metrics.prompt_tokens,
            generated_tokens: generated_tokens.len(),
            tokens_per_second: if elapsed > 0.0 { generated_tokens.len() as f64 / elapsed } else { 0.0 },
            time_to_first_token_ms: self.session.metrics.ttft_ms(),
            total_time_secs: elapsed,
        }));

        println!("{}", self.session.metrics.summary());
        Ok(())
    }

    fn prefill_chunks(
        &mut self,
        all_tokens: &[u32],
        embedding_table: &Tensor,
        _final_norm_weight: &Tensor,
        _lm_head: &candle_core::quantized::QMatMul,
        streamer: &WeightStreamer,
    ) -> Result<()> {
        let n_full_chunks = all_tokens.len() / PREFILL_CHUNK_SIZE;
        if n_full_chunks == 0 { return Ok(()); }

        println!("Chunked prefill: {} tokens → {} chunks of {}", all_tokens.len(), n_full_chunks, PREFILL_CHUNK_SIZE);

        for chunk_idx in 0..n_full_chunks {
            let chunk_start = chunk_idx * PREFILL_CHUNK_SIZE;
            let chunk_end = chunk_start + PREFILL_CHUNK_SIZE;
            let chunk_tokens = &all_tokens[chunk_start..chunk_end];

            let token_tensor = Tensor::new(chunk_tokens, &self.device)?;
            let mut hidden = embedding_table.index_select(&token_tensor, 0)?;
            hidden = hidden.unsqueeze(0)?;

            for layer_id in 0..self.config.n_layers {
                let cache = self.session.kv_cache.load(layer_id)?;
                let (next_hidden, next_cache) = self.dispatcher.forward_layer(
                    layer_id,
                    &hidden,
                    Some(&cache),
                    chunk_start,
                    Some(&self.rope_cache),
                    self.session.dual_rope.as_ref(),
                    None,
                    Some(&self.session.tp_config)
                )?;
                hidden = next_hidden;
                self.session.kv_cache.save(layer_id, next_cache)?;
            }
            println!("  Chunk {}/{} prefilled ({} tokens at pos {})", chunk_idx + 1, n_full_chunks, PREFILL_CHUNK_SIZE, chunk_start);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_step(
        &mut self,
        step: usize,
        all_tokens: &[u32],
        embedding_table: &Tensor,
        final_norm_weight: &Tensor,
        lm_head: &candle_core::quantized::QMatMul,
        _streamer: Option<&WeightStreamer>,
        prefill_done: bool,
    ) -> Result<u32> {
        let (input_tokens, start_pos) = if step == 0 && !prefill_done {
            (all_tokens, 0)
        } else if step == 0 && prefill_done {
            let remaining_start = (all_tokens.len() / PREFILL_CHUNK_SIZE) * PREFILL_CHUNK_SIZE;
            if remaining_start < all_tokens.len() {
                (&all_tokens[remaining_start..], remaining_start)
            } else {
                (&all_tokens[all_tokens.len() - 1..], all_tokens.len() - 1)
            }
        } else {
            (&all_tokens[all_tokens.len() - 1..], all_tokens.len() - 1)
        };

        let token_tensor = Tensor::new(input_tokens, &self.device)?;
        let mut hidden = embedding_table.index_select(&token_tensor, 0)?;
        hidden = hidden.unsqueeze(0)?;

        let layer_loop_start = Instant::now();
        for layer_id in 0..self.config.n_layers {
            let cache = self.session.kv_cache.load(layer_id)?;
            let (next_hidden, next_cache) = self.dispatcher.forward_layer(
                layer_id, 
                &hidden, 
                Some(&cache), 
                start_pos, 
                Some(&self.rope_cache), 
                self.session.dual_rope.as_ref(), 
                None, 
                Some(&self.session.tp_config)
            )?;
            hidden = next_hidden;
            self.session.kv_cache.save(layer_id, next_cache)?;
        }

        if step == 0 {
            let layer_total = layer_loop_start.elapsed();
            eprintln!("\n  ✓ Prefill: {} layers in {:.1}ms ({:.1}ms/layer avg)", self.config.n_layers, layer_total.as_secs_f64() * 1000.0, layer_total.as_secs_f64() * 1000.0 / self.config.n_layers as f64);
        }

        hidden = ops::rms_norm(&hidden, final_norm_weight, self.config.rms_norm_eps)?;
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        self.session.metrics.last_hidden = Some(last_hidden.clone());

        let logits = lm_head.forward(&last_hidden)?;
        let logits = logits.squeeze(0)?;

        self.policy.sampler.sample_constrained(&logits, all_tokens, self.policy.gbnf.as_ref()).map_err(|e| anyhow::anyhow!("Sampling failed: {e}"))
    }

    pub fn reset(&mut self) { self.session.reset(); }
    pub fn reset_kv_cache(&mut self) { self.reset(); }
    pub fn metrics(&self) -> &InferenceMetrics { &self.session.metrics }
    pub fn device(&self) -> &candle_core::Device { &self.device }

    pub(crate) fn generate_wavefront(
        &mut self,
        all_tokens: &[u32],
        embedding_table: &Tensor,
        final_norm_weight: &Tensor,
        _lm_head: &candle_core::quantized::QMatMul,
        streamer: &WeightStreamer,
    ) -> Result<WavefrontResult> {
        let mut stats = crate::wavefront::CycleStats::default();
        let start = Instant::now();

        let last_hidden = self.session.metrics.last_hidden.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Medusa draft requires previous hidden state")
        })?;

        let m_heads = self.policy.medusa_heads.as_ref().unwrap();
        let lm_head_matrix = self.session.lm_head_tensor.as_ref().cloned().unwrap_or_else(|| {
            Tensor::zeros((self.config.vocab_size, self.config.hidden_dim), DType::F16, &self.device).unwrap()
        });
        
        let draft_start = Instant::now();
        let bundle = m_heads.draft(last_hidden, &lm_head_matrix, all_tokens)?;
        stats.draft_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        stats.tokens_drafted = bundle.n_heads;

        let mut active_indices = Vec::new();
        if let Some(bank) = &self.policy.sparsity_bank {
            for layer_id in 0..self.config.n_layers {
                let mut layer_preds = Vec::new();
                for h in &bundle.hiddens {
                    let mask = bank.predict_mask(h, layer_id)?;
                    layer_preds.push(mask.active_indices);
                }
                let union_mask = crate::wavefront::compute_union_mask(&layer_preds, self.config.intermediate_dim);
                active_indices.push(union_mask);
            }
        }
        
        let io_start = Instant::now();
        if !active_indices.is_empty() {
            stats.sparsity_density = active_indices.iter().map(|m| m.density).sum::<f32>() / active_indices.len() as f32;
        }
        stats.io_ms = io_start.elapsed().as_secs_f64() * 1000.0;
        stats.bytes_saved = (33e9 * (1.0 - stats.sparsity_density)) as usize;

        let verify_start = Instant::now();
        let start_pos = all_tokens.len();
        let draft_ids: Vec<u32> = bundle.tokens.iter().map(|&t| t as u32).collect();

        let logits = self.verify_tree(&draft_ids, None, start_pos + draft_ids.len(), streamer)?;

        let mut accepted_tokens = Vec::new();
        let mut eos_reached = false;
        
        let target_tokens = logits.argmax(candle_core::D::Minus1)?;
        let target_ids: Vec<u32> = target_tokens.to_vec1()?;

        for (i, &draft_token) in draft_ids.iter().enumerate() {
            let target_pred = target_ids[i];
            if draft_token == target_pred {
                accepted_tokens.push(draft_token);
                if draft_token == self.config.eos_token_id { eos_reached = true; break; }
            } else {
                accepted_tokens.push(target_pred);
                if target_pred == self.config.eos_token_id { eos_reached = true; }
                break;
            }
        }

        let accepted_count = accepted_tokens.len();
        self.truncate_kv(start_pos + accepted_count);

        stats.verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        stats.tokens_accepted = accepted_count;

        self.policy.wavefront_health.update(accepted_count, bundle.n_heads);
        self.session.wavefront_session.record(&stats);

        if self.config.arch == crate::model_variant::ModelVariant::Qwen2 {
             eprintln!("\r  {}", stats.display());
        }

        Ok(WavefrontResult { accepted_tokens, eos_reached })
    }
}

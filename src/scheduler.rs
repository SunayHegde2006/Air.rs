use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use crate::dispatcher::{GenerateConfig, TokenChunk, FinishReason, Dispatcher};
use crate::generator::InferenceGenerator;
use crate::tokenizer::Tokenizer;
use crate::weight_streamer::WeightStreamer;
use futures_util::stream::BoxStream;
use anyhow::Result;

/// A high-leverage orchestrator for multiple concurrent inference requests.
/// Hides the complexity of token queuing, speculative selection, and GBNF enforcement.
pub struct RequestOrchestrator {
    model_name: String,
    config: Arc<crate::model::ModelConfig>,
    request_tx: mpsc::Sender<SchedulerMessage>,
}

enum SchedulerMessage {
    Generate {
        config: GenerateConfig,
        response_tx: mpsc::Sender<Result<TokenChunk>>,
    },
    Shutdown,
}

impl RequestOrchestrator {
    pub fn new(
        model_name: String,
        mut generator: InferenceGenerator,
        tokenizer: Tokenizer,
        streamer: Arc<WeightStreamer>,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::channel::<SchedulerMessage>(32);
        let config = Arc::new(generator.config.clone());

        // The background "tick" loop for the engine
        tokio::spawn(async move {
            let mut draft_models = std::collections::HashMap::new();

            while let Some(msg) = request_rx.recv().await {
                match msg {
                    SchedulerMessage::Generate { config: req_cfg, response_tx } => {
                        // 1. Prepare grammar
                        if let Some(g) = &req_cfg.gbnf {
                             let constraint = crate::gbnf::GbnfConstraint::from_str(g, tokenizer.vocab_vec().clone()).unwrap();
                             generator.set_grammar(constraint);
                        } else {
                            generator.clear_grammar();
                        }

                        // 2. Prepare Speculative Decoder if needed
                        if let Some(draft_path) = &req_cfg.draft_model {
                            let (draft_gen, draft_streamer) = draft_models.entry(draft_path.clone()).or_insert_with(|| {
                                let ds = Arc::new(WeightStreamer::open(draft_path).unwrap());
                                let metadata = crate::loader::GgufLoader::extract_metadata(ds.content());
                                let cfg = crate::model::ModelConfig::from_gguf_metadata(&metadata);
                                let g = InferenceGenerator::new(cfg, crate::sampler::SamplerConfig::default()).unwrap();
                                (g, ds)
                            });

                            let gbnf_ref = generator.policy.gbnf.clone();
                            let mut spec = crate::speculative::SpeculativeDecoder::new(
                                &mut generator,
                                draft_gen,
                                crate::speculative::SpeculativeConfig::default()
                            ).unwrap();

                            let result = spec.generate(&tokenizer, &req_cfg.prompt, req_cfg.max_tokens, &streamer, draft_streamer, gbnf_ref.as_ref());
                            match result {
                                Ok(text) => {
                                    let _ = response_tx.send(Ok(TokenChunk::Token { id: 0, text })).await;
                                    let _ = response_tx.send(Ok(TokenChunk::Stop { finish_reason: FinishReason::Stop })).await;
                                }
                                Err(e) => {
                                    let _ = response_tx.send(Ok(TokenChunk::Stop { finish_reason: FinishReason::Error(e.to_string()) })).await;
                                }
                            }
                        } else {
                            // 3. Standard generation stream
                            let mut rx = generator.generate_stream(&tokenizer, &req_cfg.prompt, req_cfg.max_tokens, &streamer);
                            while let Some(event) = rx.recv().await {
                                match event {
                                    crate::generator::GenerationEvent::Token(text) => {
                                        if response_tx.send(Ok(TokenChunk::Token { id: 0, text })).await.is_err() { break; }
                                    }
                                    crate::generator::GenerationEvent::Done(_) => {
                                        let _ = response_tx.send(Ok(TokenChunk::Stop { finish_reason: FinishReason::Stop })).await;
                                        break;
                                    }
                                    crate::generator::GenerationEvent::Error(m) => {
                                        let _ = response_tx.send(Ok(TokenChunk::Stop { finish_reason: FinishReason::Error(m) })).await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    SchedulerMessage::Shutdown => break,
                }
            }
        });

        Self { model_name, config, request_tx }
    }
}

impl Dispatcher for RequestOrchestrator {
    fn generate(&self, config: GenerateConfig) -> BoxStream<'static, Result<TokenChunk>> {
        let (tx, rx) = mpsc::channel(32);
        let request_tx = self.request_tx.clone();
        
        tokio::spawn(async move {
            let msg = SchedulerMessage::Generate { config, response_tx: tx };
            if let Err(e) = request_tx.send(msg).await {
                eprintln!("Scheduler error: {}", e);
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    fn list_models(&self) -> Vec<String> {
        vec![self.model_name.clone()]
    }
}

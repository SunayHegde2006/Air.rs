use axum::{Json, response::IntoResponse};
use serde::{Deserialize, Serialize};
use air_engine::pipeline::ExecutionEngine;
use air_engine::sampler::SamplingParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

pub async fn chat_completions_handler(
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let mut engine = ExecutionEngine::new(32000, 1024);
    let mut history = vec![1u32];
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        ..Default::default()
    };

    let max_toks = req.max_tokens.unwrap_or(16).min(128);
    let mut content = String::from("Response: ");

    for _ in 0..max_toks {
        if let Ok(tok) = engine.step(&mut history, &params) {
            content.push_str(&format!("token_{} ", tok));
        }
    }

    let response = ChatCompletionResponse {
        id: "chatcmpl-air-2.0.0".to_string(),
        object: "chat.completion".to_string(),
        created: 1770000000,
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Json(response)
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

pub async fn list_models_handler() -> impl IntoResponse {
    let response = ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: "air-model".to_string(),
            object: "model".to_string(),
            created: 1770000000,
            owned_by: "air-rs".to_string(),
        }],
    };

    Json(response)
}

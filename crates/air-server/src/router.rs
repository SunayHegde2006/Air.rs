use crate::handlers::{chat_completions_handler, list_models_handler};
use axum::{Router, routing::{get, post}};

pub fn create_router() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/models", get(list_models_handler))
}

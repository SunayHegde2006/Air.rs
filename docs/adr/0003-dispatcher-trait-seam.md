# ADR-0003: `Dispatcher` trait seam — decouple HTTP handlers from inference engine

**Date:** 2026-05-03
**Status:** Accepted

## Context

`api.rs` (877 lines) mixes three concerns in one file:
1. HTTP request/response parsing (Axum handlers, serde types)
2. Model lifecycle and metadata registry (`ApiState`, `ModelEntry`)
3. Inference routing — currently a stub comment: _"Simulate inference — in production this calls InferenceGenerator"_

`ApiState` holds only `model_name: String` — it cannot route to multiple models. Adding `ModelMux` (issue #4) would require deep surgery inside the handler code. HTTP handlers are untestable without spinning up a full Axum server.

## Decision

### Introduce a `Dispatcher` trait

```rust
pub trait Dispatcher: Send + Sync {
    fn dispatch(
        &self,
        model_id: &str,
        prompt: &str,
        config: GenerateConfig,
    ) -> BoxStream<'static, TokenChunk>;
}

pub enum TokenChunk {
    Token(String),
    Done(FinishReason),
}

pub enum FinishReason { Stop, Length, Error(String) }
```

`dispatch` always returns a stream. Non-streaming handlers collect it: `stream.collect::<String>()`. The `Done(FinishReason)` variant signals the final chunk — handlers use it to emit `finish_reason` in the OpenAI wire format.

### `ApiState` holds `Arc<dyn Dispatcher>`

```rust
pub struct ApiState {
    pub dispatcher: Arc<dyn Dispatcher>,
    pub models: RwLock<Vec<ModelEntry>>,
}
```

Handlers become thin:
```rust
async fn chat_completions(State(state): State<Arc<ApiState>>, ...) {
    let stream = state.dispatcher.dispatch(&req.model, &prompt, config);
    // convert TokenChunk → SSE frames
}
```

### Three adapters

| Adapter | Status | Wraps |
|---|---|---|
| `SingleModelDispatcher` | v0.2.0 (current) | one `InferenceGenerator` |
| `ModelMuxDispatcher` | v0.3.0 (issue #4) | `ModelMux` — routes by `model_id` |
| `MockDispatcher` | test-only (`#[cfg(test)]`) | returns canned `Vec<TokenChunk>` |

`create_router_with_model(name)` → updated to accept `Arc<dyn Dispatcher>`.

### Invariant preserved by the enum

`TokenChunk` is a tagged enum — it's a compile-time invariant that `Done` cannot carry token text and `Token` cannot carry a finish reason. The handler `match` arm at the HTTP boundary is the only place OpenAI wire format is applied.

## Consequences

**Positive:**
- HTTP handlers testable with `MockDispatcher` — no Axum test server, no GGUF, no GPU
- Swapping single-model for `ModelMux` is one callsite change (`Arc<dyn Dispatcher>` construction), zero handler changes
- `api.rs` responsibility narrows to HTTP parsing + SSE serialisation; inference routing is behind the seam
- `TokenChunk` enum prevents the "partial Done" bug class at compile time

**Negative:**
- `BoxStream` requires `futures::StreamExt` / `tokio_stream` — adds one dependency if not already present
- Existing `create_router_with_model(name)` signature changes — callers must now pass a `Dispatcher`, not just a model name string

## Alternatives rejected

**`ApiState` holds `ModelMux` directly:** Couples the HTTP layer to a specific inference runtime. Can't test handlers without a live engine. Rejected — fails the seam test (one adapter is hypothetical, not real).

**Two trait methods (`generate` + `stream`):** Doubles the interface width for no gain — non-streaming is always `collect()`. Rejected — violates the deep-module principle.

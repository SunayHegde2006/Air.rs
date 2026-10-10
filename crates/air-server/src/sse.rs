use axum::response::sse::{Event, Sse};
use futures_util::stream::{self, Stream};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::StreamExt;

pub fn stream_tokens(
    tokens: Vec<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::iter(tokens).map(|token| {
        let json = format!(
            r#"{{"choices": [{{"delta": {{"content": "{}"}}, "finish_reason": null}}]}}"#,
            token
        );
        Ok(Event::default().data(json))
    });

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(1)))
}

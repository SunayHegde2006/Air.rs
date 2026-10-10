#![allow(clippy::all, dead_code, unused_imports)]

pub mod handlers;
pub mod router;
pub mod sse;

pub use handlers::*;
pub use router::*;
pub use sse::*;

use air_core::error::{AirError, Result};
use std::net::SocketAddr;

pub async fn start_server(port: u16) -> Result<()> {
    let app = create_router();
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Air.rs server listening on {}", addr);
    axum::serve(listener, app)
        .await
        .map_err(|e| AirError::Server(e.to_string()))
}

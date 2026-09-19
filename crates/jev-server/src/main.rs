//! jev-server entrypoint.
//!
//! Configuration is environment-only (no config file in M0):
//! * `JEV_BASE_URL`   — OpenAI-compatible base, e.g. `http://10.10.10.115:8014/v1`
//! * `JEV_MODEL`      — model id the server knows, e.g. `qwen3.8-27b`
//! * `JEV_API_KEY`    — optional (default `EMPTY`)
//! * `JEV_LISTEN`     — optional (default `127.0.0.1:8080`)
//! * `JEV_TOKENIZER`  — optional path to `tokenizer.json` for slot verification

mod api;

use api::{router, AppState};
use jev_backend::{OpenAiCompatBackend, OpenAiCompatConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let base_url = std::env::var("JEV_BASE_URL")
        .unwrap_or_else(|_| "http://10.10.10.115:8014/v1".to_string());
    let model = std::env::var("JEV_MODEL").unwrap_or_else(|_| "qwen3.8-27b".to_string());
    let api_key = std::env::var("JEV_API_KEY").unwrap_or_else(|_| "EMPTY".to_string());
    let listen = std::env::var("JEV_LISTEN").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let backend = OpenAiCompatBackend::new(OpenAiCompatConfig {
        base_url,
        model,
        api_key,
        timeout_secs: 120,
    })?;

    let app = router(AppState { backend: Arc::new(backend), max_prompt_tokens: 32_000 });
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("jev-server listening on {listen}");
    axum::serve(listener, app).await?;
    Ok(())
}

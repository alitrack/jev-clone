//! jev-server entrypoint.
//!
//! Configuration is environment-only (no config file in M0):
//! * `JEV_BASE_URL`   — OpenAI-compatible base, e.g. `http://10.10.10.115:8014/v1`
//! * `JEV_MODEL`      — model id the server knows, e.g. `qwen3.8-27b`
//! * `JEV_API_KEY`    — optional (default `EMPTY`)
//! * `JEV_LISTEN`     — optional (default `127.0.0.1:8080`)
//! * `JEV_TOKENIZER`  — reserved, **not read in M0**: slot verification always goes
//!   through the endpoint's `/tokenize`; a local `tokenizer.json` path is M1 work
//!   (specs/M0.md §4 B3, "fallback path if time allows")
//!
//! Startup is fail-fast on purpose: before the port is bound, the endpoint's
//! `/tokenize` is asked to confirm the *probed* slot assumption
//! (`"Answer:\n"` -> `[15666, 25, 198]`, appending `A` -> exactly one more token,
//! id `32`). If that does not hold, the answer position we read log-probabilities
//! from is not the position we think it is, so the process refuses to start rather
//! than answer with confident numbers about the wrong token.

use jev_backend::tokenizer::{PROBED_PROMPT, PROBED_PROMPT_IDS, PROBED_SLOT_ID, PROBED_SLOT_LETTER};
use jev_backend::{HttpTokenizer, OpenAiCompatBackend, OpenAiCompatConfig};
use jev_server::api::{router, AppState};
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
        base_url: base_url.clone(),
        model: model.clone(),
        api_key: api_key.clone(),
        timeout_secs: 120,
    })?;

    // Slot verification needs the tokenizer BEFORE the first request: the
    // /tokenize endpoint is a hard prerequisite, so check it now and fail fast
    // with a clear message instead of at request time.
    let tokenizer = Arc::new(HttpTokenizer::new(base_url.clone(), model.clone(), api_key.clone()));
    tokenizer.verify_slot_assumption().await.map_err(|e| {
        anyhow::anyhow!(
            "slot self-check failed against JEV_BASE_URL {base_url} (JEV_MODEL {model}): {e}. \
             M0 requires /tokenize to tokenize {PROBED_PROMPT:?} as {PROBED_PROMPT_IDS:?} and \
             {PROBED_SLOT_LETTER:?} as the single token [{PROBED_SLOT_ID}]; an endpoint that \
             does not cannot read the answer slot, so jev-server refuses to start."
        )
    })?;
    tracing::info!(
        "slot self-check passed: {PROBED_PROMPT:?} = {PROBED_PROMPT_IDS:?}, \
         {PROBED_SLOT_LETTER:?} = [{PROBED_SLOT_ID}]"
    );

    let app = router(AppState {
        backend: Arc::new(backend),
        tokenizer,
        max_prompt_tokens: 32_000,
    });
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("jev-server listening on {listen}");
    axum::serve(listener, app).await?;
    Ok(())
}

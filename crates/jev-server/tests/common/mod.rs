//! Shared scaffolding for the `jev-server` end-to-end tests.
//!
//! [`AppState`](jev_server::api::AppState) holds a concrete `HttpTokenizer`, so the
//! "fake tokenizer" the tests need is a *stub `/tokenize` server* on an ephemeral
//! localhost port rather than a hand-written `SlotVerifier` implementation.
//!
//! The stub tokenizes one token per UTF-8 byte. That is not a real tokenizer, and
//! the tests do not pretend otherwise: what it reproduces is exactly the structural
//! property `verify_slots` asserts — `encode("A")` is a single token (`[65]`) and
//! `encode(prompt + "A") == encode(prompt) ++ [65]` — which is what makes it a valid
//! stand-in for the slot algorithm. Model behaviour comes from `MockBackend`, whose
//! readouts are scripted, so these tests verify the *assembly* of answers, never
//! model quality.
#![allow(dead_code)]

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Records every tokenize request and answers with byte-per-token ids.
///
/// Like the `jev-backend` stub, it **enforces the live endpoint's field name**:
/// `prompt` is the only accepted field, anything else gets the real 400. A stub
/// that accepts any body would sign off on a client speaking the wrong dialect —
/// that is exactly how a wrong field name once passed every test in this suite and
/// only failed against 115:8014.
#[derive(Clone, Default)]
pub struct TokenizeStub {
    seen: Arc<Mutex<Vec<String>>>,
}

impl TokenizeStub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every `text` the endpoint was asked to tokenize, in order.
    pub fn texts(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

fn ids_for(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

async fn tokenize(
    axum::extract::State(stub): axum::extract::State<TokenizeStub>,
    Json(body): Json<Value>,
) -> Response {
    let text = match body.get("prompt").and_then(|t| t.as_str()) {
        Some(text) => text.to_string(),
        None => {
            stub.seen
                .lock()
                .unwrap()
                .push(format!("<rejected: {body}>"));
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "application/json")],
                json!({
                    "object": "error",
                    "message": "1 validation error: Value error, Exactly one of \
                                'prompt' or 'messages' must be provided.",
                    "type": "BadRequest",
                    "code": 400,
                })
                .to_string(),
            )
                .into_response();
        }
    };
    stub.seen.lock().unwrap().push(text.clone());
    let ids = ids_for(&text);
    Json(json!({
        "tokens": ids,
        "count": ids.len(),
        "max_model_len": 262_144,
    }))
    .into_response()
}

/// Bind the stub on an ephemeral localhost port; returns its base URL.
pub async fn spawn_tokenize_stub(stub: TokenizeStub) -> String {
    let app = Router::new().route("/tokenize", post(tokenize)).with_state(stub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

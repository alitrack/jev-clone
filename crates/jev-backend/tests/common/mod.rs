//! Shared scaffolding for the `jev-backend` integration tests: tiny axum stub
//! servers on ephemeral localhost ports.
//!
//! Hand-written stubs, not `wiremock`: `wiremock` is absent from this machine's
//! offline cargo cache (`docs/dev-env.md` §1 — all cargo commands run `--offline`),
//! and both endpoints under test are a few lines of axum. Nothing in here leaves
//! `127.0.0.1`; no GPU, no external service.
//!
//! The stubs are *shaped* like the probed endpoints (specs/M0.md §1), but they are
//! not model servers: they let the tests pin down parsing, error mapping and the
//! request body, which is all that can be checked without a GPU.
#![allow(dead_code)]

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// POST /v1/completions stub
// ---------------------------------------------------------------------------

/// Answers `POST /v1/completions` with a fixed status + body and records every
/// request body it received.
#[derive(Clone)]
pub struct CompletionsStub {
    status: u16,
    body: String,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl CompletionsStub {
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Every request body the stub received, in order.
    pub fn requests(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }

    /// The single request body the stub received (panics when it saw != 1).
    pub fn only_request(&self) -> Value {
        let seen = self.seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "expected exactly one request, got {seen:?}");
        seen[0].clone()
    }
}

async fn completions(State(stub): State<CompletionsStub>, Json(body): Json<Value>) -> Response {
    stub.seen.lock().unwrap().push(body);
    (
        StatusCode::from_u16(stub.status).expect("stub status is a valid HTTP status"),
        [(header::CONTENT_TYPE, "application/json")],
        stub.body.clone(),
    )
        .into_response()
}

/// Bind a stub on an ephemeral localhost port; returns the OpenAI-compatible
/// base URL (`http://127.0.0.1:<port>/v1`) the backend should be configured with.
///
/// The listener is bound before returning, so the first client request can never
/// race the server startup.
pub async fn spawn_completions_stub(stub: CompletionsStub) -> String {
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(stub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/v1")
}

// ---------------------------------------------------------------------------
// POST /tokenize (+ /v1/tokenize) stub
// ---------------------------------------------------------------------------

/// Answers `POST /tokenize` and `POST /v1/tokenize` with a *structural* fake
/// tokenizer: one token per UTF-8 byte.
///
/// That is not a model tokenizer, but it reproduces exactly the property the slot
/// protocol needs, which is what the verifier asserts: `encode("A")` is one token
/// (`[65]`), and `encode(prompt + "A") == encode(prompt) ++ [65]`. Text-only
/// properties are honoured; no real vocabulary is involved.
///
/// **The stub enforces the real endpoint's request contract**, including its
/// field name: it reads `prompt` and answers 400 for anything else, exactly as
/// SGLang does (probed live: `{"model":..,"text":"Answer:\n"}` -> 400 "Exactly one
/// of 'prompt' or 'messages' must be provided"). A stub that accepts whatever the
/// client sends would happily agree with a client that speaks the wrong dialect —
/// which is how a wrong field name once survived the whole test suite and only
/// showed up against the real endpoint.
///
/// Per-text `overrides` let a test script specific encodings (e.g. the probed
/// `"Answer:\n" -> [15666, 25, 198]`).
#[derive(Clone, Default)]
pub struct TokenizeStub {
    overrides: HashMap<String, Vec<u32>>,
    fail_primary: bool,
    fail_fallback: bool,
    seen: Arc<Mutex<Vec<(String, String)>>>,
}

impl TokenizeStub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Force a specific encoding for `text`.
    pub fn with_override(mut self, text: impl Into<String>, ids: Vec<u32>) -> Self {
        self.overrides.insert(text.into(), ids);
        self
    }

    /// Make `POST /tokenize` (the primary path) answer 500.
    pub fn failing_primary(mut self) -> Self {
        self.fail_primary = true;
        self
    }

    /// Make `POST /v1/tokenize` (the fallback path) answer 500 too.
    pub fn failing_fallback(mut self) -> Self {
        self.fail_fallback = true;
        self
    }

    /// `(path, text)` for every received request, in order.
    pub fn requests(&self) -> Vec<(String, String)> {
        self.seen.lock().unwrap().clone()
    }

    pub fn request_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

fn fake_ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

async fn tokenize_one(stub: &TokenizeStub, path: &str, body: Value, fail: bool) -> Response {
    // Faithful to the live endpoint: `prompt` is the only accepted field.
    let text = match body.get("prompt").and_then(|t| t.as_str()) {
        Some(text) => text.to_string(),
        None => {
            stub.seen
                .lock()
                .unwrap()
                .push((path.to_string(), format!("<rejected: {body}>")));
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
    stub.seen
        .lock()
        .unwrap()
        .push((path.to_string(), text.clone()));
    if fail {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("stub {path} is down"),
        )
            .into_response();
    }
    let ids = stub
        .overrides
        .get(&text)
        .cloned()
        .unwrap_or_else(|| fake_ids(&text));
    // Field names mirror the live response verbatim (`max_model_len`, not
    // `max_token_len` as an earlier probe note claimed — re-probed 2026-09-19).
    Json(json!({
        "tokens": ids,
        "count": ids.len(),
        "max_model_len": 262_144,
    }))
    .into_response()
}

async fn tokenize(State(stub): State<TokenizeStub>, Json(body): Json<Value>) -> Response {
    let fail = stub.fail_primary;
    tokenize_one(&stub, "/tokenize", body, fail).await
}

async fn tokenize_v1(State(stub): State<TokenizeStub>, Json(body): Json<Value>) -> Response {
    let fail = stub.fail_fallback;
    tokenize_one(&stub, "/v1/tokenize", body, fail).await
}

/// Bind the tokenize stub on an ephemeral localhost port; returns the base URL
/// (`http://127.0.0.1:<port>`) the tokenizer should be configured with.
pub async fn spawn_tokenize_stub(stub: TokenizeStub) -> String {
    let app = Router::new()
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize_v1))
        .with_state(stub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

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
// Which dialect the tokenize stub speaks
// ---------------------------------------------------------------------------

/// The two live endpoints disagree on the request field name, and the failure
/// mode differs in kind: the reference endpoint *rejects* the wrong field
/// loudly (400), while llama.cpp ignores it and answers an empty tokenization
/// with a 200. The stub has to be able to reproduce both, or the client's
/// detection of the silent one cannot be tested at all.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Dialect {
    /// The reference endpoint (SGLang): `prompt` only, 400 for anything else.
    #[default]
    Reference,
    /// llama.cpp: the field is `content`; a `prompt`-only body gets
    /// `200 {"tokens": []}` — the silently-empty shape.
    LlamaCpp,
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
/// `"Answer:\n" -> [15666, 25, 198]`). Which request fields this endpoint accepts
/// is [`Dialect`]; the default is the reference dialect.
#[derive(Clone, Default)]
pub struct TokenizeStub {
    overrides: HashMap<String, Vec<u32>>,
    fail_primary: bool,
    fail_fallback: bool,
    /// Which request field this endpoint accepts (see [`Dialect`]).
    dialect: Dialect,
    seen: Arc<Mutex<Vec<(String, String)>>>,
    /// The body field each request used, in order (`prompt` / `content`).
    fields: Arc<Mutex<Vec<String>>>,
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

    /// Speak llama.cpp's dialect: `content` is the field, `prompt` is ignored
    /// with a 200 and an empty `tokens` array.
    pub fn llama_cpp_dialect(mut self) -> Self {
        self.dialect = Dialect::LlamaCpp;
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

    /// The body field each request used, in order.
    pub fn fields(&self) -> Vec<String> {
        self.fields.lock().unwrap().clone()
    }

    pub fn request_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

fn fake_ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// What the stub made of one request body.
enum FieldRead {
    /// Understood: the field name it used and the text it carries.
    Text(&'static str, String),
    /// Present, but this dialect ignores it — answer 200 with an *empty*
    /// `tokens` array, which is exactly what llama.cpp answers for `prompt`.
    Ignored(&'static str),
    /// Nothing usable — answer 400, as the reference endpoint does.
    Rejected,
}

fn read_field(body: &Value, dialect: Dialect) -> FieldRead {
    match dialect {
        Dialect::Reference => match body.get("prompt").and_then(|t| t.as_str()) {
            Some(text) => FieldRead::Text("prompt", text.to_string()),
            None => FieldRead::Rejected,
        },
        Dialect::LlamaCpp => {
            if let Some(text) = body.get("content").and_then(|t| t.as_str()) {
                FieldRead::Text("content", text.to_string())
            } else if body.get("prompt").is_some() {
                FieldRead::Ignored("prompt")
            } else {
                FieldRead::Rejected
            }
        }
    }
}

async fn tokenize_one(stub: &TokenizeStub, path: &str, body: Value, fail: bool) -> Response {
    let (field, text) = match read_field(&body, stub.dialect) {
        FieldRead::Text(field, text) => (field, text),
        FieldRead::Ignored(field) => {
            stub.seen
                .lock()
                .unwrap()
                .push((path.to_string(), format!("<ignored field {field}>")));
            stub.fields.lock().unwrap().push(field.to_string());
            // A 200 with nothing in it, not an error: the client has to notice by
            // itself that a non-empty text produced no tokens.
            return Json(json!({
                "tokens": [],
                "count": 0,
                "max_model_len": 262_144,
            }))
            .into_response();
        }
        FieldRead::Rejected => {
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
    stub.fields.lock().unwrap().push(field.to_string());
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

// ---------------------------------------------------------------------------
// POST /v1/completions stub that rejects an array `prompt` (llama.cpp's dialect)
// ---------------------------------------------------------------------------

/// Answers `POST /v1/completions` the way llama.cpp's server does: a single
/// string `prompt` is answered normally, an **array** `prompt` is rejected with
/// `400 {"error":{"message":"type must be string, but is an array"}}`.
///
/// Records every request body, so a test can assert the shape of a fallback: one
/// rejected array request, then one single-prompt request per prompt, in order.
#[derive(Clone)]
pub struct ArrayRejectingCompletionsStub {
    /// `usage.prompt_tokens` reported for each single-prompt answer.
    prompt_tokens: u64,
    seed: Arc<Mutex<u64>>,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl ArrayRejectingCompletionsStub {
    pub fn new(prompt_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            seed: Arc::new(Mutex::new(0)),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Every request body the stub received, in order.
    pub fn requests(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }
}

async fn array_rejecting_completions(
    State(stub): State<ArrayRejectingCompletionsStub>,
    Json(body): Json<Value>,
) -> Response {
    stub.seen.lock().unwrap().push(body.clone());
    match body.get("prompt") {
        Some(Value::String(_)) => {
            // Distinct per-call logprob so a test can prove the readouts stayed in
            // prompt order instead of collapsing to one value.
            let lp = {
                let mut seed = stub.seed.lock().unwrap();
                let lp = -0.1 * (*seed as f64 + 1.0);
                *seed += 1;
                lp
            };
            let prompt_tokens = stub.prompt_tokens;
            Json(json!({
                "choices": [{
                    "text": "A",
                    "index": 0,
                    "logprobs": { "top_logprobs": [ { "A": lp, "B": -2.0 } ] }
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 1,
                    "total_tokens": prompt_tokens + 1
                }
            }))
            .into_response()
        }
        Some(Value::Array(_)) => (
            StatusCode::BAD_REQUEST,
            [(header::CONTENT_TYPE, "application/json")],
            json!({
                "error": {
                    "message": "type must be string, but is an array",
                    "type": "invalid_request_error",
                    "code": 400
                }
            })
            .to_string(),
        )
            .into_response(),
        _ => (
            StatusCode::BAD_REQUEST,
            [(header::CONTENT_TYPE, "text/plain")],
            "prompt must be a string".to_string(),
        )
            .into_response(),
    }
}

/// Bind the array-rejecting completions stub on an ephemeral localhost port;
/// returns the OpenAI-compatible base URL (`http://127.0.0.1:<port>/v1`).
pub async fn spawn_array_rejecting_completions_stub(stub: ArrayRejectingCompletionsStub) -> String {
    let app = Router::new()
        .route("/v1/completions", post(array_rejecting_completions))
        .with_state(stub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/v1")
}

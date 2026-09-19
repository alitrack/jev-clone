//! No-GPU end-to-end tests for `POST /v1/systemone` (specs/M0.md §4 B5.3).
//!
//! The app assembled here is byte-for-byte the app `main.rs` serves: the same
//! `router`, the same `AppState`, the same `HttpTokenizer`. Only the two things a
//! GPU is needed for are substituted — the model readout (`MockBackend`, scripted)
//! and the tokenizer endpoint (a local stub that reproduces the slot properties
//! `verify_slots` asserts). Everything between the HTTP request and the JSON
//! response is the production code path, driven through
//! `tower::ServiceExt::oneshot` (no port, no network beyond localhost).
//!
//! What these tests therefore prove: answer *assembly* — argmax/label mapping,
//! legend text, rounding, usage totals, independence between questions, and the
//! error contract. What they cannot prove: anything about model quality.

mod common;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use common::{spawn_tokenize_stub, TokenizeStub};
use jev_backend::{BackendError, DecisionBackend, HttpTokenizer, MockBackend, Readout};
use jev_core::{confidence_from_probabilities, score_expectation, softmax, ConfidenceMode};
use jev_server::api::{router, AppState};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

const MAX_PROMPT_TOKENS: usize = 32_000;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn readout(pairs: &[(&str, f64)], prompt_tokens: u64) -> Readout {
    Readout {
        top: pairs
            .iter()
            .map(|(t, l)| ((*t).to_string(), *l))
            .collect::<BTreeMap<_, _>>(),
        prompt_tokens,
        completion_tokens: 1,
    }
}

/// A scripted backend (one readout per question, in question order) that also
/// records `(prompt, top_k)` for every call the API layer makes.
struct RecordingBackend {
    model: String,
    readouts: Vec<Readout>,
    cursor: AtomicUsize,
    calls: Mutex<Vec<(String, usize)>>,
}

impl RecordingBackend {
    fn calls(&self) -> Vec<(String, usize)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl DecisionBackend for RecordingBackend {
    async fn token_logprobs(&self, prompt: &str, top_k: usize) -> Result<Readout, BackendError> {
        self.calls.lock().unwrap().push((prompt.to_string(), top_k));
        let i = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.readouts
            .get(i)
            .cloned()
            .ok_or_else(|| BackendError::Http(format!("scripted backend exhausted at call {i}")))
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}

/// A recording backend plus the `dyn` handle the app takes.
fn backend(model: &str, readouts: Vec<Readout>) -> (Arc<RecordingBackend>, Arc<dyn DecisionBackend>) {
    let rec = Arc::new(RecordingBackend {
        model: model.to_string(),
        readouts,
        cursor: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    });
    let handle: Arc<dyn DecisionBackend> = rec.clone();
    (rec, handle)
}

fn mock(model: &str, readouts: Vec<Readout>) -> Arc<dyn DecisionBackend> {
    Arc::new(MockBackend::new(model, readouts))
}

async fn app_with(
    backend: Arc<dyn DecisionBackend>,
    stub: TokenizeStub,
    max_prompt_tokens: usize,
) -> Router {
    let base = spawn_tokenize_stub(stub).await;
    let tokenizer = Arc::new(HttpTokenizer::new(base, "test-model", "EMPTY"));
    router(AppState {
        backend,
        tokenizer,
        max_prompt_tokens,
    })
}

async fn app(backend: Arc<dyn DecisionBackend>, stub: TokenizeStub) -> Router {
    app_with(backend, stub, MAX_PROMPT_TOKENS).await
}

async fn post(app: Router, body: &Value) -> (StatusCode, Value, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.expect("the router answers");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let json = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, json, text)
}

/// Same rounding the API contract mandates (specs/M0.md §4 B4.3).
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

fn probabilities(answer: &Value) -> BTreeMap<String, f64> {
    answer["probabilities"]
        .as_object()
        .expect("answer carries a probability map")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().expect("probability is a number")))
        .collect()
}

// ---------------------------------------------------------------------------
// choice
// ---------------------------------------------------------------------------

#[tokio::test]
async fn choice_answer_is_the_argmax_label_for_every_declared_option() {
    let app = app(
        mock("mock-model", vec![readout(&[("A", -2.0), ("B", -0.1), ("C", -3.0)], 11)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "Customer paid twice.",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Does the customer request a refund?",
                    "criteria": { "no": null, "maybe": null, "yes": null }
                }
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["model"], json!("mock-model"));

    let ans = &body["answers"]["q1"];
    assert_eq!(ans["type"], json!("choice"));
    // Slots follow the criteria keys in dictionary order: A="maybe", B="no",
    // C="yes". B carries the largest logprob (-0.1), so the answer is "no".
    assert_eq!(ans["choice"], json!("no"), "answer: {ans}");

    let expected = softmax(&[-2.0, -0.1, -3.0]);
    let probs = probabilities(ans);
    assert_eq!(
        probs.keys().cloned().collect::<Vec<_>>(),
        vec!["maybe".to_string(), "no".to_string(), "yes".to_string()],
        "every declared option must be scored, keyed by its label"
    );
    for (i, key) in ["maybe", "no", "yes"].iter().enumerate() {
        assert!(
            (probs[*key] - round6(expected[i])).abs() < 1e-12,
            "{key}: {} != {}",
            probs[*key],
            expected[i]
        );
    }

    let sum: f64 = probs.values().sum();
    assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");

    let conf = ans["confidence"].as_f64().expect("choice carries confidence");
    assert!((0.0..=1.0).contains(&conf), "confidence out of range: {conf}");
    assert!(
        (conf - round6(confidence_from_probabilities(&expected, ConfidenceMode::NormalizedEntropy)))
            .abs()
            < 1e-12,
        "confidence {conf} is not the documented normalized-entropy value"
    );
}

#[tokio::test]
async fn uniform_distribution_rounds_to_six_decimals_and_picks_the_first_slot() {
    let app = app(
        mock("mock-model", vec![readout(&[("A", 0.0), ("B", 0.0), ("C", 0.0)], 5)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, text) = post(
        app,
        &json!({
            "state": "x",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": { "a": null, "b": null, "c": null }
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let ans = &body["answers"]["q1"];
    // 1/3 on the wire is 0.333333, not 0.3333333333333333.
    assert_eq!(ans["probabilities"]["a"].as_f64(), Some(0.333333));
    assert_eq!(ans["probabilities"]["b"].as_f64(), Some(0.333333));
    assert_eq!(ans["probabilities"]["c"].as_f64(), Some(0.333333));
    assert!(
        !text.contains("3333333333"),
        "dirty float leaked onto the wire: {text}"
    );
    // Ties resolve to the first declared slot (numpy.argmax convention).
    assert_eq!(ans["choice"], json!("a"));
}

// ---------------------------------------------------------------------------
// score
// ---------------------------------------------------------------------------

#[tokio::test]
async fn score_answer_expectation_and_legend_of_level_descriptions() {
    let app = app(
        mock("mock-model", vec![readout(&[("A", -1.0), ("B", -0.5), ("C", -2.0)], 17)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "Ticket #42",
            "questions": {
                "q1": {
                    "type": "score",
                    "instructions": "How good was the reply?",
                    "criteria": ["unusable", "acceptable", "excellent"]
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let ans = &body["answers"]["q1"];
    assert_eq!(ans["type"], json!("score"));

    let expected = softmax(&[-1.0, -0.5, -2.0]);
    let exp = score_expectation(&expected);
    assert!(
        (ans["score"].as_f64().unwrap() - round6(exp)).abs() < 1e-12,
        "score {} is not Σ i·pᵢ = {exp}",
        ans["score"]
    );

    // `legend` is index -> **level description text**, never the index again.
    let legend = ans["legend"].as_object().expect("score carries a legend");
    assert_eq!(
        legend.keys().cloned().collect::<Vec<_>>(),
        vec!["0".to_string(), "1".to_string(), "2".to_string()],
        "legend keys are the level indices as strings"
    );
    assert_eq!(legend["0"], json!("unusable"));
    assert_eq!(legend["1"], json!("acceptable"));
    assert_eq!(legend["2"], json!("excellent"));
    assert_ne!(
        legend["1"],
        json!("1"),
        "regression: the legend must hold descriptions, not index strings"
    );

    let probs = probabilities(ans);
    assert_eq!(
        probs.keys().cloned().collect::<Vec<_>>(),
        vec!["0".to_string(), "1".to_string(), "2".to_string()]
    );
    for (i, key) in ["0", "1", "2"].iter().enumerate() {
        assert!((probs[*key] - round6(expected[i])).abs() < 1e-12);
    }

    let conf = ans["confidence"].as_f64().expect("score carries confidence");
    assert!((0.0..=1.0).contains(&conf), "confidence out of range: {conf}");
}

#[tokio::test]
async fn score_legend_renders_structured_levels_as_pretty_json() {
    // A level may be any entry (specs/M0.md A1: objects/arrays go through
    // `render_entry`), and the legend must show the same text the prompt does.
    let app = app(
        mock("mock-model", vec![readout(&[("A", -0.5), ("B", -1.5)], 9)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "x",
            "questions": {
                "q1": {
                    "type": "score",
                    "instructions": "Rate.",
                    "criteria": ["plain text", { "tone": "harsh", "level": 2 }]
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let legend = body["answers"]["q1"]["legend"].as_object().unwrap();
    assert_eq!(legend["0"], json!("plain text"));
    let structured = legend["1"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(structured).expect("legend holds rendered JSON");
    assert_eq!(parsed["tone"], json!("harsh"));
    assert_eq!(parsed["level"], json!(2));
}

// ---------------------------------------------------------------------------
// noul
// ---------------------------------------------------------------------------

#[tokio::test]
async fn noul_answer_is_the_yes_slot_probability_without_confidence() {
    let app = app(
        mock("mock-model", vec![readout(&[("A", -0.5), ("B", -1.5)], 23)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "Reply sent within an hour.",
            "questions": {
                "q1": {
                    "type": "noul",
                    "instructions": "Did support reply the same day?",
                    "criteria": { "true": "same day", "false": "slower" }
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let ans = &body["answers"]["q1"];
    assert_eq!(ans["type"], json!("noul"));

    // Slot A is `true` (yes) by construction: it is the first slot the renderer emits.
    let expected = softmax(&[-0.5, -1.5])[0];
    assert!(
        (ans["noul"].as_f64().unwrap() - round6(expected)).abs() < 1e-12,
        "noul {} != P(slot A) = {expected}",
        ans["noul"]
    );
    assert!(
        ans.get("confidence").is_none(),
        "noul answers carry no confidence (contract.rs): {ans}"
    );
    assert!(ans.get("probabilities").is_none(), "noul answers carry no map: {ans}");
}

// ---------------------------------------------------------------------------
// request-level contract
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_questions_is_422() {
    let app = app(mock("mock-model", vec![]), TokenizeStub::new()).await;
    let (status, body, _) = post(app, &json!({ "state": "x", "questions": {} })).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("at least one question"),
        "unexpected error message: {msg}"
    );
}

#[tokio::test]
async fn a_declared_slot_missing_from_the_top_k_is_an_error_not_a_zero() {
    // The backend never returns slot C. Refusing loudly is the whole point: scoring
    // it as 0 would silently bias the answer.
    let app = app(
        mock("mock-model", vec![readout(&[("A", -0.5), ("B", -1.0)], 4)]),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "x",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Pick.",
                    "criteria": { "a": null, "b": null, "c": null }
                }
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("missing"), "{msg}");
    assert!(msg.contains('C'), "the missing slot must be named: {msg}");
}

#[tokio::test]
async fn an_over_budget_prompt_is_422_and_never_truncated() {
    let app = app_with(
        mock("mock-model", vec![readout(&[("A", -0.5), ("B", -1.0)], 4)]),
        TokenizeStub::new(),
        4,
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "a state that is definitely longer than four tokens",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Pick.",
                    "criteria": { "a": null, "b": null }
                }
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("budget"), "{msg}");
}

#[tokio::test]
async fn usage_totals_input_tokens_and_counts_one_output_token_per_question() {
    let app = app(
        mock(
            "mock-model",
            vec![
                readout(&[("A", -0.5), ("B", -1.0)], 11),
                readout(&[("A", -0.2), ("B", -2.0)], 25),
            ],
        ),
        TokenizeStub::new(),
    )
    .await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "x",
            "questions": {
                "q_choice": {
                    "type": "choice",
                    "instructions": "Pick.",
                    "criteria": { "a": null, "b": null }
                },
                "q_noul": {
                    "type": "noul",
                    "instructions": "Yes or no?"
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    assert_eq!(body["answers"].as_object().unwrap().len(), 2);
    assert_eq!(body["usage"]["input_tokens"], json!(36), "11 + 25");
    assert_eq!(body["usage"]["output_tokens"], json!(2), "one generated token per question");
}

// ---------------------------------------------------------------------------
// what the API layer asks the backend for
// ---------------------------------------------------------------------------

#[tokio::test]
async fn top_k_covers_every_slot_and_the_question_id_never_reaches_the_model() {
    let letters = ["A", "B", "C", "D", "E", "F", "G", "H"];
    let logprobs: Vec<(&str, f64)> = letters
        .iter()
        .enumerate()
        .map(|(i, l)| (*l, -0.1 * (i as f64 + 1.0)))
        .collect();
    let (rec, handle) = backend("mock-model", vec![readout(&logprobs, 30)]);
    let app = app(handle, TokenizeStub::new()).await;

    // 8 options -> top_k = max(8 + 5, 10) = 13.
    let criteria: BTreeMap<String, Value> = (0..8).map(|i| (format!("opt{i}"), Value::Null)).collect();

    let (status, body, _) = post(
        app,
        &json!({
            "state": "x",
            "questions": {
                "q_alpha": {
                    "type": "choice",
                    "instructions": "Which option fits best?",
                    "criteria": criteria
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let calls = rec.calls();
    assert_eq!(calls.len(), 1, "one readout per question");
    let (prompt, top_k) = &calls[0];
    assert_eq!(*top_k, 13, "top_k must cover every slot plus margin (max(n+5, 10))");
    assert!(prompt.contains("Answer:\n"), "prompt must end at the answer slot");
    assert!(prompt.contains("Which option fits best?"), "instructions missing");
    assert!(
        !prompt.contains("q_alpha"),
        "question ids are never sent to the model: {prompt}"
    );
}

#[tokio::test]
async fn questions_in_one_request_are_independent() {
    let (rec, handle) = backend(
        "mock-model",
        vec![
            readout(&[("A", -0.5), ("B", -1.0)], 10),
            readout(&[("A", -0.1), ("B", -2.5)], 10),
        ],
    );
    let app = app(handle, TokenizeStub::new()).await;

    let (status, body, _) = post(
        app,
        &json!({
            "state": "shared state",
            "questions": {
                "q_choice": {
                    "type": "choice",
                    "instructions": "FIRST instructions",
                    "criteria": { "a": null, "b": null }
                },
                "q_noul": {
                    "type": "noul",
                    "instructions": "SECOND instructions"
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let calls = rec.calls();
    assert_eq!(calls.len(), 2);
    let (first, second) = (&calls[0].0, &calls[1].0);
    assert!(first.contains("FIRST instructions"));
    assert!(second.contains("SECOND instructions"));
    assert!(
        !second.contains("FIRST instructions"),
        "one question's text must not leak into another's prompt: {second}"
    );
    assert_ne!(first, second, "each question gets its own prompt");
    // 2 slots -> the documented floor of 10 applies.
    assert_eq!(calls[0].1, 10);
    assert_eq!(calls[1].1, 10);
}

#[tokio::test]
async fn healthz_reports_ok() {
    let app = app(mock("mock-model", vec![]), TokenizeStub::new()).await;
    let req = Request::builder()
        .method("GET")
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], json!("ok"));
}

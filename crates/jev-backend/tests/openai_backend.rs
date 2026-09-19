//! `OpenAiCompatBackend` against hand-written stub servers (specs/M0.md §4 B5.2).
//!
//! What can be verified without a GPU: the request body we send, the response
//! shapes we accept, and the error mapping. The stub is an axum server on an
//! ephemeral localhost port — no `wiremock` (not in this machine's offline cargo
//! cache), no network, no model.

mod common;

use common::{spawn_completions_stub, CompletionsStub};
use jev_backend::{BackendError, DecisionBackend, OpenAiCompatBackend, OpenAiCompatConfig};
use serde_json::json;

const PROMPT: &str = "State:\nCustomer paid twice.\n\nQuestion:\nRefund?\n\nAnswer:\n";

fn backend(base_url: &str) -> OpenAiCompatBackend {
    OpenAiCompatBackend::new(OpenAiCompatConfig {
        base_url: base_url.to_string(),
        model: "test-model".to_string(),
        api_key: "secret-key".to_string(),
        timeout_secs: 10,
    })
    .expect("backend configures")
}

/// A well-formed `/completions` response, as probed on the live endpoint
/// (specs/M0.md §1): `top_logprobs[0]` maps token *text* to logprob.
fn ok_body() -> String {
    json!({
        "choices": [{
            "text": "A",
            "logprobs": { "top_logprobs": [ { "A": -0.229, " A": -1.234, "B": -1.854, "yes": -3.5 } ] }
        }],
        "usage": { "prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43 }
    })
    .to_string()
}

#[tokio::test]
async fn parses_top_logprobs_and_usage() {
    let stub = CompletionsStub::new(200, ok_body());
    let base = spawn_completions_stub(stub).await;

    let readout = backend(&base)
        .token_logprobs(PROMPT, 10)
        .await
        .expect("well-formed response parses");

    assert_eq!(readout.top.len(), 4);
    assert_eq!(readout.top.get("A").copied(), Some(-0.229));
    assert_eq!(readout.top.get(" A").copied(), Some(-1.234));
    assert_eq!(readout.top.get("B").copied(), Some(-1.854));
    // `usage.prompt_tokens` comes from the server; `completion_tokens` is 1 by
    // construction (exactly one token is generated, purely to read the distribution).
    assert_eq!(readout.prompt_tokens, 42);
    assert_eq!(readout.completion_tokens, 1);

    // The slot rule survives the backend round trip: A and " A" are one slot.
    let slot = readout.slot_logprob("A").expect("A is in the top-k");
    let expected = ((-0.229_f64).exp() + (-1.234_f64).exp()).ln();
    assert!((slot - expected).abs() < 1e-12, "got {slot}, expected {expected}");
}

#[tokio::test]
async fn request_body_carries_model_prompt_and_logprobs() {
    let stub = CompletionsStub::new(200, ok_body());
    let base = spawn_completions_stub(stub.clone()).await;

    backend(&base)
        .token_logprobs(PROMPT, 7)
        .await
        .expect("stub answers 200");

    let sent = stub.only_request();
    // The probed request shape (specs/M0.md §4 B1): raw completion, one token,
    // greedy, top-k logprobs. `logprobs` must be the caller's `top_k`, untweaked.
    assert_eq!(sent["model"], json!("test-model"));
    assert_eq!(sent["prompt"], json!(PROMPT));
    assert_eq!(sent["max_tokens"], json!(1));
    assert_eq!(sent["logprobs"], json!(7));
    assert!(sent["logprobs"].is_number(), "logprobs must be a number, got {}", sent["logprobs"]);
    assert!(sent["prompt"].is_string(), "prompt must be a string");
    assert_eq!(sent["temperature"], json!(0));
}

#[tokio::test]
async fn null_top_logprobs_is_no_logprobs() {
    let body = json!({
        "choices": [{ "text": "x", "logprobs": { "top_logprobs": null } }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 1 }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    assert!(
        matches!(err, BackendError::NoLogprobs),
        "expected NoLogprobs, got {err:?}"
    );
}

#[tokio::test]
async fn empty_top_logprobs_is_no_logprobs() {
    let body = json!({
        "choices": [{ "logprobs": { "top_logprobs": [{}] } }],
        "usage": { "prompt_tokens": 5 }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    assert!(matches!(err, BackendError::NoLogprobs), "got {err:?}");
}

#[tokio::test]
async fn missing_logprobs_field_is_no_logprobs() {
    let body = json!({ "choices": [{ "text": "A" }], "usage": { "prompt_tokens": 5 } }).to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    assert!(matches!(err, BackendError::NoLogprobs), "got {err:?}");
}

#[tokio::test]
async fn http_400_with_logprobs_not_supported_is_http_error() {
    // NInfer's answer, quoted verbatim from the probe (specs/M0.md §1): this is the
    // string a caller uses to tell "this endpoint cannot be a readout backend" from
    // every other 4xx.
    let body = json!({
        "error": { "code": "logprobs_not_supported", "message": "logprobs=true is not supported" }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(400, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    match err {
        BackendError::Http(msg) => {
            assert!(
                msg.contains("logprobs_not_supported"),
                "message must carry the body so callers can recognise the endpoint: {msg}"
            );
            assert!(msg.contains("400"), "message must carry the status: {msg}");
        }
        other => panic!("expected BackendError::Http, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_json_is_a_decode_error() {
    let base = spawn_completions_stub(CompletionsStub::new(200, "<html>nope</html>")).await;
    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    assert!(matches!(err, BackendError::Decode(_)), "got {err:?}");
}

#[tokio::test]
async fn non_numeric_logprob_is_a_decode_error() {
    let body = json!({
        "choices": [{ "logprobs": { "top_logprobs": [ { "A": "high" } ] } }],
        "usage": { "prompt_tokens": 5 }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;
    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    assert!(matches!(err, BackendError::Decode(_)), "got {err:?}");
}

#[tokio::test]
async fn a_full_500_char_ascii_error_body_is_capped_at_500_chars() {
    let body = "x".repeat(600);
    let base = spawn_completions_stub(CompletionsStub::new(500, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    let msg = expect_http(err);
    assert!(
        msg.ends_with(&"x".repeat(500)),
        "expected the body capped at 500 chars, got {} chars",
        msg.chars().count()
    );
}

#[tokio::test]
async fn non_ascii_error_body_cut_at_byte_500_does_not_panic() {
    // Regression test for the byte-slice truncation bug: the body is exactly 500
    // characters, but 500 *bytes* lands in the middle of the 34th `中`, so
    // `&text[..500]` panicked on a server error whose body happens to be Chinese.
    let body = format!("{}{}", "a".repeat(400), "中".repeat(100));
    assert_eq!(body.chars().count(), 500);
    assert!(body.len() > 500, "the byte cut must land inside a character");
    // Proof this body exercises the bug: the old code truncated with
    // `&body[..500]`, and byte 500 is *not* a char boundary, so that slice panics.
    assert!(
        std::str::from_utf8(&body.as_bytes()[..500]).is_err(),
        "byte index 500 must fall inside a multi-byte character"
    );
    let base = spawn_completions_stub(CompletionsStub::new(500, body.clone())).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    let msg = expect_http(err);
    assert!(msg.contains(&body), "the whole 500-char body must survive: {msg}");
}

#[tokio::test]
async fn long_non_ascii_error_body_is_capped_by_chars_not_bytes() {
    // 1000 CJK characters = 3000 bytes: old byte slicing panicked (byte 500 falls
    // inside a character); the cap must keep 500 *characters*.
    let body = "中".repeat(1000);
    assert!(std::str::from_utf8(&body.as_bytes()[..500]).is_err());
    let base = spawn_completions_stub(CompletionsStub::new(500, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 10).await.unwrap_err();
    let msg = expect_http(err);
    assert_eq!(
        msg.matches('中').count(),
        500,
        "expected exactly 500 characters of the body: {msg}"
    );
}

#[tokio::test]
async fn usage_missing_still_yields_a_readout() {
    let body = json!({
        "choices": [{ "logprobs": { "top_logprobs": [ { "A": -0.1, "B": -3.0 } ] } }]
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let readout = backend(&base).token_logprobs(PROMPT, 10).await.unwrap();
    assert_eq!(readout.prompt_tokens, 0, "no usage block -> unknown token count");
    assert_eq!(readout.completion_tokens, 1);
}

#[tokio::test]
async fn connection_failure_is_an_http_error() {
    // Nothing listens on port 1; the backend must report a transport problem as
    // `BackendError::Http`, not as a decode/no-logprobs failure.
    let err = backend("http://127.0.0.1:1/v1")
        .token_logprobs(PROMPT, 10)
        .await
        .unwrap_err();
    assert!(matches!(err, BackendError::Http(_)), "got {err:?}");
}

#[tokio::test]
async fn model_name_is_the_configured_model() {
    let base = spawn_completions_stub(CompletionsStub::new(200, ok_body())).await;
    assert_eq!(backend(&base).model_name(), "test-model");
}

fn expect_http(err: BackendError) -> String {
    match err {
        BackendError::Http(msg) => msg,
        other => panic!("expected BackendError::Http, got {other:?}"),
    }
}

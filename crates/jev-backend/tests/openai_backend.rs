//! `OpenAiCompatBackend` against hand-written stub servers (specs/M0.md §4 B5.2).
//!
//! What can be verified without a GPU: the request body we send, the response
//! shapes we accept, and the error mapping. The stub is an axum server on an
//! ephemeral localhost port — no `wiremock` (not in this machine's offline cargo
//! cache), no network, no model.

mod common;

use common::{
    spawn_array_rejecting_completions_stub, spawn_completions_stub, ArrayRejectingCompletionsStub,
    CompletionsStub,
};
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

// ---------------------------------------------------------------------------
// Batched readout (`token_logprobs_batch`) — specs/M1.md §1
// ---------------------------------------------------------------------------
//
// This is the path the whole prefix-reuse story rides on, and it is the one place
// where a mistake is invisible to the caller: if the readouts come back mapped to
// the wrong questions, every answer in the batch is quietly attached to somebody
// else's prompt. The wire order of `choices` is not promised to match the request
// order either, so these tests pin the mapping down on purpose-built responses.

fn prompts() -> Vec<String> {
    vec![
        "State:\nsame prefix\n\nQuestion:\nfirst?\n\nAnswer:\n".to_string(),
        "State:\nsame prefix\n\nQuestion:\nsecond?\n\nAnswer:\n".to_string(),
        "State:\nsame prefix\n\nQuestion:\nthird?\n\nAnswer:\n".to_string(),
    ]
}

/// One choice carrying a single, uniquely identifiable logprob.
fn choice(index: usize, token: &str, logprob: f64) -> serde_json::Value {
    json!({
        "text": token,
        "index": index,
        "logprobs": { "top_logprobs": [ { token: logprob } ] },
    })
}

fn batch_body(choices: Vec<serde_json::Value>, prompt_tokens: u64) -> String {
    json!({
        "choices": choices,
        "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": 3, "total_tokens": 3 },
    })
    .to_string()
}

#[tokio::test]
async fn the_batched_request_sends_a_prompt_array() {
    let stub = CompletionsStub::new(
        200,
        batch_body(vec![choice(0, "A", -0.1), choice(1, "B", -0.2)], 900),
    );
    let base = spawn_completions_stub(stub.clone()).await;

    backend(&base)
        .token_logprobs_batch(&prompts()[..2], 100)
        .await
        .expect("well-formed batch parses");

    let body = stub.only_request();
    let sent = body["prompt"].as_array().expect("`prompt` must be an array for a batch");
    assert_eq!(sent.len(), 2, "one array element per prompt: {body}");
    assert_eq!(sent[0].as_str(), Some(prompts()[0].as_str()));
    assert_eq!(sent[1].as_str(), Some(prompts()[1].as_str()));
    assert_eq!(body["max_tokens"], json!(1), "one token is generated, only to read the distribution");
    assert_eq!(body["logprobs"], json!(100), "top_k must reach the wire");
}

#[tokio::test]
async fn batched_choices_are_ordered_by_index_not_by_wire_order() {
    // The stub answers in REVERSE wire order but labels each choice with the
    // prompt it belongs to. If the backend trusted the array position, readout 0
    // would carry the third prompt's distribution.
    let stub = CompletionsStub::new(
        200,
        batch_body(
            vec![
                choice(2, "C", -0.3),
                choice(1, "B", -0.2),
                choice(0, "A", -0.1),
            ],
            900,
        ),
    );
    let base = spawn_completions_stub(stub).await;

    let readouts = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .expect("a permutation of 0..N in any wire order is accepted");
    assert_eq!(readouts.len(), 3);
    assert_eq!(readouts[0].top.get("A").copied(), Some(-0.1), "prompt 0 keeps its own readout");
    assert_eq!(readouts[1].top.get("B").copied(), Some(-0.2), "prompt 1 keeps its own readout");
    assert_eq!(readouts[2].top.get("C").copied(), Some(-0.3), "prompt 2 keeps its own readout");
}

#[tokio::test]
async fn a_duplicate_batch_index_fails_the_whole_call() {
    // Indices 0,1,1 mean two readouts claim the same prompt and one prompt has
    // none: unanswerable, so the call must fail rather than pick a winner.
    let stub = CompletionsStub::new(
        200,
        batch_body(
            vec![choice(0, "A", -0.1), choice(1, "B", -0.2), choice(1, "C", -0.3)],
            900,
        ),
    );
    let base = spawn_completions_stub(stub).await;

    let err = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .unwrap_err();
    let msg = match err {
        BackendError::Decode(msg) => msg,
        other => panic!("expected BackendError::Decode, got {other:?}"),
    };
    assert!(msg.contains("permutation"), "the reason must be named: {msg}");
}

#[tokio::test]
async fn a_choice_count_mismatch_fails_the_whole_call() {
    // 2 choices for 3 prompts: whatever we did with the third would be a guess.
    let stub = CompletionsStub::new(
        200,
        batch_body(vec![choice(0, "A", -0.1), choice(1, "B", -0.2)], 900),
    );
    let base = spawn_completions_stub(stub).await;

    let err = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .unwrap_err();
    match err {
        BackendError::Decode(msg) => {
            assert!(msg.contains("2 choices for 3 prompts"), "got: {msg}");
            assert!(msg.contains("all-or-nothing"), "got: {msg}");
        }
        other => panic!("expected BackendError::Decode, got {other:?}"),
    }
}

#[tokio::test]
async fn batch_choices_without_index_fall_back_to_array_position() {
    // Some OpenAI-compatible servers omit `index` on single-element batches; the
    // documented fallback is the array position.
    let body = json!({
        "choices": [
            { "text": "A", "logprobs": { "top_logprobs": [ { "A": -0.1 } ] } },
            { "text": "B", "logprobs": { "top_logprobs": [ { "B": -0.2 } ] } },
        ],
        "usage": { "prompt_tokens": 900, "completion_tokens": 2, "total_tokens": 2 },
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let readouts = backend(&base)
        .token_logprobs_batch(&prompts()[..2], 100)
        .await
        .expect("missing `index` falls back to position");
    assert_eq!(readouts[0].top.get("A").copied(), Some(-0.1));
    assert_eq!(readouts[1].top.get("B").copied(), Some(-0.2));
}

#[tokio::test]
async fn one_batched_usage_block_lands_on_the_first_readout_only() {
    // The endpoint reports one usage block for the whole batch. It is attached to
    // the first readout so that summing the vector reproduces the endpoint's own
    // number exactly; a fake per-prompt split would invent data.
    let stub = CompletionsStub::new(
        200,
        batch_body(
            vec![choice(0, "A", -0.1), choice(1, "B", -0.2), choice(2, "C", -0.3)],
            900,
        ),
    );
    let base = spawn_completions_stub(stub).await;

    let readouts = backend(&base).token_logprobs_batch(&prompts(), 100).await.unwrap();
    assert_eq!(readouts[0].prompt_tokens, 900);
    assert_eq!(readouts[1].prompt_tokens, 0);
    assert_eq!(readouts[2].prompt_tokens, 0);
    assert_eq!(
        readouts.iter().map(|r| r.prompt_tokens).sum::<u64>(),
        900,
        "the vector must reproduce the endpoint's own accounting"
    );
    assert!(readouts.iter().all(|r| r.completion_tokens == 1));
}

#[tokio::test]
async fn a_batched_choice_without_usable_logprobs_fails_the_whole_call() {
    let body = batch_body(
        vec![
            choice(0, "A", -0.1),
            json!({ "text": "B", "index": 1, "logprobs": { "top_logprobs": null } }),
        ],
        900,
    );
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base)
        .token_logprobs_batch(&prompts()[..2], 100)
        .await
        .unwrap_err();
    assert!(matches!(err, BackendError::NoLogprobs), "got {err:?}");
}

#[tokio::test]
async fn batched_http_500_is_an_http_error() {
    let base = spawn_completions_stub(CompletionsStub::new(500, "upstream is down")).await;
    let err = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .unwrap_err();
    assert!(expect_http(err).contains("500"));
}

#[tokio::test]
async fn batched_invalid_json_is_a_decode_error() {
    let base = spawn_completions_stub(CompletionsStub::new(200, "not json at all")).await;
    let err = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .unwrap_err();
    assert!(matches!(err, BackendError::Decode(_)), "got {err:?}");
}

// ---------------------------------------------------------------------------
// llama.cpp's wire dialect (probed 2026-09-21, M3 Ultra + llama.cpp server)
//
// Two things differ from the reference endpoint, and neither is cosmetic:
//   1. the distribution is `logprobs.content[0].top_logprobs` — an *array* of
//      token objects, with the legacy `logprobs.top_logprobs[0]` map absent;
//   2. an array `prompt` is refused outright (400 "type must be string, but is an
//      array"), so the M1 batched readout cannot run there as specified.
// ---------------------------------------------------------------------------

/// A `/completions` response in llama.cpp's shape.
fn llama_cpp_body() -> String {
    json!({
        "choices": [{
            "text": "A",
            "index": 0,
            "logprobs": {
                "content": [{
                    "token": "A",
                    "logprob": -0.229,
                    "bytes": [65],
                    "top_logprobs": [
                        {"token": "A", "logprob": -0.229, "bytes": [65]},
                        {"token": " A", "logprob": -1.234, "bytes": [32, 65]},
                        {"token": "B", "logprob": -1.854, "bytes": [66]}
                    ]
                }]
            }
        }],
        "usage": { "prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43 }
    })
    .to_string()
}

#[tokio::test]
async fn parses_llama_cpp_content_logprobs() {
    let base = spawn_completions_stub(CompletionsStub::new(200, llama_cpp_body())).await;

    let readout = backend(&base)
        .token_logprobs(PROMPT, 20)
        .await
        .expect("llama.cpp's `logprobs.content[]` shape parses");

    assert_eq!(readout.top.len(), 3);
    assert_eq!(readout.top.get("A").copied(), Some(-0.229));
    assert_eq!(readout.top.get(" A").copied(), Some(-1.234));
    assert_eq!(readout.prompt_tokens, 42);
    // Not merely parseable: the slot readout has to work on it, and both surface
    // spellings of the slot are summed in log space by `slot_logprob`:
    // ln(e^-0.229 + e^-1.234) = 0.0829.
    let slot = readout.slot_logprob("A").expect("the A slot is present");
    assert!(
        (slot - 0.0829).abs() < 0.0005,
        "both spellings of A must be summed, got {slot}"
    );
}

#[tokio::test]
async fn the_legacy_map_wins_when_both_shapes_are_present() {
    // A server that emits both must still be read the way M0 froze: as the map.
    let mut both: serde_json::Value = serde_json::from_str(&llama_cpp_body()).unwrap();
    both["choices"][0]["logprobs"]["top_logprobs"] = json!([{ "B": -9.0 }]);
    let base = spawn_completions_stub(CompletionsStub::new(200, both.to_string())).await;

    let readout = backend(&base).token_logprobs(PROMPT, 20).await.unwrap();

    assert_eq!(readout.top.get("B").copied(), Some(-9.0));
    assert_eq!(readout.top.len(), 1, "the legacy map is the one that counts");
}

#[tokio::test]
async fn a_content_entry_without_a_token_is_a_decode_error() {
    let body = json!({
        "choices": [{ "logprobs": { "content": [{ "top_logprobs": [{ "logprob": -1.0 }] }] } }],
        "usage": { "prompt_tokens": 1 }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 20).await.unwrap_err();

    assert!(matches!(err, BackendError::Decode(_)), "got {err:?}");
}

#[tokio::test]
async fn an_empty_content_distribution_is_no_logprobs() {
    let body = json!({
        "choices": [{ "logprobs": { "content": [
            { "token": "A", "logprob": -1.0, "top_logprobs": [] }
        ] } }],
        "usage": { "prompt_tokens": 1 }
    })
    .to_string();
    let base = spawn_completions_stub(CompletionsStub::new(200, body)).await;

    let err = backend(&base).token_logprobs(PROMPT, 20).await.unwrap_err();

    assert!(matches!(err, BackendError::NoLogprobs), "got {err:?}");
}

#[tokio::test]
async fn a_400_on_the_array_prompt_falls_back_to_sequential_requests() {
    let stub = ArrayRejectingCompletionsStub::new(7);
    let base = spawn_array_rejecting_completions_stub(stub.clone()).await;
    let b = backend(&base);

    let readouts = b
        .token_logprobs_batch(&prompts(), 100)
        .await
        .expect("the sequential fallback answers where the batch was refused");

    assert_eq!(readouts.len(), 3, "one readout per prompt, all-or-nothing");
    let sent = stub.requests();
    assert_eq!(
        sent.len(),
        prompts().len() + 1,
        "one rejected array request, then one request per prompt: {sent:?}"
    );
    assert!(
        sent[0]["prompt"].is_array(),
        "the batched shape must be attempted first, not skipped"
    );
    for (i, body) in sent[1..].iter().enumerate() {
        assert_eq!(
            body["prompt"].as_str(),
            Some(prompts()[i].as_str()),
            "the fallback must ask the prompts in order, one by one"
        );
        assert!(
            body["prompt"].is_string(),
            "the fallback sends a string prompt (that is the whole point)"
        );
        assert_eq!(
            readouts[i].top.get("A").copied(),
            Some(-0.1 * (i as f64 + 1.0)),
            "readout {i} must carry its own answer, not a neighbour's"
        );
        assert_eq!(
            readouts[i].prompt_tokens, 7,
            "in the fallback each readout carries its own `usage`, one request each"
        );
    }
    assert_eq!(
        b.batch_fallbacks(),
        1,
        "the fallback must be counted so no report can present it as a shared prefill"
    );
}

#[tokio::test]
async fn only_a_shape_rejection_triggers_the_fallback() {
    // A rate limit, an auth failure or a server fault is about the *call*, not the
    // request shape. Falling back there would launder a throttled endpoint into a
    // "successful" benchmark run — and, worse, into readouts the run would treat as
    // measurements (found in review, 2026-09-21: an array 429 was answered by three
    // sequential requests and reported as readouts).
    for status in [401, 403, 429, 500, 503] {
        let stub = CompletionsStub::new(status, "nope");
        let base = spawn_completions_stub(stub.clone()).await;
        let b = backend(&base);

        let err = b.token_logprobs_batch(&prompts(), 100).await.unwrap_err();

        assert!(
            expect_http(err).contains(&status.to_string()),
            "{status}: the endpoint's status must survive"
        );
        assert_eq!(
            stub.requests().len(),
            1,
            "{status} must not be retried sequentially"
        );
        assert_eq!(b.batch_fallbacks(), 0, "{status} is not a capability gap");
    }
}

#[tokio::test]
async fn the_sequential_fallback_does_not_swallow_a_failing_single_request() {
    // Every request 400s: the array attempt triggers the fallback, and the
    // fallback's own failures must surface as an error — never as readouts.
    let stub = CompletionsStub::new(400, "nope");
    let base = spawn_completions_stub(stub.clone()).await;

    let err = backend(&base)
        .token_logprobs_batch(&prompts(), 100)
        .await
        .unwrap_err();

    assert!(expect_http(err).contains("400"), "the endpoint's status must survive");
    assert_eq!(
        stub.requests().len(),
        2,
        "the array attempt, then the first sequential retry — which fails, so the rest are \
         not sent (no half-batch: the call returns an error instead of partial readouts)"
    );
}

#[tokio::test]
async fn an_empty_batch_makes_no_request_at_all() {
    let stub = CompletionsStub::new(200, ok_body());
    let base = spawn_completions_stub(stub.clone()).await;

    let readouts = backend(&base).token_logprobs_batch(&[], 100).await.unwrap();
    assert!(readouts.is_empty());
    assert!(
        stub.requests().is_empty(),
        "an empty prompt list is answered locally: the live endpoint rejects an empty `prompt`"
    );
}

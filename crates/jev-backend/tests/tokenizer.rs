//! `HttpTokenizer` and the probed slot self-check (specs/M0.md §4 B3/B4/B5).
//!
//! The tokenizer is a *prerequisite* for serving at all: it decides whether the
//! position we read log-probabilities from really is the answer slot. These tests
//! pin down the endpoint fallback, the cache, and the fail-fast self-check — all
//! without a GPU, against axum stubs on localhost.

mod common;

use common::{spawn_tokenize_stub, TokenizeStub};
use jev_backend::tokenizer::{
    check_probed_slot_assumption, PROBED_PROMPT, PROBED_PROMPT_IDS, PROBED_SLOT_ID,
    PROBED_SLOT_LETTER,
};
use jev_backend::{HttpTokenizer, Prefetch};
use jev_core::SlotVerifier;
use std::collections::HashMap;

const LETTER: &str = "A";
const CACHED_TEXT: &str = "Answer:\nA";

fn tokenizer(base_url: &str) -> HttpTokenizer {
    HttpTokenizer::new(base_url, "test-model", "EMPTY")
}

/// The byte-per-token encodings `TokenizeStub` produces by default.
fn ids_of(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// The probed encodings (specs/M0.md §1) as a plain map, for driving
/// [`check_probed_slot_assumption`] without any HTTP.
fn probed_map() -> HashMap<String, Vec<u32>> {
    HashMap::from([
        (PROBED_PROMPT.to_string(), PROBED_PROMPT_IDS.to_vec()),
        (PROBED_SLOT_LETTER.to_string(), vec![PROBED_SLOT_ID]),
        (
            format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"),
            vec![15666, 25, 198, 32],
        ),
    ])
}

struct FakeVerifier {
    map: HashMap<String, Vec<u32>>,
}

impl FakeVerifier {
    fn new(map: HashMap<String, Vec<u32>>) -> Self {
        Self { map }
    }

    /// Clone the probed map with one entry replaced (a broken tokenizer).
    fn with(text: &str, ids: Vec<u32>) -> Self {
        let mut map = probed_map();
        map.insert(text.to_string(), ids);
        Self::new(map)
    }
}

impl SlotVerifier for FakeVerifier {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.map
            .get(text)
            .unwrap_or_else(|| panic!("fake verifier has no entry for {text:?}"))
            .clone()
    }

    fn decode_token(&self, _id: u32) -> String {
        "<unk>".to_string()
    }
}

// ---------------------------------------------------------------------------
// check_probed_slot_assumption: the three assertions
// ---------------------------------------------------------------------------

#[test]
fn self_check_passes_on_the_probed_encodings() {
    check_probed_slot_assumption(&FakeVerifier::new(probed_map()))
        .expect("the probed encodings satisfy the slot assumption");
}

#[test]
fn self_check_rejects_a_different_prompt_tokenization() {
    // e.g. the wrong model, or a tokenizer with a different merge order.
    let v = FakeVerifier::with(PROBED_PROMPT, vec![1, 2, 3]);
    let err = check_probed_slot_assumption(&v).unwrap_err().to_string();
    assert!(err.contains("slot self-check failed"), "{err}");
    assert!(err.contains("15666"), "the expected ids must be named: {err}");
}

#[test]
fn self_check_rejects_a_slot_letter_that_is_not_a_single_token() {
    let v = FakeVerifier::with(PROBED_SLOT_LETTER, vec![32, 33]);
    let err = check_probed_slot_assumption(&v).unwrap_err().to_string();
    assert!(err.contains("not the single token"), "{err}");
}

#[test]
fn self_check_rejects_a_slot_letter_with_the_wrong_token_id() {
    let mut map = probed_map();
    map.insert(PROBED_SLOT_LETTER.to_string(), vec![99]);
    // Keep the concatenation self-consistent so only the id assertion can fire.
    map.insert(format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"), vec![15666, 25, 198, 99]);
    let err = check_probed_slot_assumption(&FakeVerifier::new(map))
        .unwrap_err()
        .to_string();
    assert!(err.contains("not the single token [32]"), "{err}");
}

#[test]
fn self_check_rejects_a_re_tokenized_prompt_tail() {
    // The letter is fine, but appending it changed the prompt's own tokens — the
    // readout position would no longer be the answer slot.
    let v = FakeVerifier::with(
        &format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"),
        vec![15666, 25, 199, 32],
    );
    let err = check_probed_slot_assumption(&v).unwrap_err().to_string();
    assert!(err.contains("re-tokenized"), "{err}");
}

// ---------------------------------------------------------------------------
// HttpTokenizer over stub endpoints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn self_check_passes_against_a_stub_returning_the_probed_ids() {
    let stub = TokenizeStub::new()
        .with_override(PROBED_PROMPT, PROBED_PROMPT_IDS.to_vec())
        .with_override(PROBED_SLOT_LETTER, vec![PROBED_SLOT_ID])
        .with_override(
            format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"),
            vec![15666, 25, 198, 32],
        );
    let base = spawn_tokenize_stub(stub).await;

    tokenizer(&base)
        .verify_slot_assumption()
        .await
        .expect("stub serves the probed encodings");
}

#[tokio::test]
async fn self_check_fails_against_a_stub_with_a_different_tokenizer() {
    // The stub's byte-per-token fake does not agree with the probe: startup must
    // fail instead of serving readouts from an unverified position.
    let base = spawn_tokenize_stub(TokenizeStub::new()).await;

    let err = tokenizer(&base).verify_slot_assumption().await.unwrap_err();
    assert!(err.to_string().contains("slot self-check failed"), "{err}");
}

#[tokio::test]
async fn prefetch_then_encode_serves_from_cache() {
    let stub = TokenizeStub::new();
    let base = spawn_tokenize_stub(stub.clone()).await;
    let tk = tokenizer(&base);

    // The synchronous verifier path panics on a cache miss, so `prefetch` must
    // populate the cache for every text the verification reads.
    tk.prefetch(&[CACHED_TEXT.to_string(), LETTER.to_string()])
        .await
        .expect("stub tokenizes");

    assert_eq!(tk.encode(CACHED_TEXT), ids_of(CACHED_TEXT));
    assert_eq!(tk.encode(LETTER), vec![65]);
    assert_eq!(
        stub.request_count(),
        2,
        "prefetch must fetch each distinct text exactly once"
    );

    // Second read is a cache hit: no additional request reaches the stub.
    assert_eq!(tk.try_encode(CACHED_TEXT).await.unwrap(), tk.encode(CACHED_TEXT));
    assert_eq!(
        stub.request_count(),
        2,
        "the repeat read must not hit the endpoint again"
    );
}

#[tokio::test]
async fn encode_without_prefetch_panics_with_a_clear_message() {
    let base = spawn_tokenize_stub(TokenizeStub::new()).await;
    let tk = tokenizer(&base);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tk.encode("this text was never prefetched")
    }));
    assert!(
        result.is_err(),
        "encode on an uncached text must panic (programming error, not a silent 0)"
    );
}

#[tokio::test]
async fn falls_back_to_v1_tokenize_when_tokenize_fails() {
    let stub = TokenizeStub::new().failing_primary();
    let base = spawn_tokenize_stub(stub.clone()).await;

    let ids = tokenizer(&base)
        .try_encode("Answer:\n")
        .await
        .expect("the /v1/tokenize fallback answers");

    assert_eq!(ids, ids_of("Answer:\n"));
    let paths: Vec<String> = stub.requests().into_iter().map(|(p, _)| p).collect();
    assert_eq!(
        paths,
        vec!["/tokenize".to_string(), "/v1/tokenize".to_string()],
        "the primary path must be tried first, the fallback second"
    );
    // The stub reads `prompt` and nothing else, so a non-empty text here is proof
    // the client speaks the endpoint's dialect.
    let texts: Vec<String> = stub.requests().into_iter().map(|(_, t)| t).collect();
    assert_eq!(
        texts,
        vec!["Answer:\n".to_string(), "Answer:\n".to_string()],
        "the text must travel in `prompt` (the live endpoint rejects any other field)"
    );
}

#[tokio::test]
async fn stub_rejects_the_wrong_field_name_like_the_live_endpoint() {
    // Guards the guard. The reason a wrong request field name once survived this
    // whole suite is that the stub agreed with the client instead of with the
    // server. Hand-write the body the client used to send and require the 400 the
    // real SGLang returns (probed live 2026-09-19).
    let base = spawn_tokenize_stub(TokenizeStub::new()).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/tokenize"))
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","text":"Answer:\n"}"#)
        .send()
        .await
        .expect("the stub answers");
    assert_eq!(
        resp.status(),
        400,
        "a body without `prompt` must be rejected, exactly like the live endpoint"
    );
    let body = resp.text().await.unwrap();
    assert!(body.contains("Exactly one of"), "{body}");
}

#[tokio::test]
async fn reports_both_failures_when_both_tokenize_paths_are_down() {
    let stub = TokenizeStub::new().failing_primary().failing_fallback();
    let base = spawn_tokenize_stub(stub).await;

    let err = tokenizer(&base).try_encode("Answer:\n").await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("/tokenize"), "{msg}");
    assert!(msg.contains("/v1/tokenize"), "{msg}");
}

#[tokio::test]
async fn non_ok_tokenize_body_is_handled_without_panicking() {
    // The primary stub answers 500 with a non-ASCII body; the fallback is up, so
    // this also exercises the char-safe truncation on the tokenizer path.
    let stub = TokenizeStub::new().failing_primary();
    let base = spawn_tokenize_stub(stub).await;

    assert!(tokenizer(&base).try_encode("Answer:\n").await.is_ok());
}

#[tokio::test]
async fn decode_token_returns_the_letter_it_observed() {
    let stub = TokenizeStub::new();
    let base = spawn_tokenize_stub(stub).await;
    let tk = tokenizer(&base);

    tk.prefetch(&[LETTER.to_string()]).await.unwrap();
    assert_eq!(tk.decode_token(65), "A", "id 65 was cached as the single token `A`");
    // Nothing was observed for id 7: diagnostics must say so, not invent text.
    assert_eq!(tk.decode_token(7), "<unk>");
}

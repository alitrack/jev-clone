//! `HttpTokenizer` and the probed slot self-check (specs/M0.md §4 B3/B4/B5).
//!
//! The tokenizer is a *prerequisite* for serving at all: it decides whether the
//! position we read log-probabilities from really is the answer slot. These tests
//! pin down the endpoint fallback, the cache, and the fail-fast self-check — all
//! without a GPU, against axum stubs on localhost.

mod common;

use common::{spawn_tokenize_stub, TokenizeStub};
use jev_backend::tokenizer::{
    check_probed_slot_assumption, check_slot_letters, slot_letters, SlotCheck, PROBED_PROMPT,
    PROBED_PROMPT_IDS, PROBED_SLOT_ID, PROBED_SLOT_LETTER,
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

// ---------------------------------------------------------------------------
// The field-name negotiation (llama.cpp's dialect, probed 2026-09-21)
//
// llama.cpp's `/tokenize` takes `content`, not `prompt` — and unlike SGLang it
// does not reject the field it does not know: it answers `200 {"tokens": []}`.
// A client that accepts that as a tokenization caches an empty prompt and reads
// its logprobs from the wrong position without a single error anywhere.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_reference_field_is_tried_first_and_accepted() {
    let stub = TokenizeStub::new();
    let base = spawn_tokenize_stub(stub.clone()).await;

    let ids = tokenizer(&base).try_encode("Answer:\n").await.unwrap();

    assert_eq!(ids, ids_of("Answer:\n"));
    assert_eq!(
        stub.request_count(),
        1,
        "the reference endpoint must cost exactly one request (no negotiation)"
    );
    assert_eq!(stub.fields(), vec!["prompt".to_string()]);
}

#[tokio::test]
async fn a_failing_endpoint_costs_two_requests_and_names_one_field() {
    // A failing endpoint never said it ignores the field, so there is no dialect
    // to negotiate: it keeps M0's semantics — the reference field on both URL
    // spellings, one error naming only that field. Probing `content` as well
    // doubled the requests and buried the real cause among four failures
    // (found in review, 2026-09-21).
    let stub = TokenizeStub::new().failing_primary().failing_fallback();
    let base = spawn_tokenize_stub(stub.clone()).await;

    let err = tokenizer(&base)
        .try_encode("Answer:\n")
        .await
        .unwrap_err()
        .to_string();

    assert_eq!(stub.request_count(), 2, "two URL spellings, not four attempts");
    assert!(
        err.contains("\"prompt\""),
        "the reference field must be named: {err}"
    );
    assert!(
        !err.contains("content"),
        "the other dialect must not appear when it was never probed: {err}"
    );
}

#[tokio::test]
async fn a_llama_cpp_endpoint_is_reached_through_its_own_field() {
    let stub = TokenizeStub::new().llama_cpp_dialect();
    let base = spawn_tokenize_stub(stub.clone()).await;

    let ids = tokenizer(&base)
        .try_encode("Answer:\n")
        .await
        .expect("the `content` field is tried once `prompt` comes back empty");

    assert_eq!(ids, ids_of("Answer:\n"), "the real tokenization, not the empty one");
    let fields = stub.fields();
    assert_eq!(
        fields.first().map(String::as_str),
        Some("prompt"),
        "the reference field must be tried first: {fields:?}"
    );
    assert_eq!(
        fields.last().map(String::as_str),
        Some("content"),
        "the endpoint's own field must be the one that finally answers: {fields:?}"
    );
}

#[tokio::test]
async fn an_empty_tokenization_is_never_cached_as_a_prompt() {
    // The silent shape must not poison the cache: a later read of the same text
    // has to return the ids the endpoint gave for its own field.
    let stub = TokenizeStub::new().llama_cpp_dialect();
    let base = spawn_tokenize_stub(stub).await;
    let tk = tokenizer(&base);

    let first = tk.try_encode("Answer:\n").await.unwrap();
    let second = tk.try_encode("Answer:\n").await.unwrap();

    assert!(!first.is_empty() && !second.is_empty());
    assert_eq!(first, second, "the cache must hold the good encoding");
    assert_eq!(tk.encode("Answer:\n"), ids_of("Answer:\n"));
}

#[tokio::test]
async fn the_tokenization_combination_is_remembered_for_later_texts() {
    // Startup probes tens of texts (`letters` mode: 41). Re-walking the whole
    // ladder for each of them would be 123 requests where 41 suffice.
    let stub = TokenizeStub::new().llama_cpp_dialect();
    let base = spawn_tokenize_stub(stub.clone()).await;
    let tk = tokenizer(&base);

    tk.try_encode("Answer:\n").await.unwrap();
    let after_first = stub.request_count();
    tk.try_encode("B").await.unwrap();

    assert_eq!(
        stub.request_count(),
        after_first + 1,
        "the negotiated combination must be tried first for the next text"
    );
    assert_eq!(stub.fields().last().map(String::as_str), Some("content"));
}

#[tokio::test]
async fn the_error_names_every_attempt_that_was_actually_made() {
    // Hard failures on both URL spellings leave no dialect evidence: the endpoint
    // never said it ignores the field, so only the reference field is probed, and
    // the message names what was tried. Probing `content` too would double the
    // requests and list failures that were never asked for (review, 2026-09-21).
    let stub = TokenizeStub::new().failing_primary().failing_fallback();
    let base = spawn_tokenize_stub(stub).await;

    let msg = tokenizer(&base).try_encode("Answer:\n").await.unwrap_err().to_string();

    for needle in ["/tokenize", "/v1/tokenize", "[prompt]"] {
        assert!(msg.contains(needle), "{needle} missing from: {msg}");
    }
    assert!(
        !msg.contains("[content]"),
        "a field that was never probed must not be named: {msg}"
    );
}

// ---------------------------------------------------------------------------
// SlotCheck modes (`JEV_SLOT_CHECK`)
// ---------------------------------------------------------------------------

/// A vocabulary whose prompt tail tokenizes differently from the reference:
/// `"Answer:\n"` = `[16141, 510, 32]`, as probed on Qwen3-4B-Instruct-2507
/// (2026-09-21), with single-token slot letters and a stable boundary. It covers
/// the renderer's full letter range (`A`..`Z`), not just the first 20: a check
/// that stopped at `T` would miss a broken `Z`.
fn foreign_vocabulary() -> HashMap<String, Vec<u32>> {
    let prompt = vec![16141u32, 510, 32];
    let mut map = HashMap::new();
    map.insert(PROBED_PROMPT.to_string(), prompt.clone());
    for (i, letter) in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().enumerate() {
        let id = 300 + i as u32;
        map.insert(letter.to_string(), vec![id]);
        let combined: Vec<u32> = prompt.iter().copied().chain(std::iter::once(id)).collect();
        map.insert(format!("{PROBED_PROMPT}{letter}"), combined);
    }
    map
}

#[test]
fn slot_letters_follow_the_renderers_limit() {
    // Tied to `render.rs::letter_at` so the self-check cannot fall behind the
    // renderer: a stale hand-written list is how `Z` went unchecked (review).
    let letters = slot_letters();
    assert_eq!(letters.len(), jev_core::render::MAX_LETTER_SLOTS);
    assert_eq!(letters.first().map(String::as_str), Some("A"));
    assert_eq!(letters.last().map(String::as_str), Some("Z"));
}

#[test]
fn letters_mode_accepts_a_vocabulary_that_strict_rejects() {
    let v = FakeVerifier::new(foreign_vocabulary());

    let strict_err = check_probed_slot_assumption(&v).unwrap_err().to_string();
    assert!(
        strict_err.contains("15666"),
        "strict mode pins the reference ids and must say so: {strict_err}"
    );

    check_slot_letters(&v).expect("the slot property itself holds, so the readout is sound");
    SlotCheck::Letters
        .run(&v)
        .expect("the mode must dispatch to the tokenizer-independent check");
}

#[test]
fn letters_mode_still_rejects_a_multi_token_letter() {
    // `Z` on purpose: the check has to reach the *last* letter the renderer can
    // name, not stop at the twentieth.
    let mut map = foreign_vocabulary();
    map.insert("Z".to_string(), vec![325, 326]);
    let err = check_slot_letters(&FakeVerifier::new(map)).unwrap_err().to_string();
    assert!(err.contains("not a single token"), "{err}");
    assert!(err.contains('Z'), "the failing letter must be named: {err}");
}

#[tokio::test]
async fn an_empty_text_does_not_poison_the_dialect_memory() {
    // Empty text tokenizes to nothing on *every* field, so it is no evidence of a
    // dialect. Taking it as evidence sends the next real text down the ladder from
    // the wrong spelling and costs an extra round trip (review, 2026-09-21:
    // measured 5 requests instead of 3 after one empty encode).
    let stub = TokenizeStub::new().llama_cpp_dialect();
    let base = spawn_tokenize_stub(stub.clone()).await;
    let tk = tokenizer(&base);

    assert!(tk.try_encode("").await.unwrap().is_empty(), "empty in, empty out");
    let before = stub.request_count();
    tk.try_encode("Answer:\n").await.unwrap();

    assert_eq!(
        stub.request_count() - before,
        3,
        "the real text must walk the documented ladder (the field on both URL spellings, then the \
         other field) rather than start from a spelling remembered from the empty text"
    );
    assert_eq!(stub.fields().last().map(String::as_str), Some("content"));
}

#[test]
fn letters_mode_still_rejects_a_re_tokenized_tail() {
    let mut map = foreign_vocabulary();
    map.insert(format!("{PROBED_PROMPT}C"), vec![16141, 511, 32, 302]);
    let err = check_slot_letters(&FakeVerifier::new(map)).unwrap_err().to_string();
    assert!(err.contains("re-tokenized"), "{err}");
}

#[test]
fn the_mode_mapping_is_explicit_and_never_guesses() {
    assert_eq!(SlotCheck::parse(None).unwrap(), SlotCheck::Strict, "unset = strict");
    assert_eq!(SlotCheck::parse(Some("")).unwrap(), SlotCheck::Strict);
    assert_eq!(SlotCheck::parse(Some("  Strict  ")).unwrap(), SlotCheck::Strict);
    assert_eq!(SlotCheck::parse(Some("letters")).unwrap(), SlotCheck::Letters);
    let err = SlotCheck::parse(Some("letterz")).unwrap_err().to_string();
    assert!(
        err.contains("not a known mode"),
        "a typo must not silently fall back to the default: {err}"
    );
    assert!(err.contains(SlotCheck::ENV), "the variable must be named: {err}");
}

#[tokio::test]
async fn letters_mode_runs_against_a_stub_endpoint() {
    // The byte-per-token stub satisfies the tokenizer-independent property for
    // A..T, while failing the strict probe — exactly the situation `letters`
    // exists for (a foreign vocabulary whose answer slot is still readable).
    let base = spawn_tokenize_stub(TokenizeStub::new()).await;
    let tk = tokenizer(&base);

    tk.verify_slot_check(SlotCheck::Letters)
        .await
        .expect("every slot letter is one token and does not re-tokenize the tail");
    assert!(
        tk.verify_slot_check(SlotCheck::Strict).await.is_err(),
        "the same stub must still fail the strict probe (it is not the reference tokenizer)"
    );
}

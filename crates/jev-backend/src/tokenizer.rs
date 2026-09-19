//! Tokenizer access for slot verification.
//!
//! Primary path: the served endpoint exposes `POST {base_url}/tokenize`
//! (verified on SGLang: `{"tokens":[15666,25,198,32],"count":4}`), so no local
//! `tokenizer.json` is required and the tokenizer is guaranteed to be the one the
//! model actually uses. Both `/tokenize` and `/v1/tokenize` work; try `/tokenize`
//! first, fall back to `/v1/tokenize`.
//!
//! `decode_token` is only used for diagnostics and for backends that match slots by
//! token *id*. Since `/tokenize` returns ids only, the letter's surface text is
//! taken to be the letter itself (verified: appending `A` to `Answer:\n` yields the
//! single token `32`, with no whitespace variant), and readout matching additionally
//! tolerates a leading-space variant (`"A"` vs `" A"`).
//!
//! Fallback path (implement if time allows): load a local `tokenizer.json` with the
//! `tokenizers` crate when `JEV_TOKENIZER` is set.

use async_trait::async_trait;
use jev_core::SlotVerifier;
use std::sync::Mutex;

/// Slot-verification client backed by the served endpoint's `/tokenize`.
pub struct HttpTokenizer {
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) api_key: String,
    pub(crate) client: reqwest::Client,
    /// decoding a token id is not offered by `/tokenize`; cache is kept so repeated
    /// verification of the same prompt is cheap within one process.
    pub(crate) cache: Mutex<std::collections::HashMap<String, Vec<u32>>>,
}

impl HttpTokenizer {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let _ = (&base_url, &model, &api_key);
        todo!("worker B: build reqwest client (10s connect, 30s total)")
    }

    /// Encode via the endpoint, using `self.cache` for repeat prompts.
    /// Errors are surfaced by returning an empty vec only in `encode`; callers that
    /// need to distinguish failure must use `try_encode`.
    pub async fn try_encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let _ = text;
        todo!("worker B: POST /tokenize (fallback /v1/tokenize), read `tokens`")
    }
}

/// `SlotVerifier` is synchronous while the HTTP path is async, so the trait is
/// implemented by pre-fetching the encodings the verifier needs.
///
/// `HttpTokenizer::prefetch(&[text])` must be called for the prompt and for
/// `prompt + letter` for every candidate letter before `verify_slots` runs; the
/// synchronous `encode` then serves from the cache and panics only if a value is
/// missing (a programming error, not a runtime condition).
#[async_trait]
pub trait Prefetch {
    async fn prefetch(&self, texts: &[String]) -> anyhow::Result<()>;
}

impl SlotVerifier for HttpTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let _ = text;
        todo!("worker B: serve from cache; panic with a clear message if missing")
    }

    fn decode_token(&self, id: u32) -> String {
        let _ = id;
        todo!("worker B: no detokenize endpoint in M0 — return the ascii letter when id \
               round-trips in the cache, else a placeholder")
    }
}

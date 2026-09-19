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
//! Fallback path: **not implemented, and deliberately not stubbed.**
//!
//! The plan was to load a local `tokenizer.json` with the `tokenizers` crate when
//! `JEV_TOKENIZER` points at one (for endpoints that expose no `/tokenize`). That
//! crate is absent from this machine's offline cargo cache and cargo cannot reach
//! crates.io (every mirror fails its TLS handshake — `docs/dev-env.md` §1), so
//! there is no way to build it here without network access. `JEV_TOKENIZER` is
//! therefore documented in `jev-server`'s env list and read by nothing: an
//! environment variable that appears to enable a fallback while doing nothing
//! would be a lie in the one place a reader looks for truth (specs/M1.md §5③).

use async_trait::async_trait;
use jev_core::SlotVerifier;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

/// Slot-verification client backed by the served endpoint's `/tokenize`.
pub struct HttpTokenizer {
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) api_key: String,
    pub(crate) client: reqwest::Client,
    /// decoding a token id is not offered by `/tokenize`; cache is kept so repeated
    /// verification of the same prompt is cheap within one process.
    pub(crate) cache: Mutex<HashMap<String, Vec<u32>>>,
}

// ---------------------------------------------------------------------------
// The M0 slot assumption, as probed on the reference endpoint (specs/M0.md §1).
//
// The whole readout protocol rests on one fact: `"Answer:\n"` is three tokens and
// appending an uppercase letter adds *exactly one* token, leaving the prompt's own
// tokens untouched. If that does not hold, the logprob we read at position 0 after
// `Answer:\n` is not the answer slot's probability — the server would then return
// confident numbers about the wrong position. So `main.rs` checks it before
// binding the port and refuses to start when it fails (fail-fast).
// ---------------------------------------------------------------------------

/// The prompt tail whose tokenization is probed.
pub const PROBED_PROMPT: &str = "Answer:\n";
/// Probed token ids of [`PROBED_PROMPT`] (`[15666, 25, 198]`).
pub const PROBED_PROMPT_IDS: [u32; 3] = [15666, 25, 198];
/// The probed slot letter.
pub const PROBED_SLOT_LETTER: &str = "A";
/// Probed single-token id of [`PROBED_SLOT_LETTER`].
pub const PROBED_SLOT_ID: u32 = 32;

/// The two tokenize URLs to try for a base URL, in order: `(primary, fallback)`.
///
/// A base that already ends in the OpenAI-style `/v1` segment gets the bare
/// `/tokenize` as its fallback; any other base gets the `/v1`-prefixed one. This
/// keeps both spellings in play for every documented configuration without ever
/// producing a doubled segment (`/v1/v1/tokenize`).
pub fn tokenize_urls(base_url: &str) -> (String, String) {
    let base = base_url.trim_end_matches('/');
    match base.strip_suffix("/v1") {
        Some(root) => (format!("{base}/tokenize"), format!("{root}/tokenize")),
        None => (format!("{base}/tokenize"), format!("{base}/v1/tokenize")),
    }
}

/// Verify the probed slot assumption against `verifier`.
///
/// Three assertions, all fatal: the prompt tail has the probed ids, the slot letter
/// is a *single* token with the probed id, and appending it does not re-tokenize the
/// prompt (`encode(prompt + letter) == encode(prompt) ++ [slot_id]`).
///
/// The caller must have populated the verifier's cache for [`PROBED_PROMPT`],
/// [`PROBED_SLOT_LETTER`] and their concatenation first (see
/// [`HttpTokenizer::verify_slot_assumption`]); `HttpTokenizer::encode` panics on a
/// cache miss by design.
pub fn check_probed_slot_assumption(verifier: &dyn SlotVerifier) -> anyhow::Result<()> {
    let prompt_ids = verifier.encode(PROBED_PROMPT);
    if prompt_ids != PROBED_PROMPT_IDS {
        anyhow::bail!(
            "slot self-check failed: tokenizing {PROBED_PROMPT:?} gave {prompt_ids:?}, \
             expected {PROBED_PROMPT_IDS:?}"
        );
    }

    let letter_ids = verifier.encode(PROBED_SLOT_LETTER);
    if letter_ids != [PROBED_SLOT_ID] {
        anyhow::bail!(
            "slot self-check failed: {PROBED_SLOT_LETTER:?} is not the single token \
             [{PROBED_SLOT_ID}], got {letter_ids:?}"
        );
    }

    let combined = verifier.encode(&format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"));
    let expected: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(std::iter::once(PROBED_SLOT_ID))
        .collect();
    if combined != expected {
        anyhow::bail!(
            "slot self-check failed: appending {PROBED_SLOT_LETTER:?} re-tokenized the \
             prompt — {combined:?} != {expected:?} (the readout position is not the \
             answer slot)"
        );
    }

    Ok(())
}

impl HttpTokenizer {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            // A tokenizer endpoint is a prerequisite for the server to start at
            // all (fail-fast self-check in main.rs); if the client itself cannot
            // be built, there is no sane fallback.
            .expect("failed to build reqwest client");
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: api_key.into(),
            client,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Encode via the endpoint, using `self.cache` for repeat prompts.
    /// The synchronous `SlotVerifier::encode` serves from this same cache and
    /// panics on a miss; use this async method (or `prefetch`) to populate it.
    pub async fn try_encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        if let Some(hit) = self.cache.lock().unwrap().get(text) {
            return Ok(hit.clone());
        }
        let tokens = self.encode_one(text).await?;
        self.cache.lock().unwrap().insert(text.to_string(), tokens.clone());
        Ok(tokens)
    }

    /// Fail-fast startup self-check: fetch the three texts the probed slot
    /// assumption needs and run [`check_probed_slot_assumption`] on them.
    ///
    /// A server that cannot pass this must not answer a single request — a wrong
    /// tokenizer silently moves the readout position and every probability it
    /// returns is about some other token.
    pub async fn verify_slot_assumption(&self) -> anyhow::Result<()> {
        let combined = format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}");
        self.prefetch(&[
            PROBED_PROMPT.to_string(),
            PROBED_SLOT_LETTER.to_string(),
            combined,
        ])
        .await?;
        check_probed_slot_assumption(self)
    }

    /// POST to `{base}/tokenize`; on a non-2xx or parse failure, retry the *other*
    /// spelling of the tokenize path. Returns the `tokens` array.
    ///
    /// Both spellings are live-verified on the reference endpoint
    /// (specs/M0.md §1). Which one is "the other" depends on the base URL: a
    /// base that already carries the OpenAI-style `/v1` segment (the documented
    /// default, `http://10.10.10.115:8014/v1`) makes the primary `{base}/tokenize`
    /// — i.e. `/v1/tokenize` — so the fallback must drop the segment rather than
    /// append a second one. Appending blindly would produce `/v1/v1/tokenize`,
    /// which no server serves.
    async fn encode_one(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let (primary, fallback) = tokenize_urls(&self.base_url);
        // The request field name is load-bearing, not cosmetic. Live probes on
        // 115:8014 (both `/v1/tokenize` and `/tokenize`, 2026-09-19):
        //   {"model": .., "prompt": "Answer:\n"} -> 200 {"tokens":[15666,25,198],..}
        //   {"model": .., "text":   "Answer:\n"} -> 400 "Exactly one of 'prompt' or
        //                                           'messages' must be provided."
        // Sending `text` makes every tokenize call fail, which used to be invisible
        // because the tests all ran against a stub that did not check the body.
        let body = serde_json::json!({"model": self.model, "prompt": text});

        match self.encode_at(&self.client, &primary, &body).await {
            Ok(ids) => Ok(ids),
            Err(primary_err) => {
                tracing::warn!("primary tokenize endpoint {primary} failed ({primary_err}); falling back to {fallback}");
                self.encode_at(&self.client, &fallback, &body).await.map_err(|e| {
                    anyhow::anyhow!("tokenize failed on {primary} ({primary_err}) and on {fallback} ({e})")
                })
            }
        }
    }

    /// One request to `url`; parses the `tokens` array from a successful response.
    async fn encode_at(
        &self,
        client: &reqwest::Client,
        url: &str,
        body: &serde_json::Value,
    ) -> anyhow::Result<Vec<u32>> {
        let resp = client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let t = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "tokenize endpoint {url} returned {status}: {}",
                // Truncate by `char`, not by byte: a byte slice panics when the
                // cut lands inside a multi-byte character (error bodies are often
                // non-ASCII).
                t.chars().take(500).collect::<String>()
            );
        }
        let v: serde_json::Value = resp.json().await?;
        let tokens = v
            .get("tokens")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("tokenize response has no `tokens` array"))?;
        let ids = tokens
            .iter()
            .map(|n| n.as_u64())
            .collect::<Option<Vec<u64>>>()
            .ok_or_else(|| anyhow::anyhow!("tokenize `tokens` entries are not integers"))?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<u32>, _>>()?;
        Ok(ids)
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

#[async_trait]
impl Prefetch for HttpTokenizer {
    async fn prefetch(&self, texts: &[String]) -> anyhow::Result<()> {
        for text in texts {
            self.try_encode(text).await?;
        }
        Ok(())
    }
}

impl SlotVerifier for HttpTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self.cache.lock().unwrap().get(text) {
            Some(ids) => ids.clone(),
            None => panic!(
                "HttpTokenizer::encode({text:?}): text is not in the cache — you must \
                 call `prefetch` (or `try_encode`) for this text before running the \
                 synchronous slot verification"
            ),
        }
    }

    /// M0 has no detokenize endpoint, so an id can only be identified when it was
    /// already observed: a *single-token* encoding in the cache whose text is a
    /// single uppercase letter decodes back to that letter. Anything else is
    /// `"<unk>"` (diagnostics only).
    fn decode_token(&self, id: u32) -> String {
        let cache = self.cache.lock().unwrap();
        for (text, ids) in cache.iter() {
            if ids.len() == 1 && ids[0] == id && text.len() == 1 {
                if let Some(c) = text.chars().next() {
                    if c.is_ascii_uppercase() {
                        return text.clone();
                    }
                }
            }
        }
        "<unk>".to_string()
    }
}

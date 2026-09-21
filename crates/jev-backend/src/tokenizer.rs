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
    /// The `(url, field)` combination that last produced a tokenization, tried
    /// first on the next text (see `encode_one`).
    pub(crate) dialect: Mutex<Option<(String, String)>>,
}

/// The failure message for the one shape that must never be mistaken for a
/// result: a 200 whose `tokens` array is empty for a non-empty input.
fn empty_tokens_note(url: &str, field: &str) -> String {
    format!("{url} [{field}] -> 200 with an empty `tokens` array (the field was not honoured)")
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

/// The request field names tried for the tokenize text, in order.
///
/// `prompt` is the *reference* endpoint's field and stays first because it is
/// load-bearing there (specs/M0.md §1: sending `text` gives
/// 400 "Exactly one of 'prompt' or 'messages' must be provided"). `content` is
/// llama.cpp's field (probed 2026-09-21).
///
/// Why a fallback rather than a config knob: llama.cpp does **not** reject an
/// unknown field. It answers `200 {"tokens": []}` — a silently empty
/// tokenization. That is the worst possible failure shape here, because an empty
/// id list cached under a non-empty text moves the readout position without any
/// error anywhere. So an empty `tokens` array for a non-empty input is treated as
/// "the field was not honoured" and the next spelling is tried.
pub const TOKENIZE_FIELDS: [&str; 2] = ["prompt", "content"];

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

/// Every slot letter the renderer can name: `A`..`Z`.
///
/// Derived from [`jev_core::render::MAX_LETTER_SLOTS`] rather than typed out, so
/// the self-check cannot drift behind the renderer: a hand-written `A`..`T` list
/// silently stopped checking the last six letters, and a multi-token `Z` is
/// exactly the bug this check exists to catch (found in review, 2026-09-21 —
/// `letter_at()` in `render.rs` goes up to `MAX_LETTER_SLOTS` = 26).
pub fn slot_letters() -> Vec<String> {
    (0..jev_core::render::MAX_LETTER_SLOTS)
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect()
}

/// How strictly the startup slot self-check verifies the tokenizer
/// (env `JEV_SLOT_CHECK`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotCheck {
    /// Default. Asserts the probed ids of specs/M0.md §1 *exactly*: the prompt
    /// tail is `[15666, 25, 198]` and the slot letter `A` is id 32. Tied to the
    /// reference vocabulary (Qwen3.8-27B, and anything sharing its tokenizer —
    /// including the ternary-Bonsai 27B build).
    Strict,
    /// Asserts only what the readout protocol actually requires, for *any*
    /// vocabulary: every slot letter is a single token, and appending it does not
    /// re-tokenize the prompt tail.
    ///
    /// `Strict` additionally pins how the prompt tail itself tokenizes, which the
    /// readout never depends on. Probed 2026-09-21 against Qwen3-4B-Instruct-2507:
    /// `"Answer:\n"` = `[16141, 510, 32]` — different ids, because `\n` merges
    /// into the same token as `Answer:` — while the slot property still holds
    /// exactly. `Strict` therefore refused to start a server whose probabilities
    /// are perfectly readable, and the refusal read as "this model cannot serve",
    /// which is a worse lie than the looseness it avoided.
    Letters,
}

impl SlotCheck {
    /// Environment variable selecting the mode; unset means [`SlotCheck::Strict`].
    pub const ENV: &'static str = "JEV_SLOT_CHECK";

    /// Read the mode from the environment. An unknown value is a startup error,
    /// never a silent fallback to the default: a typo must not quietly change
    /// which tokenizer assumptions are enforced.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(std::env::var(Self::ENV).ok().as_deref())
    }

    /// The pure half of [`Self::from_env`]: `None` (unset) means [`Self::Strict`].
    /// Split out so tests can exercise the mapping without mutating the process
    /// environment (which is shared by every test in the binary).
    pub fn parse(raw: Option<&str>) -> anyhow::Result<Self> {
        match raw {
            None => Ok(Self::Strict),
            Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "" | "strict" | "probed" => Ok(Self::Strict),
                "letters" | "slots" => Ok(Self::Letters),
                other => anyhow::bail!(
                    "{}={other:?} is not a known mode (use `strict` or `letters`)",
                    Self::ENV
                ),
            },
        }
    }

    /// The texts that must be in the verifier's cache for [`Self::run`].
    pub fn prefetch_texts(&self) -> Vec<String> {
        match self {
            Self::Strict => vec![
                PROBED_PROMPT.to_string(),
                PROBED_SLOT_LETTER.to_string(),
                format!("{PROBED_PROMPT}{PROBED_SLOT_LETTER}"),
            ],
            Self::Letters => {
                let mut texts = vec![PROBED_PROMPT.to_string()];
                for letter in slot_letters() {
                    texts.push(format!("{PROBED_PROMPT}{letter}"));
                    texts.push(letter);
                }
                texts
            }
        }
    }

    /// Run the check. Fatal by design (fail-fast, specs/M0.md §4 B3).
    pub fn run(&self, verifier: &dyn SlotVerifier) -> anyhow::Result<()> {
        match self {
            Self::Strict => check_probed_slot_assumption(verifier),
            Self::Letters => check_slot_letters(verifier),
        }
    }
}

/// The tokenizer-independent half of the slot assumption: every slot letter is a
/// *single* token, and appending it to the prompt tail does not re-tokenize it
/// (`encode(prompt + letter) == encode(prompt) ++ [letter_id]`).
///
/// Those two facts are what the readout needs. If a letter were multi-token, the
/// logprob at the position after the prompt would be about the first *piece* of
/// the answer, not the answer; if appending re-tokenized the tail, the position
/// would move. Which ids the tail gets is irrelevant to that.
pub fn check_slot_letters(verifier: &dyn SlotVerifier) -> anyhow::Result<()> {
    let prompt_ids = verifier.encode(PROBED_PROMPT);
    if prompt_ids.is_empty() {
        anyhow::bail!("slot self-check failed: {PROBED_PROMPT:?} tokenizes to nothing");
    }

    for letter in slot_letters() {
        let ids = verifier.encode(&letter);
        if ids.len() != 1 {
            anyhow::bail!(
                "slot self-check failed: slot letter {letter:?} is not a single token — \
                 got {ids:?} (the readout would then be about part of a letter)"
            );
        }
        let expected: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(std::iter::once(ids[0]))
            .collect();
        let combined = verifier.encode(&format!("{PROBED_PROMPT}{letter}"));
        if combined != expected {
            anyhow::bail!(
                "slot self-check failed: appending {letter:?} re-tokenized the prompt — \
                 {combined:?} != {expected:?} (the readout position is not the answer slot)"
            );
        }
    }

    Ok(())
}

impl HttpTokenizer {
    /// Fail-fast startup self-check under an explicit [`SlotCheck`] mode: fetch
    /// the texts that mode needs, then run it.
    pub async fn verify_slot_check(&self, mode: SlotCheck) -> anyhow::Result<()> {
        self.prefetch(&mode.prefetch_texts()).await?;
        mode.run(self)
    }

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
            dialect: Mutex::new(None),
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
        self.verify_slot_check(SlotCheck::Strict).await
    }

    /// POST the text until one `(field, url)` combination answers with a usable
    /// `tokens` array, in two phases:
    ///
    /// 1. [`TOKENIZE_FIELDS`]`[0]` (`prompt`) on both URL spellings — the
    ///    documented reference path. Its behaviour is unchanged from M0: the
    ///    primary URL first, the other spelling second.
    /// 2. [`TOKENIZE_FIELDS`]`[1]` (`content`) on both spellings — llama.cpp's
    ///    field. Reached only when phase 1 did not succeed, which is either the
    ///    silent shape (a 200 with an empty `tokens` array for non-empty text) or
    ///    a total failure (an endpoint that speaks only the other dialect must not
    ///    hide behind a URL spelling that happens to be missing).
    ///
    /// Both URL spellings are live-verified on the reference endpoint
    /// (specs/M0.md §1). Which one is "the other" depends on the base URL: a
    /// base that already carries the OpenAI-style `/v1` segment (the documented
    /// default, `http://10.10.10.115:8014/v1`) makes the primary `{base}/tokenize`
    /// — i.e. `/v1/tokenize` — so the fallback must drop the segment rather than
    /// append a second one. Appending blindly would produce `/v1/v1/tokenize`,
    /// which no server serves.
    async fn encode_one(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let (primary, fallback) = tokenize_urls(&self.base_url);
        let mut attempts: Vec<String> = Vec::new();

        // Once a combination has worked, it is tried first: startup probes tens of
        // texts (`letters` mode: 41), and an endpoint that needs the fallback
        // would otherwise re-walk the entire ladder for every one of them.
        let memo = self.dialect.lock().unwrap().clone();
        if let Some((url, field)) = memo {
            match self.attempt(&url, &field, text).await {
                Ok(Some(ids)) => return Ok(ids),
                Ok(None) => attempts.push(empty_tokens_note(&url, &field)),
                Err(e) => attempts.push(format!("{url} [{field}] (remembered) -> {e}")),
            }
        }

        // Phase 1 — the reference dialect: its field on both URL spellings. This
        // is M0's behaviour, unchanged, and it wins on the reference endpoint.
        let mut field_ignored = false;
        for url in [&primary, &fallback] {
            match self.attempt(url, TOKENIZE_FIELDS[0], text).await {
                Ok(Some(ids)) => return Ok(ids),
                Ok(None) => {
                    field_ignored = true;
                    attempts.push(empty_tokens_note(url, TOKENIZE_FIELDS[0]));
                }
                Err(e) => attempts.push(format!("{url} [{}] -> {e}", TOKENIZE_FIELDS[0])),
            }
        }

        // An endpoint that failed *every* phase-1 attempt (404, 5xx, unreachable)
        // never told us it ignores the field, so this is not a dialect difference:
        // stop here with M0's semantics — two requests, one `Http` error that
        // mentions only the reference field. Walking on would double the request
        // count on a failing endpoint and bury the real cause among four
        // failures (found in review, 2026-09-21: the reference endpoint's failure
        // path went from 2 requests to 4).
        if !field_ignored {
            anyhow::bail!(
                "tokenize failed on both URL spellings with the reference field {:?}: {}",
                TOKENIZE_FIELDS[0],
                attempts.join("; ")
            );
        }

        // Phase 2 — the other dialect's field (llama.cpp's `content`), same two
        // spellings. Reached only when the endpoint answered phase 1 and ignored
        // the field (the silent-ignore shape, e.g. `200 {"tokens":[]}`).
        for url in [&primary, &fallback] {
            match self.attempt(url, TOKENIZE_FIELDS[1], text).await {
                Ok(Some(ids)) => {
                    tracing::info!(
                        "tokenize: {url} answers the {:?} field (not {:?}) — this endpoint's \
                         dialect differs from the reference",
                        TOKENIZE_FIELDS[1],
                        TOKENIZE_FIELDS[0]
                    );
                    return Ok(ids);
                }
                Ok(None) => attempts.push(empty_tokens_note(url, TOKENIZE_FIELDS[1])),
                Err(e) => attempts.push(format!("{url} [{}] -> {e}", TOKENIZE_FIELDS[1])),
            }
        }

        anyhow::bail!(
            "tokenize failed on every URL/field combination: {}",
            attempts.join("; ")
        )
    }

    /// One `(url, field)` attempt: build the body, post it, and remember the
    /// combination when it produces a tokenization.
    async fn attempt(
        &self,
        url: &str,
        field: &str,
        text: &str,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let body = serde_json::json!({"model": self.model, field: text});
        let ids = self.encode_field(url, &body, text).await?;
        // Only a *non-empty* answer identifies the dialect. Empty text
        // legitimately tokenizes to nothing on every field, so memoising there
        // would remember whichever spelling happened to be tried first and send
        // the next non-empty text down the ladder from the wrong starting point
        // (found in review, 2026-09-21: 5 requests instead of 3 after one empty
        // encode). A non-empty text that yields nothing is already reported as
        // "field not honoured" (`Ok(None)`), which never reaches here.
        if ids.as_ref().is_some_and(|ids| !ids.is_empty()) {
            *self.dialect.lock().unwrap() = Some((url.to_string(), field.to_string()));
        }
        Ok(ids)
    }

    /// One `(url, field)` attempt. `Ok(None)` means the endpoint answered 200
    /// with an *empty* `tokens` array for non-empty text — the shape llama.cpp
    /// returns when it does not know the field name. That is not a tokenization,
    /// and caching it would silently move the readout position, so it is reported
    /// as "field not honoured" rather than as a result.
    async fn encode_field(
        &self,
        url: &str,
        body: &serde_json::Value,
        text: &str,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let ids = self.encode_at(&self.client, url, body).await?;
        if ids.is_empty() && !text.is_empty() {
            return Ok(None);
        }
        Ok(Some(ids))
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

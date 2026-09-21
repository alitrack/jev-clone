//! OpenAI-compatible readout backend (raw completion + `logprobs`).
//!
//! Works against any server exposing `POST {base_url}/completions` with the
//! `logprobs` parameter: SGLang (verified), vLLM, llama.cpp's server, LM Studio.
//! Not usable against endpoints that reject `logprobs` (e.g. NInfer) — those fail
//! loudly with `logprobs_not_supported`, which we surface as `BackendError::Http`.

use crate::{BackendError, DecisionBackend, Readout};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    /// e.g. `http://10.10.10.115:8014/v1`
    pub base_url: String,
    /// model id as the server knows it, e.g. `qwen3.8-27b`
    pub model: String,
    pub api_key: String,
    pub timeout_secs: u64,
}

pub struct OpenAiCompatBackend {
    cfg: OpenAiCompatConfig,
    client: reqwest::Client,
    /// How many batched readouts were served by the sequential fallback because
    /// the endpoint rejected an array `prompt` (see `token_logprobs_batch`).
    batch_fallbacks: AtomicU64,
}

impl OpenAiCompatBackend {
    pub fn new(cfg: OpenAiCompatConfig) -> Result<Self, BackendError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| BackendError::Http(format!("failed to build http client: {e}")))?;
        Ok(Self {
            cfg,
            client,
            batch_fallbacks: AtomicU64::new(0),
        })
    }

    /// Number of batched calls served by the sequential fallback (see
    /// [`DecisionBackend::token_logprobs_batch`]).
    ///
    /// Non-zero means the endpoint cannot take an array `prompt` at all, so a
    /// "shared-prefix" number measured against it rests on the server's own
    /// per-request cache — not on one shared prefill. Any report quoting such a
    /// number must say so, or the number claims an optimisation that never ran.
    pub fn batch_fallbacks(&self) -> u64 {
        self.batch_fallbacks.load(Ordering::Relaxed)
    }

    fn completions_url(&self) -> String {
        format!("{}/completions", self.cfg.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl DecisionBackend for OpenAiCompatBackend {
    /// POST to the *raw* completion endpoint with `max_tokens: 1` and read the
    /// top-k distribution at position 0. Raw completion is mandated (not chat):
    /// a chat template can prepend thinking tokens that would consume the single
    /// generated position (observed live: the model answered `We` instead of the
    /// answer slot). The one generated token's text is discarded — we only read
    /// the distribution.
    async fn token_logprobs(&self, prompt: &str, top_k: usize) -> Result<Readout, BackendError> {
        // Body fields are load-bearing (see specs/M0.md §1): the endpoint was
        // probed with exactly this shape.
        let body = serde_json::json!({
            "model": self.cfg.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": top_k,
        });

        let resp = self
            .client
            .post(self.completions_url())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| BackendError::Http(format!("request to completions endpoint failed: {e}")))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            BackendError::Http(format!("failed to read response body (status {status}): {e}"))
        })?;
        if !status.is_success() {
            // Keep the first 500 *characters* of the body: it is how callers
            // identify `logprobs_not_supported` endpoints and other server
            // errors. Truncation is by `char`, never by byte: a byte slice
            // would panic when the cut lands inside a multi-byte UTF-8
            // character, and error bodies routinely carry non-ASCII text.
            return Err(BackendError::Http(format!(
                "completions endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>()
            )));
        }

        // Parse the response. Every structural failure is a Decode error — the
        // endpoint answered but not with a shape we understand.
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| BackendError::Decode(format!("invalid JSON: {e}")))?;

        let choice = v
            .get("choices")
            .and_then(|c| c.get(0))
            .ok_or(BackendError::NoLogprobs)?;
        let top = choice_top_logprobs(choice)?;

        let prompt_tokens = v
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0);

        Ok(Readout {
            top,
            prompt_tokens,
            // We requested exactly one generated token (it only exists so the
            // server reports the distribution at that position; its text is
            // discarded).
            completion_tokens: 1,
        })
    }

    /// One POST for N prompts: `prompt` is an **array** (supported by the probe,
    /// specs/M1.md §1 — 3 prompts sharing a `state` prefix returned 3 choices in
    /// less wall time than 1 single request, i.e. the server prefills the shared
    /// prefix once).
    ///
    /// All-or-nothing by design: a response with a different number of choices
    /// than prompts, choices whose `index` fields are not a permutation of
    /// `0..N`, or any element without usable logprobs fails the whole call. A
    /// partial success would silently mis-align prompts and probabilities.
    async fn token_logprobs_batch(
        &self,
        prompts: &[String],
        top_k: usize,
    ) -> Result<Vec<Readout>, BackendError> {
        // Nothing to ask for. The endpoint rejects an empty `prompt`, and the
        // server only ever calls this with >= 1 prompt.
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "model": self.cfg.model,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": top_k,
        });

        let resp = self
            .client
            .post(self.completions_url())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| BackendError::Http(format!("request to completions endpoint failed: {e}")))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            BackendError::Http(format!("failed to read response body (status {status}): {e}"))
        })?;
        if !status.is_success() {
            // A *client* error means the endpoint refuses this request shape at
            // all — llama.cpp rejects an array `prompt` with
            // `400 {"error":{"message":"type must be string, but is an array"}}`
            // (probed 2026-09-21). That is a capability gap, not a wrong answer,
            // so fall back to the sequential path the trait's default already
            // provides instead of failing the run. Server faults (5xx) are
            // propagated unchanged: re-asking a broken server N times helps
            // nobody, and the caller must see the endpoint is down.
            let body = text.chars().take(500).collect::<String>();
            // Only a *request-shape* rejection means "this endpoint cannot batch":
            // 400 for llama.cpp's `type must be string, but is an array`, plus the
            // other shape-level codes. A 429 (rate limit), 401/403 (auth) or
            // 408/413 is about the *call*, not the shape — falling back there would
            // turn a throttled or unauthorised endpoint into a "successful"
            // benchmark run (found in review, 2026-09-21: an array 429 was silently
            // answered by three sequential requests and reported as readouts).
            if matches!(status.as_u16(), 400 | 404 | 405 | 415 | 422) {
                tracing::warn!(
                    "batched readout rejected by {} with {status}: {body} — falling back to {} \
                     sequential single-prompt requests (no shared prefill on this endpoint)",
                    self.completions_url(),
                    prompts.len()
                );
                self.batch_fallbacks.fetch_add(1, Ordering::Relaxed);
                let mut readouts = Vec::with_capacity(prompts.len());
                for prompt in prompts {
                    readouts.push(self.token_logprobs(prompt, top_k).await?);
                }
                return Ok(readouts);
            }
            return Err(BackendError::Http(format!(
                "completions endpoint returned {status}: {body}"
            )));
        }

        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| BackendError::Decode(format!("invalid JSON: {e}")))?;

        let choices = v
            .get("choices")
            .and_then(|c| c.as_array())
            .ok_or_else(|| BackendError::Decode("response has no `choices` array".to_string()))?;
        if choices.len() != prompts.len() {
            return Err(BackendError::Decode(format!(
                "endpoint returned {} choices for {} prompts (a batched readout is \
                 all-or-nothing — refusing to guess which answer belongs to which prompt)",
                choices.len(),
                prompts.len()
            )));
        }

        // `index` is the server's own statement of which prompt a choice belongs
        // to; the wire order is not guaranteed to match the request order. A
        // choice without `index` is taken at its array position (some
        // OpenAI-compatible servers omit it for single-element batches).
        let mut indexed: Vec<(usize, &serde_json::Value)> = Vec::with_capacity(choices.len());
        for (pos, choice) in choices.iter().enumerate() {
            let index = match choice.get("index") {
                Some(n) => n.as_u64().ok_or_else(|| {
                    BackendError::Decode(format!("choice {pos} has a non-integer `index`: {n}"))
                })? as usize,
                None => pos,
            };
            indexed.push((index, choice));
        }
        indexed.sort_by_key(|(index, _)| *index);
        for (pos, (index, _)) in indexed.iter().enumerate() {
            if *index != pos {
                return Err(BackendError::Decode(format!(
                    "batched choices' `index` fields are not a permutation of 0..{}: \
                     saw {index} at position {pos} after sorting",
                    prompts.len()
                )));
            }
        }

        // The endpoint reports ONE usage block for the whole batch, and backends
        // disagree on its size: the reference endpoint counted it as N x prefix
        // (the radix cache does not discount it — specs/M1.md §1), while
        // llama.cpp b11065 reported the shared prefix once (909 vs 18933 for the
        // same work, measured 2026-09-21). We reinterpret it neither way.
        // It is attached to the first readout so that summing the vector
        // reproduces that number exactly; the remaining readouts report 0 rather
        // than a fake per-prompt split. The public `usage.input_tokens` therefore
        // keeps M0's meaning: the endpoint's own prompt-token accounting for the
        // request.
        let batch_prompt_tokens = v
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0);

        let mut readouts = Vec::with_capacity(prompts.len());
        for (position, (_, choice)) in indexed.into_iter().enumerate() {
            readouts.push(Readout {
                top: choice_top_logprobs(choice)?,
                prompt_tokens: if position == 0 { batch_prompt_tokens } else { 0 },
                completion_tokens: 1,
            });
        }
        Ok(readouts)
    }

    fn model_name(&self) -> String {
        self.cfg.model.clone()
    }
}

/// Extract one choice's top-k distribution into a [`BTreeMap`], accepting both
/// wire shapes that are live in the wild.
///
/// * **legacy map** (SGLang / vLLM, probed 2026-09-19): `logprobs.top_logprobs[0]`
///   is already a token-text -> logprob object. This is the shape specs/M0.md §1
///   froze, so it is tried first and its behaviour is unchanged.
/// * **content array** (llama.cpp's `/v1/completions`, probed 2026-09-21):
///   `logprobs.content[0].top_logprobs` is an *array* of
///   `{"token": "A", "logprob": -0.229, "bytes": [65]}` objects, and the legacy
///   key is absent. The old parser answered `NoLogprobs` to that, which made a
///   fully supported endpoint (llama.cpp has source-level `logprobs`, and its
///   server emits them) look like one that cannot serve a readout at all.
///
/// Shared by the single-prompt and batched paths so both agree exactly on which
/// shapes are `NoLogprobs` (field absent / `null` / empty distribution) and which
/// are `Decode` (a logprob that is not a number). A missing map is `NoLogprobs`
/// rather than an empty distribution because an empty distribution would make
/// every declared slot look absent — the caller must not be able to mistake
/// "the endpoint sent nothing" for "the model put no mass anywhere".
fn choice_top_logprobs(choice: &serde_json::Value) -> Result<BTreeMap<String, f64>, BackendError> {
    let logprobs = choice.get("logprobs").ok_or(BackendError::NoLogprobs)?;

    // Shape 1: `logprobs.top_logprobs[0]` is a token-text -> logprob object.
    if let Some(map) = logprobs
        .get("top_logprobs")
        .and_then(|t| t.get(0))
        .and_then(|t| t.as_object())
    {
        if !map.is_empty() {
            let mut top = BTreeMap::new();
            for (token, logprob) in map {
                top.insert(token.clone(), as_logprob(token, logprob)?);
            }
            return Ok(top);
        }
    }

    // Shape 2: `logprobs.content[0].top_logprobs` is an array of token objects.
    if let Some(entries) = logprobs
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|e| e.get("top_logprobs"))
        .and_then(|t| t.as_array())
    {
        if !entries.is_empty() {
            let mut top = BTreeMap::new();
            for entry in entries {
                let token = entry.get("token").and_then(|t| t.as_str()).ok_or_else(|| {
                    BackendError::Decode(format!(
                        "`logprobs.content[0].top_logprobs` entry has no `token` string: {entry}"
                    ))
                })?;
                let logprob = entry.get("logprob").ok_or_else(|| {
                    BackendError::Decode(format!(
                        "`logprobs.content[0].top_logprobs` entry has no `logprob`: {entry}"
                    ))
                })?;
                top.insert(token.to_string(), as_logprob(token, logprob)?);
            }
            return Ok(top);
        }
    }

    Err(BackendError::NoLogprobs)
}

/// One logprob value, with the offending token text in the error message.
fn as_logprob(token: &str, logprob: &serde_json::Value) -> Result<f64, BackendError> {
    logprob.as_f64().ok_or_else(|| {
        BackendError::Decode(format!("logprob for token {token:?} is not a number"))
    })
}

//! OpenAI-compatible readout backend (raw completion + `logprobs`).
//!
//! Works against any server exposing `POST {base_url}/completions` with the
//! `logprobs` parameter: SGLang (verified), vLLM, llama.cpp's server, LM Studio.
//! Not usable against endpoints that reject `logprobs` (e.g. NInfer) — those fail
//! loudly with `logprobs_not_supported`, which we surface as `BackendError::Http`.

use crate::{BackendError, DecisionBackend, Readout};
use async_trait::async_trait;
use std::collections::BTreeMap;
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
}

impl OpenAiCompatBackend {
    pub fn new(cfg: OpenAiCompatConfig) -> Result<Self, BackendError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| BackendError::Http(format!("failed to build http client: {e}")))?;
        Ok(Self { cfg, client })
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
            // Same char-safe truncation as the single-prompt path: error bodies
            // routinely carry non-ASCII text, and a byte cut would panic.
            return Err(BackendError::Http(format!(
                "completions endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>()
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

        // The endpoint reports ONE usage block for the whole batch, counted as
        // N x prefix (the radix cache does not discount it — specs/M1.md §1).
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

/// Extract one choice's top-k distribution (`logprobs.top_logprobs[0]`, a token
/// text -> logprob map) into a [`BTreeMap`].
///
/// Shared by the single-prompt and batched paths so both agree exactly on which
/// shapes are `NoLogprobs` (field absent / `null` / empty map) and which are
/// `Decode` (a logprob that is not a number). A missing map is `NoLogprobs`
/// rather than an empty distribution because an empty distribution would make
/// every declared slot look absent — the caller must not be able to mistake
/// "the endpoint sent nothing" for "the model put no mass anywhere".
fn choice_top_logprobs(choice: &serde_json::Value) -> Result<BTreeMap<String, f64>, BackendError> {
    let top_logprobs = choice
        .get("logprobs")
        .and_then(|l| l.get("top_logprobs"))
        .and_then(|t| t.get(0))
        .and_then(|t| t.as_object())
        .ok_or(BackendError::NoLogprobs)?;
    if top_logprobs.is_empty() {
        return Err(BackendError::NoLogprobs);
    }

    let mut top = BTreeMap::new();
    for (token, logprob) in top_logprobs {
        let lp = logprob.as_f64().ok_or_else(|| {
            BackendError::Decode(format!("logprob for token {token:?} is not a number"))
        })?;
        top.insert(token.clone(), lp);
    }
    Ok(top)
}

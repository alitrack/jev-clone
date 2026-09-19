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

        let top_logprobs = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("logprobs"))
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

    fn model_name(&self) -> String {
        self.cfg.model.clone()
    }
}

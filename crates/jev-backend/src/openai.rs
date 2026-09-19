//! OpenAI-compatible readout backend (raw completion + `logprobs`).
//!
//! Works against any server exposing `POST {base_url}/completions` with the
//! `logprobs` parameter: SGLang (verified), vLLM, llama.cpp's server, LM Studio.
//! Not usable against endpoints that reject `logprobs` (e.g. NInfer) — those fail
//! loudly with `logprobs_not_supported`, which we surface as `BackendError::Http`.

use crate::{BackendError, DecisionBackend, Readout};
use async_trait::async_trait;

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
        let _ = cfg;
        todo!("worker B: build the client (reqwest, with connect/total timeouts)")
    }
}

#[async_trait]
impl DecisionBackend for OpenAiCompatBackend {
    async fn token_logprobs(&self, prompt: &str, top_k: usize) -> Result<Readout, BackendError> {
        let _ = (prompt, top_k);
        todo!(
            "worker B: POST <base_url>/completions with \
             {{\"model\":..,\"prompt\":prompt,\"max_tokens\":1,\"temperature\":0,\"logprobs\":top_k}} \
             then read choices[0].logprobs.top_logprobs[0] (token text -> logprob). \
             Empty/null logprobs => BackendError::NoLogprobs. Include usage.prompt_tokens."
        )
    }

    fn model_name(&self) -> String {
        self.cfg.model.clone()
    }
}

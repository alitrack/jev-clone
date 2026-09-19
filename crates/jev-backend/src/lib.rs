//! jev-backend — pluggable readout backends.
//!
//! A backend answers exactly one question: *what is the log-probability the served
//! model assigns to each candidate token at the position right after the prompt?*
//! Nothing else. Everything above it (rendering, slot verification, probability
//! math, the HTTP contract) is backend-agnostic, so the same server runs against a
//! local llama.cpp process, a SGLang/vLLM endpoint, or a mock in tests.
//!
//! ## Readout protocol (verified on the live endpoint, 2026-09-19)
//!
//! `POST <base>/completions` with `max_tokens: 1, temperature: 0, logprobs: K`
//! returns, for that single position, the top-K tokens with their log-probabilities
//! (`choices[0].logprobs.top_logprobs[0]`, a token-text -> logprob map).
//!
//! Deliberate choices, both learned the hard way:
//! * **raw completion, not chat.** A chat template can prepend a thinking scaffold,
//!   and the mandated thinking tokens consume the single generated position
//!   (observed: the model answered `We` instead of the answer slot). Raw completion
//!   has no template, so position 0 after `Answer:\n` *is* the answer slot.
//! * **exactly one token is generated, and its text is discarded.** We never parse
//!   prose; we read a distribution. Backends that can do a pure prefill (llama.cpp,
//!   candle) must override this and return the same shape.
//!
//! Known endpoint limitation: the NInfer server refuses `logprobs=true`
//! (`logprobs_not_supported`). Such an endpoint cannot serve as a readout backend —
//! it is only usable as a generation baseline for comparison.

use async_trait::async_trait;
use jev_core::CoreError;
use std::collections::BTreeMap;
use thiserror::Error;

pub mod mock;
pub mod openai;
pub mod tokenizer;

pub use mock::MockBackend;
pub use openai::{OpenAiCompatBackend, OpenAiCompatConfig};
pub use tokenizer::{HttpTokenizer, Prefetch};

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("http error: {0}")]
    Http(String),
    #[error("backend returned no top_logprobs (endpoint may not support logprobs)")]
    NoLogprobs,
    #[error("backend response could not be parsed: {0}")]
    Decode(String),
    #[error(transparent)]
    Core(#[from] CoreError),
}

/// One readout: the top-K token log-probabilities at the answer position.
#[derive(Debug, Clone, PartialEq)]
pub struct Readout {
    /// token text -> log-probability
    pub top: BTreeMap<String, f64>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Readout {
    /// Log-sum-exp of every key that denotes `slot_token`, tolerating the leading /
    /// trailing whitespace variants the tokenizer treats as distinct tokens
    /// (`"A"` vs `" A"`). Returns `None` when the slot never appears in the top-k.
    ///
    /// Both variants are *added* rather than compared because they are alternative
    /// surface forms of the same slot: the probability of "the model answers A" is
    /// the total mass it puts on any spelling of A.
    pub fn slot_logprob(&self, slot_token: &str) -> Option<f64> {
        // Nested helper: log(e^a + e^b), the stable way (shift by the max).
        fn lse(a: f64, b: f64) -> f64 {
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        }
        // Every key whose strip() equals the slot denotes the same slot ("A" and
        // " A" are two surface forms of one token boundary), so their logprobs
        // are ADDED in log space (log-sum-exp) — never compared.
        let target = slot_token.trim();
        let mut acc: Option<f64> = None;
        for (text, lp) in &self.top {
            if text.trim() == target {
                acc = Some(match acc {
                    Some(a) => lse(a, *lp),
                    None => *lp,
                });
            }
        }
        acc
    }

}

#[async_trait]
pub trait DecisionBackend: Send + Sync {
    /// Log-probabilities at the position following `prompt`, for the top `top_k`
    /// tokens. Implementations must not add chat templates or thinking scaffolds.
    async fn token_logprobs(&self, prompt: &str, top_k: usize) -> Result<Readout, BackendError>;

    /// Read the answer-slot distribution for N prompts in one shot.
    ///
    /// The returned vector has exactly one [`Readout`] per input prompt, in the
    /// same order as `prompts` (never a partial result: either every prompt got a
    /// readout or the call failed). The `state` prefix the prompts share is what
    /// makes this cheap — a backend that can prefill it once (SGLang's radix
    /// cache, a local prefill-only backend) amortises the prefix across all N.
    ///
    /// The default implementation is a sequential loop over
    /// [`token_logprobs`](Self::token_logprobs): correct, backend-agnostic, and
    /// exactly as slow as N single requests. Backends that can batch (the
    /// OpenAI-compatible one sends `prompt` as an array) override it.
    async fn token_logprobs_batch(
        &self,
        prompts: &[String],
        top_k: usize,
    ) -> Result<Vec<Readout>, BackendError> {
        let mut readouts = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            readouts.push(self.token_logprobs(prompt, top_k).await?);
        }
        Ok(readouts)
    }

    /// Identifier surfaced in the response's `model` field (kept stable for eval).
    fn model_name(&self) -> String;
}
